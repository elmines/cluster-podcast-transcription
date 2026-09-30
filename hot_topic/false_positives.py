#!/usr/bin/env python3

import argparse
import csv
from itertools import batched
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from .silver_label import CONFIDENCE_FIELD, NOISE_FIELD, NOISE_LABEL, SOURCE_FIELD, transcript_paths


def noise_class_index(config) -> int:
    id2label = getattr(config, "id2label", None) or {}
    for key, name in id2label.items():
        if str(name).lower() == NOISE_LABEL:
            return int(key)
    raise ValueError(f"Model config has no {NOISE_LABEL!r} label in id2label={dict(id2label)}")


def original_fields(fieldnames) -> list[str]:
    added = {SOURCE_FIELD, NOISE_FIELD, CONFIDENCE_FIELD}
    return [name for name in fieldnames if name not in added]


def select_false_positives(rows, probabilities, predictions, positive_index: int):
    """Lines the model calls noise whose silver label is not noise."""
    chosen = []
    for row, probability, prediction in zip(rows, probabilities, predictions, strict=True):
        if int(row[NOISE_FIELD]) == 0 and int(prediction) == positive_index:
            chosen.append({**row, CONFIDENCE_FIELD: float(probability)})
    return chosen


def sort_by_confidence(rows):
    rows.sort(key=lambda row: row[CONFIDENCE_FIELD], reverse=True)
    return rows


def length_sort_indices(samples):
    """Descending token-length order, so each batch needs less padding."""
    seq_lens = np.array([len(sample["input_ids"]) for sample in samples])
    return np.flip(np.argsort(seq_lens)).tolist()


def score_rows(model, tokenizer, collator, rows, batch_size, max_length, noise_index, device):
    tokenized = [
        tokenizer(row["text"], truncation=True, max_length=max_length)
        for row in rows
    ]
    sort_inds = length_sort_indices(tokenized)
    tokenized = [tokenized[index] for index in sort_inds]
    rows = [rows[index] for index in sort_inds]

    probabilities = []
    predictions = []
    for samples in batched(tokenized, batch_size):
        encoded = collator(list(samples))
        batch = {key: value.to(device) for key, value in encoded.items()}
        logits = model(**batch).logits
        probs = torch.softmax(logits, dim=-1)
        probabilities.extend(probs[:, noise_index].tolist())
        predictions.extend(probs.argmax(dim=-1).tolist())
    return select_false_positives(rows, probabilities, predictions, noise_index)


def iter_silver_files(root: Path):
    paths = transcript_paths(root)
    if not paths:
        raise ValueError(f"no CSV files under {root}")
    fieldnames = None
    found_rows = False
    for path in tqdm(paths, desc='Processing files'):
        with path.open(newline="") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None:
                continue
            if "text" not in reader.fieldnames or NOISE_FIELD not in reader.fieldnames or SOURCE_FIELD not in reader.fieldnames:
                raise ValueError(f"{path} must have text, {SOURCE_FIELD}, and {NOISE_FIELD} columns")
            if fieldnames is None:
                fieldnames = list(reader.fieldnames)
            elif list(reader.fieldnames) != fieldnames:
                raise ValueError(f"{path} columns {list(reader.fieldnames)} do not match {fieldnames}")
            rows = list(reader)
            if rows:
                found_rows = True
                yield rows
    if not found_rows:
        raise ValueError(f"{root} has no labeled lines")


def silver_fieldnames(root: Path) -> list[str]:
    for path in transcript_paths(root):
        with path.open(newline="") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames:
                return list(reader.fieldnames)
    raise ValueError(f"{root} has no labeled lines")


def output_fieldnames(fields) -> list[str]:
    return [*fields, SOURCE_FIELD, NOISE_FIELD, CONFIDENCE_FIELD]


def write_false_positives(path: Path, rows, fields):
    fieldnames = output_fieldnames(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **{field: row.get(field, "") for field in fields},
                SOURCE_FIELD: row[SOURCE_FIELD],
                NOISE_FIELD: row[NOISE_FIELD],
                CONFIDENCE_FIELD: f"{row[CONFIDENCE_FIELD]:.6f}",
            })


def main(raw_args=None):
    parser = argparse.ArgumentParser(
        description=(
            "Score every silver-labeled line and write model false positives, "
            "highest confidence first. A false positive is a line predicted as noise "
            "whose silver label is not noise."
        )
    )
    parser.add_argument("-i", "--input", default="out/silver_labels", type=Path,
                        help="Directory of per-transcript silver-label CSVs")
    parser.add_argument("-o", "--output", default="out/ad_false_positives.csv", type=Path)
    parser.add_argument("--model", required=True, type=Path,
                        help="Directory saved by hot_topic.train_classifier")
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--max-length", default=256, type=int)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args(raw_args)

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.max_length < 1:
        parser.error("--max-length must be positive")

    input_dir = args.input.resolve()
    model_path = args.model.resolve()
    if not input_dir.is_dir():
        parser.error(f"--input is not a directory: {input_dir}")
    if not model_path.is_dir():
        parser.error(f"--model is not a directory: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=args.trust_remote_code)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        trust_remote_code=args.trust_remote_code,
    )
    try:
        noise_index = noise_class_index(model.config)
    except ValueError as exc:
        parser.error(str(exc))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    try:
        fields = original_fields(silver_fieldnames(input_dir))
    except ValueError as exc:
        parser.error(str(exc))

    collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")
    false_positives = []
    line_count = 0
    try:
        files = iter_silver_files(input_dir)
    except ValueError as exc:
        parser.error(str(exc))
    with torch.inference_mode():
        for rows in files:
            line_count += len(rows)
            false_positives.extend(score_rows(
                model,
                tokenizer,
                collator,
                rows,
                args.batch_size,
                args.max_length,
                noise_index,
                device,
            ))

    sort_by_confidence(false_positives)
    output_path = args.output.resolve()
    write_false_positives(output_path, false_positives, fields)
    print(f"Wrote {len(false_positives)} false positives from {line_count} lines -> {output_path}")


if __name__ == "__main__":
    main()
