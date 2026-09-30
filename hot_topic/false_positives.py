#!/usr/bin/env python3

import argparse
import csv
from itertools import batched
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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


def predict_batch(model, tokenizer, texts, max_length: int, noise_index: int, device):
    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    model_inputs = {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
    }
    logits = model(**model_inputs).logits
    probabilities = torch.softmax(logits, dim=-1)
    predictions = probabilities.argmax(dim=-1)
    return probabilities[:, noise_index].tolist(), predictions.tolist()


def iter_silver_rows(root: Path):
    paths = transcript_paths(root)
    if not paths:
        raise ValueError(f"no CSV files under {root}")
    fieldnames = None
    for path in paths:
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
            yield from reader
    if fieldnames is None:
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
    parser.add_argument("--batch-size", default=64, type=int)
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

    false_positives = []
    line_count = 0
    with torch.inference_mode():
        try:
            rows = iter_silver_rows(input_dir)
        except ValueError as exc:
            parser.error(str(exc))
        for batch in tqdm(batched(rows, args.batch_size), desc="Predicting noise"):
            batch = list(batch)
            line_count += len(batch)
            probabilities, predictions = predict_batch(
                model,
                tokenizer,
                [row["text"] for row in batch],
                args.max_length,
                noise_index,
                device,
            )
            false_positives.extend(
                select_false_positives(batch, probabilities, predictions, noise_index)
            )

    sort_by_confidence(false_positives)
    output_path = args.output.resolve()
    write_false_positives(output_path, false_positives, fields)
    print(f"Wrote {len(false_positives)} false positives from {line_count} lines -> {output_path}")


if __name__ == "__main__":
    main()
