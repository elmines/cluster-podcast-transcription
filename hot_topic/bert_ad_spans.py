#!/usr/bin/env python3

import argparse
import csv
import json
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from .silver_label import NOISE_FIELD, NOISE_LABEL, SOURCE_FIELD


def batched(iterable, size):
    iterator = iter(iterable)
    while batch := list(islice(iterator, size)):
        yield batch


def noise_class_index(config) -> int:
    id2label = getattr(config, "id2label", None) or {}
    for key, name in id2label.items():
        if str(name).lower() == NOISE_LABEL:
            return int(key)
    raise ValueError(f"Model config has no {NOISE_LABEL!r} label in id2label={dict(id2label)}")


def length_sort_indices(samples):
    """Descending token-length order, so each batch needs less padding."""
    seq_lens = np.array([len(sample["input_ids"]) for sample in samples])
    return np.flip(np.argsort(seq_lens)).tolist()


def predicted_spans(rows, predictions, positive_index: int, false_positives: bool = False):
    """Return contiguous half-open spans for lines predicted as noise."""
    spans = []
    start = None
    for index, (row, prediction) in enumerate(zip(rows, predictions, strict=True)):
        positive = int(prediction) == positive_index
        if false_positives:
            positive = positive and int(row[NOISE_FIELD]) == 0
        if positive and start is None:
            start = index
        elif not positive and start is not None:
            spans.append({"start": start, "end": index})
            start = None
    if start is not None:
        spans.append({"start": start, "end": len(rows)})
    return spans


def score_rows(model, tokenizer, collator, rows, batch_size, max_length, noise_index, device):
    tokenized = [
        tokenizer(row["text"], truncation=True, max_length=max_length)
        for row in rows
    ]
    sort_inds = length_sort_indices(tokenized)
    tokenized = [tokenized[index] for index in sort_inds]

    predictions = []
    for samples in batched(tokenized, batch_size):
        encoded = collator(list(samples))
        batch = {key: value.to(device) for key, value in encoded.items()}
        predictions.extend(model(**batch).logits.argmax(dim=-1).tolist())

    unsorted_predictions = [None] * len(predictions)
    for sorted_index, original_index in enumerate(sort_inds):
        unsorted_predictions[original_index] = predictions[sorted_index]
    return unsorted_predictions


def input_paths(root: Path, inputs) -> list[Path]:
    root = root.resolve()
    paths = []
    for input_path in inputs:
        path = input_path.resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"--input file is outside --root: {input_path}")
        if not path.is_file():
            raise ValueError(f"--input file does not exist: {path}")
        paths.append(path)
    return paths


def output_path(root: Path, output_dir: Path, input_path: Path) -> Path:
    return (output_dir / input_path.relative_to(root)).with_suffix(".json")


def read_rows(path: Path, false_positives: bool = False):
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            return [], None
        required = {"text"}
        if false_positives:
            required.update({NOISE_FIELD, SOURCE_FIELD})
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        return list(reader), reader.fieldnames


def result_for_rows(rows, predictions, positive_index, false_positives, source_path=None):
    source = rows[0].get(SOURCE_FIELD) if rows else None
    if not source:
        source = str(source_path) if source_path is not None else ""
    return {
        "source": source,
        "line_count": len(rows),
        "spans": predicted_spans(rows, predictions, positive_index, false_positives),
    }


def main(raw_args=None):
    parser = argparse.ArgumentParser(
        description="Extract BERT-predicted advertising spans from silver-labeled transcript CSVs."
    )
    parser.add_argument("-i", "--input", dest="inputs", nargs="+", required=True, type=Path,
                        help="Silver-labeled transcript CSV files under --root")
    parser.add_argument("--root", default=Path("out/silver_labels"), type=Path,
                        help="Root directory under which every --input file must be found")
    parser.add_argument("-o", "--output", default="out/bert_ad_spans", type=Path,
                        help="Directory for JSON outputs, mirroring --input")
    parser.add_argument("--model", default="out/noise_classifier", type=Path,
                        help="Directory saved by hot_topic.train_classifier")
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--max-length", default=256, type=int)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--false-positives",
        action="store_true",
        help="Only write BERT-positive spans whose lines are not silver-labeled as noise",
    )
    args = parser.parse_args(raw_args)

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.max_length < 1:
        parser.error("--max-length must be positive")

    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"--root is not a directory: {root}")
    model_path = args.model.resolve()
    if not model_path.is_dir():
        parser.error(f"--model is not a directory: {model_path}")
    output_dir = args.output.resolve()
    if output_dir == root or output_dir.is_relative_to(root) or root.is_relative_to(output_dir):
        parser.error("--output must be a different directory from --root")
    try:
        paths = input_paths(root, args.inputs)
    except ValueError as exc:
        parser.error(str(exc))

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
    collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")

    written = 0
    with torch.inference_mode():
        for path in tqdm(paths, desc="Writing transcripts"):
            try:
                rows, _ = read_rows(path, args.false_positives)
            except ValueError as exc:
                parser.error(str(exc))
            predictions = score_rows(
                model, tokenizer, collator, rows, args.batch_size,
                args.max_length, noise_index, device,
            ) if rows else []
            result = result_for_rows(
                rows, predictions, noise_index, args.false_positives, source_path=path
            )
            destination = output_path(root, output_dir, path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("w", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2)
                stream.write("\n")
            written += 1
    print(f"Wrote {written} JSON files -> {output_dir}")


if __name__ == "__main__":
    main()