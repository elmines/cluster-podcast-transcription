#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from .silver_label import NOISE_FIELD, NOISE_LABEL, NOT_NOISE_LABEL, transcript_paths


def class_balanced_split(labels, val_fraction: float, seed: int):
    """Keep every minority-class row, match that count from the majority class,
    then split each class so training and validation have the same count per class.
    """
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")

    labels = np.asarray(labels)
    classes = np.unique(labels)
    if len(classes) < 2:
        raise ValueError(f"Need at least two classes to split, found {classes.tolist()}")

    rng = np.random.default_rng(seed)
    per_class = []
    for cls in classes:
        indices = np.flatnonzero(labels == cls)
        if len(indices) < 2:
            raise ValueError(
                f"Class {int(cls)} has {len(indices)} row(s); need at least 2 for a class-balanced split"
            )
        per_class.append(indices)

    minority = min(len(indices) for indices in per_class)
    train_parts = []
    val_parts = []
    for indices in per_class:
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        shuffled = shuffled[:minority]
        n_val = int(round(len(shuffled) * val_fraction))
        n_val = min(max(n_val, 1), len(shuffled) - 1)
        val_parts.append(shuffled[:n_val])
        train_parts.append(shuffled[n_val:])

    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def load_noise_labels(root: Path):
    """Return labels in transcript-path order, plus each file's (path, start, count)."""
    paths = transcript_paths(root)
    if not paths:
        raise ValueError(f"no CSV files under {root}")
    labels = []
    spans = []
    for path in tqdm(paths, desc="Reading silver labels"):
        start = len(labels)
        with path.open(newline="") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None:
                spans.append((path, start, 0))
                continue
            if "text" not in reader.fieldnames or NOISE_FIELD not in reader.fieldnames:
                raise ValueError(f"{path} must have text and {NOISE_FIELD} columns")
            for row in reader:
                labels.append(int(row[NOISE_FIELD]))
        spans.append((path, start, len(labels) - start))
    if not labels:
        raise ValueError(f"{root} has no labeled lines")
    return np.asarray(labels, dtype=np.int8), spans


def load_texts_for_indices(spans, indices: np.ndarray) -> dict[int, str]:
    wanted = sorted(int(index) for index in indices)
    found = {}
    cursor = 0
    for path, start, count in tqdm(spans, desc="Loading split texts"):
        end = start + count
        while cursor < len(wanted) and wanted[cursor] < start:
            cursor += 1
        if count == 0 or cursor >= len(wanted) or wanted[cursor] >= end:
            continue
        needed = set()
        scan = cursor
        while scan < len(wanted) and wanted[scan] < end:
            needed.add(wanted[scan])
            scan += 1
        with path.open(newline="") as source:
            reader = csv.DictReader(source)
            for offset, row in enumerate(reader):
                index = start + offset
                if index in needed:
                    found[index] = row["text"]
        cursor = scan
        if len(found) == len(wanted):
            return found
    if len(found) != len(wanted):
        raise ValueError(f"Missing {len(wanted) - len(found)} rows while reloading silver labels")
    return found


def class_counts(labels):
    values, counts = np.unique(labels, return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}


def tokenize_texts(tokenizer, texts, max_length: int, batch_size: int = 1024):
    collected = None
    for start in tqdm(range(0, len(texts), batch_size), desc="Tokenizing"):
        batch = tokenizer(
            texts[start:start + batch_size],
            truncation=True,
            max_length=max_length,
        )
        if collected is None:
            collected = {key: [] for key in batch}
        for key, values in collected.items():
            values.extend(batch[key])
    return collected


class EncodedDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: values[index] for key, values in self.encodings.items()}
        item["labels"] = int(self.labels[index])
        return item


def main(raw_args=None):
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune an AutoModelForSequenceClassification model on silver noise labels. "
            "Training and validation are class-balanced, and training stops early on validation loss."
        )
    )
    parser.add_argument("-i", "--input", default="out/silver_labels", type=Path,
                        help="Directory of per-transcript silver-label CSVs")
    parser.add_argument("-o", "--output", default="out/noise_classifier", type=Path,
                        help="Directory to save the best model and tokenizer")
    parser.add_argument("--model", default="answerdotai/ModernBERT-base",
                        help="Any Hugging Face checkpoint AutoModelForSequenceClassification can load")
    parser.add_argument("--val-fraction", default=0.2, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--epochs", default=10, type=int, help="Maximum epochs before early stopping")
    parser.add_argument("--patience", default=3, type=int,
                        help="Stop after this many validations without improved validation loss")
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--learning-rate", default=2e-5, type=float)
    parser.add_argument("--max-length", default=256, type=int)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args(raw_args)

    if not 0 < args.val_fraction < 1:
        parser.error("--val-fraction must be between 0 and 1")
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.patience < 1:
        parser.error("--patience must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.max_length < 1:
        parser.error("--max-length must be positive")

    input_dir = args.input.resolve()
    if not input_dir.is_dir():
        parser.error(f"--input is not a directory: {input_dir}")

    try:
        labels, spans = load_noise_labels(input_dir)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Loaded {len(labels)} lines with class counts {class_counts(labels)}")
    try:
        train_idx, val_idx = class_balanced_split(labels, args.val_fraction, args.seed)
    except ValueError as exc:
        parser.error(str(exc))

    texts = load_texts_for_indices(spans, np.concatenate((train_idx, val_idx)))
    train_texts = [texts[int(index)] for index in train_idx]
    val_texts = [texts[int(index)] for index in val_idx]
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    print(
        f"Class-balanced split: train {len(train_labels)} {class_counts(train_labels)}, "
        f"validation {len(val_labels)} {class_counts(val_labels)}"
    )

    id2label = {0: NOT_NOISE_LABEL, 1: NOISE_LABEL}
    label2id = {NOT_NOISE_LABEL: 0, NOISE_LABEL: 1}
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        problem_type="single_label_classification",
        ignore_mismatched_sizes=True,
        trust_remote_code=args.trust_remote_code,
    )

    train_dataset = EncodedDataset(tokenize_texts(tokenizer, train_texts, args.max_length), train_labels)
    val_dataset = EncodedDataset(tokenize_texts(tokenizer, val_texts, args.max_length), val_labels)

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        bf16=use_bf16,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    print(f"Saved best model to {output_dir}")


if __name__ == "__main__":
    main()
