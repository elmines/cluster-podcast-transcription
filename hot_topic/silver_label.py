#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

from tqdm import tqdm

from .constants.blacklist import AD_PATT


SOURCE_FIELD = "source_csv"
NOISE_FIELD = "noise"
CONFIDENCE_FIELD = "confidence"

NOISE_LABEL = "noise"
NOT_NOISE_LABEL = "not_noise"


def is_noise(text: str) -> bool:
    return AD_PATT.search(text) is not None


def transcript_paths(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.csv") if path.is_file())


def silver_output_path(root: Path, output_dir: Path, transcript: Path) -> Path:
    return output_dir / transcript.relative_to(root)


def label_fieldnames(original_fields: list[str]) -> list[str]:
    fieldnames = [SOURCE_FIELD, *original_fields]
    if NOISE_FIELD not in fieldnames:
        fieldnames.append(NOISE_FIELD)
    return fieldnames


def main(raw_args=None):
    parser = argparse.ArgumentParser(
        description="Silver-label transcript lines as noise using the blacklist patterns."
    )
    parser.add_argument("--root", default="out/whisper_segmented", type=Path,
                        help="Directory of transcript CSVs (searched recursively)")
    parser.add_argument("-o", "--output", default="out/silver_labels", type=Path,
                        help="Directory for labeled CSVs, mirroring --root")
    parser.add_argument("-n", type=int, help="Label only the first N transcript files, in path order")
    args = parser.parse_args(raw_args)

    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"--root is not a directory: {root}")
    if args.n is not None and args.n < 1:
        parser.error("-n must be positive")

    output_dir = args.output.resolve()
    if output_dir.exists() and not output_dir.is_dir():
        parser.error(f"--output is not a directory: {output_dir}")
    if output_dir == root or output_dir.is_relative_to(root) or root.is_relative_to(output_dir):
        parser.error("--output must be a different directory from --root")

    paths = transcript_paths(root)
    if args.n is not None:
        paths = paths[:args.n]
    if not paths:
        parser.error(f"no CSV files under {root}")

    fieldnames = None
    line_count = 0
    noise_count = 0
    for path in tqdm(paths, desc="Silver-labeling transcripts"):
        with path.open(newline="") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None:
                continue
            if "text" not in reader.fieldnames:
                parser.error(f"{path} has no text column")
            original_fields = list(reader.fieldnames)
            file_fieldnames = label_fieldnames(original_fields)
            if fieldnames is None:
                fieldnames = file_fieldnames
            elif file_fieldnames != fieldnames:
                parser.error(f"{path} columns {original_fields} do not match {fieldnames}")

            destination = silver_output_path(root, output_dir, path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            source_csv = str(path.resolve())
            with destination.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for row in reader:
                    noise = int(is_noise(row["text"]))
                    row[SOURCE_FIELD] = source_csv
                    row[NOISE_FIELD] = noise
                    writer.writerow(row)
                    line_count += 1
                    noise_count += noise

    print(
        f"Labeled {line_count} lines from {len(paths)} files "
        f"(noise={noise_count}, not_noise={line_count - noise_count}) -> {output_dir}"
    )


if __name__ == "__main__":
    main()
