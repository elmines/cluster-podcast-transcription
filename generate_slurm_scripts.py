#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import shlex
import stat


def chmodx(out_path):
    os.chmod(out_path, os.stat(out_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def write_code(out_path, bash_code):
    with open(out_path, "w") as handle:
        handle.write(bash_code)
    chmodx(out_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate balanced Slurm scripts for whisper-cli transcription jobs."
    )
    parser.add_argument("num_jobs", type=int, help="Number of Slurm jobs to generate")
    parser.add_argument(
        "input_dir",
        help="Root directory containing nested MP3 files to transcribe",
    )
    return parser.parse_args()


def load_config(repo_dir):
    with (repo_dir / "config.json").open() as handle:
        config = json.load(handle)
    return config["email"], config["l4_partition"], config["whisper_root"]


def discover_mp3_files(input_dir):
    discovered = []
    for path in input_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() == ".mp3":
            discovered.append(path)
    return sorted(discovered)


def output_csv_path(out_dir, input_dir, mp3_path):
    relative_path = mp3_path.relative_to(input_dir)
    return out_dir / f"{relative_path}.csv"


def balance_files(files, num_jobs):
    buckets = [{"files": [], "bytes": 0} for _ in range(num_jobs)]
    for mp3_path, size_bytes, output_path in sorted(files, key=lambda item: (-item[1], str(item[0]))):
        bucket = min(buckets, key=lambda item: (item["bytes"], len(item["files"])))
        bucket["files"].append((mp3_path, size_bytes, output_path))
        bucket["bytes"] += size_bytes
    return buckets


def shell_quote(value):
    return shlex.quote(str(value))


def build_command_block(job_files, repo_dir, whisper_root):
    if not job_files:
        return "echo 'No MP3 files assigned to this job.'"

    lines = [
        f"WHISPER_ROOT={shell_quote(whisper_root)}",
        f"REPO_DIR={shell_quote(repo_dir)}",
        'WHISPER_BIN="$WHISPER_ROOT/build/bin/whisper-cli"',
        'WHISPER_MODEL="$WHISPER_ROOT/models/ggml-medium.bin"',
        'cd "$REPO_DIR"',
        'mp3_files=(',
    ]

    for mp3_path, _size_bytes, output_path in job_files:
        lines.append(f"  {shell_quote(mp3_path)}")

    lines.extend(
        [
            ")",
            "output_files=(",
        ]
    )

    for _mp3_path, _size_bytes, output_path in job_files:
        lines.append(f"  {shell_quote(output_path)}")

    lines.extend(
        [
            ")",
            'for index in "${!mp3_files[@]}"; do',
            '  source="${mp3_files[$index]}"',
            '  destination="${output_files[$index]}"',
            '  csv_name="$(basename \"$source\").csv"',
            '  mkdir -p "$(dirname \"$destination\")"',
            '  if [ -f "$destination" ]; then',
            '    echo "Skipping existing $destination"',
            '    continue',
            '  fi',
            '  echo "Transcribing $source"',
            '  rm -f "$csv_name"',
            '  "$WHISPER_BIN" --beam-size 1 -ocsv -np --model "$WHISPER_MODEL" "$source" > /dev/null',
            '  if [ -f "$csv_name" ]; then',
            '    mv "$csv_name" "$destination"',
            '  else',
            '    echo "Missing csv output for $source"',
            '  fi',
            'done',
        ]
    )

    return "\n".join(lines)


slurm_template = """#!/bin/bash

#SBATCH --time=3:00:00
#SBATCH --job-name={name}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16gb
#SBATCH --mail-user={user_email}
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

export XDG_RUNTIME_DIR=$SLURM_TMPDIR
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
module load cuda/12.4.1 gcc/12.2.0
date
hostname
cd {repo_dir}
pwd

{command}
"""


def main():
    args = parse_args()
    if args.num_jobs <= 0:
        raise SystemExit("num_jobs must be a positive integer")

    repo_dir = Path(__file__).resolve().parent
    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"input_dir does not exist or is not a directory: {input_dir}")

    out_dir = repo_dir / "out"
    slurm_dir = repo_dir / "slurm_scripts"
    out_dir.mkdir(parents=True, exist_ok=True)
    slurm_dir.mkdir(parents=True, exist_ok=True)

    for existing_script in slurm_dir.glob("*.sh"):
        existing_script.unlink()

    user_email, l4_part, whisper_root = load_config(repo_dir)

    discovered = discover_mp3_files(input_dir)
    pending = []
    for mp3_path in discovered:
        output_path = output_csv_path(out_dir, input_dir, mp3_path)
        pending.append((mp3_path, mp3_path.stat().st_size, output_path))

    buckets = balance_files(pending, args.num_jobs)
    name_width = max(2, len(str(args.num_jobs)))

    for index, bucket in enumerate(buckets, start=1):
        job_name = f"whisper_{index:0{name_width}d}_of_{args.num_jobs:0{name_width}d}"
        script_path = slurm_dir / f"{job_name}.sh"
        command = build_command_block(bucket["files"], repo_dir, whisper_root)
        script = slurm_template.format(
            name=job_name,
            partition=l4_part,
            user_email=user_email,
            repo_dir=repo_dir,
            command=command,
        )
        write_code(script_path, script)

    print(f"Discovered MP3 files: {len(discovered)}")
    print(f"Queued for generation: {len(pending)}")
    for index, bucket in enumerate(buckets, start=1):
        print(f"Job {index:0{name_width}d}: {len(bucket['files'])} files, {bucket['bytes']} bytes")
    print(f"Wrote scripts to: {slurm_dir}")


if __name__ == "__main__":
    main()


