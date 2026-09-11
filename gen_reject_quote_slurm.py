#!/usr/bin/env python3

import json
import os
from pathlib import Path
import shlex
import stat


JOBS = [
	("3:15:00", "meta-llama/Llama-3.3-70B-Instruct", "b200", "gpu:1"),
	("3:15:00", "openai/gpt-oss-120b"              , "b200", "gpu:1"),
	("3:15:00", "google/gemma-4-31B-it"            , "b200", "gpu:1"),
]


def shell_quote(value):
	return shlex.quote(str(value))


def normalize_model_name(model):
	return model.replace("/", "--")


def chmodx(out_path):
	os.chmod(out_path, os.stat(out_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def write_code(out_path, bash_code):
	with out_path.open("w") as handle:
		handle.write(bash_code)
	chmodx(out_path)


def load_config(repo_dir):
	with (repo_dir / "config.json").open() as handle:
		return json.load(handle)


def build_script(repo_dir, duration, partition, email, model, input_paths, gres):
	command = [
		"uv",
		"run",
		"python",
		"-m",
		"hot_topic.reject_quote",
		"-i",
		*(shell_quote(path) for path in input_paths),
		"--model",
		shell_quote(model),
	]
	command_str = " \\\n\t".join(command)

	return f"""#!/bin/bash

#SBATCH --time={shell_quote(duration)}
#SBATCH --job-name=quote_reject_{shell_quote(normalize_model_name(model))}
#SBATCH --partition={shell_quote(partition)}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres={shell_quote(gres)}
#SBATCH --mem=48gb
#SBATCH --mail-user={shell_quote(email)}
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

module load cuda/13.0.2 git
export XDG_RUNTIME_DIR=$SLURM_TMPDIR
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
date
hostname
cd {shell_quote(repo_dir)}
pwd

{command_str}
"""


def main():
	repo_dir = Path(__file__).resolve().parent
	config = load_config(repo_dir)
	partitions = {
		"rtx": config["rtx_partition"],
		"b200": config["b200_partition"],
	}
	input_paths = [
		repo_dir / "out" / f"gen--{normalize_model_name(model)}" / "topic_quotes.csv"
		for _, model, _, _ in JOBS
	]

	for duration, model, partition_name, gres in JOBS:
		model_dir_name = normalize_model_name(model)
		script_path = repo_dir / "slurm_scripts" / f"reject_quote_{model_dir_name}.sh"
		script_path.parent.mkdir(parents=True, exist_ok=True)

		script = build_script(
			repo_dir,
			duration,
			partitions[partition_name],
			config["email"],
			model,
			input_paths,
			gres,
		)
		write_code(script_path, script)
		print(f"Wrote script to: {script_path}")


if __name__ == "__main__":
	main()
