#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
import shlex
import stat


GPU_CONFIG = {
	"l4": ("l4_partition", "l4"),
	"rtx": ("rtx_partition", "rtx6000"),
}


def shell_quote(value):
	return shlex.quote(str(value))


def chmodx(out_path):
	os.chmod(out_path, os.stat(out_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def write_code(out_path, bash_code):
	with out_path.open("w") as handle:
		handle.write(bash_code)
	chmodx(out_path)


def parse_args():
	parser = argparse.ArgumentParser(
		description="Generate a Slurm script for candidate topic generation."
	)
	parser.add_argument("--gpu", choices=sorted(GPU_CONFIG), default="rtx", help="GPU kind: l4 or rtx")
	parser.add_argument("--duration", default="20:00:00", help="Slurm time limit, for example 3:00:00")
	parser.add_argument("--model", help="Hugging Face model name")
	return parser.parse_args()


def load_config(repo_dir):
	with (repo_dir / "config.json").open() as handle:
		return json.load(handle)


def build_script(repo_dir, duration, gpu, partition, email, model, output_dir):
	topic_output = output_dir / "topics.csv"
	quote_output = output_dir / "topic_quotes.csv"
	command = " \\\n\t".join(
		[
			"python",
			"-m",
			"hot_topic.gen",
			"-i",
			shell_quote(repo_dir / "resegmented"),
			"-o",
			shell_quote(topic_output),
			"--o-quote",
			shell_quote(quote_output),
			"--model",
			shell_quote(model),
		]
	)

	return f"""#!/bin/bash

#SBATCH --time={shell_quote(duration)}
#SBATCH --job-name=topic_disc_{shell_quote(model.replace('/', '-'))}
#SBATCH --partition={shell_quote(partition)}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:{gpu}:1
#SBATCH --mem=48gb
#SBATCH --mail-user={shell_quote(email)}
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

export XDG_RUNTIME_DIR=$SLURM_TMPDIR
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
date
hostname
cd {shell_quote(repo_dir)}
pwd

mkdir -p {shell_quote(output_dir)}
{command}
"""


def main():
	args = parse_args()
	repo_dir = Path(__file__).resolve().parent
	config = load_config(repo_dir)
	partition_key, gpu_name = GPU_CONFIG[args.gpu]
	partition = config[partition_key]
	model_dir_name = args.model.replace("/", "--")
	output_dir = repo_dir / f"{model_dir_name}-out"
	script_path = repo_dir / "slurm_scripts" / f"topic_disc_{model_dir_name}.sh"
	script_path.parent.mkdir(parents=True, exist_ok=True)

	script = build_script(
		repo_dir,
		args.duration,
		gpu_name,
		partition,
		config["email"],
		args.model,
		output_dir,
	)
	write_code(script_path, script)
	print(f"Wrote script to: {script_path}")
	print(f"Output directory: {output_dir}")


if __name__ == "__main__":
	main()
