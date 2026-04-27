# cluster-podcast-transcription

Generate balanced Slurm jobs for transcribing nested MP3 collections with `whisper-cli`.

The metascript, [generate_slurm_scripts.py](generate_slurm_scripts.py), scans an input directory recursively for MP3 files, groups them into `N` Slurm scripts by file size, and writes the generated jobs into [slurm_scripts/](slurm_scripts/). Each generated job transcribes its assigned files and writes CSV output into [out/](out/) using the same directory structure as the input tree. If a CSV already exists, the generated job skips that file so interrupted runs can be resumed.

Example:

```bash
python generate_slurm_scripts.py 4 /path/to/nested/mp3s
```

Before running the generated jobs, make a [config.json](config.json) that points to your `whisper_root`, Slurm partition, and email address.

Example `config.json`:

```json
{
	"email": "your.name@example.edu",
	"l4_partition": "your-gpu-partition",
	"whisper_root": "/absolute/path/to/whisper.cpp"
}
```
The Slurm partition would compromise anonymity--ask the repo creator the name of the partition if you're trying to run this yourself.