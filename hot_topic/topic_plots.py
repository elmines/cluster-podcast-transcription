import argparse
import re
from pathlib import Path

import plotly.express as px
import polars as pl
from tqdm import tqdm


def _plot_filename(topic_name: str) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", topic_name).strip("._")
    return f"politics_vs_{safe_name or 'topic'}.png"


def make_topic_plots(input_path: str | Path, output_dir: str | Path, min_rows: int = 0) -> list[Path]:
    all_data = pl.read_csv(input_path)
    total_episodes = all_data.height
    data = all_data.filter(pl.col("nrows") >= min_rows)
    remaining_episodes = data.height
    topic_names = [
        name for name in data.columns
        if name not in {"filepath", "nrows", "politics"}
    ]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_paths = []
    for topic_name in tqdm(topic_names, desc='Processing topics'):
        title = f"Politics vs. {topic_name}"
        if remaining_episodes < total_episodes:
            title += f" ({remaining_episodes} / {total_episodes} Episodes)"
        figure = px.scatter(
            data,
            x="politics",
            y=topic_name,
            range_x=[0, 1],
            range_y=[0, 1],
            hover_data=["filepath", "nrows"],
            labels={"politics": "Politics", topic_name: topic_name},
            title=title,
        )
        plot_path = output_path / _plot_filename(topic_name)
        figure.write_image(plot_path)
        plot_paths.append(plot_path)
    return plot_paths


def main(raw_args=None):
    parser = argparse.ArgumentParser(description="Plot topic percentages against politics")
    parser.add_argument("-i", "--input", default="out/topic_percents.csv",
                        help="Topic percentages CSV path")
    parser.add_argument("-o", "--output", default="out/topic_plots/",
                        help="Output directory for Plotly PNG files")
    parser.add_argument("--min-rows", type=int, default=0,
                        help="Exclude entries with fewer than this many rows")
    args = parser.parse_args(raw_args)

    make_topic_plots(args.input, args.output, args.min_rows)


if __name__ == "__main__":
    main()