
import argparse
import glob
import csv
import polars as pl
from tqdm import tqdm


def main(raw_args=None):
    parser = argparse.ArgumentParser(description="Calculate topic percentages for scored line CSVs")
    parser.add_argument("-i", "--input", default="./out/line_scores/**/*.csv",
                        help="Glob for line score CSV files")
    parser.add_argument("-o", "--output", default="out/topic_percents.csv",
                        help="Output CSV path")
    parser.add_argument("--thresh", type=float, default=0.8,
                        help="Score threshold for counting a topic")
    args = parser.parse_args(raw_args)

    file_paths = glob.glob(args.input)
    topic_names = pl.scan_csv(file_paths[0]).collect_schema().names()
    field_names = ["filepath", "nrows"] + topic_names

    raw_rows = []
    for csv_path in tqdm(file_paths):
        result = pl \
            .scan_csv(csv_path) \
            .select(
                [pl.lit(csv_path).alias('filepath'), pl.len().alias('nrows')] + \
                [((pl.col(c) > args.thresh).sum() / pl.len()).alias(c) for c in topic_names]
            ) \
            .collect()
        raw_rows.append(result.row(0))
    with open(args.output, 'w') as w:
        writer = csv.writer(w)
        writer.writerow(field_names)
        writer.writerows(raw_rows)


if __name__ == "__main__":
    main()




