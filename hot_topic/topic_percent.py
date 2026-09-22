
import polars as pl
import glob
from tqdm import tqdm
import csv

if __name__ == "__main__":

    topic_names = [
        "politics","not_politics","entertainment","trade","psychology","art","technology","sports","astronomy","education","health","finance","environment","religion","history","outdoors","business","law","agriculture"
    ]
    field_names = ["filepath", "nrows"] + topic_names

    raw_rows = []
    thresh = 0.8
    file_paths = glob.glob("./out/line_scores/**/*.csv")
    for csv_path in tqdm(file_paths):
        result = pl \
            .scan_csv(csv_path) \
            .select(
                [pl.lit(csv_path).alias('filepath'), pl.len().alias('nrows')] + \
                [ ((pl.col(c) > thresh).sum() / pl.len()).alias(c) for c in topic_names]
            ) \
            .collect()
        raw_rows.append(result.row(0))
    with open('out/topic_percents.csv', 'w') as w:
        writer = csv.writer(w)
        writer.writerow(field_names)
        writer.writerows(raw_rows)




