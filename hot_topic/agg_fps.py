"""
Usage python -m hot_topic.agg_fps [out/ad_false_positives.csv] [out path or sys.stdout]
"""
import sys
import polars as pl

def main(raw_args=None):
    if raw_args is None:
        raw_args = sys.argv[1:]
    in_path = raw_args[0] if raw_args else "out/ad_false_positives.csv"
    out_path = raw_args[1] if len(raw_args) > 1 else "out/agg_false_positives.csv"
    pl.scan_csv(in_path) \
        .select([pl.col('text'), pl.col('confidence')]) \
        .group_by('text') \
        .agg([pl.len().alias('count'), pl.max('confidence')]) \
        .sort(['count', "confidence"], descending=True) \
        .collect() \
        .write_csv(out_path)

if __name__ == '__main__':
    main()