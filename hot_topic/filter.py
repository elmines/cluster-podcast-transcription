import argparse
import os

import pandas as pd

def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i-quotes", default="out/gen/scored_topic_quotes.csv", type=os.path.abspath)
    parser.add_argument("-i", default="out/gen/topics.csv", type=os.path.abspath)
    parser.add_argument("-o", default="out/gen/filtered_topics.csv", type=os.path.abspath)

    parser.add_argument("--quote-thresh",
                        type=float,
                        default=0.15,
                        help="Minimum quote quality relevance score to keep the quote. Set to -1 to ignore")

    args = parser.parse_args(raw_args)
    quotes_path = args.i_quotes
    topics_path = args.i
    out_path = args.o
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    quote_thresh = args.quote_thresh

    if quote_thresh > -1:
        quote_score_col = 'quote_score'
        quotes_df = pd.read_csv(quotes_path)
        if quote_score_col not in quotes_df.columns:
            raise ValueError(f"{quotes_path} missing {quote_score_col}")
        old_len = len(quotes_df)
        quotes_df = quotes_df[quotes_df[quote_score_col] >= quote_thresh]
        if (filtered_out := old_len - len(quotes_df)):
            print(f"Filtered out {filtered_out} quotes which fell below the threshold {quote_thresh}")

    attributed_topics = set(quotes_df['topic'])
    topics_df = pd.read_csv(topics_path)
    old_len = len(topics_df)
    topics_df = topics_df[topics_df['topic'].apply(attributed_topics.__contains__)]
    if (filtered_out := old_len - len(topics_df)):
        print(f"Filtered out {filtered_out} topics which have no attributable quote")

    topics_df.to_csv(out_path)


if __name__ == "__main__":
    main()