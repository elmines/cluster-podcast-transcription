import argparse
import os
from itertools import combinations, batched
from collections import defaultdict
import json
import pdb

from tqdm import tqdm
import pandas as pd

def filter_by_freq(topics_df: pd.DataFrame, quotes_df: pd.DataFrame, min_freq: int) -> pd.DataFrame:
    topic_counts = quotes_df['topic'].value_counts()
    new_topics_df = topics_df[topics_df['topic'].apply(lambda t: topic_counts[t] >= min_freq)]
    if (rem_topics := len(topics_df) - len(new_topics_df)):
        print(f"Filtered out {rem_topics} topics with frequency < {min_freq}")
        avail_topics = set(new_topics_df['topic'])
        new_quotes_df = quotes_df[quotes_df['topic'].apply(avail_topics.__contains__)]
        if (rem_quotes := len(quotes_df) - len(new_quotes_df)):
            print(f"Filtered out {rem_quotes} quotes for removed topics")
    else:
        new_quotes_df = quotes_df
    return new_topics_df, new_quotes_df

def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i-quotes", default="out/gen/scored_topic_quotes.csv", type=os.path.abspath)
    parser.add_argument("-i", default="out/gen/topics.csv", type=os.path.abspath)

    parser.add_argument("-o", default="out/gen/", type=os.path.abspath)

    parser.add_argument("--min-freq",
                        type=int,
                        default=2,
                        help="Minimum topic occurrence frequency PRIOR to merging. Set to 0 to ignore")
    parser.add_argument("--min-freq-postmerge",
                        type=int,
                        default=0,
                        help="Minimum topic occurrence frequence AFTER merging. Set to 0 to ignore")

    parser.add_argument("--quote-thresh",
                        type=float,
                        default=0.15,
                        help="Minimum quote quality relevance score to keep the quote. Set to -1 to ignore")
    parser.add_argument("--redund-thresh",
                        type=float,
                        default=0.5,
                        help="Minimum cosine similarity to merge redundant topics. Set to 1 to ignore"
    )
    parser.add_argument("--embed-batch-size",
                        type=int,
                        default=32,
                        help="Batch size for embedding topics")
    parser.add_argument("--sim-batch-size",
                        type=int,
                        default=256,
                        help="Batch size for computing cosine similarity between embeddings")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")

    args = parser.parse_args(raw_args)
    quotes_path = args.i_quotes
    topics_path = args.i
    out_dir = args.o
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    out_topics_path = os.path.join(out_dir, "filtered_topics.csv")
    out_quotes_path = os.path.join(out_dir, "filtered_quotes.csv")
    out_merges_path = os.path.join(out_dir, "topic_merges.json")
    model_name = args.model
    quote_thresh = args.quote_thresh
    redund_thresh = args.redund_thresh
    embed_batch_size = args.embed_batch_size
    sim_batch_size = args.sim_batch_size
    min_freq = args.min_freq
    min_freq_postmerge = args.min_freq_postmerge


    for op in [out_topics_path, out_quotes_path, out_merges_path]:
        os.makedirs(os.path.dirname(op), exist_ok=True)

    topics_df = pd.read_csv(topics_path)
    old_len = len(topics_df)
    topics_df.drop_duplicates(subset='topic', inplace=True)
    if (filtered_out := old_len - len(topics_df)):
        print(f"Filtered out {filtered_out} rows corresponding to duplicate topics")
    topics_df.set_index('topic')

    quotes_df = pd.read_csv(quotes_path)
    if quote_thresh > -1:
        quote_score_col = 'quote_score'
        if quote_score_col not in quotes_df.columns:
            raise ValueError(f"{quotes_path} missing {quote_score_col}")
        old_len = len(quotes_df)
        quotes_df = quotes_df[quotes_df[quote_score_col] >= quote_thresh]
        if (filtered_out := old_len - len(quotes_df)):
            print(f"Filtered out {filtered_out} quotes which fell below the threshold {quote_thresh}")

    # We're not entr
    old_len = len(quotes_df)
    quotes_df.drop_duplicates(subset=['episode_file', 'topic'], inplace=True)
    if (filtered_out := old_len - len(quotes_df)):
        print(f"Filtered out {filtered_out} quotes that were for the same topic in the same episode")

    attributed_topics = set(quotes_df['topic'])
    old_len = len(topics_df)
    topics_df = topics_df[topics_df['topic'].apply(attributed_topics.__contains__)]
    if (filtered_out := old_len - len(topics_df)):
        print(f"Filtered out {filtered_out} topics which have no attributable quote")

    if (min_freq > 0):
        topics_df, quotes_df = filter_by_freq(topics_df, quotes_df, min_freq)

    # For now doing a crude all-pairs comparison
    if (redund_thresh < 1):
        from sentence_transformers import SentenceTransformer
        from torch.nn.functional import cosine_similarity
        from scipy.cluster.hierarchy import DisjointSet
        import torch
        import numpy as np

        topic_strs = list(topics_df.apply(lambda row: row['topic'] + " : " + row['topic_desc'], axis='columns'))
        # sentence transformers returns numpy by default
        with torch.no_grad():
            model = SentenceTransformer(model_name).cuda()
            cuda_device = model.device
            topic_strs_iter = tqdm(topic_strs, desc="Computing topic embeddings")
            all_embeddings = torch.tensor(
                np.concatenate([model.encode(topic_batch) for topic_batch in batched(topic_strs_iter, embed_batch_size)]),
                device=cuda_device
            )
            model = None

            topic_inds = list(range(len(topic_strs)))
            ind_pairs = list(combinations(topic_inds, 2))
            sim_scores = []
            pair_iter = tqdm(ind_pairs, desc="Computing cosine similarities for embedding pairs")
            for pairs_batch in batched(pair_iter, sim_batch_size):
                sim_scores.append(
                    cosine_similarity(
                        all_embeddings[[i for i,_ in pairs_batch]],
                        all_embeddings[[j for _,j in pairs_batch]]
                    )
                )
            sim_scores = torch.concatenate(sim_scores).detach().cpu().numpy()

        topics_by_inds = topics_df['topic'].iloc
        ds = DisjointSet(topic_inds)
        meets_thresh = sim_scores >= redund_thresh
        inds_to_merge = np.array(ind_pairs)[meets_thresh]
        scores_above_thresh = sim_scores[meets_thresh]
        for (ind_a, ind_b), score in zip(inds_to_merge, scores_above_thresh):
            ds.merge(ind_a, ind_b)
            print(f"Merge: {score:.3f},{topics_by_inds[ind_a]},{topics_by_inds[ind_b]}")


        rep_to_members = dict()
        topic_remapping = dict()
        for subset in ds.subsets():
            str_subset = [topics_by_inds[i] for i in subset]
            # Use the shortest topic name as the rep
            simple_name = min(str_subset, key=len)
            topic_remapping.update({s:simple_name for s in str_subset})
            rep_to_members[simple_name] = str_subset
            if len(str_subset) > 1:
                print(f"Final set: {str_subset} --> {simple_name}")

        with open(out_merges_path, 'w') as w:
            json.dump(rep_to_members, w, indent=2)
        print(f"Wrote {out_merges_path}")

        reduced_topics = set(topic_remapping.values())
        old_len = len(topics_df)
        topics_df = topics_df[topics_df['topic'].apply(reduced_topics.__contains__)]
        if (filtered_out := old_len - len(topics_df)):
            print(f"Filtered out {filtered_out} topics based on embedding similarity")

        quotes_df['topic'] = quotes_df['topic'].apply(topic_remapping.__getitem__)

    if (min_freq_postmerge > 0):
        topics_df, quotes_df = filter_by_freq(topics_df, quotes_df, min_freq_postmerge)

    topics_df.to_csv(out_topics_path)
    print(f"Wrote {out_topics_path}")

    rem_topics = set(topics_df['topic'])
    old_len = len(quotes_df)
    quotes_df = quotes_df[quotes_df['topic'].apply(rem_topics.__contains__)]
    if (filtered_out := old_len - len(quotes_df)):
        print(f"Filtered out {filtered_out} quotes that no longer have a topic")
    quotes_df.to_csv(out_quotes_path)
    print(f"Wrote {out_quotes_path}")

if __name__ == "__main__":
    main()