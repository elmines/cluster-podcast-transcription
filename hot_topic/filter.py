import argparse
import os
from itertools import combinations, batched

from tqdm import tqdm
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
    parser.add_argument("--redund-thresh",
                        type=float,
                        default=0.5,
                        help="Minimum cosine similarity to merge redundant topics. Set to -1 to ignore"
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
    out_path = args.o
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    model_name = args.model
    quote_thresh = args.quote_thresh
    redund_thresh = args.redund_thresh
    embed_batch_size = args.embed_batch_size
    sim_batch_size = args.sim_batch_size

    
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

    # For now doing a crude all-pairs comparison
    if (redund_thresh > -1):
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


        reduced_topics = set()
        for subset in ds.subsets():
            str_subset = [topics_by_inds[i] for i in subset]
            # Use the shortest topic name as the rep
            simple_name = min(str_subset, key=len)
            reduced_topics.add(simple_name)
            if len(str_subset) > 1:
                print(f"Final set: {str_subset} --> {simple_name}")

        old_len = len(topics_df)
        topics_df = topics_df[topics_df['topic'].apply(reduced_topics.__contains__)]
        if (filtered_out := old_len - len(topics_df)):
            print(f"Filtered out {filtered_out} topics based on embedding similarity")

    topics_df.to_csv(out_path)


if __name__ == "__main__":
    main()