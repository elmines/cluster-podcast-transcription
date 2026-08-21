#!/usr/bin/env python3

import csv
import os
from itertools import batched
import argparse

import torch
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity
from tqdm import tqdm

def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", default="out/generated_topics.csv", type=os.path.abspath)
    parser.add_argument("-o", default="out/scored_topics.csv", type=os.path.abspath)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")

    args = parser.parse_args(raw_args)
    in_path = args.i
    out_path = args.o
    model_name = args.model 
    batch_size = 32

    model = SentenceTransformer(model_name).cuda()
    with open(in_path, 'r') as r:
        reader = csv.DictReader(r)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)
    row_iter = tqdm(rows, desc="Processing rows")
    sim_scores = []
    with torch.no_grad():
        for batch in batched(row_iter, batch_size):
            topics = [row['topic'] for row in batch]
            quotes = [row['episode_quote'] for row in batch]

            all_embeddings = torch.tensor(
                model.encode(topics + quotes),
                device=model.device
            )
            sim_scores.extend( cosine_similarity(all_embeddings[:len(batch)], all_embeddings[len(batch):]).detach().cpu().tolist() )
    with open(out_path, 'w') as w:
        writer = csv.DictWriter(w, fieldnames=fieldnames + ['topic_score'])
        writer.writeheader()
        writer.writerows([{**row, 'topic_score': score} for row, score in zip(rows, sim_scores)])


if __name__ == "__main__":
    main()