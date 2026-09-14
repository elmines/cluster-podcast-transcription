#!/usr/bin/env python3

import argparse
import csv
import os
from itertools import batched, product
from pathlib import Path
from collections import OrderedDict

from line_profiler import profile

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


POLITICS_TOPIC = "politics"
NOT_POLITICS_TOPIC = "not_politics"

DEFAULT_TEMPLATE = "This sentence is related to {}"
NON_POLI_PROMPT = "This sentence is not related to politics"

@profile
def main(raw_args=None):
    parser = argparse.ArgumentParser(description="Score transcript lines against generated topics")
    parser.add_argument("-i", default="out/resegmented", type=os.path.abspath)
    parser.add_argument("-t", "--topics", required=True, type=os.path.abspath)
    parser.add_argument("-o", default="out/line_scores", type=os.path.abspath)
    parser.add_argument("--model", default="MoritzLaurer/ModernBERT-large-zeroshot-v2.0")
    parser.add_argument("--prec", default=6, type=int, help="Precision in digits to which to round ")
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument('-n', type=int)
    args = parser.parse_args(raw_args)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    topic_path = args.topics
    batch_size = args.batch_size
    digits_prec = args.prec
    in_dir = args.i
    out_dir = args.o 
    n = args.n
    score_format = '{:' + str(digits_prec) + "f}"

    print(f"PROCESS: {os.getpid()}")

    topics_to_prompts = OrderedDict()
    topics_to_prompts[POLITICS_TOPIC] = DEFAULT_TEMPLATE.format(POLITICS_TOPIC)
    topics_to_prompts[NOT_POLITICS_TOPIC] = NON_POLI_PROMPT
    with open(topic_path, newline="") as source:
        topics  = map(lambda     x: x['topic'].strip(), csv.DictReader(source))
        entries = map(lambda topic: (topic, DEFAULT_TEMPLATE.format(topic)), topics)
        topics_to_prompts.update(entries)
    topic_count = len(topics_to_prompts)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    for index, label in getattr(model.config, "id2label", {}).items():
        if "entail" in str(label).lower():
            entailment_label = int(index)
            break
    else:
        raise ValueError("Could not identify the entailment label in the NLI model")

    device = next(model.parameters()).device

    input_paths = sorted(Path(in_dir).glob("**/*.csv"))
    if n is not None:
        input_paths = input_paths[:n]
    for input_path in tqdm(input_paths, desc="Scoring transcript lines"):
        relative_path = input_path.relative_to(in_dir)
        output_path = os.path.join(out_dir, relative_path)

        with open(input_path, newline="") as source:
            rows = list(csv.DictReader(source))
        lines = [row["text"] for row in rows]

        pairs = list(product(lines, topics_to_prompts.values()))
        scores = []
        with torch.inference_mode():
            for pair_batch in batched(pairs, batch_size):
                premises, hypotheses = zip(*pair_batch)
                inputs = tokenizer(
                    list(premises),
                    list(hypotheses),
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(device)
                logits = model(**inputs).logits

                entailment = logits[:, entailment_label]
                # Combines all the logits for non-entailment classes into 1 logit representing non-entailment
                non_entailment = torch.logsumexp(
                    torch.cat((logits[:, :entailment_label], logits[:, entailment_label + 1:]), dim=1),
                    dim=1,
                )
                batch_scores = torch.softmax(torch.stack((non_entailment, entailment), dim=1), dim=1)[:, 1]

                cpu_scores = batch_scores.detach()
                cpu_scores = batch_scores.cpu()
                cpu_scores = batch_scores.tolist()
                scores.extend(cpu_scores)

        # Makes our CSV file easier to read
        # The format command does rounding as needed
        scores = [ score_format.format(s) for s in scores]
    
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", newline="") as destination:
            writer = csv.writer(destination)
            writer.writerow(list(topics_to_prompts))
            writer.writerows([
                scores[offset:offset + topic_count]
                for offset in range(0, len(scores), topic_count)
            ])


if __name__ == "__main__":
    main()