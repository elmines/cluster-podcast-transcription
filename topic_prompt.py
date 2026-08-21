#!/usr/bin/env python3

import argparse
import csv
import glob
import os
import re

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import xgrammar as xgr

def main(raw_args=None):
    parser = argparse.ArgumentParser(description="Generate candidate topics")
    parser.add_argument("-i", type=os.path.abspath, default="out/resegmented")
    parser.add_argument("-o", type=os.path.abspath, default="out/generated_topics.csv")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")

    args = parser.parse_args(raw_args)
    data_dir = args.i
    out_path = args.o

    model_name = args.model
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    system_prompt = r"""
Answer with one topic and quotation from the transcript per line.
The topic should be 1 to 30 characters.
The substring should be 1 to 256 characters.
Separate the two with a colon.
Do not add quote marks to your substring
"""

    standard_messages = [
        {"role": "system", "content": system_prompt},
    ]


    file_names = []
    texts = []

    # Sample one episode from each show
    show_dirs = [d_path for d_path in glob.glob(os.path.join(data_dir, "*")) if os.path.isdir(d_path)]
    for show_dir in tqdm(show_dirs, desc="Processing shows"):
        for csv_path in glob.glob(os.path.join(show_dir, "*.csv")):
            with open(csv_path, 'r') as r:
                texts.append("".join(row['text'] for row in csv.DictReader(r)))
            file_names.append(csv_path)

            # Debug statement
            break


    output_pattern = re.compile('([a-z ]+) : (.+)')

    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)

    grammar = xgr.GrammarCompiler(tokenizer_info).compile_grammar(r"""
root ::=  () | entry ("\n" entry){0,2}
entry ::= [a-z][a-z ]{0,29} " : " [^\r\n]{1,512}
topic ::= [a-z][a-z ]{0,29}
    """)


    out_rows = [

    ]

    orig_topics = ["politics", "business", "sports", "entertainment"]
    cur_topics = sorted(orig_topics)
    set_topics = set(cur_topics)
    text_iter = tqdm(texts, desc="Processing texts")
    for file_path, text in zip(file_names, text_iter):
        user_prefix = f"""
You are tasked with performing topic modeling over podcast transcripts.
Here are the topics you've discovered so far:
{','.join(cur_topics)}

Given a podcast transcript, generate a list of no more than 3 topics.
Each topic should be on a separate line and be followed by a colon
and a substring of the transcript as your justification for the topic.

Consider the sample podcast transcript:
\"I really want to take a Mediterranean cruise. There's so much history on the coasts of Greece and Turkey. And we can enjoy some good ouzo.\"

Your output could be as follows:
travel : want to take a Mediterranean cruise
history : There's so much history
food and drink : enjoy some good ouzo

Now here's the actual transcript:
"""


        tokenized_prompt = tokenizer.apply_chat_template(standard_messages + [
            {"role": "user", "content": user_prefix + "\n" + text}
        ],
        tokenize=True, add_generation_prompt=True, return_tensors='pt').to(model.device)
        output = model.generate(**tokenized_prompt,
                                logits_processor=[xgr.contrib.hf.LogitsProcessor(grammar)],
                                max_new_tokens=2048
        )
        input_length = tokenized_prompt['input_ids'].shape[-1]
        decoded = tokenizer.decode(output[0][input_length:], skip_special_tokens=True)

        # The quote they give must be from the text itself--avoid hallucinations
        matches = output_pattern.findall(decoded)
        # We ignore quotes that were hallucinated
        valid_matches = [ (topic, quote) for (topic, quote) in matches if quote in text]
        set_topics = set_topics | {t for t, _ in valid_matches}
        cur_topics = sorted(set_topics)

        for topic, quote in valid_matches:
            out_rows.append(
                (file_path, topic, quote)
            )

    fields = ["episode_file", "topic", "episode_quote"]
    with open(out_path, 'w') as w:
        writer = csv.writer(w)
        writer.writerow(fields)
        writer.writerows(out_rows)

if __name__ == "__main__":
    main()



