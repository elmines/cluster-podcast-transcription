#!/usr/bin/env python3

import argparse
import csv
import glob
import os
import re

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import xgrammar as xgr

from .utils import partial_format, tokenized_with_trunc
from .constants import GEN_USER_PROMPT, DEFAULT_TOPICS

def format_topic(topic_name, topic_desc):
    return f"{topic_name} : {topic_desc}"

def main(raw_args=None):
    parser = argparse.ArgumentParser(description="Generate candidate topics")
    parser.add_argument("-i", type=os.path.abspath, default="out/resegmented")
    parser.add_argument("-o", type=os.path.abspath, default="out/generated_topics.csv")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")

    args = parser.parse_args(raw_args)
    data_dir = args.i
    out_path = args.o

    model_name = args.model
    max_new_tokens = 2048


    # Sample one episode from each show
    show_dirs = [d_path for d_path in glob.glob(os.path.join(data_dir, "*")) if os.path.isdir(d_path)]
    csvs_by_show_dir = {}
    for show_dir in tqdm(show_dirs, desc="Processing shows"):
        show_file_names = []
        for csv_path in glob.glob(os.path.join(show_dir, "*.csv")):
            show_file_names.append(csv_path)
        csvs_by_show_dir[show_dir] = show_file_names
        

    file_names = []
    # Round robin allocation of the csv paths
    while csvs_by_show_dir:
        for show in show_dirs:
            if (rem_paths := csvs_by_show_dir.get(show, [])):
                file_names.append(rem_paths.pop())
            else:
                csvs_by_show_dir.pop(show, None)

    file_names = file_names[:200]

    texts = []
    for csv_path in file_names:
        with open(csv_path, 'r') as r:
            texts.append("".join(row['text'] for row in csv.DictReader(r)))

    output_pattern = re.compile(r'\[1\] ([a-z ]+) : ([a-z ]+) : (.+)')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    grammar = xgr.GrammarCompiler(tokenizer_info).compile_grammar(r"""
root ::=  () | entry ("\n" entry){0,2}
entry ::= "[1] " [a-z][a-z ]{0,29} " : " [a-z][a-z ]{0,255} " : " [^\r\n]{1,512}
    """)
    system_prompt = r"""
Answer with one topic per line.
Use the following format:
[1] topic : topic desc : episode quote

The topic should be 1 to 30 characters.
The topic description should be 1 to 256 characters.
The episode quote should be 1 to 512 characters.
Do not add quote marks to your episode quote.
"""

    orig_topics = DEFAULT_TOPICS
    set_topics = {t for (t,*_) in orig_topics}
    topic_str = "\n".join( format_topic(name, desc) for name, desc in orig_topics)

    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    model_context_window = getattr(model.config, "max_position_embeddings", 96000)
    max_input = model_context_window - max_new_tokens

    out_rows = []
    text_iter = tqdm(texts, desc="Processing texts")
    for file_path, text in zip(file_names, text_iter):
        user_prefix = partial_format(GEN_USER_PROMPT, Topics=topic_str)
        tokenized_prompt = tokenized_with_trunc(tokenizer,
                                            [{"role": "system", "content": system_prompt}],
                                            user_prefix,
                                            text,
                                            max_input).to(model.device)
        output = model.generate(**tokenized_prompt,
                                logits_processor=[xgr.contrib.hf.LogitsProcessor(grammar)],
                                max_new_tokens=max_new_tokens)
        input_length = tokenized_prompt['input_ids'].shape[-1]
        decoded = tokenizer.decode(output[0][input_length:], skip_special_tokens=True)

        # The quote they give must be from the text itself--avoid hallucinations
        matches = output_pattern.findall(decoded)
        # We ignore quotes that were hallucinated
        valid_matches = [ (topic, desc, quote) for (topic, desc, quote) in matches if quote in text]
        out_rows.extend((file_path, *match_res) for match_res in valid_matches)
        if (new_topics := [(t, desc) for (t, desc, *_) in valid_matches if t not in set_topics]):
            set_topics.update(t for t, _ in new_topics)
            topic_str += "\n" + "\n".join(format_topic(name, desc) for name, desc in new_topics)


    fields = ["episode_file", "topic", "desc", "episode_quote"]
    with open(out_path, 'w') as w:
        writer = csv.writer(w)
        writer.writerow(fields)
        writer.writerows(out_rows)

if __name__ == "__main__":
    main()



