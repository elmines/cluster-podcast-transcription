#!/usr/bin/env python3

import argparse
import csv
import glob
import os
import re
from collections import OrderedDict

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import xgrammar as xgr

from .csv_writer import get_csv_writer
from .utils import partial_format, tokenized_with_trunc, read_transcription_text, preprocess
from .constants import GEN_USER_PROMPT, GEN_SYSTEM_PROMPT, DEFAULT_TOPICS, GEN_GRAMMAR, NOISE_PROMPT

def format_topic(topic_name, topic_desc):
    return f"{topic_name} : {topic_desc}"

def main(raw_args=None):
    parser = argparse.ArgumentParser(description="Generate candidate topics")
    parser.add_argument("-i", type=os.path.abspath, default="out/resegmented")
    parser.add_argument("--o-quote", type=os.path.abspath, default="out/gen/topic_quotes.csv")
    parser.add_argument("-o"      , type=os.path.abspath, default="out/gen/topics.csv")
    parser.add_argument("-n", type=int)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--buffer-size", default=8, type=int)

    args = parser.parse_args(raw_args)
    data_dir = args.i
    topic_path = args.o
    n = args.n
    quote_path = args.o_quote
    buffer_size = args.buffer_size
    os.makedirs(os.path.dirname(topic_path), exist_ok=True)
    os.makedirs(os.path.dirname(quote_path), exist_ok=True)


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

    if n:
        file_names = file_names[:n]

    texts = []
    for csv_path in tqdm(file_names, desc="Reading texts into memory"):
        texts.append(read_transcription_text(csv_path))
    texts = [preprocess(t) for t in tqdm(texts, desc="Stripping music markers from transcripts")]

    output_pattern = re.compile(r'\[1\] ([a-z ]+) : ([a-z ]+) : (.+)')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    grammar = xgr.GrammarCompiler(tokenizer_info).compile_grammar(GEN_GRAMMAR)
    system_prompt = GEN_SYSTEM_PROMPT

    orig_topics = DEFAULT_TOPICS
    topic_to_desc = OrderedDict(orig_topics)
    topic_str = "\n".join( format_topic(name, desc) for name, desc in orig_topics)

    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    model_context_window = getattr(model.config, "max_position_embeddings", 96000)
    max_input = model_context_window - max_new_tokens

    write_topics = get_csv_writer(topic_path, ['topic', 'topic_desc'])
    write_quotes = get_csv_writer(quote_path, ['episode_file', 'topic', 'episode_quote'])
    buffered_topics = []
    buffered_quotes = []

    text_iter = tqdm(texts, desc="Processing texts")
    for file_path, text in zip(file_names, text_iter):
        user_prefix = partial_format(GEN_USER_PROMPT, Topics=topic_str, Noise=NOISE_PROMPT)
        tokenized_prompts = tokenized_with_trunc(tokenizer,
                                               [{"role": "system", "content": system_prompt}],
                                               user_prefix,
                                               text,
                                               max_input)
        matches = []
        for tokenized_prompt in tokenized_prompts:
            prompt = {key: value.to(model.device) for key, value in tokenized_prompt.items()}
            output = model.generate(**prompt,
                                    logits_processor=[xgr.contrib.hf.LogitsProcessor(grammar)],
                                    max_new_tokens=max_new_tokens)
            input_length = prompt['input_ids'].shape[-1]
            decoded = tokenizer.decode(output[0][input_length:], skip_special_tokens=True)
            matches.extend(output_pattern.findall(decoded))
        # The quote they give must be from the text itself--avoid hallucinations

        # OrderedDict implicitly handles duplicate topics
        # We prefer the first mention on the topic (probably has a better quote)
        # So we use [::-1] to reverse the order they're added to the dictionary...
        valid_matches = OrderedDict([
            (topic, (desc, quote))
            for (topic, desc, quote) in matches[::-1]
            if quote.strip() and quote in text # quote.strip() is to verify the quote isn't just whitespace, or an empty string
        ])
        # ... and use reversed() here to get them back in the order the model gave them (if we ever need that)
        valid_matches = [(k, desc, quote) for k,(desc, quote) in reversed(valid_matches.items())]

        if (new_topics := [(t, desc) for (t, desc, *_) in valid_matches if t not in topic_to_desc]):
            topic_to_desc.update(new_topics)
            topic_str += "\n" + "\n".join(format_topic(name, desc) for name, desc in new_topics)
            buffered_topics.extend(new_topics)
        buffered_quotes.extend((file_path, topic, quote) for (topic, _, quote) in valid_matches)
        if max(len(buffered_quotes), len(buffered_topics)) >= buffer_size:
            write_topics(buffered_topics)
            write_quotes(buffered_quotes)
            buffered_topics = []
            buffered_quotes = []
    write_topics(buffered_topics)
    write_quotes(buffered_quotes)

if __name__ == "__main__":
    main()



