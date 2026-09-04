from typing import List, Any, Tuple
import csv
import os
import re

from transformers import PreTrainedTokenizerFast

_MUSIC_PATT = re.compile("|".join([
    r"\[.*?MUSIC.*?\]",
    r"\(.*?music\)",
    r"\(singing.*?\)",
]), flags=re.IGNORECASE)

_WHITE_PATT = re.compile(r"\s+")

def preprocess(s: str) -> str:
    rval, count = _MUSIC_PATT.subn("", s)
    # print(f"{count} instances of music removed")
    # Cleans up large blocks of whitespace created by that earlier sub operation
    rval = _WHITE_PATT.sub(" ", rval)
    return rval


def read_transcription_text(csv_path):
    with open(csv_path, 'r') as r:
        return "".join(row['text'] for row in csv.DictReader(r))

############# From Chat GPT ########################
class PartialFormatDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"

def partial_format(s, **kwargs):
    return s.format_map(PartialFormatDict(kwargs))
###################################################

def extract_show_and_episode(p: os.PathLike) -> Tuple[str, str]:
    show_id = int(os.path.basename(os.path.dirname(p)))
    episode_id = os.path.basename(p).split('.')[0]
    return show_id, episode_id

def tokenized_with_trunc(tokenizer: PreTrainedTokenizerFast,
                         messages: List[Any],
                         prompt_template: str,
                         document: str,
                         max_len: int):
    base_template = messages + [{"role": "user", "content": prompt_template.format(Document="")}]
    base_prompt = tokenizer.apply_chat_template(
        base_template,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors='pt'
    )
    base_len = base_prompt['input_ids'].shape[-1]
    # print(f"base_len = {base_len}")

    document_tokens = tokenizer.tokenize(document)
    # print(f"document_tokens = {len(document_tokens)}")
    if not document_tokens:
        return [
            tokenizer.apply_chat_template(
                messages + [{"role": "user", "content": prompt_template.format(Document=document)}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors='pt'
            )
        ]

    # If the base prompt is already at or above max_len, we still need to split the
    # document into separate chunks rather than erroring out. In that case, treat the
    # document chunk size as max_len so every chunk remains a valid standalone prompt.
    remaining_capacity = max_len - base_len if base_len < max_len else max_len
    remaining_capacity = max(1, remaining_capacity)

    chunks = []
    for start in range(0, len(document_tokens), remaining_capacity):
        chunk_tokens = document_tokens[start:start + remaining_capacity]
        assert chunk_tokens == document_tokens
        chunk_document = tokenizer.decode(tokenizer.convert_tokens_to_ids(chunk_tokens))
        chunks.append(
            tokenizer.apply_chat_template(
                messages + [{"role": "user", "content": prompt_template.format(Document=chunk_document)}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors='pt'
            )
        )
        # print(f"\tchunk = {chunks[-1]['input_ids'].shape}")
    return chunks