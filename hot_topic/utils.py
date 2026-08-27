from typing import List, Any, Tuple
import sys
import os

from transformers import PreTrainedTokenizerFast

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

    trunc_document = document
    while True:
        prompt = tokenizer.apply_chat_template(
            messages + [{"role": "user", "content": prompt_template.format(Document=trunc_document)}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors='pt'
        )
        if (excess := prompt['input_ids'].shape[-1] - max_len) > 0:
            trunc_document = tokenizer.decode(
                tokenizer.convert_tokens_to_ids(tokenizer.tokenize(document))[:-excess]
            )
            if not trunc_document:
                print("Prompt left no room for document", file=sys.stderr)
        else:
            break
    return prompt