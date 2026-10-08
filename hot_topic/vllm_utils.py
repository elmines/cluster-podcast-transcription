import copy
import json
from typing import Optional

from vllm.sampling_params import StructuredOutputsParams
from xgrammar import get_model_structural_tag
from xgrammar.structural_tag import GrammarFormat, RegexFormat, JSONSchemaFormat

def make_structured_outputs_params(model_name: str,
                                    *,
                                   grammar: Optional[str] = None,
                                   regex: Optional[str] = None,
                                   json_schema: Optional[str | dict] = None,
                                   reasoning: bool = True,
                                   ) -> StructuredOutputsParams:
    if sum(bool(x) for x in [grammar, regex, json_schema]) != 1:
        raise ValueError("Must specifiy exactly one of {grammar, regex}")
    if "llama" in model_name or 'gemma' in model_name:
        return StructuredOutputsParams(grammar=grammar, regex=regex, json=json_schema)
    elif 'gpt-oss' in model_name:
        # Have to make an expanded grammar that supports the Harmony format
        structural_tag = get_model_structural_tag(
            "harmony",
            tools=[],
            tool_choice="none",
            reasoning=reasoning,
        )
        structural_tag = copy.deepcopy(structural_tag)

        if grammar:
            fmt_object = GrammarFormat(grammar=grammar)
        elif regex:
            fmt_object = RegexFormat(pattern=regex)
        else:
            fmt_object = JSONSchemaFormat(json_schema=json_schema)

        for tag in structural_tag.format.tags:
            # Only care about constraining the final answer
            if tag.begin == '<|channel|>final<|message|>':
                tag.content = fmt_object
        return StructuredOutputsParams(structural_tag=json.dumps(structural_tag.model_dump()))
    else:
        raise ValueError(f"Unsupported model {model_name}")

def make_ans_extraction(model_name):
    if 'gpt-oss' in model_name:
        GPT_SENTINEL = 'assistantfinal'
        def f(text):
            if ((left_index := text.find(GPT_SENTINEL)) == -1):
                return ''
            # Only take the "first final answer"
            content_start = left_index + len(GPT_SENTINEL)
            if ((right_index := text.find('assistant', content_start)) == -1):
                return text[content_start:]
            return text[content_start:right_index]
    else:
        f = lambda x: x
    return lambda x: f(x).strip()