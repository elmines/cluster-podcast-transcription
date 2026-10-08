#!/usr/bin/env python3

import argparse
import asyncio
import csv
import json
from collections import Counter
from multiprocessing import Pool, cpu_count
import re
import sys
import time
import torch
from pathlib import Path
from uuid import uuid4

from tqdm import tqdm
from vllm import AsyncLLMEngine, SamplingParams
from vllm.config import DeviceConfig, ModelConfig, VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.renderers import ChatParams, BaseRenderer, renderer_from_config

from .vllm_utils import make_structured_outputs_params, make_ans_extraction


MAX_LINES_PER_REQUEST = 300
CHUNK_OVERLAP = 24

AD_SPAN_SCHEMA = {
    "type": "object",
    "properties": {
        "spans": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "integer", "minimum": 0},
                    "end": {"type": "integer", "minimum": 1},
                    "reason": {"type": "string"},
                },
                "required": ["start", "end"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["spans"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You identify advertising in podcast transcripts.

Reasoning: low

Return one JSON object matching the provided schema, with a `spans` array. Each span must contain integer `start` and `end` fields and may contain a short `reason` field.
Spans use half-open intervals: `start` is inclusive and `end` is exclusive. Indices refer to the 0-based line numbers supplied in the transcript.
"""

USER_PROMPT = """Read the transcript below and identify every contiguous range of transcript lines that is advertising. Include sponsor reads, product or service promotions, affiliate messages, fundraising or membership appeals, and promotions for the podcast or its network. Do not label ordinary discussion of a product, news, or a guest's work unless it is promotional.

Return JSON matching the schema. Use half-open spans: `start` is the first advertising line and `end` is one past the last advertising line. For example, {"start": 0, "end": 3} labels lines 0, 1, and 2.

Use the supplied line indices exactly. If there is no advertising, return {"spans": []}. Do not include any text outside the JSON object.

TRANSCRIPT:
"""


def read_transcript(path):
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        rows = list(reader)
    if reader.fieldnames:
        missing = {"start", "end", "text"} - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return rows


def format_transcript(rows, offset=0):
    return "\n".join(f"[{offset + index}] {row['text'].strip()}" for index, row in enumerate(rows))


def warning(path, message):
    print(f"{path.name}: Warning: {message}", file=sys.stderr)


def extract_json_object(text, path, extract_answer):
    text = extract_answer(text)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "spans" in value:
            return value
    warning(path, f"LLM response did not contain a spans JSON object: {text[:500]!r}")
    return {"spans": []}


def validate_spans(payload, line_count, offset, path):
    if not isinstance(payload, dict) or not isinstance(payload.get("spans"), list):
        warning(path, "LLM response must contain a spans array")
        return []
    validated = []
    lower = offset
    upper = offset + line_count
    for span in payload["spans"]:
        if not isinstance(span, dict) or not isinstance(span.get("start"), int) or not isinstance(span.get("end"), int):
            warning(path, f"Malformed span: {span!r}")
            continue
        start, end = span["start"], span["end"]
        if not lower <= start < end <= upper:
            warning(path, f"Span {span!r} is outside half-open line range [{lower}, {upper})")
            continue
        validated.append({"start": start, "end": end, "reason": str(span.get("reason", ""))})
    return validated


def merge_spans(spans):
    merged = []
    for span in sorted(spans, key=lambda item: (item["start"], item["end"])):
        if merged and span["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], span["end"])
            if span.get("reason") and span["reason"] not in merged[-1].get("reason", ""):
                merged[-1]["reason"] = f"{merged[-1].get('reason', '')}; {span['reason']}".strip("; ")
        else:
            merged.append(dict(span))
    return merged


def render_chunk(renderer, offset, chunk):
    return renderer.render_chat(
        [[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT + format_transcript(chunk, offset)},
        ]],
        ChatParams(chat_template_kwargs={"add_generation_prompt": True}),
    )[1][0]


def prompt_token_count(rendered_prompt):
    try:
        return len(rendered_prompt["prompt_token_ids"])
    except (KeyError, TypeError):
        return len(rendered_prompt.prompt_token_ids)


def truncate_chunk_line(renderer, offset, chunk, max_prompt_tokens, path):
    row = chunk[0]
    text = row["text"]
    low, high = 0, len(text)
    best = None
    while low <= high:
        middle = (low + high) // 2
        candidate = [dict(row, text=text[:middle])]
        rendered_prompt = render_chunk(renderer, offset, candidate)
        if prompt_token_count(rendered_prompt) <= max_prompt_tokens:
            best = (candidate, rendered_prompt)
            low = middle + 1
        else:
            high = middle - 1
    if best is None:
        raise ValueError(
            f"cannot fit even an empty transcript line in the prompt budget for {path}"
        )
    warning(path, "truncated a transcript line to fit the prompt token budget")
    return best


def transcript_chunks(rows, renderer, max_model_len, max_new_tokens, path):
    max_prompt_tokens = max_model_len - max_new_tokens
    if max_prompt_tokens < 1:
        raise ValueError("max_model_len must be greater than max_new_tokens")

    chunks = []
    offset = 0
    while offset < len(rows):
        line_count = min(MAX_LINES_PER_REQUEST, len(rows) - offset)
        while line_count > 1:
            chunk = rows[offset:offset + line_count]
            rendered_prompt = render_chunk(renderer, offset, chunk)
            if prompt_token_count(rendered_prompt) <= max_prompt_tokens:
                break
            line_count -= 1
        else:
            chunk = rows[offset:offset + 1]
            rendered_prompt = render_chunk(renderer, offset, chunk)
            if prompt_token_count(rendered_prompt) > max_prompt_tokens:
                chunk, rendered_prompt = truncate_chunk_line(
                    renderer, offset, chunk, max_prompt_tokens, path
                )

        chunks.append((offset, chunk, rendered_prompt))
        offset += max(1, len(chunk) - CHUNK_OVERLAP)
    return chunks


async def generate_text(engine, prompt, sampling_params, request_id):
    final_output = None

    async for output in engine.generate(prompt, sampling_params, request_id):
        final_output = output
        completion = output.outputs[0]
    if final_output is None:
        raise RuntimeError(f"no output received for request {request_id}")
    completion = final_output.outputs[0]
    return completion.text

async def label_transcript(
    engine, renderer: BaseRenderer, input_path, sampling_params,
    extract_answer, max_model_len, max_new_tokens,
):
    rows = read_transcript(input_path)
    all_spans = []
    chunks = transcript_chunks(
        rows, renderer, max_model_len, max_new_tokens, input_path
    ) if rows else []
    rendered_prompts = [rendered_prompt for _, _, rendered_prompt in chunks]
    requests = [
        generate_text(
            engine,
            rendered_prompt,
            sampling_params,
            f"{input_path.name}-{uuid4()}",
        )
        for rendered_prompt in rendered_prompts
    ]
    responses = await asyncio.gather(*requests)
    for (offset, chunk, _), response in zip(chunks, responses):
        payload = extract_json_object(response, input_path, extract_answer)
        all_spans.extend(validate_spans(payload, len(chunk), offset, input_path))
    return {
        "source": str(input_path),
        "line_count": len(rows),
        "spans": merge_spans(all_spans),
    }


def input_paths(root, inputs):
    root = root.resolve()
    paths = []
    for input_path in inputs:
        path = input_path.resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"--input file is outside --root: {input_path}")
        if not path.is_file():
            raise ValueError(f"--input file does not exist: {path}")
        paths.append(path)
    return paths


def output_path(root, output_dir, input_path):
    relative = input_path.relative_to(root)
    return (output_dir / relative).with_suffix(".json")


def make_renderer(args):
    model_config = ModelConfig(
        model=args.model,
        max_model_len=args.max_model_len,
    )
    config = VllmConfig(
        model_config=model_config,
        device_config=DeviceConfig(device="cpu"),
    )
    return renderer_from_config(config)


_ESTIMATE_RENDERER = None
_ESTIMATE_MAX_MODEL_LEN = None
_ESTIMATE_MAX_NEW_TOKENS = None


def initialize_estimator(model, max_model_len, max_new_tokens):
    global _ESTIMATE_RENDERER, _ESTIMATE_MAX_MODEL_LEN, _ESTIMATE_MAX_NEW_TOKENS
    _ESTIMATE_RENDERER = make_renderer(
        argparse.Namespace(model=model, max_model_len=max_model_len)
    )
    _ESTIMATE_MAX_MODEL_LEN = max_model_len
    _ESTIMATE_MAX_NEW_TOKENS = max_new_tokens


def estimate_file_token_lengths(input_path):
    rows = read_transcript(input_path)
    chunks = transcript_chunks(
        rows,
        _ESTIMATE_RENDERER,
        _ESTIMATE_MAX_MODEL_LEN,
        _ESTIMATE_MAX_NEW_TOKENS,
        input_path,
    ) if rows else []
    frequencies = Counter()
    for _, _, rendered_prompt in chunks:
        token_count = prompt_token_count(rendered_prompt)
        frequencies[token_count // 5000 * 5000] += 1
    return frequencies


def estimate_token_lengths(args, input_files):
    frequencies = Counter()
    worker_count = min(args.num_workers, len(input_files))
    with Pool(
        processes=worker_count,
        initializer=initialize_estimator,
        initargs=(args.model, args.max_model_len, args.max_new_tokens),
    ) as pool:
        for file_frequencies in tqdm(
            pool.imap_unordered(estimate_file_token_lengths, input_files),
            total=len(input_files),
            desc="Counting tokens from chunks in input files",
        ):
            frequencies.update(file_frequencies)

    writer = csv.writer(sys.stdout)
    writer.writerow(("token_length", "count"))
    for bucket_start in sorted(frequencies):
        writer.writerow((f"{bucket_start}-{bucket_start + 5000}", frequencies[bucket_start]))


async def run(args, input_files, output_dir):
    tensor_parallel_size = args.tensor_parallel_size
    engine_args = AsyncEngineArgs(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        structured_outputs=make_structured_outputs_params(
            args.model,
            json_schema=AD_SPAN_SCHEMA,
            reasoning=False
        ),
    )
    extract_answer = make_ans_extraction(args.model)
    async def process(path):
        destination = output_path(args.root.resolve(), output_dir, path)
        if destination.exists():
            print(f"Skipping transcript {path}: output already exists at {destination}")
            return 0
        result = await label_transcript(
            engine,
            engine.renderer,
            path,
            sampling_params,
            extract_answer,
            args.max_model_len,
            args.max_new_tokens,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
        return 1

    n_transcribed = 0
    duration = -time.time()
    tasks = [asyncio.create_task(process(path)) for path in input_files]
    with tqdm(total=len(tasks), desc="Writing transcripts") as progress:
        for task in asyncio.as_completed(tasks):
            n_transcribed += await task
            progress.update(1)
    duration += time.time()

    device_names = ' '.join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))
    print("n_transcribed", "duration", "devices", "devices_used", sep=',')
    print(n_transcribed, duration, device_names, tensor_parallel_size, sep=',')


def main(raw_args=None):
    parser = argparse.ArgumentParser(description="Extract advertising spans from transcript CSVs")
    parser.add_argument("-i", "--input", dest="inputs", nargs="+", required=True, type=Path,
                        help="Transcript CSV files under --root")
    parser.add_argument("--root", default=Path("out/whisper_segmented"), type=Path,
                        help="Root directory under which every --input file must be found")
    parser.add_argument("-o", "--output", default=Path("out/ad_spans"), type=Path,
                        help="Directory for JSON outputs, mirroring --root")
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--tensor-parallel-size", type=int, default=torch.cuda.device_count())
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(4, cpu_count() - 1),
        help="Worker processes for CPU-side transcript preparation",
    )
    parser.add_argument(
        "--estimate-tokens",
        action="store_true",
        help="Count prompt lengths without creating an LLM engine",
    )
    args = parser.parse_args(raw_args)

    if args.max_model_len < 1 or args.max_new_tokens < 1 or args.max_num_seqs < 1 or args.num_workers < 1:
        parser.error("model length, token count, sequence count, and worker count must be positive")
    if args.max_model_len <= args.max_new_tokens:
        parser.error("--max-model-len must be greater than --max-new-tokens")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")

    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"--root is not a directory: {root}")
    output_dir = args.output.resolve()
    if output_dir == root or output_dir.is_relative_to(root) or root.is_relative_to(output_dir):
        parser.error("--output must be a different directory from --root")
    try:
        input_files = input_paths(root, args.inputs)
    except ValueError as exc:
        parser.error(str(exc))

    if args.estimate_tokens:
        estimate_token_lengths(args, input_files)
    else:
        asyncio.run(run(args, input_files, output_dir))


if __name__ == "__main__":
    main()