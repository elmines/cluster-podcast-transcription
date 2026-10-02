import json
import sys
import pdb
import csv
import re
from multiprocessing import Pool, cpu_count
from operator import itemgetter
from collections import defaultdict

from tqdm import tqdm

from .constants.blacklist import _CASE_AD_PATTS, _CASE_INSENS_PATTS


_TRACKED_PATT = None


def _compile_tracked_pattern():
    patterns = []
    raw_patterns = []
    for case_insensitive, source_patterns in (
        (True, _CASE_INSENS_PATTS),
        (False, _CASE_AD_PATTS),
    ):
        for pattern in source_patterns:
            raw_patterns.append(pattern)
            name = f"pattern_{len(patterns)}"
            if case_insensitive:
                pattern = f"(?i:{pattern})"
            patterns.append(f"(?P<{name}>{pattern})")
    return re.compile("|".join(patterns)), raw_patterns


def _pattern_matches(match):
    return next(
        name for name, value in match.groupdict().items() if value is not None
    )


def _init_worker():
    global _TRACKED_PATT
    _TRACKED_PATT, _ = _compile_tracked_pattern()


def _process_path(path):
    repl_list = defaultdict(set)
    pattern_matches = defaultdict(set)
    with open(path) as source:
        rows = csv.DictReader(source)
        for row in rows:
            text = row['text']
            match = _TRACKED_PATT.search(text)
            if match:
                repl_list[match.group()].add(text)
                pattern_matches[_pattern_matches(match)].add(text)
    return repl_list, pattern_matches


def main(raw_args=None):
    if not raw_args:
        raw_args = sys.argv[1:]

    paths = raw_args
    repl_list = defaultdict(set)
    pattern_matches = defaultdict(set)
    _, patterns = _compile_tracked_pattern()
    worker_count = max(1, cpu_count() - 1)
    with Pool(worker_count, initializer=_init_worker) as pool:
        results = pool.imap_unordered(_process_path, paths)
        for local_repl_list, local_pattern_matches in tqdm(
            results, total=len(paths), desc='Cleaning text'
        ):
            for replacement, texts in local_repl_list.items():
                repl_list[replacement].update(texts)
            for pattern, texts in local_pattern_matches.items():
                pattern_matches[pattern].update(texts)

    repl_count = {k:len(v) for k,v in repl_list.items()}
    replacements = sorted(map(list, repl_count.items()), key=lambda pair: pair[1], reverse=True)
    unused_patterns = [
        pattern
        for index, pattern in enumerate(patterns)
        if f"pattern_{index}" not in pattern_matches
    ]

    json.dump({
        "replacements": replacements,
        "unused_patterns": unused_patterns,
    }, sys.stdout, indent=2)


if __name__ == "__main__":
    main()