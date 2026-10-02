import json
import sys
import pdb
import csv
import re
from operator import itemgetter
from collections import defaultdict

from tqdm import tqdm

from .constants.blacklist import _CASE_AD_PATTS, _CASE_INSENS_PATTS


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

def main(raw_args=None):
    if not raw_args:
        raw_args = sys.argv[1:]

    paths = raw_args
    repl_list = defaultdict(set)
    pattern_matches = defaultdict(set)
    tracked_patt, patterns = _compile_tracked_pattern()
    get_text = itemgetter('text')
    for p in tqdm(paths, desc='Cleaning text'):
        with open(p) as r:
            rows = list(csv.DictReader(r))
        texts = map(get_text, rows)
        matches = map(lambda x: (x, tracked_patt.search(x)), texts)
        matches = filter(lambda x: x[1], matches)
        for t, m in matches:
            repl_list[m.group()].add(t)
            pattern_matches[_pattern_matches(m)].add(t)

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