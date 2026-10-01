import json
import sys
import pdb
import csv
from operator import itemgetter
from collections import defaultdict

from tqdm import tqdm

from .constants.blacklist import AD_PATT

def main(raw_args=None):
    if not raw_args:
        raw_args = sys.argv[1:]

    paths = raw_args
    repl_list = defaultdict(set)
    get_text = itemgetter('text')
    for p in tqdm(paths, desc='Cleaning text'):
        with open(p) as r:
            rows = list(csv.DictReader(r))
        texts = map(get_text, rows)
        matches = map(lambda x: (x, AD_PATT.search(x)), texts)
        matches = filter(lambda x: x[1], matches)
        for t, m in matches:
            repl_list[m.group()].add(t)

    repl_count = {k:len(v) for k,v in repl_list.items()}
    replacements = sorted(map(list, repl_count.items()), key=lambda pair: pair[1], reverse=True)

    json.dump({
        "replacements": replacements
    }, sys.stdout, indent=2)


if __name__ == "__main__":
    main()