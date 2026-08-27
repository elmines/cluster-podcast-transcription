#!/usr/bin/env python3

import csv
import os
import sqlite3
import argparse
import glob

from .utils import extract_show_and_episode

SHOW_SCHEMA = r"""
create table if not exists show(
    id   integer primary key,
    title text
)
"""

EPISODE_SCHEMA = r"""
create table if not exists episode(
    show_id integer not null,
    id text not null,
    title text,

    primary key (show_id, id),
    foreign key(show_id) references show(id)
)
"""

LINE_SCHEMA = r"""
create table if not exists line(
    id integer primary key,
    show_id integer not null,
    episode_id text not null,
    transcription text,
    start_ms integer,
    stop_ms integer,

    foreign key(show_id) references show(id),
    foreign key(episode_id) references episode(id)
)
"""

def make_tables(cur: sqlite3.Cursor):
    cur.execute(SHOW_SCHEMA)
    cur.execute(EPISODE_SCHEMA)
    cur.execute(LINE_SCHEMA)


def main(raw_args=None):
    parser = argparse.ArgumentParser(description="Convert raw CSV transcriptions of data to SQLite database")
    parser.add_argument("-i", default="out/resegmented", type=os.path.abspath, help="Directory of transcribed CSVS")
    parser.add_argument("-f", action="store_true", help="Force overwrite of existing -o file")
    parser.add_argument("-o", default="out/podcasts.sqlite", type=os.path.abspath, help="Output path")
    args = parser.parse_args(raw_args)

    in_path = args.i
    out_path = args.o
    if os.path.exists(out_path):
        if args.f:
            os.remove(out_path)
        else:
            raise ValueError(f"{out_path} exists. Use -f to force overwrite")

    patt = os.path.join(in_path, "**", "*.csv")
    csv_paths = glob.glob(patt)

    show_ids = set()
    show_episode_ids = []

    for show_id, episode_id in map(extract_show_and_episode, csv_paths):
        show_ids.add(show_id)
        show_episode_ids.append((show_id, episode_id))
    show_ids = sorted(show_ids)


    with sqlite3.connect(out_path) as con:
        cursor = con.cursor()
        make_tables(cursor)

        cursor.executemany(
            "INSERT INTO show(id) VALUES (?)",
            [(id,) for id in show_ids]
        )

        cursor.executemany(
            "INSERT INTO episode(show_id, id) VALUES (?, ?)",
            show_episode_ids
        )

        for p in csv_paths:
            print(p)
            show_id, episode_id = extract_show_and_episode(p)
            with open(p, 'r') as r:
                csv_rows = list(csv.DictReader(r))
            cursor.executemany(
                "INSERT INTO line(show_id, episode_id, transcription, start_ms, stop_ms) VALUES (?,?,?,?,?)",
                [(show_id, episode_id, row['text'], row['start'], row['end']) for row in csv_rows]
            )


if __name__ == "__main__":
    main()
