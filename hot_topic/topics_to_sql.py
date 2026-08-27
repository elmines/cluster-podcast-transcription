
import os
import sqlite3
import argparse
import pandas as pd

from .sql import force_drop
from .utils import extract_show_and_episode

TOPIC_SCHEMA = r"""
create table {table_name}(
    id integer primary key,
    name text,
    desc text
)
"""

QUOTE_SCHEMA = r"""
create table {table_name}(
    id integer primary key,
    topic_id integer,
    show_id integer,
    episode_id text,
    quote text,

    foreign key(topic_id) references {topic_table},
    foreign key(show_id) references show(id),
    foreign key(episode_id) references episode(id)
)
"""

TOPIC_QUOTE_VIEW = r"""
    CREATE VIEW {view_name} AS
    SELECT
        q.show_id,
        q.episode_id,
        t.name as topic_name,
        t.desc as topic_desc,
        q.quote as episode_quote
    FROM
        {quote_table} q LEFT JOIN {topic_table} t ON q.topic_id = t.id
"""

def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-db", default="out/podcasts.sqlite", type=os.path.abspath)
    parser.add_argument("-i", required=True, type=os.path.abspath)
    parser.add_argument("--model", default="llmgen", help="Stem for table names")
    args = parser.parse_args(raw_args)

    db_path = args.db
    if not os.path.exists(db_path):
        raise ValueError(f"{db_path} doesn't exist")
    model_stem = args.model
    in_dir = args.i
    topics_path = os.path.join(in_dir, "filtered_topics.csv")
    quotes_path = os.path.join(in_dir, "filtered_quotes.csv")

    topics_df = pd.read_csv(topics_path, index_col="topic")
    if topics_df.index.has_duplicates:
        raise ValueError(f"{topics_path} has duplicate topics")

    quotes_df = pd.read_csv(quotes_path)

    t_topics = f"topics_{model_stem}"
    t_quotes = f"quotes_{model_stem}"
    view_name = f"readable_topics_{model_stem}"
    with sqlite3.connect(db_path) as con:
        cursor = con.cursor()
        for t_name in [t_topics, t_quotes]:
            force_drop(cursor, t_name)
        force_drop(cursor, view_name, "view")

        cursor.execute(TOPIC_SCHEMA.format(table_name=t_topics))
        cursor.execute(QUOTE_SCHEMA.format(table_name=t_quotes, topic_table=t_topics))

        topic_insert = [(idx, row['topic_desc']) for idx,row in topics_df.iterrows()]
        cursor.executemany(f"INSERT INTO {t_topics}(name, desc) VALUES (?,?)",
                           topic_insert)
        print("Populated topics table")

        cursor.execute(f"SELECT name,id from {t_topics}")
        name_to_id = dict(cursor.fetchall())

        quotes_df[ ['show_id', 'episode_id'] ] = quotes_df['episode_file'].apply(extract_show_and_episode).apply(pd.Series)
        quotes_df['topic_id'] = quotes_df['topic'].apply(name_to_id.__getitem__)

        cursor.executemany(f"INSERT INTO {t_quotes}(topic_id, show_id, episode_id, quote) VALUES (?,?,?,?)",
                           quotes_df[['topic_id', 'show_id', 'episode_id', 'episode_quote']].itertuples(index=False)
        )
        print("Populated quotes table")
        cursor.execute(TOPIC_QUOTE_VIEW.format(view_name=view_name, topic_table=t_topics, quote_table=t_quotes))
        print("Created view")



if __name__ == "__main__":
    main()