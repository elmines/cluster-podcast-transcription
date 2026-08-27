import sqlite3

def force_drop(cursor: sqlite3.Cursor, table_name):
    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")