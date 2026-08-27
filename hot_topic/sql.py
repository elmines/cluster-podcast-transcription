import sqlite3

def force_drop(cursor: sqlite3.Cursor, table_name, entity="TABLE"):
    cursor.execute(f"DROP {entity} IF EXISTS {table_name}")