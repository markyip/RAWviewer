#!/usr/bin/env python3
import os
import sqlite3

db = os.path.expanduser("~/.rawviewer_cache/exif_cache.db")
sem = os.path.expanduser("~/.rawviewer_cache/semantic_index.db")
pat = "k:\\photos\\japan trip\\%"

def count(conn, table):
    expr = "lower(replace(file_path, '/', '\\'))"
    n = conn.execute(
        f"SELECT COUNT(*) FROM {table} WHERE {expr} LIKE lower(?)",
        (pat,),
    ).fetchone()[0]
    n_ct = conn.execute(
        f"SELECT COUNT(*) FROM {table} WHERE {expr} LIKE lower(?) "
        "AND capture_time IS NOT NULL AND capture_time != ''",
        (pat,),
    ).fetchone()[0]
    n_blob = conn.execute(
        f"SELECT COUNT(*) FROM {table} WHERE {expr} LIKE lower(?) AND exif_data IS NOT NULL",
        (pat,),
    ).fetchone()[0]
    return n, n_ct, n_blob

if os.path.isfile(db):
    c = sqlite3.connect(db)
    n, n_ct, n_blob = count(c, "exif_cache")
    print(f"exif_cache: rows={n} capture_col={n_ct} has_blob={n_blob}")
    rows = c.execute(
        "SELECT file_path, capture_time FROM exif_cache "
        "WHERE lower(file_path) LIKE '%japan trip%' LIMIT 8"
    ).fetchall()
    for r in rows:
        print(" ", r)
else:
    print("missing", db)

if os.path.isfile(sem):
    c2 = sqlite3.connect(sem)
    n, n_ct, _ = count(c2, "semantic_index")
    print(f"semantic_index: rows={n} capture_col={n_ct}")
