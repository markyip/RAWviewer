import sqlite3
import os

db = os.path.expanduser('~/.rawviewer_cache/semantic_index.db')
if not os.path.exists(db):
    print(f"Database NOT found: {db}")
else:
    conn = sqlite3.connect(db)
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM semantic_index WHERE embedding IS NOT NULL')
    print(f"Indexed embeddings: {c.fetchone()[0]}")
    c.execute('SELECT COUNT(*) FROM semantic_index')
    print(f"Total files in DB: {c.fetchone()[0]}")
    conn.close()
