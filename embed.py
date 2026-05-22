#!/usr/bin/env python3
"""
Embed all documents in corpus.db using local nomic-embed-text via Ollama.
Stores vectors as binary BLOBs (float32) in the embeddings table.
Safe to re-run — skips already-embedded docs.
"""

import sqlite3
import struct
import requests
import time
import sys

DB_PATH  = "data/corpus.db"
MODEL    = "nomic-embed-text"
OLLAMA   = "http://localhost:11434/api/embeddings"
DIM      = 768
BATCH_PAUSE = 0.05  # seconds between requests — keeps Ollama responsive

def embed(text: str) -> list[float]:
    r = requests.post(OLLAMA, json={"model": MODEL, "prompt": text}, timeout=30)
    r.raise_for_status()
    vec = r.json()["embedding"]
    assert len(vec) == DIM, f"Expected {DIM} dims, got {len(vec)}"
    return vec

def vec_to_blob(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)

def main():
    conn = sqlite3.connect(DB_PATH)

    # Ensure embeddings table exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            doc_id INTEGER PRIMARY KEY REFERENCES documents(id),
            vector BLOB NOT NULL
        )
    """)
    conn.commit()

    # Find docs that still need embedding
    rows = conn.execute("""
        SELECT d.id, d.title, d.body
        FROM documents d
        LEFT JOIN embeddings e ON e.doc_id = d.id
        WHERE e.doc_id IS NULL
        ORDER BY d.id
    """).fetchall()

    total    = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    done     = total - len(rows)
    print(f"Already embedded: {done}/{total}. Embedding {len(rows)} remaining...")

    for i, (doc_id, title, body) in enumerate(rows):
        text = f"{title}\n{body}"[:2000]
        try:
            vec  = embed(text)
            blob = vec_to_blob(vec)
            conn.execute("INSERT INTO embeddings(doc_id, vector) VALUES (?,?)", (doc_id, blob))
            if (i + 1) % 20 == 0:
                conn.commit()
                pct = (done + i + 1) / total * 100
                print(f"  {done + i + 1}/{total}  ({pct:.0f}%)")
            time.sleep(BATCH_PAUSE)
        except Exception as e:
            print(f"  [error] doc {doc_id} '{title}': {e}", file=sys.stderr)

    conn.commit()
    final = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    print(f"\nDone. Embeddings stored: {final}/{total}")
    conn.close()

if __name__ == "__main__":
    main()
