#!/usr/bin/env python3
"""
Fetch 240 sports-related Wikipedia pages and store in SQLite.
Usage: python3 ingest.py
"""

import sqlite3
import requests
import time
import json
from pathlib import Path

DB_PATH = "data/corpus.db"
TARGET = 240

SPORTS_CATEGORIES = [
    "Summer Olympic sports",
    "Winter Olympic sports",
    "Association football",
    "Basketball",
    "Tennis",
    "Athletics (sport)",
    "Swimming (sport)",
    "Cricket",
    "Rugby union",
    "American football",
    "Baseball",
    "Volleyball",
    "Golf",
    "Boxing",
    "Cycling",
    "Ice hockey",
    "Motorsport",
    "Wrestling",
    "Gymnastics",
    "Rowing (sport)",
    "Badminton",
    "Table tennis",
    "Martial arts",
    "Equestrian sport",
    "Archery",
    "Fencing",
    "Weightlifting",
    "Triathlon",
    "Skiing",
    "Snowboarding",
]

def init_db(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            title   TEXT NOT NULL,
            body    TEXT NOT NULL,
            source  TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_title ON documents(title)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            doc_id  INTEGER PRIMARY KEY REFERENCES documents(id),
            vector  BLOB
        )
    """)
    conn.commit()

HEADERS = {"User-Agent": "lecai-retrieval-project/1.0 (tadeyemo32@gmail.com)"}

def get_category_pages(category, limit=30):
    """Return up to `limit` page titles from a Wikipedia category."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmlimit": limit,
        "cmtype": "page",
        "format": "json",
    }
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=10)
        r.raise_for_status()
        members = r.json()["query"]["categorymembers"]
        return [m["title"] for m in members]
    except Exception as e:
        print(f"  [warn] category '{category}' failed: {e}")
        return []

def get_page_extract(title):
    """Return the plain-text intro extract of a Wikipedia page."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "format": "json",
    }
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=10)
        r.raise_for_status()
        pages = r.json()["query"]["pages"]
        page = next(iter(pages.values()))
        extract = page.get("extract", "").strip()
        if len(extract) < 100:
            return None
        return extract
    except Exception as e:
        print(f"  [warn] extract for '{title}' failed: {e}")
        return None

EXTRA_PAGES = [
    "Ultimate Fighting Championship",
    "Olympic Games",
    "FIFA World Cup",
    "Super Bowl",
    "Isle of Man TT",
    "MotoGP",
    "Formula One",
    "Tour de France",
    "Wimbledon Championships",
    "NBA Finals",
    "UEFA Champions League",
    "Rugby World Cup",
    "The Ashes",
    "Boston Marathon",
    "Indianapolis 500",
    "Kentucky Derby",
    "Ryder Cup",
    "Stanley Cup",
    "World Series",
    "Premier League",
]

def main():
    Path("data").mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    existing = conn.execute("SELECT title FROM documents").fetchall()
    seen = {row[0] for row in existing}
    print(f"Already have {len(seen)} documents.")

    titles_to_fetch = []
    for t in EXTRA_PAGES:
        if t not in seen and t not in titles_to_fetch:
            titles_to_fetch.append(t)

    for cat in SPORTS_CATEGORIES:
        pages = get_category_pages(cat, limit=20)
        for t in pages:
            if t not in seen and t not in titles_to_fetch:
                titles_to_fetch.append(t)
        if len(seen) + len(titles_to_fetch) >= TARGET:
            break
        time.sleep(0.2)

    print(f"Fetching extracts for {len(titles_to_fetch)} pages...")
    inserted = 0
    for title in titles_to_fetch:
        if len(seen) + inserted >= TARGET:
            break
        extract = get_page_extract(title)
        if extract:
            conn.execute(
                "INSERT INTO documents (title, body, source) VALUES (?, ?, ?)",
                (title, extract, "wikipedia"),
            )
            inserted += 1
            if inserted % 20 == 0:
                conn.commit()
                print(f"  {len(seen) + inserted} / {TARGET}")
        time.sleep(0.15)

    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    print(f"\nDone. Total documents in DB: {total}")
    conn.close()

if __name__ == "__main__":
    main()
