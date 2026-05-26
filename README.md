# LAC AI: Retrieval System (Assignment 1)

## Why I picked this assignment

I'm a software engineer, not an ML researcher. Retrieval has a clear correctness signal: you either surface the right document or you don't. The failure modes are specific and worth thinking about. I used a sports Wikipedia corpus because it's broad enough to produce varied queries, but narrow enough that I could write 20 test cases where I actually knew the right answer.

---

## Corpus

**240 Wikipedia sports articles** fetched via the Wikipedia API (`ingest.py`). Categories span Olympic sports, team sports, combat sports, motorsport, and major events (Formula One, FIFA World Cup, UFC, Isle of Man TT, etc.).

Documents are stored in `data/corpus.db` (SQLite). Each document is the article title plus its intro extract, capped at 2000 characters. One article (Glossary of table tennis, 27k chars) crashed Ollama's context window, so I truncated everything to 2000 chars to keep the corpus uniform.

---

## Three configurations

### 1. BM25 (keyword)

Scores documents by term overlap with the query, penalising long documents and rewarding rare terms.

Pure C++, no external libraries. A compact posting list (`uint32_t doc_idx`, `uint16_t term_freq`, `uint16_t doc_len`) is built at index time. IDF is precomputed once in `build()`. Scoring uses a pre-allocated flat `vector<double>` scratch buffer indexed by position, avoiding hash map lookups on every query. Parameters: k1=1.5, b=0.75.

Sports article titles are distinctive keyword sequences. "Formula One", "Isle of Man TT", "The Ashes": BM25 finds these cleanly because the terms are rare across the corpus.

### 2. Semantic (embeddings)

Each document and query gets converted to a 768-dimensional vector using `nomic-embed-text` (local, via Ollama, no API cost, no data leaving the machine). Retrieval is nearest-neighbour search by cosine similarity.

Embeddings are generated once by `embed.py` and cached as float32 BLOBs in SQLite. At runtime, `embed_store.hpp` loads all 240 vectors into a contiguous row-major matrix. Search is a single `vDSP_mmul` call (Apple Accelerate), one matrix multiply instead of 240 individual dot products, dispatched across all CPU cores automatically.

Query embeddings are fetched at search time through a hand-rolled HTTP/1.0 client in C++ (no libcurl dependency).

Where it loses: "championship" reads as semantically similar across every major sports event. A query like "cycling race leading general classification" ranks the Tour de France 2nd or 3rd behind other championships. MRR takes the hit.

### 3. Hybrid (RRF)

Runs BM25 and Semantic in parallel (`std::async`) then fuses their ranked lists using Reciprocal Rank Fusion:

```
score(d) = 1/(60 + rank_bm25(d)) + 1/(60 + rank_semantic(d))
```

k=60 is the standard constant. Documents absent from one list get no contribution from that source. The top-20 from each source are fused to top-5.

BM25 and Semantic touch separate data structures with no shared mutable state, so `std::async` adds zero synchronisation overhead and cuts wall-clock latency on multi-core hardware.

---

## Evaluation

20 hand-written queries in `queries.toml`, each labelled with the document(s) that should appear in the top 5. Five are deliberately hard: paraphrased titles, multi-hop clues, no keyword overlap with the correct document.

Run with `make eval`.

### Results

| Metric | BM25 | Semantic | Hybrid |
|---|---|---|---|
| Recall@5 (all 20) | **1.00** | **1.00** | **1.00** |
| MRR (all 20) | **0.975** | 0.883 | **0.975** |
| p95 latency | 7 µs | 7 µs | 118 µs |
| Recall@5 (hard 5) | **1.00** | **1.00** | **1.00** |
| MRR (hard 5) | **0.900** | 0.867 | **0.900** |

All three configs hit Recall@5 = 1.0, meaning the correct document always appears somewhere in the top 5. The difference is rank: BM25 and Hybrid put it at position 1 more often (MRR 0.975 vs 0.883 for Semantic).

Hybrid p95 is 118 µs rather than 7 µs because it includes an Ollama HTTP round-trip to embed the query. BM25 has no such dependency. All three clear the 1 second constraint easily.

### Where the best config still loses

Semantic breaks on queries where the distinguishing detail is a proper noun rather than a concept. Query: *"annual competition where national league champions face off across the continent"* returns UEFA Champions League at rank 1 in BM25, but rank 4 in Semantic, behind Super Bowl, NBA Finals, and FIFA World Cup. All four are annual championship competitions, so the embedding model treats them as equally plausible. BM25 picks up "champions" and "league" as rare co-occurring terms that point directly at one document.

On a 240-document sports corpus, major events cluster in the same semantic neighbourhood ("championship", "annual", "international competition"). Semantic retrieval doesn't have enough to differentiate when the only distinguishing detail is a specific name.

BM25 has the opposite problem. Query 16: *"annual race held on an island known for dangerous road circuit since 1907"* ranks Isle of Man TT second in BM25 because the query shares no keywords with the title. Semantic gets it right at rank 1.

---

## Headline numbers

- BM25 MRR: **0.975**, best or tied on every metric
- Semantic MRR: **0.883**, lags on keyword-distinctive queries, wins on paraphrase
- Hybrid MRR: **0.975**, matches BM25; the hybrid advantage would show on a larger, noisier corpus
- All configs under **200 µs** p95 on a single laptop

---

## With one more week

Add a cross-encoder reranker on the top-20 hybrid candidates and test on a larger, noisier corpus. On 240 hand-picked articles BM25 and hybrid tie — that gap only opens up at scale where keyword matching gets ambiguous. I'd also fix the tokeniser (currently lowercase-alpha only, drops hyphenated terms) and add a stemmer to improve BM25 recall on word variants.

---

## How to run

```bash
# Ollama must be running for semantic and hybrid modes
ollama serve &
ollama pull nomic-embed-text

# 1. Ingest corpus (one-time)
python3 ingest.py

# 2. Generate embeddings (one-time)
python3 embed.py

# 3. Build C++ binary
make

# 4. Run evaluation: all 3 configs, 20 queries, metrics table
make eval

# 5. Interactive CLI
make ui

# 6. Single query
./lecai bm25     "olympic swimming world record"
./lecai semantic "annual road race in the French Alps"
./lecai hybrid   "chess and fighting hybrid sport"
```

**Requirements:** macOS (uses Apple Accelerate), clang++ with C++23, SQLite3, Node.js >=18, Python >=3.11, Ollama.
