#!/usr/bin/env python3
"""
Evaluation harness — runs all 20 queries against bm25 / dense / hybrid,
computes Recall@5, MRR, and p95 latency per config.
"""

import subprocess
import re
import time
import sys
import tomllib
from pathlib import Path
from dataclasses import dataclass, field

BIN        = "./lecai"
QUERY_FILE = "queries.toml"
MODES      = ["bm25", "semantic", "hybrid"]
TOP_K      = 5
N_REPEATS  = 5   # runs per query for stable latency measurement

@dataclass
class QueryResult:
    query_id:    int
    mode:        str
    hit:         bool    # correct doc appeared in top-k
    reciprocal:  float   # 1/rank if found, else 0
    latency_us:  float   # single-run µs from binary output

@dataclass
class Config:
    mode:    str
    results: list[QueryResult] = field(default_factory=list)

    def recall_at_k(self) -> float:
        return sum(r.hit for r in self.results) / len(self.results)

    def mrr(self) -> float:
        return sum(r.reciprocal for r in self.results) / len(self.results)

    def p95_latency(self) -> float:
        lats = sorted(r.latency_us for r in self.results)
        idx  = int(len(lats) * 0.95) - 1
        return lats[max(idx, 0)]


def run_query(mode: str, query_text: str) -> tuple[list[int], float]:
    """Returns (list of doc_ids in ranked order, latency_µs)."""
    try:
        out = subprocess.check_output(
            [BIN, mode, query_text],
            stderr=subprocess.DEVNULL,
            timeout=15,
            text=True,
        )
    except subprocess.TimeoutExpired:
        return [], 99_999.0
    except subprocess.CalledProcessError:
        return [], 99_999.0

    latency_us = 0.0
    lat_m = re.search(r"latency:\s*([\d.]+)\s*us", out)
    if lat_m:
        latency_us = float(lat_m.group(1))

    doc_ids: list[int] = []
    for m in re.finditer(r"^\s+\d+\.\s+\[(\d+)\]", out, re.MULTILINE):
        doc_ids.append(int(m.group(1)))

    return doc_ids, latency_us


def evaluate(queries: list[dict], mode: str) -> Config:
    cfg = Config(mode=mode)
    for q in queries:
        qid      = q["id"]
        text     = q["text"]
        relevant = set(q["relevant_ids"])

        # Run N_REPEATS times; use median latency for stability
        latencies = []
        last_ids: list[int] = []
        for _ in range(N_REPEATS):
            ids, lat = run_query(mode, text)
            last_ids = ids
            latencies.append(lat)
        latency = sorted(latencies)[N_REPEATS // 2]

        # Recall: did any relevant doc appear in top-k?
        hit = any(d in relevant for d in last_ids[:TOP_K])

        # MRR: rank of first relevant doc
        reciprocal = 0.0
        for rank, doc_id in enumerate(last_ids[:TOP_K], start=1):
            if doc_id in relevant:
                reciprocal = 1.0 / rank
                break

        cfg.results.append(QueryResult(
            query_id=qid,
            mode=mode,
            hit=hit,
            reciprocal=reciprocal,
            latency_us=latency,
        ))
    return cfg


def print_table(configs: list[Config], queries: list[dict]) -> None:
    # ── Per-query breakdown ──────────────────────────────────────────────────
    hard_ids = {q["id"] for q in queries if q.get("hard")}

    print("\n── Per-query results ───────────────────────────────────────────────────────")
    header = f"{'ID':>3}  {'Query':<48}  {'H':1}  " + "  ".join(f"{m.upper():>7}" for m in MODES)
    print(header)
    print("─" * len(header))

    by_id: dict[int, dict[str, QueryResult]] = {}
    for cfg in configs:
        for r in cfg.results:
            by_id.setdefault(r.query_id, {})[cfg.mode] = r

    for q in queries:
        qid   = q["id"]
        label = q["text"][:46]
        hard  = "H" if q.get("hard") else " "
        cols  = []
        for mode in MODES:
            r = by_id[qid].get(mode)
            if r:
                mark = "✓" if r.hit else "✗"
                cols.append(f"{mark} {r.reciprocal:.2f}".rjust(7))
            else:
                cols.append("   n/a".rjust(7))
        print(f"{qid:>3}  {label:<48}  {hard}  " + "  ".join(cols))

    # ── Aggregate metrics ────────────────────────────────────────────────────
    print("\n── Aggregate metrics ───────────────────────────────────────────────────────")
    row = f"{'Metric':<20}" + "".join(f"  {m.upper():>10}" for m in MODES)
    print(row)
    print("─" * len(row))

    metrics = [
        ("Recall@5 (all)",  lambda c: c.recall_at_k()),
        ("MRR (all)",        lambda c: c.mrr()),
        ("p95 latency (µs)", lambda c: c.p95_latency()),
    ]
    for label, fn in metrics:
        vals = "".join(f"  {fn(c):>10.4f}" for c in configs)
        print(f"{label:<20}{vals}")

    # Hard-only subset
    for cfg in configs:
        cfg_hard = Config(mode=cfg.mode,
                          results=[r for r in cfg.results if r.query_id in hard_ids])
    hard_configs = []
    for cfg in configs:
        hc = Config(mode=cfg.mode,
                    results=[r for r in cfg.results if r.query_id in hard_ids])
        hard_configs.append(hc)

    print()
    hard_metrics = [
        ("Recall@5 (hard)", lambda c: c.recall_at_k()),
        ("MRR (hard)",       lambda c: c.mrr()),
    ]
    for label, fn in hard_metrics:
        vals = "".join(f"  {fn(c):>10.4f}" for c in hard_configs)
        print(f"{label:<20}{vals}")

    print()


def main() -> None:
    if not Path(BIN).exists():
        print(f"Binary '{BIN}' not found — run `make` first.", file=sys.stderr)
        sys.exit(1)

    with open(QUERY_FILE, "rb") as f:
        data = tomllib.load(f)
    queries: list[dict] = data["query"]
    print(f"Loaded {len(queries)} queries ({sum(1 for q in queries if q.get('hard'))} hard)")

    configs: list[Config] = []
    for mode in MODES:
        print(f"Running {mode.upper()}...", end=" ", flush=True)
        t0  = time.time()
        cfg = evaluate(queries, mode)
        configs.append(cfg)
        print(f"done ({time.time()-t0:.1f}s)")

    print_table(configs, queries)


if __name__ == "__main__":
    main()
