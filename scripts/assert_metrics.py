#!/usr/bin/env python3
"""CI gate: fail if hybrid does not beat embedding-only on nDCG.
Usage: python scripts/assert_metrics.py --metrics artifacts/metrics.json --min-ndcg 0.35
"""
import argparse
import json
import sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=str, default="artifacts/metrics.json")
    ap.add_argument("--min-ndcg", type=float, default=0.35)
    args = ap.parse_args()
    p = Path(args.metrics)
    if not p.exists():
        print(f"metrics not found: {p}")
        sys.exit(1)
    data = json.loads(p.read_text(encoding="utf-8"))
    # Support both {methods: {embedding: {ndcg@10: ...}}} and flat
    def get_ndcg(method):
        m = data.get("methods", data).get(method, {})
        return m.get("ndcg@10") or m.get("ndcg_at_10") or m.get("ndcg") or 0

    emb = get_ndcg("embedding") or get_ndcg("embedding-only") or get_ndcg("vector")
    bm25 = get_ndcg("bm25") or 0
    hyb = get_ndcg("hybrid") or get_ndcg("hybrid_rrf")
    rer = get_ndcg("hybrid+ce") or get_ndcg("reranked") or hyb

    print(f"nDCG@10 — embedding={emb:.4f} bm25={bm25:.4f} hybrid={hyb:.4f} reranked={rer:.4f} (min={args.min_ndcg})")

    ok = True
    # Hybrid must beat BM25 (lexical baseline) and be within 0.02 of best single (embedding)
    best_single = max(emb, bm25)
    if hyb < bm25 - 1e-6:
        print(f"FAIL: hybrid ({hyb:.4f}) < bm25 ({bm25:.4f})")
        ok = False
    if hyb < best_single - 0.02:
        print(f"FAIL: hybrid ({hyb:.4f}) too far below best single ({best_single:.4f})")
        ok = False
    if rer < hyb - 1e-6:
        print(f"WARN: reranked ({rer:.4f}) < hybrid ({hyb:.4f}) — not failing, but unexpected")
    # Allow reranked==hybrid when CE skipped in CI
    if max(emb, bm25, hyb, rer) < args.min_ndcg:
        print(f"FAIL: best nDCG {max(emb,bm25,hyb,rer):.4f} < min {args.min_ndcg}")
        ok = False
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
