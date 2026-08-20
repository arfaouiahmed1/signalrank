#!/usr/bin/env python3
"""
Ingest jobs.jsonl into PostgreSQL/pgvector (with tsvector auto-generated).
Usage: python scripts/ingest.py --jobs data/raw/jobs.jsonl
       python scripts/ingest.py --jobs data/raw/jobs.jsonl --no-embed  # skip embeddings (CI)
Requires DATABASE_URL env.
"""
import argparse
import json
import os
from pathlib import Path

def ingest_no_embed(jobs_path: Path):
    """Fallback: inserts without embeddings (hybrid BM25 still works)."""
    import psycopg
    dsn = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/signalrank")
    jobs = [json.loads(l) for l in open(jobs_path, encoding="utf-8") if l.strip()]
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(open("backend/db/pgvector_init.sql", encoding="utf-8").read())
            cur.execute("DELETE FROM qrels; DELETE FROM jobs;")
            for j in jobs:
                cur.execute(
                    "INSERT INTO jobs (id, title, company, location, description, skills, source) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                    (j["id"], j["title"], j["company"], j.get("location"), j["description"], json.dumps(j.get("skills",[])), j.get("source","synthetic")),
                )
        conn.commit()
    print(f"[ingest] Inserted {len(jobs)} jobs (no embeddings)")

def ingest_with_embed(jobs_path: Path, model_name: str):
    import json as _json
    import psycopg
    from sentence_transformers import SentenceTransformer
    import numpy as np

    dsn = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/signalrank")
    jobs = [_json.loads(l) for l in open(jobs_path, encoding="utf-8") if l.strip()]
    print(f"[ingest] Loading embed model {model_name}...")
    model = SentenceTransformer(model_name)
    texts = [f"{j['title']} — {j['company']}. {j['description']}" for j in jobs]
    print(f"[ingest] Encoding {len(texts)} jobs...")
    embs = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=False)
    # psycopg vector adapter expects list
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(open("backend/db/pgvector_init.sql", encoding="utf-8").read())
            cur.execute("DELETE FROM qrels; DELETE FROM jobs;")
            for j, emb in zip(jobs, embs):
                cur.execute(
                    "INSERT INTO jobs (id, title, company, location, description, skills, source, embedding) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                    (j["id"], j["title"], j["company"], j.get("location"), j["description"], _json.dumps(j.get("skills",[])), j.get("source","synthetic"), emb.tolist()),
                )
        conn.commit()
    print(f"[ingest] Inserted {len(jobs)} jobs with embeddings ({model_name})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=str, default="data/raw/jobs.jsonl")
    ap.add_argument("--no-embed", action="store_true", help="skip embedding (faster CI)")
    ap.add_argument("--model", type=str, default=os.getenv("EMBED_MODEL","sentence-transformers/all-MiniLM-L6-v2"))
    args = ap.parse_args()
    p = Path(args.jobs)
    if not p.exists():
        raise SystemExit(f"jobs not found: {p} — run fetch scripts first")
    if args.no_embed:
        ingest_no_embed(p)
    else:
        ingest_with_embed(p, args.model)

if __name__ == "__main__":
    main()
