import json
import os
import re
import tempfile
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Security: rate limiting + hardened CORS
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse
from fastapi import Request

limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])
# Allowed origins: Pages + HF Space + local dev — configurable via ALLOWED_ORIGINS env (comma-separated)
_allowed = os.getenv("ALLOWED_ORIGINS", "https://arfaouiahmed1.github.io,https://ahmedarfaoui99-signalrank.hf.space,http://localhost:3000,http://localhost:8000")
ALLOWED_ORIGINS = [o.strip() for o in _allowed.split(",") if o.strip()]

app = FastAPI(
    title="SignalRank API",
    description="CV → hybrid search (BM25 + pgvector) → cross-encoder rerank → ranked jobs. Metrics: P@K/R@K/MRR/nDCG.",
    version="0.1.0",
)

app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded. Try again later."})

@app.middleware("http")
async def _security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    resp.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    resp.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src https://fonts.gstatic.com; connect-src 'self' https://ahmedarfaoui99-signalrank.hf.space https://arfaouiahmed1.github.io; img-src 'self' data: https:; frame-ancestors 'none'"
    # HSTS only when not localhost (Pages/HF are HTTPS)
    if "https://" in str(request.url) or request.headers.get("x-forwarded-proto") == "https":
        resp.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return resp

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# --- helpers: try to use pgvector, fallback to in-memory ---

def _load_jobs_jsonl(path: str = "data/raw/jobs.jsonl") -> List[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

def _extract_pdf_text(file_bytes: bytes) -> str:
    try:
        from pypdf import PdfReader
        import io
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        raise HTTPException(400, f"PDF parse failed: {e}")

def _pg_search(cv_text: str, top_k: int = 100):
    """Try pgvector + FTS; fallback to in-memory if DB unavailable."""
    dsn = os.getenv("DATABASE_URL")
    # Attempt DB
    if dsn:
        try:
            import psycopg
            from app.retrieval.bm25 import pg_bm25_search
            from sentence_transformers import SentenceTransformer
            from app.config import get_settings
            cfg = get_settings()
            # vector search
            model = SentenceTransformer(cfg.embed_model)
            q_emb = model.encode([cv_text], normalize_embeddings=True)[0]
            with psycopg.connect(dsn) as conn:
                # BM25 via FTS
                bm25_rows = pg_bm25_search(conn, cv_text, top_k=top_k)
                # Vector search — cosine via pgvector operator <=>
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id, title, company, location, description, skills, source, embedding <=> %s::vector AS distance FROM jobs ORDER BY embedding <=> %s::vector LIMIT %s",
                        (q_emb.tolist(), q_emb.tolist(), top_k),
                    )
                    cols = [d[0] for d in cur.description]
                    vec_rows = [dict(zip(cols, r)) for r in cur.fetchall()]
                    # convert distance to score
                    for r in vec_rows:
                        r["vector_score"] = 1 - float(r.get("distance", 1))
                # Normalize to hybrid format
                bm25_fmt = [{"job": r, "score": float(r.get("bm25_score",0)), "rank": i+1, "id": r["id"]} for i, r in enumerate(bm25_rows)]
                vec_fmt  = [{"job": r, "score": float(r.get("vector_score",0)), "rank": i+1, "id": r["id"]} for i, r in enumerate(vec_rows)]
                return bm25_fmt, vec_fmt
        except Exception as e:
            print(f"[_pg_search] DB fallback: {e}")

    # Fallback: in-memory
    jobs = _load_jobs_jsonl()
    if not jobs:
        return [], []
    from app.retrieval.bm25 import BM25Index
    from app.retrieval.embed import embed_texts
    import numpy as np

    bm25 = BM25Index(jobs)
    bm25_res = bm25.search(cv_text, top_k=top_k)

    # vector fallback — embed jobs on fly (cached in prod via ingest)
    try:
        from sentence_transformers import SentenceTransformer
        cfg_model = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        model = SentenceTransformer(cfg_model)
        job_texts = [f"{j['title']} — {j['company']}. {j['description']}" for j in jobs]
        job_embs = model.encode(job_texts, normalize_embeddings=True)
        q_emb = model.encode([cv_text], normalize_embeddings=True)[0]
        sims = (job_embs @ q_emb).tolist()
        vec_scored = sorted([(s, j) for s, j in zip(sims, jobs)], key=lambda x: x[0], reverse=True)[:top_k]
        vec_fmt = [{"job": j, "score": float(s), "rank": i+1, "id": j["id"]} for i, (s, j) in enumerate(vec_scored)]
        # bm25 already formatted
        bm25_fmt = [{"job": r["job"], "score": r["score"], "rank": r["rank"], "id": r["job"]["id"]} for r in bm25_res]
        return bm25_fmt, vec_fmt
    except Exception as e:
        print(f"[_pg_search] embed fallback failed: {e}")
        # BM25 only
        bm25_fmt = [{"job": r["job"], "score": r["score"], "rank": r["rank"], "id": r["job"]["id"]} for r in bm25_res]
        return bm25_fmt, []

class RankRequest(BaseModel):
    cv_text: str
    k: int = 10
    method: str = "hybrid+ce"  # embedding | bm25 | hybrid | hybrid+ce | hybrid+lgbm

    def model_post_init(self, __context):
        # Pydantic v2 hook — validate after init
        if len(self.cv_text) < 20:
            raise ValueError("cv_text too short (min 20 chars)")
        if len(self.cv_text) > 20000:
            raise ValueError("cv_text too long (max 20000 chars)")
        if not 1 <= self.k <= 50:
            raise ValueError("k must be 1..50")
        if self.method not in ("embedding", "vector", "bm25", "hybrid", "hybrid+ce", "hybrid+lgbm"):
            raise ValueError(f"method must be one of embedding/bm25/hybrid/hybrid+ce/hybrid+lgbm, got {self.method}")

@app.get("/health")
def health():
    return {"status": "ok", "service": "signalrank", "version": "0.1.0"}

@app.get("/jobs")
def list_jobs(limit: int = Query(20, le=500), offset: int = 0):
    jobs = _load_jobs_jsonl()
    return {"total": len(jobs), "jobs": jobs[offset: offset+limit]}

@app.post("/rank")
@limiter.limit("30/minute")
async def rank(
    request: Request,
    cv_text: Optional[str] = Form(None),
    cv_file: Optional[UploadFile] = File(None),
    k: int = Form(10),
    method: str = Form("hybrid+ce"),
):
    # also support JSON body
    # If called as JSON via RankRequest, FastAPI will route to rank_json below
    text = cv_text or ""
    if cv_file is not None:
        # 5 MB limit — prevents abuse on Pages/HF Space (public)
        data = await cv_file.read()
        if len(data) > 5 * 1024 * 1024:
            raise HTTPException(413, "File too large (max 5 MB)")
        if cv_file.filename and cv_file.filename.lower().endswith(".pdf"):
            text = _extract_pdf_text(data)
        else:
            # basic content-type check
            if cv_file.content_type and "text" not in cv_file.content_type and "pdf" not in cv_file.content_type and "octet-stream" not in cv_file.content_type:
                raise HTTPException(400, f"Unsupported file type: {cv_file.content_type}")
            text = data.decode("utf-8", errors="ignore")
    if not text or len(text.strip()) < 20:
        raise HTTPException(400, "Provide cv_text (>=20 chars) or cv_file (pdf/txt)")

    k = max(1, min(int(k), 50))
    if method not in ("embedding", "vector", "bm25", "hybrid", "hybrid+ce", "hybrid+lgbm"):
        raise HTTPException(400, f"method must be one of embedding/bm25/hybrid/hybrid+ce/hybrid+lgbm, got {method}")
    if len(text) > 20000:
        raise HTTPException(400, "cv_text too long (max 20000 chars)")
    text = text.strip()

    from app.retrieval.hybrid import hybrid_search
    from app.rerank.cross_encoder import rerank_with_ce, is_ce_available
    from app.rerank.lgbm import lgbm_rerank, is_lgbm_available

    bm25_res, vec_res = _pg_search(text, top_k=100)

    if method == "bm25":
        fused = bm25_res[:k]
        results = [{"rank": i+1, "job": r["job"], "score": r["score"], "method": "bm25", "bm25_score": r["score"]} for i, r in enumerate(fused)]
        return {"method": method, "k": k, "results": results, "meta": {"bm25": len(bm25_res), "vector": len(vec_res)}}

    if method == "embedding" or method == "vector":
        fused = vec_res[:k]
        results = [{"rank": i+1, "job": r["job"], "score": r["score"], "method": "embedding", "vector_score": r["score"]} for i, r in enumerate(fused)]
        return {"method": method, "k": k, "results": results, "meta": {"bm25": len(bm25_res), "vector": len(vec_res)}}

    # hybrid (default path)
    fused = hybrid_search(text, bm25_res, vec_res, top_k=100)

    if method == "hybrid":
        top = fused[:k]
        results = []
        for i, c in enumerate(top, 1):
            job = c.get("job") or c
            results.append({"rank": i, "job": job, "score": c.get("rrf_score",0), "rrf_score": c.get("rrf_score",0), "method": "hybrid"})
        return {"method": method, "k": k, "results": results, "meta": {"bm25": len(bm25_res), "vector": len(vec_res), "fused": len(fused)}}

    # hybrid+ce (default)
    # attach vector/bm25 scores for explainability
    vec_by_id = {r["id"]: r["score"] for r in vec_res}
    bm25_by_id = {r["id"]: r["score"] for r in bm25_res}
    for c in fused:
        c["vector_score"] = vec_by_id.get(c.get("id"), 0)
        c["bm25_score"] = bm25_by_id.get(c.get("id"), 0)

    reranked = rerank_with_ce(text, fused, top_k=k)
    # optional LGBM on top
    if method == "hybrid+lgbm" and is_lgbm_available():
        reranked = lgbm_rerank(text, reranked, top_k=k)
        method_out = "hybrid+lgbm"
    else:
        method_out = "hybrid+ce" if is_ce_available() else "hybrid"

    results = []
    for i, c in enumerate(reranked, 1):
        job = c.get("job") or c
        # highlight query tokens in description snippet
        results.append({
            "rank": i,
            "job": job,
            "score": c.get("ce_score", c.get("rrf_score",0)),
            "ce_score": c.get("ce_score"),
            "rrf_score": c.get("rrf_score"),
            "bm25_score": c.get("bm25_score"),
            "vector_score": c.get("vector_score"),
            "method": method_out,
        })

    return {"method": method_out, "k": k, "results": results, "meta": {"bm25": len(bm25_res), "vector": len(vec_res), "fused": len(fused), "ce_available": is_ce_available(), "lgbm_available": is_lgbm_available()}}

@app.post("/rank/json")
@limiter.limit("30/minute")
def rank_json(request: Request, req: RankRequest):
    # convenience for JSON clients (frontend fetch with JSON)
    import asyncio
    # Reuse logic via internal call — duplicate minimal to avoid Form parsing complexity
    from app.retrieval.hybrid import hybrid_search
    from app.rerank.cross_encoder import rerank_with_ce, is_ce_available
    text = req.cv_text.strip()
    if len(text) < 20:
        raise HTTPException(400, "cv_text too short")
    k = max(1, min(int(req.k), 50))
    bm25_res, vec_res = _pg_search(text, top_k=100)
    if req.method == "bm25":
        fused = bm25_res[:k]
        return {"method": "bm25", "k": k, "results": [{"rank": i+1, "job": r["job"], "score": r["score"]} for i, r in enumerate(fused)]}
    if req.method in ("embedding","vector"):
        fused = vec_res[:k]
        return {"method": "embedding", "k": k, "results": [{"rank": i+1, "job": r["job"], "score": r["score"]} for i, r in enumerate(fused)]}
    fused = hybrid_search(text, bm25_res, vec_res, top_k=100)
    if req.method == "hybrid":
        top = fused[:k]
        return {"method": "hybrid", "k": k, "results": [{"rank": i+1, "job": c.get("job") or c, "score": c.get("rrf_score",0)} for i, c in enumerate(top)]}
    vec_by_id = {r["id"]: r["score"] for r in vec_res}
    bm25_by_id = {r["id"]: r["score"] for r in bm25_res}
    for c in fused:
        c["vector_score"] = vec_by_id.get(c.get("id"), 0)
        c["bm25_score"] = bm25_by_id.get(c.get("id"), 0)
    reranked = rerank_with_ce(text, fused, top_k=k)
    return {"method": "hybrid+ce" if is_ce_available() else "hybrid", "k": k, "results": [{"rank": i+1, "job": (c.get("job") or c), "score": c.get("ce_score", c.get("rrf_score",0)), "ce_score": c.get("ce_score"), "rrf_score": c.get("rrf_score")} for i, c in enumerate(reranked)]}

@app.get("/metrics")
def get_metrics():
    p = Path("artifacts/metrics.json")
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    p2 = Path("artifacts/metrics-full.json")
    if p2.exists():
        return json.loads(p2.read_text(encoding="utf-8"))
    return {"detail": "No metrics yet. Run: python backend/app/evaluation/compare.py --jobs data/raw/jobs.jsonl --qrels data/qrels.jsonl --out artifacts/metrics.json"}

@app.post("/ingest")
@limiter.limit("10/minute")
def ingest(request: Request, jobs_path: str = "data/raw/jobs.jsonl", no_embed: bool = False):
    # Path traversal guard — only allow inside data/
    import pathlib
    from scripts.ingest import ingest_no_embed, ingest_with_embed
    p = pathlib.Path(jobs_path)
    try:
        # Python 3.9+ is_relative_to
        if not p.resolve().is_relative_to(pathlib.Path("data").resolve()):
            raise HTTPException(403, "jobs_path must be inside data/ directory")
    except AttributeError:
        # fallback for older
        if "data" not in str(p.resolve()):
            raise HTTPException(403, "jobs_path must be inside data/")
    if not p.exists():
        raise HTTPException(404, f"jobs not found: {jobs_path}")
    if no_embed:
        ingest_no_embed(p)
    else:
        import os as _os
        model = _os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        ingest_with_embed(p, model)
    return {"status": "ok", "jobs": jobs_path, "no_embed": no_embed}
