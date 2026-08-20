from functools import lru_cache
from typing import List, Dict
import os

from app.config import get_settings

@lru_cache(maxsize=1)
def get_ce_model():
    from sentence_transformers import CrossEncoder
    cfg = get_settings()
    # cross-encoder is optional in CI — lazy load
    try:
        return CrossEncoder(cfg.ce_model)
    except Exception as e:
        print(f"[cross_encoder] load failed: {e}")
        return None

def rerank_with_ce(cv_text: str, candidates: List[Dict], top_k: int = 10) -> List[Dict]:
    """
    candidates: fused list of {job, rrf_score, rank, id} or raw jobs
    Returns reranked top_k with ce_score.
    Falls back to rrf_score order if model unavailable.
    """
    if not candidates:
        return []
    model = get_ce_model()
    if model is None:
        # fallback — return as-is truncated
        for c in candidates:
            c["ce_score"] = c.get("rrf_score", 0)
        return candidates[:top_k]

    # Build pairs
    pairs = []
    for c in candidates:
        job = c.get("job") or c
        jd = f"{job.get('title','')} at {job.get('company','')}. {job.get('description','')[:2000]}"
        pairs.append([cv_text[:2000], jd])

    scores = model.predict(pairs, batch_size=32, show_progress_bar=False)
    # attach
    for c, s in zip(candidates, scores):
        c["ce_score"] = float(s)
        # keep fused context
        c.setdefault("rrf_score", 0)

    reranked = sorted(candidates, key=lambda x: x["ce_score"], reverse=True)
    for i, x in enumerate(reranked, 1):
        x["ce_rank"] = i
    return reranked[:top_k]

def is_ce_available() -> bool:
    return get_ce_model() is not None
