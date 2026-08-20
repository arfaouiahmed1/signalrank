from typing import List, Dict
import numpy as np

from app.config import get_settings

def rrf_fuse(rank_lists: List[List[Dict]], k: int = 60, weight: List[float] | None = None) -> List[Dict]:
    """
    Reciprocal Rank Fusion.
    rank_lists: each is sorted list of {job, score, rank} or {id,...}
    Returns fused sorted list descending by rrf_score.
    """
    s = {}
    id_to_job = {}
    for li, lst in enumerate(rank_lists):
        w = (weight[li] if weight else 1.0)
        for item in lst:
            # normalize id extraction
            job = item.get("job") or item
            jid = job.get("id") or job.get("job_id") or item.get("id")
            if jid is None:
                continue
            jid = int(jid)
            rank = item.get("rank", 1)
            s[jid] = s.get(jid, 0) + w * (1.0 / (k + rank))
            if jid not in id_to_job:
                id_to_job[jid] = job

    fused = [{"job": id_to_job[jid], "rrf_score": score, "id": jid} for jid, score in s.items()]
    fused.sort(key=lambda x: x["rrf_score"], reverse=True)
    for i, x in enumerate(fused, 1):
        x["rank"] = i
    return fused


def hybrid_search(
    cv_text: str,
    bm25_results: List[Dict],
    vector_results: List[Dict],
    top_k: int | None = None,
) -> List[Dict]:
    """
    Fuse BM25 + vector via RRF. Both inputs are rank-ordered.
    Returns fused up to top_k (default settings.top_k_retrieval).
    """
    cfg = get_settings()
    k = cfg.rrf_k
    # equal weight; tune alpha via experiments notebook
    fused = rrf_fuse([bm25_results, vector_results], k=k, weight=[1.0, 1.0])
    if top_k:
        fused = fused[:top_k]
    else:
        fused = fused[: cfg.top_k_retrieval]
    return fused
