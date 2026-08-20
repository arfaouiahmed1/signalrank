"""Feature extraction for LightGBM LTR (v2). v1 ships with cross-encoder only."""
import re
from typing import Dict, List

def extract_features(cv_text: str, job: Dict, bm25_score: float = 0, cosine_score: float = 0, ce_score: float = 0) -> Dict[str, float]:
    cv_low = cv_text.lower()
    jd = (job.get("title","") + " " + job.get("description","") + " " + " ".join(job.get("skills",[]))).lower()
    # overlap
    cv_tokens = set(re.findall(r"[a-z0-9]+", cv_low))
    jd_tokens = set(re.findall(r"[a-z0-9]+", jd))
    overlap = len(cv_tokens & jd_tokens) / max(1, len(cv_tokens))
    # high-value skill hits
    high = ["langgraph","langchain","mcp","fastapi","pgvector","hybrid","ranking","recommender","personalization","mlflow","dvc","docker","pytorch","yolo","resnet","faiss"]
    high_hits = sum(1 for h in high if h in jd)
    # length features
    title_len = len(job.get("title","").split())
    desc_len = len(job.get("description","").split())

    return {
        "bm25": float(bm25_score),
        "cosine": float(cosine_score),
        "ce_score": float(ce_score),
        "overlap": float(overlap),
        "high_hits": float(high_hits),
        "title_len": float(title_len),
        "desc_len": float(desc_len),
    }

def features_to_vector(feats: Dict[str, float], order: List[str] | None = None) -> List[float]:
    order = order or ["bm25","cosine","ce_score","overlap","high_hits","title_len","desc_len"]
    return [feats.get(k, 0.0) for k in order]
