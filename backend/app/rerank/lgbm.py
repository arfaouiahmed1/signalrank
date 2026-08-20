"""LightGBM learning-to-rank stub (v2). Train offline with qrels; inference in /rank if model exists."""
from pathlib import Path
from typing import List, Dict

MODEL_PATH = Path("artifacts/lgbm_ranker.txt")

def is_lgbm_available() -> bool:
    return MODEL_PATH.exists()

def lgbm_rerank(cv_text: str, candidates: List[Dict], top_k: int = 10) -> List[Dict]:
    """If model exists, rerank by LGBM score; else passthrough."""
    if not is_lgbm_available():
        return candidates[:top_k]
    try:
        import lightgbm as lgb
        import numpy as np
        from app.rerank.features import extract_features, features_to_vector
        model = lgb.Booster(model_file=str(MODEL_PATH))
        feats = []
        for c in candidates:
            job = c.get("job") or c
            f = extract_features(cv_text, job, bm25_score=c.get("bm25_score",0), cosine_score=c.get("cosine_score",0), ce_score=c.get("ce_score",0))
            feats.append(features_to_vector(f))
        scores = model.predict(np.array(feats))
        for c, s in zip(candidates, scores):
            c["lgbm_score"] = float(s)
        reranked = sorted(candidates, key=lambda x: x["lgbm_score"], reverse=True)
        return reranked[:top_k]
    except Exception as e:
        print(f"[lgbm] rerank failed: {e}")
        return candidates[:top_k]

def train_lgbm(jobs_path: str, qrels_path: str, out_path: str = str(MODEL_PATH)):
    """Offline training — called from notebook/scripts, not from API."""
    import json, numpy as np, lightgbm as lgb
    from app.rerank.features import extract_features, features_to_vector
    from app.retrieval.embed import embed_texts

    # Load
    jobs = [json.loads(l) for l in open(jobs_path, encoding="utf-8") if l.strip()]
    qrels = {}
    for line in open(qrels_path, encoding="utf-8"):
        if not line.strip(): continue
        r = json.loads(line); qrels[r["job_id"]] = r["relevance"]
    cv_text = open("data/sample/cv_sample.txt", encoding="utf-8").read()

    # For demo: create features with dummy scores (real pipeline would compute bm25/cosine/ce)
    X, y = [], []
    job_by_id = {j["id"]: j for j in jobs}
    for jid, rel in qrels.items():
        job = job_by_id.get(jid)
        if not job: continue
        f = extract_features(cv_text, job)
        X.append(features_to_vector(f)); y.append(rel)

    if not X:
        print("[lgbm] no training data")
        return
    X = np.array(X); y = np.array(y)
    train = lgb.Dataset(X, label=y)
    params = {"objective": "regression", "metric": "rmse", "verbosity": -1}
    model = lgb.train(params, train, num_boost_round=100)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(out_path)
    print(f"[lgbm] saved {out_path}")
