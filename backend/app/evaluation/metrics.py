"""Ranking metrics: Precision@K, Recall@K, MRR, nDCG@K (graded)."""
import math
from typing import List, Dict

def precision_at_k(retrieved: List[int], relevant: set, k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for jid in retrieved[:k] if jid in relevant)
    return hits / k

def recall_at_k(retrieved: List[int], relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for jid in retrieved[:k] if jid in relevant)
    return hits / len(relevant)

def mrr(retrieved: List[int], relevant: set) -> float:
    for i, jid in enumerate(retrieved, 1):
        if jid in relevant:
            return 1.0 / i
    return 0.0

def dcg(relevances: List[int], k: int) -> float:
    s = 0.0
    for i, rel in enumerate(relevances[:k], 1):
        s += (2**rel - 1) / math.log2(i + 1)
    return s

def ndcg_at_k(retrieved: List[int], qrels: Dict[int, int], k: int) -> float:
    """qrels: job_id -> graded relevance (0/1/2). retrieved ordered."""
    rels = [qrels.get(jid, 0) for jid in retrieved]
    ideal = sorted(qrels.values(), reverse=True)
    d = dcg(rels, k)
    i = dcg(ideal, k)
    return d / i if i > 0 else 0.0

def evaluate_one(retrieved: List[int], qrels: Dict[int, int], relevant_threshold: int = 1, k: int = 10) -> Dict[str, float]:
    relevant = {jid for jid, rel in qrels.items() if rel >= relevant_threshold}
    rels_for_ndcg = qrels
    return {
        f"precision@{k}": precision_at_k(retrieved, relevant, k),
        f"recall@{k}": recall_at_k(retrieved, relevant, k),
        "mrr": mrr(retrieved, relevant),
        f"ndcg@{k}": ndcg_at_k(retrieved, rels_for_ndcg, k),
        f"ndcg@5": ndcg_at_k(retrieved, rels_for_ndcg, 5),
        f"precision@5": precision_at_k(retrieved, relevant, 5),
    }

def mean_metrics(per_query: List[Dict[str, float]]) -> Dict[str, float]:
    if not per_query:
        return {}
    keys = per_query[0].keys()
    return {k: sum(d[k] for d in per_query) / len(per_query) for k in keys}
