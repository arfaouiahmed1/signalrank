import re
from typing import List, Dict
from rank_bm25 import BM25Okapi

def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())

class BM25Index:
    def __init__(self, docs: List[Dict]):
        self.docs = docs
        self.corpus = [tokenize(f"{d['title']} {d['description']}") for d in docs]
        self.bm25 = BM25Okapi(self.corpus) if self.corpus else None

    def search(self, query: str, top_k: int = 100) -> List[Dict]:
        if not self.bm25:
            return []
        q_tokens = tokenize(query)
        scores = self.bm25.get_scores(q_tokens)
        # argsort descending
        idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [{"job": self.docs[i], "score": float(scores[i]), "rank": r+1} for r, i in enumerate(idx)]

# Postgres FTS path — used when DB available
def pg_bm25_search(conn, query: str, top_k: int = 100):
    """Use Postgres tsvector for BM25-ish ranking (ts_rank). Returns rows with rank."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, title, company, location, description, skills, source,
                   ts_rank(tsv, plainto_tsquery('english', %s)) AS bm25_score
            FROM jobs
            ORDER BY bm25_score DESC
            LIMIT %s
            """,
            (query, top_k),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
