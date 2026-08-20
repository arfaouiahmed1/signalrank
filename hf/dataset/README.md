---
language: en
license: cc0-1.0
tags: [information-retrieval, ranking, hybrid-search, pgvector, re-ranking, cross-encoder, recommendation]
pretty_name: SignalRank Jobs 500 + Qrels
size_categories: [n<1K]
---

# SignalRank — 500 Jobs + Graded Qrels

Hybrid search benchmark: 500 jobs from **Kaggle `lukebarousse/data_jobs`** + **HF `lukebarousse/data-jobs`** (deduped), normalized to `{id,title,company,location,description,skills,source}` + graded qrels `0/1/2` for CV **ahmed_cv** (AI Engineer, Tunis — `data/sample/cv_sample.txt`).

**Qrels:** weak labels via high-value skill overlap (`agentic`, `LangGraph`, `pgvector`, `hybrid search`, `ranking`, `personalization`, …) + title boost; optional LLM-as-judge (`--llm-grade` via `mistralai/Mistral-7B-Instruct-v0.3`). `build_qrels.py` disclosed in repo.

**Use:**
```python
from datasets import load_dataset
ds = load_dataset("ahmedarfaoui/signalrank-jobs", split="train")
# jobs.jsonl + qrels.jsonl in data/
```

**Eval:** `backend/app/evaluation/compare.py --jobs data/raw/jobs.jsonl --qrels data/qrels.jsonl --with-ce --out artifacts/metrics.json` → P@K/R@K/MRR/nDCG@K, ablations `embedding-only vs BM25 vs hybrid vs hybrid+CE`.

Repo: `github.com/arfaouiahmed1/signalrank` · DockerHub: `ahmedarfaoui/signalrank-api` · Space: `huggingface.co/spaces/ahmedarfaoui/signalrank`
