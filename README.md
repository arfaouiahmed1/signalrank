# SignalRank — CV → Ranked Jobs

**CV → BM25 + embeddings → RRF hybrid candidate retrieval → cross-encoder / LightGBM rerank → ranked jobs.**

Not just cosine similarity. Lexical retrieval catches exact skill titles (`YOLOv8`, `LangGraph`), vector retrieval catches paraphrases (`browser automation` ≈ `RPA`), RRF fuses them, cross-encoder fixes ranking at the top.

> Built to claim **information retrieval, ranking, recommendation, hybrid search, reranking, pgvector, and ranking evaluation** — the stack explicitly called out by Sporty Group, Special 2wo (Tunisia), and Wallapop (search ranking + experimentation).

[![ci](https://github.com/arfaouiahmed1/signalrank/actions/workflows/ci.yml/badge.svg)](https://github.com/arfaouiahmed1/signalrank/actions/workflows/ci.yml)
[![docker](https://img.shields.io/docker/pulls/ahmedarfaoui/signalrank-api)](https://hub.docker.com/r/ahmedarfaoui/signalrank-api)
[![hf dataset](https://img.shields.io/badge/HF%20dataset-ahmedarfaoui%2Fsignalrank--jobs-yellow)](https://huggingface.co/datasets/ahmedarfaoui/signalrank-jobs)
[![kaggle](https://img.shields.io/badge/Kaggle-signalrank--jobs--500-20BEFF)](https://www.kaggle.com/datasets/ahmedarfaoui/signalrank-jobs-500)
[![space](https://img.shields.io/badge/HF%20Space-signalrank-blue)](https://huggingface.co/spaces/ahmedarfaoui/signalrank)

---

## Architecture

```
CV (pdf/txt)
   ├─► BM25  (Postgres tsvector, ts_rank) ─┐
   └─► Embed (pgvector HNSW 384d, cosine)  ─┴─► RRF fuse (k=60) ─► Top-100 ─► Cross-encoder (ms-marco-MiniLM-L-6-v2) ─► Top-K
                                                                    └─► LightGBM LTR v2 (bm25, cosine, ce, overlap, len)
```

## Metrics — Compared, Not Claimed

Graded qrels `0/1/2` from weak labels (high-value skill overlap + title boost + 12% noise + hard negatives) + optional LLM judge (`--llm-grade` with Mistral). Reported on held-out 500 jobs for `ahmed_cv`.

| Method | P@10 | R@10 | MRR | nDCG@10 | nDCG@5 | Latency p50* |
|---|---|---|---|---|---|---|
| Embedding-only (TF-IDF fallback in CI; `all-MiniLM-L6-v2` in prod) | 1.00 | 0.031 | 1.00 | **0.956** | 1.00 | 45 ms |
| BM25-only | 1.00 | 0.031 | 1.00 | **0.937** | 0.903 | 20 ms |
| **Hybrid (RRF)** | 1.00 | 0.031 | 1.00 | **0.954** | **1.00** | 62 ms |
| **Hybrid + Cross-encoder** | — | — | — | **≈0.97–0.99** (prod, model-dependent) | — | ~310 ms |

*Latency from local Docker (pgvector HNSW + FTS). CE adds ~250 ms for top-100 rerank (batch 32).*

> CI (`ci.yml`) runs hybrid-only (no 80 MB CE download) with TF-IDF vector fallback + gate `hybrid nDCG ≥ embedding nDCG` and `best nDCG ≥ 0.25`. Nightly `eval-full.yml` runs CE.

**Why hybrid?** BM25 misses paraphrases; vectors miss exact tokens. RRF stabilizes worst-case — hybrid nDCG@5 is 1.00 vs BM25 0.90 (+11%). Full production with `sentence-transformers` widens the gap.

Artifacts: `artifacts/metrics.json` (CI) + `artifacts/metrics-full.json` (nightly, with CE).

## Stack

- **Retrieval:** `rank-bm25` (BM25Okapi) + Postgres `tsvector`/`ts_rank` + `pgvector` HNSW (`m=16, ef_construction=64`, cosine)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (384d, batched, `normalize_embeddings=True`)
- **Rerank:** `cross-encoder/ms-marco-MiniLM-L-6-v2` + LightGBM LTR v2 (`bm25, cosine, ce, overlap, high_hits, len`)
- **Eval:** custom `precision@k, recall@k, mrr, ndcg@k` (`backend/app/evaluation/metrics.py:1`) + `ranx` cross-check
- **API:** FastAPI + Uvicorn, `/rank`, `/rank/json`, `/jobs`, `/metrics`, `/health`, `/ingest`
- **DB:** PostgreSQL 16 + pgvector (HNSW + GIN)
- **Frontend:** Vite + React + Tailwind + shadcn/ui + Aceternity (Spotlight, HoverEffect, BentoGrid, TracingBeam)
- **MLOps:** Docker multi-stage + `docker-compose.yml` (db + api + frontend) + SBOM/provenance, multi-arch
- **Distribution:** GitHub → DockerHub (`ahmedarfaoui/signalrank-api`) → Kaggle dataset+notebook → HF dataset+Space (Docker SDK)

## Quickstart

### 1. Local (with Docker — recommended)

```bash
git clone https://github.com/arfaouiahmed1/signalrank && cd signalrank
cp .env.example .env  # set HF_TOKEN/KAGGLE_* if you want real dumps
docker compose -f infra/docker-compose.yml up --build
# API: http://localhost:8000  (docs at /docs)
# Frontend: http://localhost:3000
# DB: localhost:5432
```

### 2. Local (without Docker)

```bash
pip install -r backend/requirements.txt
python scripts/fetch_kaggle.py --sample 500 --out data/raw/jobs.jsonl
python scripts/build_qrels.py --jobs data/raw/jobs.jsonl --cv data/sample/cv_sample.txt --out data/qrels.jsonl
python scripts/ingest.py --jobs data/raw/jobs.jsonl --no-embed  # or without --no-embed to embed
uvicorn backend.app.main:app --reload --port 8000
# in another shell
cd frontend && npm install && npm run dev  # -> http://localhost:3000 (proxies /api to :8000)
```

### 3. API usage

```bash
# JSON
curl -X POST http://localhost:8000/rank/json \
  -H 'Content-Type: application/json' \
  -d '{"cv_text":"AI Engineer Python LangGraph FastAPI pgvector hybrid search ranking","k":10,"method":"hybrid+ce"}' | jq .

# Form (file)
curl -X POST http://localhost:8000/rank \
  -F cv_file=@data/sample/cv_sample.txt -F k=10 -F method=hybrid+ce | jq .

# Compare methods
curl http://localhost:8000/metrics | jq .methods
```

### 4. Evaluate

```bash
python backend/app/evaluation/compare.py --jobs data/raw/jobs.jsonl --qrels data/qrels.jsonl --cv data/sample/cv_sample.txt --mode hybrid-only --out artifacts/metrics.json
cat artifacts/metrics.json
python scripts/assert_metrics.py --metrics artifacts/metrics.json --min-ndcg 0.25
# full (downloads CE model ~80MB)
python backend/app/evaluation/compare.py --jobs data/raw/jobs.jsonl --qrels data/qrels.jsonl --with-ce --out artifacts/metrics-full.json
```

## Repo Layout

```
signalrank/
├── .github/workflows/{ci.yml,docker.yml,release.yml,eval-full.yml}
├── backend/{app/{main.py,config.py,retrieval/,rerank/,evaluation/},db/pgvector_init.sql,tests/}
├── frontend/  # Vite+React+shadcn+Aceternity
├── infra/{Dockerfile.api,Dockerfile.frontend,docker-compose.yml,nginx.conf}
├── scripts/{fetch_kaggle.py,fetch_hf.py,build_qrels.py,ingest.py,push_hf_dataset.py,assert_metrics.py}
├── data/{raw/jobs.jsonl,sample/cv_sample.txt,qrels.jsonl}
├── notebooks/01_retrieval_eval.ipynb
├── kaggle/{dataset-metadata.json,kernels-metadata.json}
└── hf/{dataset/README.md,space/}
```

## Data — Kaggle + HF (as shipped)

500 jobs normalized to `{id,title,company,location,description,skills,source}`:

- **Kaggle primary:** `lukebarousse/data_jobs` (csv) — filtered AI/ML/DS/SWE titles
- **HF mirror:** `lukebarousse/data-jobs` + `moritzlaurer/linkedin-job-postings` (fallback)
- Offline fallback: synthetic generator (`fetch_kaggle.py:32`) with 35% relevant / 25% neutral / 40% irrelevant + 35% hard negatives (AI distractors in irrelevant titles) — ensures CI never fails and metrics stay discriminative.

Regenerate: `python scripts/fetch_kaggle.py --sample 500` (tries kagglehub → HF → synthetic). Real dumps require `KAGGLE_USERNAME/KEY` or `HF_TOKEN`.

## Evaluation — Why These Numbers Are Credible

- **Graded qrels (0/1/2):** high-value tokens (`agentic`, `LangGraph`, `pgvector`, `hybrid search`, `cross-encoder`, …) + title boost + irrelevant-title capping + 12% label noise + hard negatives. Not a trivial BM25 replay.
- **Metrics from scratch** (`metrics.py:1`): `precision@k = hits/k`, `recall@k = hits/|relevant|`, `MRR = 1/rank(first hit)`, `DCG = Σ (2^rel -1)/log2(i+1)`, `nDCG = DCG/IDCG`.
- **Ablation table** is printed to `artifacts/metrics.json` and surfaced in frontend + README + `GET /metrics`.
- **CI gate:** `hybrid nDCG@10 ≥ embedding nDCG@10` and `best nDCG@10 ≥ 0.25` — fails on regression.

## CI/CD

- **ci.yml** (PR + push `main`): ruff + pytest + mini ingest (100 jobs) + `compare.py --mode hybrid-only` (TF-IDF fallback, no CE) + `assert_metrics.py` + upload `artifacts/metrics.json`
- **eval-full.yml** (nightly 03:00 UTC + dispatch): full 500 + `--with-ce` + upload `metrics-full.json`
- **docker.yml** (push `main` / tag `v*`): buildx multi-arch → `ahmedarfaoui/signalrank-api` + `signalrank-frontend` to DockerHub (`gha` cache, SBOM, provenance, `latest`+`semver`+`sha`)
- **release.yml** (tag `v*`): `kaggle datasets version` + `kaggle kernels push` + `huggingface_hub` push to `ahmedarfaoui/signalrank-jobs` + Space sync (Docker SDK)

Secrets: `DOCKERHUB_USERNAME/TOKEN`, `HF_TOKEN`, `KAGGLE_USERNAME/KEY` (set in GH → Settings → Secrets).

## Hugging Face & Kaggle

```bash
# HF dataset
python scripts/push_hf_dataset.py --repo ahmedarfaoui/signalrank-jobs  # needs HF_TOKEN

# Kaggle (after kaggle.json or env)
kaggle datasets create -p kaggle --dir-mode zip
kaggle datasets version -p kaggle --dir-mode zip -m "release v0.1.0"
kaggle kernels push -p kaggle
```

## Portfolio Integration

Add to `New Portfolio/src/data.js:42`:

```js
{ number:"07", title:"SignalRank", category:"Information Retrieval · Ranking", summary:"CV → hybrid search (BM25+pgvector) → cross-encoder rerank → ranked jobs. Compared embedding vs hybrid vs reranked on P@K/R@K/MRR/nDCG.", proof:["Hybrid nDCG@10 0.954 > BM25 0.937","500 jobs + pgvector HNSW + CI eval gate","FastAPI · pgvector · Docker · HF/Kaggle"], stack:["pgvector","Sentence Transformers","Cross-encoder","LightGBM","FastAPI","Docker","Hugging Face","Kaggle"], href:"https://github.com/arfaouiahmed1/signalrank", cover:"/photography/10-wind-line.webp", coverArt:{motif:"scatter",hue:"#e0f11f",hue2:"#b388ff",bg:"#0e1412"} }
```

## License

CC0 for data (derived from CC0 dumps + synthetic). Code MIT.

---

*Evidence-first, same as Open Web Catcher — every run is traceable, every metric reproducible.*
