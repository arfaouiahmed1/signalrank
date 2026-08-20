# Data

- `raw/jobs.jsonl` — 500 normalized jobs `{id,title,company,location,description,skills,source}` from Kaggle `lukebarousse/data_jobs` + HF mirrors, deduped; synthetic fallback if offline (see `scripts/fetch_kaggle.py:14`).
- `sample/cv_sample.txt` — anonymized CV derived from `New Portfolio/src/data.js:1` (Ahmed Arfaoui, AI Engineer, Tunis).
- `qrels.jsonl` — graded qrels `0/1/2` for `ahmed_cv` via `scripts/build_qrels.py:15` (HIGH_VALUE overlap + title boost + hard-negative cap + 12% noise). Use `--llm-grade` with `HF_TOKEN` for Mistral judge.

Regenerate:

```bash
python scripts/fetch_kaggle.py --sample 500 --out data/raw/jobs.jsonl
python scripts/build_qrels.py --jobs data/raw/jobs.jsonl --cv data/sample/cv_sample.txt --out data/qrels.jsonl
```
