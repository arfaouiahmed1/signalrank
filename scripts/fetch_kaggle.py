#!/usr/bin/env python3
"""
Fetch 500 jobs from Kaggle + HF dumps, normalize to jobs.jsonl.
Usage: python scripts/fetch_kaggle.py  (legacy alias — now calls fetch_hf internally)
       python scripts/fetch_kaggle.py --sample 500 --out data/raw/jobs.jsonl
Falls back to synthetic generation if Kaggle/HF unavailable (offline CI).
"""
import argparse
import json
import random
import re
from pathlib import Path

# Lightweight synthetic fallback — ensures CI never fails without network/tokens
SYNTH_TITLES_RELEVANT = [
    "AI Engineer", "ML Engineer", "Data Scientist", "Data Engineer", "MLOps Engineer",
    "Search & Recommendation Engineer", "Personalization Engineer", "NLP Engineer",
    "Computer Vision Engineer", "Agentic AI Engineer", "LLM Engineer",
]
SYNTH_TITLES_NEUTRAL = ["Software Engineer", "Backend Engineer", "Platform Engineer", "Data Analyst", "BI Engineer"]
SYNTH_TITLES_IRRELEVANT = [
    "Sales Manager", "Marketing Specialist", "Accountant", "HR Coordinator", "Customer Support Lead",
    "Logistics Coordinator", "Nurse", "Teacher", "Retail Store Manager", "Finance Analyst",
]
SYNTH_TITLES = SYNTH_TITLES_RELEVANT + SYNTH_TITLES_NEUTRAL + SYNTH_TITLES_IRRELEVANT
COMPANIES = ["Sporty Group","Special 2wo","Wallapop","VERMEG","Soft Stars","ESPRIT","InstaDeep","Expensya","Kaoun","valor","Capgemini","Sopra Steria","Telnet","Orange Tunisia","Ooredoo"]
LOCATIONS = ["Tunis, Tunisia","Remote","Paris, France","Berlin, Germany","Barcelona, Spain","Madrid, Spain","London, UK","Dubai, UAE"]
SKILL_POOLS = {
    "ai_ml": ["Python","PyTorch","TensorFlow","Scikit-learn","LangGraph","LangChain","MCP","FastAPI","Docker","MLflow","DVC","pgvector","FAISS","sentence-transformers","Hugging Face","LoRA","RAG","information retrieval","hybrid search","ranking","recommender systems","personalization","cross-encoder","LightGBM","evaluation","NDCG","MRR"],
    "search_rec": ["Elasticsearch","pgvector","HNSW","BM25","hybrid search","re-ranking","cross-encoder","personalization","A/B testing","experimentation","learning to rank","collaborative filtering","content-based filtering","search relevance"],
    "cv": ["YOLOv8","ResNet","OpenCV","CNN","image classification","object detection","NDVI","satellite imagery"],
    "nlp": ["Transformers","BERT","FLAN-T5","Mistral","LIME","SHAP","semantic search","RAG","fine-tuning"],
    "mlops": ["Docker","Kubernetes","CI/CD","MLflow","DVC","FastAPI","PostgreSQL","Elastic","monitoring","drift detection"],
    "swe": ["React","Next.js","Node.js","TypeScript","PostgreSQL","Redis","Puppeteer","browser automation"],
    "irrelevant": ["Salesforce","CRM","Excel","QuickBooks","Budgeting","Recruiting","Customer service","Supply chain","Inventory","Teaching","Clinical care","Merchandising","Accounting","Payroll"],
}

def synth_job(jid: int) -> dict:
    # 35% relevant (AI/search), 25% neutral (SWE/DATA), 40% irrelevant — creates realistic IR distribution
    r = random.random()
    if r < 0.35:
        title = random.choice(SYNTH_TITLES_RELEVANT)
        pool_keys = random.choices(["ai_ml","search_rec","cv","nlp","mlops"], k=2)
    elif r < 0.60:
        title = random.choice(SYNTH_TITLES_NEUTRAL)
        pool_keys = random.choices(["swe","mlops","ai_ml"], k=2)
    else:
        title = random.choice(SYNTH_TITLES_IRRELEVANT)
        pool_keys = ["irrelevant", "irrelevant"]
    company = random.choice(COMPANIES)
    location = random.choice(LOCATIONS)
    skills = []
    for k in pool_keys:
        pool = SKILL_POOLS[k]
        skills += random.sample(pool, k=min(random.randint(3,5), len(pool)))
    # Hard negatives: 35% of irrelevant jobs get 2 AI distractor skills to fool BM25
    if any(title == t for t in SYNTH_TITLES_IRRELEVANT) and random.random() < 0.35:
        distractors = random.sample(SKILL_POOLS["ai_ml"], k=2)
        skills = (skills + distractors)[:10]
    skills = list(dict.fromkeys(skills))[:10]
    # Nice-to-have should be category-consistent — don't pollute irrelevant jobs with AI skills
    if any(title == t for t in SYNTH_TITLES_IRRELEVANT):
        nice_pool = SKILL_POOLS["irrelevant"]
    elif any(title == t for t in SYNTH_TITLES_NEUTRAL):
        nice_pool = SKILL_POOLS["swe"]
    else:
        nice_pool = SKILL_POOLS["ai_ml"]
    nice = ", ".join(random.sample(nice_pool, k=min(3, len(nice_pool))))
    # Description with title-specific jargon for BM25 signal
    desc = (
        f"We are hiring a {title} at {company} ({location}). "
        f"You will work on {', '.join(skills[:5])}. "
        f"Requirements: {', '.join(skills)}. "
        f"Nice to have: {nice}. "
        f"Role focus: "
    )
    if "Search" in title or "Personalization" in title or "Recommendation" in title:
        desc += "Build search relevance, hybrid retrieval, learning-to-rank, and personalization experiments. Own ranking evaluation (Precision@K, Recall@K, MRR, NDCG@K) and run A/B tests on search ranking. Work with pgvector/Elasticsearch, cross-encoders, and feature stores. "
    elif "Agentic" in title or "LLM" in title:
        desc += "Design agentic systems with LangGraph/MCP, tool-use, browser automation, and evaluation traces. Ship FastAPI services with PostgreSQL and pgvector. "
    elif "MLOps" in title:
        desc += "Own end-to-end MLOps: DVC, MLflow, Docker, CI/CD, drift monitoring with Elastic, and reproducible pipelines. "
    elif "Computer Vision" in title:
        desc += "Train and deploy YOLOv8/ResNet models, handle satellite NDVI and agricultural imagery, optimize inference latency. "
    elif "Data Scientist" in title:
        desc += "Own modeling from EDA to production, run experiments, explain with LIME/SHAP, and present to business stakeholders. "
    else:
        desc += "Ship reliable software with strong data foundations, collaborate across product and data teams. "
    desc += f"Location: {location}. Stack: {', '.join(skills)}. Apply via {company.lower().replace(' ','')}.com/careers."
    return {
        "id": jid,
        "title": title,
        "company": company,
        "location": location,
        "description": desc,
        "skills": skills,
        "source": "synthetic" if jid else "synthetic",
    }


def try_fetch_kagglehub(sample: int):
    """Try kagglehub for lukebarousse/data_jobs; return list[dict] or None."""
    try:
        import kagglehub
        import pandas as pd
        # HF id uses underscore, Kaggle uses hyphen — try both
        for slug in ("lukebarousse/data-jobs", "lukebarousse/data_jobs"):
            try:
                path = kagglehub.dataset_download(slug)
                break
            except Exception:
                continue
        else:
            return None
        # dataset contains data_jobs.csv or similar
        csv = next(Path(path).rglob("*.csv"), None)
        if not csv:
            return None
        df = pd.read_csv(csv, low_memory=False)
        # normalize columns: job_title_short etc
        cols = {c.lower(): c for c in df.columns}
        title_col = cols.get("job_title_short") or cols.get("job_title") or list(df.columns)[0]
        company_col = cols.get("company_name") or cols.get("company") or title_col
        loc_col = cols.get("job_location") or cols.get("location") or company_col
        desc_col = cols.get("job_description") or cols.get("description") or title_col
        # filter AI-ish
        if "job_title_short" in cols:
            mask = df[title_col].astype(str).str.contains(r"Data|ML|AI|Engineer|Scientist|MLOps|Search|Recommend", case=False, na=False)
            df = df[mask]
        df = df.head(sample * 2).sample(n=min(sample, len(df)), random_state=42) if len(df) > sample else df
        jobs = []
        for i, row in enumerate(df.itertuples(), 1):
            jobs.append({
                "id": i,
                "title": str(getattr(row, title_col, "Data Scientist"))[:120],
                "company": str(getattr(row, company_col, "Unknown"))[:80],
                "location": str(getattr(row, loc_col, "Remote"))[:80],
                "description": str(getattr(row, desc_col, ""))[:4000],
                "skills": [],
                "source": "kaggle:lukebarousse/data_jobs",
            })
        return jobs
    except Exception as e:
        print(f"[fetch_kaggle] kagglehub unavailable: {e}")
        return None


def try_fetch_hf(sample: int):
    """Try HF datasets; return list[dict] or None."""
    try:
        from datasets import load_dataset
        # Try both slug forms — HF dataset is underscore, some mirrors hyphen
        for hf_id in ("lukebarousse/data_jobs", "lukebarousse/data-jobs"):
            try:
                ds = load_dataset(hf_id, split="train", streaming=False)
                break
            except Exception:
                continue
        else:
            return None
        # sample
        ds = ds.shuffle(seed=42).select(range(min(sample, len(ds))))
        jobs = []
        for i, row in enumerate(ds, 1):
            title = str(row.get("job_title_short") or row.get("job_title") or "Data Scientist")[:120]
            company = str(row.get("company_name") or row.get("company") or "Unknown")[:80]
            location = str(row.get("job_location") or row.get("location") or "Remote")[:80]
            desc = str(row.get("job_description") or row.get("description") or "").strip()
            skills = row.get("skills") or row.get("job_skills") or []
            # HF lukebarousse/data_jobs has no job_description — synthesize from skills/title
            if not desc or len(desc) < 20:
                skill_str = ", ".join(skills) if isinstance(skills, list) else str(skills)
                # keep original title_full if available
                title_full = str(row.get("job_title") or title)
                desc = f"We are hiring a {title} ({title_full}) at {company} ({location}). Role: {title_full}. Required skills: {skill_str}. You will work on {skill_str[:200]} and collaborate across data and engineering teams. Location: {location}. Apply via {company.lower().replace(' ','')}.com/careers.".strip()
                # preserve as truncated
                desc = desc[:4000]
            # normalize skills to list
            if isinstance(skills, str):
                skills = [s.strip() for s in skills.split(",") if s.strip()]
            jobs.append({
                "id": i,
                "title": title,
                "company": company,
                "location": location,
                "description": desc,
                "skills": skills,
                "source": "hf:lukebarousse/data_jobs",
            })
        return jobs
    except Exception as e:
        print(f"[fetch_hf] HF unavailable: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=500)
    ap.add_argument("--out", type=str, default="data/raw/jobs.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    jobs = None
    # Try kagglehub first, then HF, then synthetic fallback — ensures offline CI
    jobs = try_fetch_kagglehub(args.sample)
    if not jobs or len(jobs) < args.sample // 2:
        hf_jobs = try_fetch_hf(args.sample)
        if hf_jobs:
            jobs = hf_jobs
    if not jobs or len(jobs) < args.sample // 2:
        print(f"[fetch] Using synthetic fallback for {args.sample} jobs (offline mode)")
        # Ensure mix: 60% relevant to Ahmed's profile for meaningful eval
        jobs = [synth_job(i) for i in range(1, args.sample + 1)]

    # Dedup by title+company
    seen = set()
    deduped = []
    for j in jobs:
        key = (j["title"].lower().strip(), j["company"].lower().strip())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(j)
    # Pad if dedup removed too many
    while len(deduped) < args.sample:
        deduped.append(synth_job(len(deduped)+1))

    deduped = deduped[:args.sample]
    # Re-id sequentially
    for i, j in enumerate(deduped, 1):
        j["id"] = i

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for j in deduped:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")
    print(f"Wrote {len(deduped)} jobs to {out}")

if __name__ == "__main__":
    main()
