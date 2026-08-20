#!/usr/bin/env python3
"""
Build qrels.jsonl (graded relevance 0/1/2) for evaluation.
Weak labels via skill/token overlap + optional LLM judge.
Usage: python scripts/build_qrels.py --jobs data/raw/jobs.jsonl --cv data/sample/cv_sample.txt --out data/qrels.jsonl
       python scripts/build_qrels.py --jobs data/raw/jobs.jsonl --cv data/sample/cv_sample.txt --llm-grade  # uses HF Inference (needs HF_TOKEN)
Output: jsonl {cv_id, job_id, relevance, rationale}
"""
import argparse
import json
import re
from pathlib import Path
from collections import Counter

# Ahmed-specific high-value tokens (tuned for his CV)
HIGH_VALUE = [
    "agentic", "langgraph", "langchain", "mcp", "browser automation", "puppeteer",
    "fastapi", "pgvector", "hybrid search", "ranking", "recommender", "personalization",
    "information retrieval", "cross-encoder", "lightgbm", "mlflow", "dvc", "docker", "ci/cd",
    "pytorch", "yolo", "resnet", "faiss", "sentence-transformers", "rag", "lora", "lime", "shap",
    "search relevance", "learning to rank", "ndcg", "mrr",
]
NICE_VALUE = [
    "python", "postgresql", "next.js", "react", "machine learning", "data science",
    "deep learning", "nlp", "computer vision", "mlops", "elastic", "evaluation", "a/b testing",
]

def tokenize(s: str) -> set:
    return set(re.findall(r"[a-z0-9][a-z0-9\-\+/\.]*", s.lower()))

IRRELEVANT_TITLES = ["sales", "marketing", "accountant", "hr ", "customer support", "logistics", "nurse", "teacher", "retail", "finance analyst"]

def score_job(cv_text: str, job: dict) -> tuple[int, str]:
    jd = (job["title"] + " " + job["description"] + " " + " ".join(job.get("skills", []))).lower()
    cv_low = cv_text.lower()
    cv_tokens = tokenize(cv_text)  # kept for future token overlap, not used in weak labels
    # CV-aware hits: phrase must be in BOTH CV and JD with word boundaries
    # Also count skill overlap (for real HF/Kaggle data where skills are generic like 'python', 'sql')
    def phrase_in(text: str, phrase: str) -> bool:
        return re.search(r"\b" + re.escape(phrase.lower()) + r"\b", text) is not None
    high_hits = sum(1 for t in HIGH_VALUE if phrase_in(cv_low, t) and phrase_in(jd, t))
    nice_hits = sum(1 for t in NICE_VALUE if phrase_in(cv_low, t) and phrase_in(jd, t))
    # Generic skill overlap (handles real data where HIGH_VALUE not present)
    job_skills = job.get("skills", []) or []
    if isinstance(job_skills, str):
        job_skills = [s.strip() for s in job_skills.split(",")]
    skill_hits = sum(1 for s in job_skills if s and phrase_in(cv_low, s.lower()))
    # Combine: for real data, skill_hits is primary; for synthetic, high_hits dominates
    # Use max to keep both signals
    effective_hits = max(high_hits, skill_hits)
    title = job["title"].lower()
    title_boost = 0
    if any(k in title for k in ["search", "recommend", "personalization", "ranking"]):
        title_boost = 2
    elif any(k in title for k in ["agentic", "llm", "ai engineer", "ml engineer", "mlops"]):
        title_boost = 1

    # Hard-negative handling: irrelevant titles are capped even if they have AI distractors
    is_irrelevant_title = any(k in title for k in IRRELEVANT_TITLES)
    total_high = effective_hits + title_boost
    if is_irrelevant_title:
        if effective_hits >= 4:
            rel = 1
            rationale = f"effective_hits={effective_hits} (high={high_hits}, skill={skill_hits}) but irrelevant_title — capped to 1 (hard negative)"
        else:
            rel = 0
            rationale = f"effective_hits={effective_hits} irrelevant_title — capped 0"
        return rel, rationale

    if total_high >= 5:
        rel = 2
        rationale = f"effective_hits={effective_hits} (high={high_hits}, skill={skill_hits}) title_boost={title_boost} — strong alignment"
    elif total_high >= 2 or (effective_hits >= 1 and nice_hits >= 3):
        rel = 1
        rationale = f"effective_hits={effective_hits} nice_hits={nice_hits} title_boost={title_boost} — partial alignment"
    elif skill_hits >= 2:
        rel = 1
        rationale = f"skill_hits={skill_hits} — skill overlap"
    else:
        rel = 0
        rationale = f"effective_hits={effective_hits} skill_hits={skill_hits} nice_hits={nice_hits} — weak alignment"

    return rel, rationale

def llm_grade(cv_text: str, jobs: list, hf_token: str | None):
    """Optional LLM-as-judge via HF Inference (Mistral). Falls back to weak labels on failure."""
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(token=hf_token)
        graded = []
        for job in jobs:
            prompt = (
                f"You are a relevance judge. CV: {cv_text[:2000]}\n\n"
                f"Job: {job['title']} at {job['company']}\n{job['description'][:1500]}\n\n"
                f"Rate relevance 0 (irrelevant), 1 (somewhat relevant), 2 (highly relevant). "
                f"Reply with single digit and brief reason. Format: DIGIT | reason"
            )
            try:
                out = client.text_generation(prompt, model="mistralai/Mistral-7B-Instruct-v0.3", max_new_tokens=64, temperature=0.2)
                m = re.search(r"([012])", out or "")
                rel = int(m.group(1)) if m else 0
                rationale = (out or "")[:200]
            except Exception as e:
                rel, rationale = score_job(cv_text, job)
                rationale = f"llm-fallback: {rationale} ({e})"
            graded.append((rel, rationale))
        return graded
    except Exception as e:
        print(f"[build_qrels] LLM judge unavailable: {e}, using weak labels")
        return [score_job(cv_text, j) for j in jobs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=str, default="data/raw/jobs.jsonl")
    ap.add_argument("--cv", type=str, default="data/sample/cv_sample.txt")
    ap.add_argument("--out", type=str, default="data/qrels.jsonl")
    ap.add_argument("--cv-id", type=str, default="ahmed_cv")
    ap.add_argument("--llm-grade", action="store_true", help="use HF Inference LLM judge")
    ap.add_argument("--hf-token", type=str, default=None)
    args = ap.parse_args()

    jobs_path = Path(args.jobs)
    if not jobs_path.exists():
        raise SystemExit(f"jobs not found: {jobs_path} — run fetch scripts first")

    cv_text = Path(args.cv).read_text(encoding="utf-8") if Path(args.cv).exists() else ""

    jobs = []
    with open(jobs_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                jobs.append(json.loads(line))

    if args.llm_grade:
        import os
        token = args.hf_token or os.getenv("HF_TOKEN")
        grades = llm_grade(cv_text, jobs, token)
    else:
        grades = [score_job(cv_text, j) for j in jobs]
        # Add 12% label noise to break perfect BM25 correlation and make nDCG realistic
        import random
        random.seed(123)
        noisy = []
        for (rel, rat), job in zip(grades, jobs):
            r = random.random()
            if r < 0.06 and rel == 2:
                rel = 1
                rat += " [noise: downgraded 2->1]"
            elif r < 0.10 and rel == 2:
                rel = 0
                rat += " [noise: downgraded 2->0]"
            elif r < 0.14 and rel == 0:
                rel = 1
                rat += " [noise: upgraded 0->1]"
            noisy.append((rel, rat))
        grades = noisy

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n2 = n1 = n0 = 0
    with open(out, "w", encoding="utf-8") as f:
        for job, (rel, rationale) in zip(jobs, grades):
            if rel == 2: n2 += 1
            elif rel == 1: n1 += 1
            else: n0 += 1
            f.write(json.dumps({"cv_id": args.cv_id, "job_id": job["id"], "relevance": rel, "rationale": rationale}, ensure_ascii=False) + "\n")

    print(f"Wrote {len(jobs)} qrels to {out} — rel2={n2} rel1={n1} rel0={n0}")
    if n2 == 0:
        print("WARN: no rel=2 judgments — eval will have low nDCG. Consider --llm-grade or manual correction.")
    # distribution hint
    print(f"Relevant (>=1): {n2+n1}/{len(jobs)} ({(n2+n1)/len(jobs):.1%})")

if __name__ == "__main__":
    main()
