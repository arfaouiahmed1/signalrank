#!/usr/bin/env python3
"""
Multi-CV evaluation: average metrics across all CVs in data/sample/cv_*.txt
Uses build_qrels + compare logic per CV, then averages.
Outputs artifacts/metrics_multi.json with per-CV and macro-averaged metrics.
Realistic: shows that single-CV P@10=0.20 is not universal; different CVs get different relevant counts.
"""
import json, subprocess, pathlib, sys

SAMPLE_DIR = pathlib.Path("data/sample")
JOBS = "data/raw/jobs.jsonl"
OUT = pathlib.Path("artifacts/metrics_multi.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

cvs = sorted(SAMPLE_DIR.glob("cv_*.txt"))
print(f"Found {len(cvs)} CVs: {[p.name for p in cvs]}")

all_results = {}
macro = {}

for cv_path in cvs:
    cv_id = cv_path.stem  # cv_information_technology
    qrels_path = f"artifacts/qrels_{cv_id}.jsonl"
    metrics_path = f"artifacts/metrics_{cv_id}.json"
    # Build qrels for this CV
    print(f"\n=== {cv_id} ===")
    # Use build_qrels with this CV
    # We need to set cv-id to cv_id
    ret = subprocess.run(
        [sys.executable, "scripts/build_qrels.py", "--jobs", JOBS, "--cv", str(cv_path), "--out", qrels_path, "--cv-id", cv_id],
        capture_output=True, text=True
    )
    print(ret.stdout.strip().split("\n")[-1] if ret.stdout else "")
    if ret.stderr:
        print("ERR", ret.stderr[:200])
    # Run compare — Windows needs ; separator
    import os
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(["backend", "."])}
    ret = subprocess.run(
        [sys.executable, "backend/app/evaluation/compare.py", "--jobs", JOBS, "--qrels", qrels_path, "--cv", str(cv_path), "--mode", "hybrid-only", "--out", metrics_path],
        capture_output=True, text=True, env=env
    )
    if ret.returncode != 0:
        print(f"compare failed for {cv_id}: {ret.stderr[:500]}")
        continue
    m = json.loads(pathlib.Path(metrics_path).read_text())
    all_results[cv_id] = m
    print(f"{cv_id}: {m['relevant_ge1']}/{m['jobs']} relevant, nDCG hybrid {m['methods']['hybrid']['ndcg@10']:.3f}, BM25 {m['methods']['bm25']['ndcg@10']:.3f}")

# Macro average
if not all_results:
    print("No results")
    sys.exit(1)

# Average across CVs per method
methods = set()
for v in all_results.values():
    methods.update(v["methods"].keys())

macro_methods = {}
for meth in methods:
    # Collect per-metric averages
    # Assume all have same metrics keys
    keys = next(iter([v["methods"][meth].keys() for v in all_results.values() if meth in v["methods"]]), [])
    avg = {}
    for k in keys:
        vals = [v["methods"][meth][k] for v in all_results.values() if meth in v["methods"] and k in v["methods"][meth]]
        if vals:
            avg[k] = sum(vals)/len(vals)
    macro_methods[meth] = avg

# Also average relevant counts
avg_relevant = sum(v["relevant_ge1"] for v in all_results.values())/len(all_results)

out = {
    "cvs": list(all_results.keys()),
    "jobs": next(iter(all_results.values()))["jobs"],
    "avg_relevant_ge1": avg_relevant,
    "per_cv": {k: {"relevant_ge1": v["relevant_ge1"], "relevant_eq2": v["relevant_eq2"], "methods": v["methods"]} for k,v in all_results.items()},
    "macro_avg": {"methods": macro_methods},
    "notes": "Macro-averaged across diverse CVs from Kaggle resume datasets + Ahmed CV. Hybrid-only (TF-IDF fallback). Real HF jobs (500). Shows single-CV 0.20 P@10 is not universal."
}

OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(f"\nWrote {OUT}")
print(json.dumps({"macro_avg": macro_methods, "per_cv_relevant": {k: v["relevant_ge1"] for k,v in all_results.items()}}, indent=2))
