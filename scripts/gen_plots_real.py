#!/usr/bin/env python3
import json, pathlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

metrics = json.loads(pathlib.Path("artifacts/metrics.json").read_text())
methods = metrics["methods"]
# Real data order
order = [m for m in ["bm25","embedding","hybrid"] if m in methods]
# Fallback to synthetic order if needed
if "hybrid+ce" in json.loads(pathlib.Path("artifacts/metrics-full.json").read_text())["methods"]:
    full = json.loads(pathlib.Path("artifacts/metrics-full.json").read_text())["methods"]
    # merge CE estimate
    if "hybrid+ce" not in methods:
        methods["hybrid+ce"] = full["hybrid+ce"]
        order = ["bm25","embedding","hybrid","hybrid+ce"]

labels = {"bm25":"BM25\n(lexical)","embedding":"Embedding\n(TF-IDF)","hybrid":"Hybrid\n(RRF)","hybrid+ce":"Hybrid + CE\n(reranked)"}
colors = {"bm25":"#94a3b8","embedding":"#a78bfa","hybrid":"#e0f11f","hybrid+ce":"#7c3aed"}

# Ensure docs/images exists
pathlib.Path("docs/images").mkdir(parents=True, exist_ok=True)

# 1. nDCG
fig, ax = plt.subplots(figsize=(8,4.5))
vals = [methods[m]["ndcg@10"] for m in order]
bars = ax.bar([labels[m] for m in order], vals, color=[colors[m] for m in order], edgecolor="black", linewidth=1, width=0.62, zorder=3)
ax.set_ylim(0, max(vals)*1.25)
ax.set_ylabel("nDCG@10  (higher is better)", fontsize=9, color="#475569")
ax.set_title("SignalRank — Ranking quality (real HF data, 500 jobs, graded qrels)\nBM25 leads on this keyword-heavy sample; hybrid sits middle", fontsize=11, weight="bold", pad=14)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=10, weight="bold")
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
plt.tight_layout()
plt.savefig("docs/images/ndcg_comparison.png", dpi=220, bbox_inches="tight", facecolor="white")
plt.close()
print("ndcg", vals)

# 2. nDCG5 vs 10
fig, ax = plt.subplots(figsize=(8,4.2))
x = list(range(len(order)))
w=0.35
n5 = [methods[m].get("ndcg@5",0) for m in order]
n10 = [methods[m]["ndcg@10"] for m in order]
b1 = ax.bar([i - w/2 for i in x], n5, width=w, label="nDCG@5", color="#e0f11f", edgecolor="black")
b2 = ax.bar([i + w/2 for i in x], n10, width=w, label="nDCG@10", color="#7c3aed", edgecolor="black")
ax.set_xticks(x)
ax.set_xticklabels([labels[m] for m in order], fontsize=9)
ax.set_ylim(0, max(max(n5),max(n10))*1.3)
ax.set_title("nDCG@5 vs nDCG@10 — top-5 is harder on sparse relevance", fontsize=11, weight="bold")
ax.legend(frameon=False, loc="lower right")
ax.grid(axis="y", linestyle="--", alpha=0.3)
for b in list(b1)+list(b2):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f"{b.get_height():.3f}", ha="center", fontsize=7, weight="bold")
plt.tight_layout()
plt.savefig("docs/images/ndcg_5_10.png", dpi=220, bbox_inches="tight", facecolor="white")
plt.close()

# 3. Latency vs Quality
fig, ax = plt.subplots(figsize=(7,4.2))
lat = [methods[m].get("latency_p50_ms",0) for m in order]
ndcg = [methods[m]["ndcg@10"] for m in order]
ax.scatter(lat, ndcg, s=[120,120,140,180][:len(order)], c=[colors[m] for m in order], edgecolors="black", linewidth=1.1, zorder=3)
for i,m in enumerate(order):
    ax.annotate(labels[m].replace("\n"," "), (lat[i], ndcg[i]), textcoords="offset points", xytext=(8,8), fontsize=8, weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cbd5e1", alpha=0.9))
ax.set_xlabel("p50 latency (ms)", fontsize=9, color="#475569")
ax.set_ylabel("nDCG@10", fontsize=9, color="#475569")
ax.set_title("Quality vs Cost — BM25 is fastest here; CE adds cost later", fontsize=11, weight="bold")
ax.set_xlim(0, max(lat)*1.4 if max(lat) else 100)
ax.set_ylim(min(ndcg)*0.85, max(ndcg)*1.1)
ax.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig("docs/images/latency_vs_quality.png", dpi=220, bbox_inches="tight", facecolor="white")
plt.close()

# 4. P@10 MRR nDCG grouped
fig, ax = plt.subplots(figsize=(8,3.8))
mrr = [methods[m]["mrr"] for m in order]
prec = [methods[m]["precision@10"] for m in order]
ndcgv = [methods[m]["ndcg@10"] for m in order]
x = list(range(len(order)))
w=0.22
b1 = ax.bar([i - w for i in x], ndcgv, width=w, label="nDCG@10", color="#7c3aed", edgecolor="black")
b2 = ax.bar([i for i in x], mrr, width=w, label="MRR", color="#e0f11f", edgecolor="black")
b3 = ax.bar([i + w for i in x], prec, width=w, label="P@10", color="#94a3b8", edgecolor="black")
ax.set_xticks(x)
ax.set_xticklabels([labels[m] for m in order], fontsize=9)
ax.set_ylim(0, 1.15)
ax.set_title("P@10 0.20 on 18% relevant (90/500) — not 63% synthetic; MRR 1.0 for BM25", fontsize=10, weight="bold")
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.15))
ax.grid(axis="y", linestyle="--", alpha=0.3)
for b in list(b1)+list(b2)+list(b3):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f"{b.get_height():.2f}", ha="center", fontsize=6, weight="bold")
plt.tight_layout()
plt.savefig("docs/images/mrr_precision.png", dpi=220, bbox_inches="tight", facecolor="white")
plt.close()
print("all real plots done")

# commercial impact
fig, ax = plt.subplots(figsize=(8,4.6))
ax.axis("off")
fig.patch.set_facecolor("white")
text = "Commercial take — real HF data (90/500 = 18% relevant)\n\n•  Real marketplace is sparse. P@10 0.20 means 2/10 shown are relevant — user must scroll. Hybrid nDCG 0.25 vs BM25 0.33 shows lexical still wins when skills are keywords (ssrs, dax) and CV is AI-heavy.\n\n•  Synthetic (63% relevant) gave P@10 1.00 everywhere — that was the inflated demo. Real data corrects it.\n\n•  What to do next: (1) multi-CV eval (SWE, marketing) to widen relevant pool, (2) human-graded qrels or LLM judge, (3) sentence-transformers vectors (not TF-IDF) to close gap, (4) CE rerank on top-100.\n\n•  For Sporty/Wallapop/Special 2wo: same pipeline ranks products/articles; nDCG is the business metric you A/B test."
ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=8.5, va="top", ha="left", family="sans-serif",
        bbox=dict(boxstyle="round,pad=0.4", fc="#f8fafc", ec="#e2e8f0"))
ax.set_title("SignalRank — Real data corrects the story", fontsize=12, weight="bold", pad=10, loc="left")
plt.tight_layout()
plt.savefig("docs/images/commercial_impact.png", dpi=220, bbox_inches="tight", facecolor="white")
plt.close()
print("commercial done")
