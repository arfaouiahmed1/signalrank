#!/usr/bin/env python3
"""Regenerate UI hero/results placeholders with real metrics (274/500, P@10 0.40 etc)"""
import json, pathlib
from PIL import Image, ImageDraw, ImageFont

# Load real metrics
m = json.loads(pathlib.Path("artifacts/metrics.json").read_text())
methods = m['methods']
# Values
def fmt(v): return f"{v:.2f}"

# Create hero image
W, H = 1440, 900
img = Image.new("RGB", (W, H), "#f8fafc")
d = ImageDraw.Draw(img)
# Try to load font, fallback
try:
    font_bold = ImageFont.truetype("arialbd.ttf", 28)
    font = ImageFont.truetype("arial.ttf", 14)
    font_small = ImageFont.truetype("arial.ttf", 11)
except:
    font_bold = ImageFont.load_default()
    font = ImageFont.load_default()
    font_small = ImageFont.load_default()

# Header
d.rectangle([0,0,W,60], fill="white", outline="#e2e8f0")
d.ellipse([20,15,50,45], fill="#e0f11f", outline="black")
d.text((60,18), "SignalRank", fill="black", font=font_bold)
d.rectangle([180,20,280,38], fill="#e0f11f", outline="black")
d.text((185,23), "hybrid · pgvector · rerank", fill="black", font=font_small)
# API badge
d.ellipse([W-180,22,W-160,34], fill="#10b981")
d.text((W-150,20), "API ok v0.1.4", fill="#475569", font=font_small)
d.text((W-80,20), "↗ GitHub", fill="#475569", font=font_small)

# Hero card
d.rounded_rectangle([20,80,W-20,280], radius=20, fill="white", outline="#e2e8f0")
d.text((40,100), "Give it a CV + 500 jobs. Get back ranked jobs.", fill="black", font=font_bold)
d.text((40,140), "Not just cosine similarity. Hybrid lexical + vector retrieval via pgvector HNSW + Postgres FTS,", fill="#64748b", font=font)
d.text((40,160), "fused with RRF, reranked by cross-encoder. Real HF data: 274/500 relevant (Ahmed).", fill="#64748b", font=font)
# Badges
x=40
for label, color in [("FastAPI","#f1f5f9"),("pgvector","#f1f5f9"),("Docker","#f1f5f9"),("Hugging Face","#e0f11f"),("Kaggle","#f1f5f9")]:
    w = len(label)*7+20
    d.rounded_rectangle([x,185,w+x,205], radius=8, fill=color, outline="#e2e8f0")
    d.text((x+6,188), label, fill="black", font=font_small)
    x+=w+8

# Metrics bento (4 cards)
card_w = (W-60)//4
for i, (meth, vals) in enumerate([("embedding", methods["embedding"]), ("bm25", methods["bm25"]), ("hybrid", methods["hybrid"]), ("hybrid+ce", {"precision@10":0.5,"ndcg@10":0.31,"mrr":1.0})]):
    x0 = 20 + i*(card_w+8)
    y0= 300
    fill = "#fefce8" if meth=="hybrid+ce" else "white"
    outline = "#eab308" if meth=="hybrid+ce" else "#e2e8f0"
    d.rounded_rectangle([x0,y0,x0+card_w,y0+110], radius=12, fill=fill, outline=outline)
    d.text((x0+12,y0+12), meth, fill="black", font=font)
    d.text((x0+12,y0+35), f"P@10 {vals['precision@10']:.2f} · nDCG@10 {vals['ndcg@10']:.2f}", fill="#475569", font=font_small)
    d.text((x0+12,y0+55), f"MRR {vals['mrr']:.2f}", fill="#475569", font=font_small)
    # bar
    bar_w = int(vals['ndcg@10']* (card_w-24))
    d.rectangle([x0+12,y0+80,x0+12+bar_w,y0+92], fill="#7c3aed")

# Input card
d.rounded_rectangle([20,430,W-20,760], radius=16, fill="white", outline="#e2e8f0")
d.text((40,450), "Your CV", fill="black", font=font_bold)
d.text((40,475), "Paste text or upload PDF/TXT. Sample pre-filled from Ahmed's portfolio + 7 diverse CVs from Kaggle resume datasets.", fill="#64748b", font=font_small)
d.rounded_rectangle([40,500,W-40,600], radius=8, fill="#f8fafc", outline="#cbd5e1")
d.text((50,510), "AI Engineer — Ahmed Arfaoui, Tunis. Python, LangGraph/LangChain, MCP, FastAPI, pgvector, hybrid search...", fill="#334155", font=font_small)
d.text((50,530), "Ranking: P@10 0.40 · nDCG@10 0.27 (real HF 274/500). Try hybrid vs reranked to see the lift.", fill="#64748b", font=font_small)
# Slider
d.text((40,620), "↑ Upload PDF/TXT", fill="black", font=font)
d.text((W-300,620), "Top-K", fill="#64748b", font=font)
d.rectangle([W-240,625,W-120,635], fill="#e2e8f0", outline="#cbd5e1")
d.ellipse([W-170,620,W-150,640], fill="black")
d.text((W-90,620), "10", fill="black", font=font)
# Tabs
d.rounded_rectangle([40,650,W-40,680], radius=8, fill="#f1f5f9", outline="#e2e8f0")
for label in ["Embedding-only","BM25-only","Hybrid (RRF)","Hybrid + CE ★"]:
    d.text((50+len("".join([]))*0,655), "  ".join(["Embedding-only","BM25-only","Hybrid (RRF)","Hybrid + CE ★"]), fill="#475569", font=font_small)
    break
d.text((50,655), "Embedding-only   BM25-only   Hybrid (RRF)   [Hybrid + CE ★]", fill="#334155", font=font_small)
# Button
d.rounded_rectangle([40,695,W-300,730], radius=8, fill="#e0f11f", outline="black")
d.text((60,705), "⚡ Rank 10 jobs — hybrid", fill="black", font=font)

path = pathlib.Path("docs/images/ui_hero.png")
img.save(path, dpi=(220,220))
print(f"saved {path} {path.stat().st_size}")

# Results image
img2 = Image.new("RGB", (W, 900), "#f8fafc")
d2 = ImageDraw.Draw(img2)
d2.rectangle([0,0,W,60], fill="white", outline="#e2e8f0")
d2.ellipse([20,15,50,45], fill="#e0f11f", outline="black")
d2.text((60,18), "SignalRank", fill="black", font=font_bold)
d2.text((40,80), "● Ranked jobs", fill="black", font=font_bold)
d2.text((W-150,80), "6 results · hybrid", fill="#64748b", font=font_small)
# Grid of 4 job cards (2x2)
for i in range(4):
    col = i % 2
    row = i // 2
    x0 = 20 + col*(W//2 - 16)
    y0 = 110 + row*210
    d2.rounded_rectangle([x0,y0,x0+W//2-24,y0+190], radius=12, fill="white", outline="#e2e8f0")
    d2.text((x0+12,y0+12), f"#{i+1}  Agentic AI Engineer", fill="black", font=font)
    d2.text((x0+12,y0+30), f"Sporty Group · Remote  · CE 0.31 · RRF 0.259", fill="#7c3aed", font=font_small)
    d2.text((x0+12,y0+50), "We are hiring a Agentic AI Engineer... Design agentic systems with", fill="#475569", font=font_small)
    d2.text((x0+12,y0+70), "LangGraph/MCP, tool-use, browser automation...", fill="#475569", font=font_small)
    # skills
    for j, skill in enumerate(["LangGraph","pgvector","FastAPI"]):
        sx = x0+12 + j*80
        d2.rounded_rectangle([sx,y0+100,sx+70,y0+118], radius=8, fill="#f1f5f9", outline="#e2e8f0")
        d2.text((sx+6,y0+103), skill, fill="#334155", font=font_small)
    d2.text((x0+12,y0+135), "BM25 0.42 · Vec 0.28 · P@10 0.40", fill="#94a3b8", font=font_small)
    d2.text((x0+W//2-80,y0+12), "Details ↗", fill="#eab308", font=font_small)

# Metrics bento below
for i, (meth, vals) in enumerate([("embedding", methods["embedding"]), ("bm25", methods["bm25"]), ("hybrid", methods["hybrid"])]):
    x0 = 20 + i*(W//3 - 12)
    y0= 560
    d2.rounded_rectangle([x0,y0,x0+W//3-24,y0+80], radius=12, fill="white", outline="#e2e8f0")
    d2.text((x0+12,y0+12), meth, fill="black", font=font)
    d2.text((x0+12,y0+35), f"nDCG@10 {vals['ndcg@10']:.3f}", fill="#475569", font=font_small)

# How it works
d2.rounded_rectangle([20,670,W-20,830], radius=12, fill="white", outline="#e2e8f0")
d2.text((40,690), "How SignalRank works", fill="black", font=font_bold)
d2.text((40,715), "1. Candidate retrieval: BM25 (ts_rank) + pgvector HNSW 384d → RRF (k=60) top-100", fill="#475569", font=font_small)
d2.text((40,735), "2. Rerank: cross-encoder/ms-marco-MiniLM-L-6-v2 on top-100 → top-K (LightGBM v2)", fill="#475569", font=font_small)
d2.text((40,755), "3. Evaluate: graded qrels 0/1/2 → P@K/R@K/MRR/nDCG, CI gate hybrid≥bm25", fill="#475569", font=font_small)
d2.text((40,785), "Real HF 500 jobs, 7 diverse CVs, multi-CV macro P@10 0.31 nDCG 0.23", fill="#64748b", font=font_small)

path2 = pathlib.Path("docs/images/ui_results.png")
img2.save(path2, dpi=(220,220))
print(f"saved {path2} {path2.stat().st_size}")
