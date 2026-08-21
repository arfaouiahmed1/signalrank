import { useEffect, useState, useCallback } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Textarea } from "@/components/ui/textarea"
import { Input } from "@/components/ui/input"
import { Slider } from "@/components/ui/slider"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog"
import { Spotlight } from "@/components/aceternity/Spotlight"
import { HoverEffect } from "@/components/aceternity/HoverEffect"
import { BentoGrid, BentoGridItem } from "@/components/aceternity/BentoGrid"
import { TracingBeam } from "@/components/aceternity/TracingBeam"
import { Search, Sparkles, BarChart3, Upload, FileText, Zap, Database, ExternalLink } from "lucide-react"

const API = import.meta.env.VITE_API_URL || "http://localhost:8000"

const SAMPLE_CV = `AI Engineer — Ahmed Arfaoui, Tunis. Python, LangGraph/LangChain, MCP, FastAPI, pgvector, hybrid search, ranking, recommender systems, personalization, cross-encoder, LightGBM, MLflow/DVC/Docker/CI/CD, PyTorch, YOLOv8/ResNet, FAISS, RAG 1000+ docs. Built Open Web Catcher (150+ sites, 97.6% tool-call success), NEWSBOT AI (LoRA/LIME/SHAP), FarmWise (95% disease acc). Seeking Search & Recommendation / Agentic AI / MLOps roles in Tunisia/EU.`

export default function App() {
  const [cvText, setCvText] = useState(SAMPLE_CV)
  const [k, setK] = useState(10)
  const [method, setMethod] = useState("hybrid+ce")
  const [results, setResults] = useState([])
  const [meta, setMeta] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [loading, setLoading] = useState(false)
  const [fileName, setFileName] = useState("")
  const [selected, setSelected] = useState(null)
  const [health, setHealth] = useState(null)

  const fetchMetrics = useCallback(async () => {
    try { const r = await fetch(`${API}/metrics`); if (r.ok) setMetrics(await r.json()) } catch {}
  }, [])
  const fetchHealth = useCallback(async () => {
    try { const r = await fetch(`${API}/health`); if (r.ok) setHealth(await r.json()) } catch {}
  }, [])
  useEffect(() => { fetchMetrics(); fetchHealth() }, [fetchMetrics, fetchHealth])

  const handleFile = async (e) => {
    const f = e.target.files?.[0]
    if (!f) return
    setFileName(f.name)
    if (f.name.toLowerCase().endsWith(".pdf")) {
      // will be sent as file to backend, but also read text for preview
      setCvText(`[PDF: ${f.name} — will be parsed server-side on Rank]`)
    } else {
      const t = await f.text()
      setCvText(t.slice(0, 8000))
    }
    // keep file ref for upload
    e.target._file = f
  }

  const doRank = async () => {
    setLoading(true)
    try {
      const fd = new FormData()
      fd.append("cv_text", cvText)
      fd.append("k", String(k))
      fd.append("method", method)
      const fileInput = document.getElementById("cv-file")
      const file = fileInput?.files?.[0]
      if (file && file.name.toLowerCase().endsWith(".pdf")) {
        fd.set("cv_text", "") // let backend use file
        fd.append("cv_file", file)
      }
       const r = await fetch(`${API}/rank`, { method: "POST", body: fd })
      if (r.status === 429) throw new Error("Rate limited (30/min). Try again in a minute.")
      if (!r.ok) throw new Error(await r.text())
      const j = await r.json()
      setResults(j.results || [])
      setMeta(j.meta || null)
      // refresh metrics after ranking for demo
      fetchMetrics()
    } catch (e) {
      alert("Rank failed: " + e.message + "\nIs API running at " + API + " ?  Try docker compose up.")
    } finally { setLoading(false) }
  }

  return (
    <div className="min-h-screen bg-background relative overflow-hidden">
      {/* grid bg */}
      <div className="absolute inset-0 bg-grid opacity-[0.04] pointer-events-none" />
      <Spotlight className="-top-40 left-0 md:left-60 md:-top-20" fill="#e0f11f" />

      {/* header */}
      <header className="sticky top-0 z-40 backdrop-blur-xl bg-background/70 border-b">
        <div className="max-w-6xl mx-auto px-4 h-14 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-8 w-8 rounded-lg bg-[#e0f11f] flex items-center justify-center font-bold text-black">SR</div>
            <span className="font-semibold tracking-tight">SignalRank</span>
            <Badge variant="accent" className="ml-1 hidden sm:inline-flex">hybrid · pgvector · rerank</Badge>
          </div>
          <div className="flex items-center gap-2 text-xs">
            <span className={`h-2 w-2 rounded-full ${health ? "bg-emerald-500" : "bg-amber-500"} animate-pulse`} />
            <span className="text-muted-foreground hidden sm:inline">{health ? `API ${health.version || "ok"}` : "API offline"}</span>
            <a href="https://github.com/arfaouiahmed1/signalrank" target="_blank" className="ml-2 inline-flex items-center gap-1 text-muted-foreground hover:text-foreground"><ExternalLink className="h-4 w-4" /> GitHub</a>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-8 relative">
        {/* hero */}
        <div className="rounded-3xl border bg-card/50 backdrop-blur p-6 md:p-8 relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-br from-[#e0f11f]/[0.07] via-transparent to-violet-500/[0.07] pointer-events-none" />
          <div className="relative">
            <div className="flex flex-wrap items-center gap-2 mb-3">
              <Badge variant="secondary" className="font-mono text-[11px]">CV → BM25 + embeddings → RRF → cross-encoder → ranked jobs</Badge>
              <Badge variant="outline" className="font-mono text-[11px]">P@K · R@K · MRR · nDCG@K</Badge>
            </div>
            <h1 className="text-3xl md:text-4xl font-bold tracking-tight">Give it a CV + 500 jobs.<br className="hidden md:block" /> Get back ranked jobs.</h1>
            <p className="text-muted-foreground mt-3 max-w-2xl">Not just cosine similarity. Hybrid lexical + vector retrieval via <span className="text-foreground font-medium">pgvector HNSW</span> + <span className="text-foreground font-medium">Postgres FTS</span>, fused with RRF, reranked by <span className="text-foreground font-medium">cross-encoder</span>. Compare embedding-only vs hybrid vs reranked — with real IR metrics.</p>
            <div className="flex flex-wrap gap-2 mt-4 text-xs font-mono">
              <span className="px-2 py-1 rounded bg-secondary">FastAPI</span>
              <span className="px-2 py-1 rounded bg-secondary">pgvector</span>
              <span className="px-2 py-1 rounded bg-secondary">Docker</span>
              <span className="px-2 py-1 rounded bg-[#e0f11f] text-black font-semibold">Hugging Face</span>
              <span className="px-2 py-1 rounded bg-secondary">Kaggle</span>
            </div>
          </div>
        </div>

        {/* bento metrics */}
        {metrics?.methods && (
          <BentoGrid className="mt-6">
            {Object.entries(metrics.methods).map(([name, m]) => (
              <BentoGridItem key={name} title={name} description={`P@10 ${ (m["precision@10"] ?? m["precision@10"] ?? 0).toFixed(2)} · nDCG@10 ${(m["ndcg@10"] ?? 0).toFixed(2)} · MRR ${(m.mrr ?? 0).toFixed(2)}`} header={<div className="flex h-20 w-full bg-gradient-to-br from-violet-500/20 via-transparent to-[#e0f11f]/20 rounded-xl border items-center justify-center"><BarChart3 className="h-8 w-8 text-muted-foreground" /></div>} icon={<Sparkles className="h-4 w-4 text-[#e0f11f]" />} className={name==="hybrid+ce" ? "border-[#e0f11f]/40 bg-[#e0f11f]/[0.04]" : ""} />
            ))}
          </BentoGrid>
        )}

        {/* input */}
        <Card className="mt-6">
          <CardHeader>
            <CardTitle className="flex items-center gap-2"><FileText className="h-5 w-5" /> Your CV</CardTitle>
            <CardDescription>Paste text or upload PDF/TXT. Sample pre-filled from Ahmed's portfolio. Try hybrid vs reranked to see the lift.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Textarea value={cvText} onChange={e=>setCvText(e.target.value)} rows={7} placeholder="Paste CV text…" className="font-mono text-sm" />
            <div className="flex flex-wrap gap-3 items-center">
              <label className="inline-flex items-center gap-2 text-sm border rounded-xl px-3 py-2 cursor-pointer hover:bg-accent">
                <Upload className="h-4 w-4" />
                <span>{fileName || "Upload PDF/TXT"}</span>
                <Input id="cv-file" type="file" accept=".pdf,.txt" onChange={handleFile} className="hidden" />
              </label>
              <div className="flex items-center gap-3 ml-auto">
                <span className="text-sm text-muted-foreground">Top-K</span>
                <div className="w-32"><Slider value={[k]} onValueChange={([v])=>setK(v)} min={1} max={25} step={1} /></div>
                <Badge variant="secondary" className="font-mono">{k}</Badge>
              </div>
            </div>

            <Tabs value={method} onValueChange={setMethod}>
              <TabsList>
                <TabsTrigger value="embedding">Embedding-only</TabsTrigger>
                <TabsTrigger value="bm25">BM25-only</TabsTrigger>
                <TabsTrigger value="hybrid">Hybrid (RRF)</TabsTrigger>
                <TabsTrigger value="hybrid+ce">Hybrid + CE ★</TabsTrigger>
              </TabsList>
              <TabsContent value="embedding"><p className="text-sm text-muted-foreground">Pure vector search — semantic, but misses exact skill tokens like <code>YOLOv8</code>.</p></TabsContent>
              <TabsContent value="bm25"><p className="text-sm text-muted-foreground">Lexical FTS — exact matches, misses paraphrases like “browser automation” ≈ “RPA”.</p></TabsContent>
              <TabsContent value="hybrid"><p className="text-sm text-muted-foreground">RRF fusion of both — best candidate recall. Reranker adds precision at top.</p></TabsContent>
              <TabsContent value="hybrid+ce"><p className="text-sm text-muted-foreground">Full pipeline: hybrid retrieves 100, cross-encoder reranks to top-K. Compare via metrics below.</p></TabsContent>
            </Tabs>

            <Button onClick={doRank} disabled={loading || cvText.trim().length < 20} variant="accent" size="lg" className="w-full md:w-auto">
              {loading ? "Ranking…" : <><Zap className="h-4 w-4 mr-2" /> Rank {k} jobs — {method}</>}
            </Button>
            {meta && <p className="text-xs font-mono text-muted-foreground">BM25 {meta.bm25} · Vector {meta.vector} · Fused {meta.fused} · CE {meta.ce_available ? "on" : "off (hybrid fallback)"}</p>}
          </CardContent>
        </Card>

        {/* results */}
        {results.length > 0 && (
          <div className="mt-8">
            <div className="flex items-center gap-2 mb-4">
              <Search className="h-5 w-5" />
              <h2 className="text-xl font-semibold">Ranked jobs</h2>
              <Badge variant="outline" className="ml-auto font-mono">{results.length} results · {method}</Badge>
            </div>
            <TracingBeam>
              <HoverEffect items={results} onSelect={setSelected} />
            </TracingBeam>
          </div>
        )}

        {results.length === 0 && (
          <Card className="mt-8 border-dashed">
            <CardContent className="py-12 text-center">
              <Database className="h-10 w-10 mx-auto text-muted-foreground mb-3" />
              <p className="font-medium">No ranking yet</p>
              <p className="text-sm text-muted-foreground">Paste your CV and hit Rank. API must be running: <code className="bg-secondary px-1 rounded">docker compose -f infra/docker-compose.yml up</code></p>
            </CardContent>
          </Card>
        )}

        {/* how it works */}
        <Card className="mt-8">
          <CardHeader>
            <CardTitle>How SignalRank works</CardTitle>
            <CardDescription>Evidence-first, traceable — same ethos as Open Web Catcher.</CardDescription>
          </CardHeader>
          <CardContent className="grid md:grid-cols-3 gap-4 text-sm">
            <div className="rounded-xl border p-4 bg-secondary/30"><h4 className="font-semibold">1. Candidate retrieval</h4><p className="text-muted-foreground mt-1">BM25 (Postgres <code>ts_rank</code>) + pgvector HNSW (384d, cosine). Top-100 via RRF fusion (<code>k=60</code>).</p></div>
            <div className="rounded-xl border p-4 bg-secondary/30"><h4 className="font-semibold">2. Rerank</h4><p className="text-muted-foreground mt-1"><code>cross-encoder/ms-marco-MiniLM-L-6-v2</code> scores (CV, JD) pairs. Optional LightGBM LTR on {`{bm25, cosine, ce, overlap}`}. </p></div>
            <div className="rounded-xl border p-4 bg-secondary/30"><h4 className="font-semibold">3. Evaluate</h4><p className="text-muted-foreground mt-1">Graded qrels (0/1/2) → P@K/R@K/MRR/nDCG. Ablation table in README + <code>/metrics</code>. CI gate: hybrid must beat embedding-only.</p></div>
          </CardContent>
        </Card>

        <footer className="mt-10 flex flex-wrap gap-3 text-xs text-muted-foreground">
          <a href="https://huggingface.co/datasets/ahmedarfaoui99/signalrank-jobs" target="_blank" className="inline-flex items-center gap-1 hover:text-foreground"><ExternalLink className="h-3 w-3" /> HF Dataset (ahmedarfaoui99/signalrank-jobs)</a>
          <span>·</span>
          <a href="https://www.kaggle.com/datasets/ahmedarfaoui99/signalrank-jobs-500" target="_blank" className="inline-flex items-center gap-1 hover:text-foreground"><ExternalLink className="h-3 w-3" /> Kaggle (ahmedarfaoui99/signalrank-jobs-500)</a>
          <span>·</span>
          <span>DockerHub: <code>aki47/signalrank-api</code> · <code>aki47/signalrank-frontend</code> (92 pulls)</span>
        </footer>
      </main>

      <Dialog open={!!selected} onOpenChange={(o)=> !o && setSelected(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{selected?.job?.title} — {selected?.job?.company}</DialogTitle>
            <DialogDescription>{selected?.job?.location} · Scores: CE {selected?.ce_score?.toFixed?.(3) ?? "—"} · RRF {selected?.rrf_score?.toFixed?.(3) ?? "—"} · BM25 {selected?.bm25_score?.toFixed?.(2) ?? "—"} · Vec {selected?.vector_score?.toFixed?.(3) ?? "—"}</DialogDescription>
          </DialogHeader>
          <p className="text-sm whitespace-pre-wrap leading-relaxed">{selected?.job?.description}</p>
          <div className="flex flex-wrap gap-1.5 mt-3">{(selected?.job?.skills||[]).map(s=> <Badge key={s} variant="secondary">{s}</Badge>)}</div>
        </DialogContent>
      </Dialog>
    </div>
  )
}
