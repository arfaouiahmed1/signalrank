import { cn } from "@/lib/utils"
import { motion, AnimatePresence } from "framer-motion"
import { useState } from "react"

export function HoverEffect({ items, className, onSelect }) {
  const [hovered, setHovered] = useState(null)
  return (
    <div className={cn("grid grid-cols-1 md:grid-cols-2 gap-4", className)}>
      {items.map((item, idx) => (
        <div key={item.job?.id ?? idx} className="relative group block p-2 h-full w-full" onMouseEnter={() => setHovered(idx)} onMouseLeave={() => setHovered(null)}>
          <AnimatePresence>
            {hovered === idx && <motion.span className="absolute inset-0 h-full w-full bg-[#e0f11f]/10 dark:bg-[#e0f11f]/10 block rounded-2xl" layoutId="hoverBackground" initial={{ opacity: 0 }} animate={{ opacity: 1, transition: { duration: 0.15 } }} exit={{ opacity: 0, transition: { duration: 0.15, delay: 0.2 } }} />}
          </AnimatePresence>
          <div className="rounded-2xl h-full w-full p-5 overflow-hidden bg-card border group-hover:border-[#e0f11f]/30 relative z-20 transition-colors">
            <div className="relative z-50">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-mono text-muted-foreground">#{String(item.rank).padStart(2,"0")}</span>
                    {item.ce_score != null && <span className="text-xs font-mono bg-[#e0f11f] text-black px-1.5 py-0.5 rounded">CE {item.ce_score.toFixed(3)}</span>}
                    {item.rrf_score != null && <span className="text-xs font-mono bg-secondary px-1.5 py-0.5 rounded">RRF {item.rrf_score.toFixed(3)}</span>}
                  </div>
                  <h4 className="font-semibold mt-1 line-clamp-1">{item.job?.title}</h4>
                  <p className="text-sm text-muted-foreground">{item.job?.company} · {item.job?.location}</p>
                </div>
                <button onClick={() => onSelect?.(item)} className="text-xs font-medium underline decoration-[#e0f11f] underline-offset-4 shrink-0">Details ↗</button>
              </div>
              <p className="text-sm mt-3 line-clamp-3 text-muted-foreground leading-relaxed">{item.job?.description?.slice(0, 280)}…</p>
              <div className="flex flex-wrap gap-1.5 mt-3">
                {(item.job?.skills || []).slice(0,6).map(s => <span key={s} className="text-[11px] font-medium border rounded-full px-2 py-0.5 bg-secondary">{s}</span>)}
              </div>
              <div className="flex gap-2 mt-3 text-[11px] font-mono text-muted-foreground">
                {item.bm25_score != null && <span>BM25 {item.bm25_score.toFixed(2)}</span>}
                {item.vector_score != null && <span>· Vec {item.vector_score.toFixed(3)}</span>}
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}
