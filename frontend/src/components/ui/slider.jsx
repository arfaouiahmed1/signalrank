import * as React from "react"
import { cn } from "@/lib/utils"
export function Slider({ value, onValueChange, min=1, max=50, step=1, className }) {
  return (
    <input type="range" min={min} max={max} step={step} value={value?.[0] ?? value} onChange={(e)=> onValueChange?.([Number(e.target.value)])} className={cn("w-full h-2 bg-secondary rounded-lg appearance-none cursor-pointer accent-primary", className)} />
  )
}
