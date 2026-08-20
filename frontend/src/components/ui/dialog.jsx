import * as React from "react"
import { cn } from "@/lib/utils"

export function Dialog({ open, onOpenChange, children }) {
  if (!open) return null
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="fixed inset-0 bg-black/60 backdrop-blur-sm" onClick={() => onOpenChange?.(false)} />
      <div className="relative z-50 w-full max-w-2xl max-h-[85vh] overflow-auto">{children}</div>
    </div>
  )
}
export function DialogContent({ className, children, ...props }) {
  return <div className={cn("m-4 bg-card border rounded-2xl shadow-xl p-6", className)} {...props}>{children}</div>
}
export function DialogHeader({ className, ...props }) { return <div className={cn("flex flex-col space-y-1.5 mb-4", className)} {...props} /> }
export function DialogTitle({ className, ...props }) { return <h3 className={cn("text-lg font-semibold", className)} {...props} /> }
export function DialogDescription({ className, ...props }) { return <p className={cn("text-sm text-muted-foreground", className)} {...props} /> }
