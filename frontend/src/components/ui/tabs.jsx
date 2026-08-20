import * as React from "react"
import { cn } from "@/lib/utils"
const TabsContext = React.createContext(null)
export function Tabs({ defaultValue, value, onValueChange, children, className }) {
  const [internal, setInternal] = React.useState(defaultValue)
  const active = value ?? internal
  const set = (v) => { if (value === undefined) setInternal(v); onValueChange?.(v) }
  return <TabsContext.Provider value={{ active, set }}><div className={cn(className)}>{children}</div></TabsContext.Provider>
}
export function TabsList({ className, ...props }) {
  return <div className={cn("inline-flex h-10 items-center justify-center rounded-xl bg-muted p-1 text-muted-foreground", className)} {...props} />
}
export function TabsTrigger({ value, children, className }) {
  const { active, set } = React.useContext(TabsContext)
  const isActive = active === value
  return <button onClick={() => set(value)} className={cn("inline-flex items-center justify-center whitespace-nowrap rounded-lg px-3 py-1.5 text-sm font-medium transition-all", isActive ? "bg-background text-foreground shadow-sm" : "hover:bg-background/50", className)}>{children}</button>
}
export function TabsContent({ value, children, className }) {
  const { active } = React.useContext(TabsContext)
  if (active !== value) return null
  return <div className={cn("mt-4 ring-offset-background focus-visible:outline-none", className)}>{children}</div>
}
