import { cn } from "@/lib/utils"
export const BentoGrid = ({ className, children }) => (
  <div className={cn("grid md:auto-rows-[18rem] grid-cols-1 md:grid-cols-3 gap-4", className)}>{children}</div>
)
export const BentoGridItem = ({ className, title, description, header, icon }) => (
  <div className={cn("row-span-1 rounded-2xl group/bento flex flex-col justify-between space-y-4 border bg-card p-5 shadow-sm hover:shadow-md transition", className)}>
    {header}
    <div>
      <div className="flex items-center gap-2">{icon}<h4 className="font-semibold">{title}</h4></div>
      <p className="text-sm text-muted-foreground mt-2">{description}</p>
    </div>
  </div>
)
