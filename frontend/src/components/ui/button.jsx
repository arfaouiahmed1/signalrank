import * as React from "react"
import { cva } from "class-variance-authority"
import { cn } from "@/lib/utils"

const buttonVariants = cva(
  "inline-flex items-center justify-center whitespace-nowrap rounded-xl text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default: "bg-primary text-primary-foreground shadow hover:bg-primary/90",
        secondary: "bg-secondary text-secondary-foreground hover:bg-secondary/80",
        outline: "border border-input bg-background hover:bg-accent hover:text-accent-foreground",
        ghost: "hover:bg-accent hover:text-accent-foreground",
        accent: "bg-[#e0f11f] text-black hover:bg-[#d4e61c] font-semibold shadow",
      },
      size: { default: "h-10 px-5 py-2", sm: "h-8 rounded-lg px-3", lg: "h-12 rounded-xl px-8", icon: "h-10 w-10" },
    },
    defaultVariants: { variant: "default", size: "default" },
  }
)

const Button = React.forwardRef(({ className, variant, size, ...props }, ref) => (
  <button className={cn(buttonVariants({ variant, size, className }))} ref={ref} {...props} />
))
Button.displayName = "Button"
export { Button, buttonVariants }
