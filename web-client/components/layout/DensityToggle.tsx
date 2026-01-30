"use client"

import { useRiskStore } from "@/stores/risk-store"
import { cn } from "@/lib/utils"
import { Grid, StretchHorizontal } from "lucide-react"

export function DensityToggle() {
    const density = useRiskStore((s) => s.density)
    const setDensity = useRiskStore((s) => s.setDensity)

    return (
        <div className="flex items-center border border-border rounded-sm bg-surface-1 p-0.5">
            <button
                onClick={() => setDensity('comfortable')}
                className={cn(
                    "p-1 rounded-[1px] transition-colors",
                    density === 'comfortable' ? "bg-surface-3 text-text-primary" : "text-text-muted hover:text-text-secondary"
                )}
                title="Comfortable Density"
            >
                <StretchHorizontal className="h-3 w-3" />
            </button>
            <button
                onClick={() => setDensity('compact')}
                className={cn(
                    "p-1 rounded-[1px] transition-colors",
                    density === 'compact' ? "bg-surface-3 text-text-primary" : "text-text-muted hover:text-text-secondary"
                )}
                title="Compact Density"
            >
                <Grid className="h-3 w-3" />
            </button>
        </div>
    )
}
