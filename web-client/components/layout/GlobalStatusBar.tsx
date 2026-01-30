"use client"

import { Wifi, ShieldCheck, Zap, Clock, User } from "lucide-react"
import { useRiskStore } from "@/stores/risk-store"
import { cn } from "@/lib/utils"
import { DensityToggle } from "@/components/layout/DensityToggle"

export function GlobalStatusBar() {
    const isShadowMode = useRiskStore((s) => s.isShadowMode)
    const riskLevel = useRiskStore((s) => s.riskLevel)

    return (
        <div className="fixed top-0 left-0 right-0 z-50 flex h-10 items-center justify-between border-b border-border bg-surface-1 px-4 text-xs font-medium">
            {/* Left: System Status */}
            <div className="flex items-center gap-6">
                <div className="flex items-center gap-2">
                    <div className="h-1.5 w-1.5 rounded-full bg-success animate-pulse" />
                    <span className="font-mono text-text-secondary">SYSTEM ONLINE</span>
                </div>

                <div className="h-4 w-[1px] bg-border-subtle" />

                <div className="flex items-center gap-2 text-text-secondary">
                    <Wifi className="h-3 w-3" />
                    <span className="font-mono tabular-nums">12ms</span>
                </div>

                <div className="h-4 w-[1px] bg-border-subtle" />

                <div className="flex items-center gap-2">
                    <span className={cn(
                        "font-mono uppercase transition-colors",
                        isShadowMode ? "text-warning" : "text-success"
                    )}>
                        {isShadowMode ? "SIMULATION MODE" : "LIVE TRADING"}
                    </span>
                </div>
            </div>

            {/* Center: Clock (Optional, can be added later) */}

            {/* Right: Risk & Account */}
            <div className="flex items-center gap-6">
                <DensityToggle />

                <div className="h-4 w-[1px] bg-border-subtle" />

                <div className="flex items-center gap-2">
                    <span className="text-text-muted">RISK STATE:</span>
                    <span className={cn(
                        "uppercase",
                        riskLevel === 'normal' ? "text-success" :
                            riskLevel === 'flatline' ? "text-danger animate-pulse" : "text-warning"
                    )}>
                        {riskLevel}
                    </span>
                </div>

                <div className="h-4 w-[1px] bg-border-subtle" />

                <div className="flex items-center gap-2 text-text-secondary hover:text-text-primary cursor-pointer">
                    <User className="h-3 w-3" />
                    <span>ADMIN</span>
                </div>
            </div>
        </div>
    )
}
