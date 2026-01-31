"use client"

import { Wifi, ShieldCheck, Zap, Clock, User } from "lucide-react"
import { useRiskStore } from "@/stores/risk-store"
import { cn } from "@/lib/utils"
import { DensityToggle } from "@/components/layout/DensityToggle"
import { useEffect, useState } from "react"

export function GlobalStatusBar() {
    const isShadowMode = useRiskStore((s) => s.isShadowMode)
    const riskLevel = useRiskStore((s) => s.riskLevel)
    const [ping, setPing] = useState<number>(0)

    // Polling for system status
    useEffect(() => {
        const fetchHealth = async () => {
            try {
                const start = performance.now()
                const res = await fetch('http://localhost:8000/health')
                const end = performance.now()

                if (res.ok) {
                    const data = await res.json()
                    useRiskStore.getState().setSystemOnline(true)
                    useRiskStore.getState().setActiveStrategies(data.active_strategies || {})
                    setPing(Math.round(end - start))
                } else {
                    useRiskStore.getState().setSystemOnline(false)
                }
            } catch (e) {
                useRiskStore.getState().setSystemOnline(false)
            }
        }

        // Initial fetch
        fetchHealth()

        // Poll every 5s
        const interval = setInterval(fetchHealth, 5000)
        return () => clearInterval(interval)
    }, [])

    const isSystemOnline = useRiskStore((s) => s.isSystemOnline)
    const activeStrategies = useRiskStore((s) => s.activeStrategies)
    const activeCount = Object.values(activeStrategies).filter(Boolean).length

    return (
        <div className="fixed top-0 left-0 right-0 z-50 flex h-10 items-center justify-between border-b border-border bg-surface-1 px-4 text-xs font-medium">
            {/* Left: System Status */}
            <div className="flex items-center gap-6">
                <div className="flex items-center gap-2">
                    <div className={cn(
                        "h-1.5 w-1.5 rounded-full animate-pulse",
                        isSystemOnline ? "bg-success" : "bg-danger"
                    )} />
                    <span className="font-mono text-text-secondary">
                        {isSystemOnline ? "SYSTEM ONLINE" : "SYSTEM OFFLINE"}
                    </span>
                </div>

                <div className="h-4 w-[1px] bg-border-subtle" />

                <div className="flex items-center gap-2 text-text-secondary">
                    <Wifi className={cn("h-3 w-3", !isSystemOnline && "text-danger")} />
                    <span className="font-mono tabular-nums">{ping}ms</span>
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

            {/* Center: Strategies ticker */}
            <div className="flex items-center gap-4 hidden md:flex">
                <span className="text-text-muted">ACTIVE STRATEGIES ({activeCount}):</span>
                <div className="flex items-center gap-3">
                    {Object.entries(activeStrategies).map(([name, enabled]) => (
                        enabled && (
                            <div key={name} className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-surface-2 border border-border-subtle">
                                <div className="h-1 w-1 rounded-full bg-success" />
                                <span className="text-[10px] uppercase tracking-wider">{name}</span>
                            </div>
                        )
                    ))}
                </div>
            </div>

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
