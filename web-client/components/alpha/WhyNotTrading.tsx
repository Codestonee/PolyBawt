"use client"

import { CheckCircle, XCircle, AlertTriangle } from "lucide-react"
import { cn } from "@/lib/utils"

interface Blocker {
    id: string
    label: string
    status: 'pass' | 'fail' | 'warn'
    message: string
    meta?: string
}

const BLOCKERS: Blocker[] = [
    { id: 'risk', label: 'Risk Checks', status: 'pass', message: 'Within daily loss limits' },
    { id: 'spread', label: 'Spread Threshold', status: 'fail', message: 'Spread > 5% (Target: <3%)', meta: 'Current: 5.2%' },
    { id: 'liquidity', label: 'Liquidity Depth', status: 'pass', message: 'Sufficient depth at BBO' },
    { id: 'oracle', label: 'Oracle Freshness', status: 'warn', message: 'Last update 4s ago', meta: 'Max: 5s' },
    { id: 'capital', label: 'Capital Available', status: 'pass', message: 'USDC available for new positions' },
]

export function WhyNotTrading() {
    const failCount = BLOCKERS.filter(b => b.status === 'fail').length

    return (
        <div className="panel p-4">
            <div className="mb-4 flex items-center justify-between">
                <h2 className="text-sm font-bold uppercase tracking-wider text-text-primary">Why Not Trading?</h2>
                <div className={cn(
                    "px-1.5 py-0.5 text-[10px] font-bold font-mono uppercase tracking-wider border rounded-[1px]",
                    failCount === 0 ? "bg-success/10 text-success border-success/20" : "bg-danger/10 text-danger border-danger/20 animate-pulse"
                )}>
                    {failCount === 0 ? "SYSTEM READY" : `${failCount} BLOCKERS`}
                </div>
            </div>

            <div className="space-y-1">
                {BLOCKERS.map((item) => (
                    <div
                        key={item.id}
                        className={cn(
                            "flex items-center justify-between p-2 border-l-2 transition-colors hover:bg-surface-2",
                            item.status === 'pass' ? "border-l-success bg-transparent" :
                                item.status === 'fail' ? "border-l-danger bg-danger/5" :
                                    "border-l-warning bg-warning/5"
                        )}
                    >
                        <div className="flex items-center gap-3">
                            {item.status === 'pass' && <CheckCircle className="h-3.5 w-3.5 text-success" />}
                            {item.status === 'fail' && <XCircle className="h-3.5 w-3.5 text-danger" />}
                            {item.status === 'warn' && <AlertTriangle className="h-3.5 w-3.5 text-warning" />}

                            <div>
                                <div className="text-xs font-medium text-text-primary">{item.label}</div>
                                <div className="text-[10px] text-text-muted">{item.message}</div>
                            </div>
                        </div>

                        {item.meta && (
                            <div className="font-mono text-[10px] text-text-secondary tabular-nums">
                                {item.meta}
                            </div>
                        )}
                    </div>
                ))}
            </div>
        </div>
    )
}
