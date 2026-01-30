"use client"

import { AlertOctagon, Wifi, Clock, ServerCrash } from "lucide-react"

const INCIDENTS = [
    { time: '04:02:15', type: 'disconnect', message: 'WebSocket connection lost', severity: 'high' },
    { time: '04:02:18', type: 'recover', message: 'Reconnected (Backoff: 3000ms)', severity: 'success' },
    { time: '08:45:00', type: 'rate_limit', message: '429 Rate Limit (Order: Submit)', severity: 'medium' },
    { time: '11:20:10', type: 'lag', message: 'High Latency Warning (>500ms)', severity: 'low' },
]

export function IncidentCenter() {
    return (
        <div className="panel p-3">
            <div className="mb-3 flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <AlertOctagon className="h-4 w-4 text-text-muted" />
                    <h2 className="text-xs font-bold uppercase tracking-wider text-text-primary">Incident Log</h2>
                </div>
                <div className="px-1.5 py-0.5 text-[10px] font-mono text-text-muted bg-surface-2 rounded-sm border border-border-subtle">
                    24H VIEW
                </div>
            </div>

            <div className="relative border-l border-border ml-2 space-y-3">
                {INCIDENTS.map((inc, i) => (
                    <div key={i} className="relative ml-4">
                        <span className="absolute -left-[21px] flex h-4 w-4 items-center justify-center rounded-full bg-surface-1 border border-border z-10">
                            {inc.type === 'disconnect' && <div className="h-1.5 w-1.5 rounded-full bg-danger" />}
                            {inc.type === 'rate_limit' && <div className="h-1.5 w-1.5 rounded-full bg-warning" />}
                            {inc.type === 'lag' && <div className="h-1.5 w-1.5 rounded-full bg-info" />}
                            {inc.type === 'recover' && <div className="h-1.5 w-1.5 rounded-full bg-success" />}
                        </span>

                        <div className="flex flex-col">
                            <span className="font-mono text-[10px] text-text-muted tabular-nums leading-tight">{inc.time}</span>
                            <span className="text-xs text-text-secondary leading-tight">{inc.message}</span>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    )
}
