"use client"

import * as React from "react"
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, ScatterChart, Scatter, ZAxis } from "recharts"
import { Activity, Clock, Zap } from "lucide-react"

// Mock Data
const LATENCY_DATA = [
    { id: 1, ui: 5, backend: 15, exchange: 45, ack: 20 },
    { id: 2, ui: 4, backend: 12, exchange: 42, ack: 18 },
    { id: 3, ui: 6, backend: 18, exchange: 55, ack: 22 },
    { id: 4, ui: 5, backend: 14, exchange: 48, ack: 19 },
    { id: 5, ui: 4, backend: 10, exchange: 38, ack: 15 },
]

export function MicrostructureMonitor() {
    const [fillQualityData, setFillQualityData] = React.useState<any[]>([])

    React.useEffect(() => {
        setFillQualityData(Array.from({ length: 20 }).map(() => ({
            expected: 0.5 + Math.random() * 0.1,
            realized: 0.5 + Math.random() * 0.1 + (Math.random() > 0.8 ? 0.02 : 0),
            size: Math.random() * 100
        })))
    }, [])
    return (
        <div className="grid grid-cols-1 gap-1 lg:grid-cols-3">
            {/* Latency Widget */}
            <div className="lg:col-span-1 panel p-3">
                <div className="mb-3 flex items-center gap-2">
                    <Clock className="h-4 w-4 text-text-muted" />
                    <span className="text-xs font-bold uppercase tracking-wider text-text-secondary">Latency Breakdown (ms)</span>
                </div>
                <div className="space-y-1">
                    {LATENCY_DATA.map((item) => (
                        <div key={item.id} className="flex h-5 w-full items-center overflow-hidden rounded-[1px] text-[10px] font-mono leading-none">
                            <div style={{ width: `${(item.ui / 80) * 100}%` }} className="h-full bg-info/80 flex items-center justify-center text-black/50">{item.ui}</div>
                            <div style={{ width: `${(item.backend / 80) * 100}%` }} className="h-full bg-warning/80 flex items-center justify-center text-black/50">{item.backend}</div>
                            <div style={{ width: `${(item.exchange / 80) * 100}%` }} className="h-full bg-danger/80 flex items-center justify-center text-white/50">{item.exchange}</div>
                            <div style={{ width: `${(item.ack / 80) * 100}%` }} className="h-full bg-success/80 flex items-center justify-center text-black/50">{item.ack}</div>
                        </div>
                    ))}
                </div>
                <div className="mt-2 flex justify-between px-1 text-[9px] text-text-muted font-mono">
                    <div className="flex items-center gap-1"><div className="h-1.5 w-1.5 rounded-full bg-info" />UI</div>
                    <div className="flex items-center gap-1"><div className="h-1.5 w-1.5 rounded-full bg-warning" />Backend</div>
                    <div className="flex items-center gap-1"><div className="h-1.5 w-1.5 rounded-full bg-danger" />Exchange</div>
                    <div className="flex items-center gap-1"><div className="h-1.5 w-1.5 rounded-full bg-success" />Ack</div>
                </div>
            </div>

            {/* Rate Limits */}
            <div className="lg:col-span-1 panel p-3 flex flex-col items-center justify-center">
                <div className="mb-2 flex items-center gap-2">
                    <Zap className="h-4 w-4 text-text-muted" />
                    <span className="text-xs font-bold uppercase tracking-wider text-text-secondary">Rate Limits</span>
                </div>
                <div className="relative flex h-20 w-20 items-center justify-center">
                    <svg className="h-full w-full -rotate-90 transform">
                        <circle cx="40" cy="40" r="36" className="stroke-surface-2" strokeWidth="6" fill="transparent" />
                        <circle cx="40" cy="40" r="36" className="stroke-success" strokeWidth="6" fill="transparent" strokeDasharray="226.19" strokeDashoffset="33.9" strokeLinecap="butt" />
                    </svg>
                    <div className="absolute flex flex-col items-center">
                        <span className="text-lg font-bold font-mono text-text-primary tabular-nums">85%</span>
                        <span className="text-[9px] uppercase text-text-muted">Remaining</span>
                    </div>
                </div>
            </div>

            {/* Fill Quality */}
            <div className="lg:col-span-1 panel p-3">
                <div className="mb-2 flex items-center gap-2">
                    <Activity className="h-4 w-4 text-text-muted" />
                    <span className="text-xs font-bold uppercase tracking-wider text-text-secondary">Fill Quality</span>
                </div>
                <div className="h-[120px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                        <ScatterChart margin={{ top: 5, right: 5, bottom: 5, left: -20 }}>
                            <XAxis type="number" dataKey="expected" name="Expected" stroke="#666" fontSize={10} tickLine={false} axisLine={false} />
                            <YAxis type="number" dataKey="realized" name="Realized" stroke="#666" fontSize={10} tickLine={false} axisLine={false} />
                            <ZAxis type="number" dataKey="size" range={[20, 150]} />
                            <Tooltip cursor={{ strokeDasharray: '2 2', stroke: '#333' }} contentStyle={{ backgroundColor: '#1E1E1E', borderColor: '#333', fontSize: '10px' }} />
                            <Scatter name="Fills" data={fillQualityData} fill="var(--color-warning)" shape="square" />
                        </ScatterChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>
    )
}
