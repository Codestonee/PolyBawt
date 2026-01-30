"use client"

import { BarChart, Bar, XAxis, Tooltip, ResponsiveContainer, Cell } from "recharts"

const DATA = [
    { bucket: '0-24h', amount: 2500, label: 'Intraday' },
    { bucket: '1-7d', amount: 1200, label: 'Swing' },
    { bucket: '7-30d', amount: 800, label: 'Medium' },
    { bucket: '>1mo', amount: 300, label: 'Long Term' },
]

export function CapitalTimeline() {
    return (
        <div className="panel p-3">
            <div className="mb-3">
                <h2 className="text-xs font-bold uppercase tracking-wider text-text-primary">Capital Timeline</h2>
                <p className="text-[10px] text-text-muted">Capital lockup duration</p>
            </div>

            <div className="h-[150px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={DATA}>
                        <XAxis
                            dataKey="bucket"
                            stroke="#666"
                            fontSize={10}
                            tickLine={false}
                            axisLine={false}
                        />
                        <Tooltip
                            cursor={{ fill: 'transparent' }}
                            contentStyle={{ backgroundColor: '#1E1E1E', borderColor: '#333', fontSize: '10px' }}
                            formatter={(value: any) => [`$${value}`, 'Locked']}
                        />
                        <Bar dataKey="amount" radius={[2, 2, 0, 0]}>
                            {DATA.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={index === 0 ? 'var(--color-success)' : index === 1 ? 'var(--color-info)' : 'var(--color-action)'} />
                            ))}
                        </Bar>
                    </BarChart>
                </ResponsiveContainer>
            </div>
        </div>
    )
}
