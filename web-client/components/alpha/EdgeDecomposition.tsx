"use client"

import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from "recharts"

const DATA = [
    { day: 'Mon', spread: 150, directional: 45, arb: 10 },
    { day: 'Tue', spread: 230, directional: -20, arb: 15 },
    { day: 'Wed', spread: 180, directional: 120, arb: 30 },
    { day: 'Thu', spread: 290, directional: 80, arb: 25 },
    { day: 'Fri', spread: 200, directional: 40, arb: 50 },
    { day: 'Sat', spread: 120, directional: 10, arb: 40 },
    { day: 'Sun', spread: 140, directional: -5, arb: 20 },
]

export function EdgeDecomposition() {
    return (
        <div className="col-span-2 panel p-3">
            <div className="mb-3">
                <h2 className="text-xs font-bold uppercase tracking-wider text-text-primary">Edge Decomposition</h2>
                <p className="text-[10px] text-text-muted">PnL attribution by strategy source</p>
            </div>

            <div className="h-[200px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={DATA} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                        <XAxis
                            dataKey="day"
                            stroke="#666"
                            fontSize={10}
                            tickLine={false}
                            axisLine={false}
                        />
                        <YAxis
                            stroke="#666"
                            fontSize={10}
                            tickLine={false}
                            axisLine={false}
                            tickFormatter={(value) => `$${value}`}
                        />
                        <Tooltip
                            cursor={{ fill: '#333', opacity: 0.1 }}
                            contentStyle={{ backgroundColor: '#1E1E1E', borderColor: '#333', fontSize: '10px' }}
                        />
                        <Legend wrapperStyle={{ fontSize: '10px', marginTop: '5px' }} iconSize={8} />

                        <Bar dataKey="spread" name="Spread Capture" stackId="a" fill="var(--color-info)" radius={[0, 0, 0, 0]} />
                        <Bar dataKey="directional" name="Directional" stackId="a" fill="var(--color-action)" radius={[0, 0, 0, 0]} />
                        <Bar dataKey="arb" name="Arbitrage" stackId="a" fill="var(--color-success)" radius={[2, 2, 0, 0]} />
                    </BarChart>
                </ResponsiveContainer>
            </div>
        </div>
    )
}
