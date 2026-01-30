"use client"

import { cn } from "@/lib/utils"

// Mock Correlation Data (Matrix)
const ASSETS = ['TRUMP', 'BIDEN', 'BTC', 'ETH', 'FED', 'CPI']
const MATRIX = [
    [1.0, -0.8, 0.4, 0.3, 0.1, 0.2],  // TRUMP
    [-0.8, 1.0, -0.3, -0.2, 0.0, -0.1], // BIDEN
    [0.4, -0.3, 1.0, 0.9, 0.5, 0.3],   // BTC
    [0.3, -0.2, 0.9, 1.0, 0.4, 0.2],   // ETH
    [0.1, 0.0, 0.5, 0.4, 1.0, 0.6],    // FED
    [0.2, -0.1, 0.3, 0.2, 0.6, 1.0],   // CPI
]

export function CorrelationMatrix() {
    return (
        <div className="panel p-3">
            <div className="mb-3">
                <h2 className="text-xs font-bold uppercase tracking-wider text-text-primary">Correlation Risk</h2>
                <p className="text-[10px] text-text-muted">Asset exposure clustering</p>
            </div>

            <div className="overflow-x-auto no-scrollbar">
                <div className="inline-block min-w-full align-middle">
                    <div className="grid grid-cols-[auto_repeat(6,1fr)] gap-px bg-surface-1">
                        {/* Header Row */}
                        <div className="h-6 w-6" /> {/* Empty corner */}
                        {ASSETS.map((asset) => (
                            <div key={`head-${asset}`} className="flex items-center justify-center text-[9px] font-bold text-text-muted">
                                {asset}
                            </div>
                        ))}

                        {/* Matrix Rows */}
                        {ASSETS.map((rowAsset, rIndex) => (
                            <>
                                {/* Row Label */}
                                <div key={`label-${rowAsset}`} className="flex items-center justify-end px-2 text-[9px] font-bold text-text-muted">
                                    {rowAsset}
                                </div>

                                {/* Cells */}
                                {MATRIX[rIndex].map((val, cIndex) => {
                                    const intensity = Math.abs(val)
                                    const isPositive = val > 0
                                    const isIdentity = rIndex === cIndex

                                    return (
                                        <div
                                            key={`${rowAsset}-${ASSETS[cIndex]}`}
                                            className={cn(
                                                "flex h-6 w-full items-center justify-center rounded-[1px] text-[9px] font-mono tabular-nums",
                                                isIdentity ? "bg-surface-2 text-text-muted" :
                                                    isPositive
                                                        ? `bg-success/${Math.round(intensity * 30)} text-success`
                                                        : `bg-danger/${Math.round(intensity * 30)} text-danger`
                                            )}
                                            title={`Correlation ${rowAsset} <> ${ASSETS[cIndex]}: ${val}`}
                                        >
                                            {isIdentity ? "1.0" : val.toFixed(1)}
                                        </div>
                                    )
                                })}
                            </>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    )
}
