"use client"

import { useState, useEffect } from "react"
import { Check, ShieldCheck, Plane, X } from "lucide-react"
import { cn } from "@/lib/utils"
import { useRiskStore } from "@/stores/risk-store"

interface CheckItem {
    id: string
    label: string
    status: 'pending' | 'ok' | 'fail'
}

export function PreFlightChecklist() {
    const isOpen = useRiskStore(s => s.isPreFlightOpen)
    const setOpen = useRiskStore(s => s.setPreFlightOpen)
    const toggleShadowMode = useRiskStore(s => s.toggleShadowMode)

    const onClose = () => setOpen(false)
    const onConfirm = () => {
        toggleShadowMode() // actually go live
        setOpen(false)
    }

    const [checks, setChecks] = useState<CheckItem[]>([
        { id: 'time', label: 'NTP Time Sync (<100ms offset)', status: 'pending' },
        { id: 'api', label: 'Polymarket API Connection', status: 'pending' },
        { id: 'wallet', label: 'Wallet Allowance Approved', status: 'pending' },
        { id: 'risk', label: 'Risk Limits Loaded', status: 'pending' },
    ])
    const [isChecking, setIsChecking] = useState(false)

    useEffect(() => {
        if (isOpen) {
            setChecks(c => c.map(x => ({ ...x, status: 'pending' })))
            setIsChecking(true)

            // Simulate sequential checks
            let delay = 500
            checks.forEach((check, i) => {
                setTimeout(() => {
                    setChecks(prev => {
                        const next = [...prev]
                        next[i].status = 'ok'
                        return next
                    })
                    if (i === checks.length - 1) setIsChecking(false)
                }, delay)
                delay += 800
            })
        }
    }, [isOpen])

    if (!isOpen) return null

    const allPassed = checks.every(c => c.status === 'ok')

    return (
        <div className="fixed inset-0 z-[200] flex items-center justify-center bg-black/80 backdrop-blur-sm">
            <div className="w-[400px] overflow-hidden rounded-xl border border-zinc-700 bg-zinc-900 shadow-2xl">
                <div className="flex items-center justify-between border-b border-zinc-800 bg-zinc-950 p-4">
                    <div className="flex items-center gap-2">
                        <Plane className="h-5 w-5 text-zinc-400" />
                        <h2 className="font-bold text-zinc-100">Pre-Flight Checks</h2>
                    </div>
                    <button onClick={onClose} className="text-zinc-500 hover:text-zinc-100">
                        <X className="h-5 w-5" />
                    </button>
                </div>

                <div className="p-6 space-y-4">
                    {checks.map((check) => (
                        <div key={check.id} className="flex items-center justify-between">
                            <span className="text-sm text-zinc-300">{check.label}</span>
                            {check.status === 'pending' && <div className="h-4 w-4 animate-spin rounded-full border-2 border-zinc-600 border-t-zinc-400" />}
                            {check.status === 'ok' && <Check className="h-4 w-4 text-emerald-500" />}
                            {check.status === 'fail' && <X className="h-4 w-4 text-red-500" />}
                        </div>
                    ))}
                </div>

                <div className="border-t border-zinc-800 bg-zinc-950 p-4">
                    <button
                        disabled={!allPassed}
                        onClick={onConfirm}
                        className={cn(
                            "flex w-full items-center justify-center gap-2 rounded-lg py-3 font-bold transition-all",
                            allPassed
                                ? "bg-emerald-500 text-black hover:bg-emerald-400 shadow-[0_0_20px_rgba(16,185,129,0.4)]"
                                : "bg-zinc-800 text-zinc-500 cursor-not-allowed"
                        )}
                    >
                        <ShieldCheck className="h-5 w-5" />
                        {isChecking ? "RUNNING CHECKS..." : "ENGAGE LIVE MODE"}
                    </button>
                </div>
            </div>
        </div>
    )
}
