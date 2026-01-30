"use client"

import * as React from "react"
import { Play, Pause, SkipBack, SkipForward } from "lucide-react"
import { useRiskStore } from "@/stores/risk-store"
import { cn } from "@/lib/utils"

export function ReplayScrubber() {
    const { currentTime, setCurrentTime } = useRiskStore()
    const [isPlaying, setIsPlaying] = React.useState(false)

    const [events, setEvents] = React.useState<Array<{ left: string, type: string }>>([])

    React.useEffect(() => {
        setEvents(Array.from({ length: 20 }).map(() => ({
            left: `${Math.random() * 100}%`,
            type: Math.random() > 0.7 ? 'trade' : 'info'
        })))
    }, [])

    return (
        <div className="w-full border-t border-zinc-800 bg-zinc-950 p-4">
            <div className="flex items-center gap-4">
                {/* Controls */}
                <div className="flex items-center gap-2">
                    <button className="rounded p-2 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100">
                        <SkipBack className="h-4 w-4" />
                    </button>
                    <button
                        onClick={() => setIsPlaying(!isPlaying)}
                        className="rounded-full bg-amber-500 p-2 text-black hover:bg-amber-400"
                    >
                        {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                    </button>
                    <button className="rounded p-2 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100">
                        <SkipForward className="h-4 w-4" />
                    </button>
                </div>

                {/* Time Display */}
                <div className="font-mono text-sm text-amber-500">
                    {currentTime ? currentTime.toLocaleTimeString() : "LIVE"}
                </div>

                {/* Timeline Bar */}
                <div className="group relative h-12 flex-1 cursor-pointer overflow-hidden rounded bg-zinc-900/50">
                    {/* Events */}
                    {events.map((ev, i) => (
                        <div
                            key={i}
                            className={cn(
                                "absolute bottom-0 h-3 w-[2px] rounded-t-full opacity-50",
                                ev.type === 'trade' ? "bg-emerald-500 h-6 opacity-80" : "bg-zinc-600"
                            )}
                            style={{ left: ev.left }}
                        />
                    ))}

                    {/* Playhead */}
                    <div className="absolute inset-y-0 left-1/2 w-[2px] bg-amber-500 shadow-[0_0_10px_2px_rgba(245,158,11,0.5)]">
                        <div className="absolute top-0 -ml-[5px] border-[6px] border-l-transparent border-r-transparent border-t-amber-500" />
                    </div>

                    {/* Hover Effect */}
                    <div className="absolute inset-0 bg-white/0 transition-colors group-hover:bg-white/5" />
                </div>
            </div>
        </div>
    )
}
