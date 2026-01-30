"use client"

import * as React from "react"
import { Command } from "cmdk"
import { Search, Shield, Zap, FileText } from "lucide-react"
import { useRiskStore } from "@/stores/risk-store"

export function CommandPalette() {
    const [open, setOpen] = React.useState(false)
    const toggleShadowMode = useRiskStore((state) => state.toggleShadowMode)
    const isShadowMode = useRiskStore((state) => state.isShadowMode)

    React.useEffect(() => {
        const down = (e: KeyboardEvent) => {
            if (e.key === "k" && (e.metaKey || e.ctrlKey)) {
                e.preventDefault()
                setOpen((open) => !open)
            }
        }
        document.addEventListener("keydown", down)
        return () => document.removeEventListener("keydown", down)
    }, [])

    return (
        <Command.Dialog
            open={open}
            onOpenChange={setOpen}
            label="God Mode"
            className="fixed left-1/2 top-1/2 z-[100] w-full max-w-[640px] -translate-x-1/2 -translate-y-1/2 overflow-hidden rounded-xl border border-zinc-800 bg-zinc-950/95 shadow-2xl backdrop-blur-md"
        >
            <div className="flex items-center border-b border-zinc-800 px-3">
                <Search className="h-5 w-5 text-zinc-500" />
                <Command.Input
                    autoFocus
                    placeholder="Type a command or search..."
                    className="flex h-12 w-full bg-transparent px-3 text-sm text-zinc-200 outline-none placeholder:text-zinc-500"
                />
                <kbd className="pointer-events-none inline-flex h-5 select-none items-center gap-1 rounded border border-zinc-700 bg-zinc-800 px-1.5 font-mono text-[10px] font-medium text-zinc-400">
                    ESC
                </kbd>
            </div>

            <Command.List className="max-h-[300px] overflow-y-auto p-2">
                <Command.Empty className="py-6 text-center text-sm text-zinc-500">
                    No results found.
                </Command.Empty>

                <Command.Group heading="Risk Controls" className="mb-2 px-2 text-[10px] uppercase tracking-wider text-zinc-500">
                    <Command.Item
                        onSelect={() => {
                            if (isShadowMode) {
                                // Going live requires check
                                useRiskStore.getState().setPreFlightOpen(true)
                            } else {
                                toggleShadowMode()
                            }
                            setOpen(false)
                        }}
                        className="flex cursor-pointer select-none items-center gap-2 rounded-md px-2 py-2 text-sm text-zinc-200 hover:bg-zinc-800 aria-selected:bg-zinc-800"
                    >
                        <Shield className="h-4 w-4" />
                        <span>{isShadowMode ? "Disable Shadow Mode (Go Live)" : "Enable Shadow Mode"}</span>
                    </Command.Item>
                </Command.Group>

                <Command.Group heading="Actions" className="mb-2 px-2 text-[10px] uppercase tracking-wider text-zinc-500">
                    <Command.Item className="flex cursor-pointer select-none items-center gap-2 rounded-md px-2 py-2 text-sm text-zinc-200 hover:bg-zinc-800 aria-selected:bg-zinc-800">
                        <Zap className="h-4 w-4" />
                        <span>Export Logs to CSV</span>
                    </Command.Item>
                    <Command.Item className="flex cursor-pointer select-none items-center gap-2 rounded-md px-2 py-2 text-sm text-zinc-200 hover:bg-zinc-800 aria-selected:bg-zinc-800">
                        <FileText className="h-4 w-4" />
                        <span>Jump to Market ID...</span>
                    </Command.Item>
                </Command.Group>
            </Command.List>
        </Command.Dialog>
    )
}
