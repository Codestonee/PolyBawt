"use client"

import { Activity, BrainCircuit, ShieldAlert, FlaskConical, LayoutDashboard, Settings, FileText } from "lucide-react"
import { cn } from "@/lib/utils"
import Link from "next/link"
import { usePathname } from "next/navigation"

const NAV_ITEMS = [
    { label: "Overview", icon: LayoutDashboard, href: "/" },
    { label: "Alpha Intelligence", icon: BrainCircuit, href: "/alpha" },
    { label: "Risk Desk", icon: ShieldAlert, href: "/risk" },
    { label: "Replay Lab", icon: FlaskConical, href: "/lab" },
]

const SYS_ITEMS = [
    { label: "Logs", icon: FileText, href: "#logs" },
    { label: "Settings", icon: Settings, href: "#settings" },
]

export function Sidebar() {
    const pathname = usePathname()

    return (
        <nav className="fixed left-0 top-10 bottom-0 w-12 flex flex-col items-center border-r border-border bg-surface-1 py-4 z-40">
            <div className="flex flex-col gap-1 w-full px-1">
                {NAV_ITEMS.map((item) => {
                    const isActive = pathname === item.href
                    return (
                        <Link
                            key={item.href}
                            href={item.href}
                            className={cn(
                                "group relative flex h-10 w-full items-center justify-center rounded-sm transition-colors",
                                isActive
                                    ? "bg-surface-2 text-text-primary border-l-2 border-action"
                                    : "text-text-muted hover:bg-surface-2 hover:text-text-secondary"
                            )}
                            title={item.label}
                        >
                            <item.icon className="h-5 w-5 stroke-[1.5]" />
                        </Link>
                    )
                })}
            </div>

            <div className="mt-auto flex flex-col gap-1 w-full px-1 border-t border-border pt-2">
                {SYS_ITEMS.map((item) => (
                    <button
                        key={item.label}
                        className="group relative flex h-10 w-full items-center justify-center rounded-sm text-text-muted hover:bg-surface-2 hover:text-text-secondary transition-colors"
                        title={item.label}
                    >
                        <item.icon className="h-4 w-4 stroke-[1.5]" />
                    </button>
                ))}
            </div>
        </nav>
    )
}
