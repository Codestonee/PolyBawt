"use client"

import { useEffect, useState } from "react"
import { WifiOff } from "lucide-react"
import { cn } from "@/lib/utils"

export function DataCanary() {
    const [lastUpdate, setLastUpdate] = useState(Date.now())
    const [isStale, setIsStale] = useState(false)

    // Mock: Update "last update" every 2 seconds typically
    // But occasionally pause to simulate lag
    useEffect(() => {
        const dataInterval = setInterval(() => {
            if (Math.random() > 0.3) {
                setLastUpdate(Date.now())
            }
        }, 2000)
        return () => clearInterval(dataInterval)
    }, [])

    // Check for staleness
    useEffect(() => {
        const checkInterval = setInterval(() => {
            const diff = Date.now() - lastUpdate
            setIsStale(diff > 5000)
        }, 1000)
        return () => clearInterval(checkInterval)
    }, [lastUpdate])

    return (
        <>
            {/* Global CSS Filter Trigger */}
            <style jsx global>{`
        body {
          transition: filter 0.5s ease;
          filter: ${isStale ? "grayscale(80%) contrast(1.2)" : "none"};
        }
      `}</style>

            {/* Warning Overlay */}
            {isStale && (
                <div className="fixed top-0 left-0 right-0 z-[100] flex items-center justify-center bg-red-500/90 py-1 shadow-lg backdrop-blur">
                    <div className="flex items-center gap-2 text-sm font-bold text-white animate-pulse">
                        <WifiOff className="h-4 w-4" />
                        <span>DATA CONNECTION UNSTABLE - LAST UPDATE {(Date.now() - lastUpdate) / 1000}s AGO</span>
                    </div>
                </div>
            )}
        </>
    )
}
