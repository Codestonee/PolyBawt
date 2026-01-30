import { WhyNotTrading } from "@/components/alpha/WhyNotTrading"
import { EdgeDecomposition } from "@/components/alpha/EdgeDecomposition"
import { CapitalTimeline } from "@/components/alpha/CapitalTimeline"
import { CorrelationMatrix } from "@/components/alpha/CorrelationMatrix"
import { BrainCircuit } from "lucide-react"

export default function AlphaPage() {
    return (
        <div className="space-y-4">
            {/* Page Header is integrated into layouts typically, but keeping simple header for now */}
            <header className="flex items-center gap-2 mb-4">
                <BrainCircuit className="h-5 w-5 text-text-secondary" />
                <h1 className="text-sm font-bold uppercase tracking-wider text-text-primary">Alpha Intelligence</h1>
            </header>

            <div className="grid grid-cols-1 gap-1 lg:grid-cols-3">
                {/* Left Col: Inspector */}
                <div className="lg:col-span-1">
                    <WhyNotTrading />
                </div>

                {/* Right Col: Charts */}
                <div className="gap-1 lg:col-span-2 grid grid-cols-1 md:grid-cols-2">
                    <div className="md:col-span-2">
                        <EdgeDecomposition />
                    </div>
                    <CapitalTimeline />
                    <CorrelationMatrix />
                </div>
            </div>
        </div>
    )
}
