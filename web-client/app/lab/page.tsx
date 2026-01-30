import { ReplayScrubber } from "@/components/lab/ReplayScrubber";

export default function LabPage() {
    return (
        <div className="flex h-[calc(100vh-theme(spacing.12))] flex-col">
            <header className="mb-6">
                <h1 className="text-2xl font-bold tracking-tight text-white">Replay Lab</h1>
                <p className="text-sm text-zinc-500">Post-mortem analysis and strategy simulation</p>
            </header>

            <div className="flex flex-1 flex-col gap-4 rounded-xl border border-zinc-800 bg-zinc-950 p-1">

                {/* Main Visualization Area */}
                <div className="relative flex-1 bg-zinc-950">
                    <div className="absolute inset-0 flex items-center justify-center">
                        <span className="font-mono text-sm text-zinc-600">CHART VISUALIZATION AREA</span>
                    </div>
                </div>

                {/* Scrubber at Bottom */}
                <ReplayScrubber />
            </div>
        </div>
    );
}
