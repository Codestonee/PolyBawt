import { KillSwitchLadder } from "@/components/risk/KillSwitchLadder";

export default function RiskPage() {
    return (
        <div className="space-y-6">
            <header>
                <h1 className="text-2xl font-bold tracking-tight text-white">Risk Desk</h1>
                <p className="text-sm text-zinc-500">Global exposure limits and kill-switches</p>
            </header>

            <div className="grid grid-cols-1 gap-12 lg:grid-cols-2">
                <section>
                    <KillSwitchLadder />
                </section>

                <section className="space-y-4">
                    {/* Placeholder for Limits Config */}
                    <div className="rounded-xl border border-zinc-800 bg-zinc-950 p-6">
                        <h2 className="mb-4 text-lg font-medium text-zinc-200">Exposure Limits</h2>
                        <div className="space-y-4">
                            <div className="space-y-2">
                                <div className="flex justify-between text-sm">
                                    <span className="text-zinc-400">Max Drawdown (Daily)</span>
                                    <span className="text-zinc-200">5.00%</span>
                                </div>
                                <div className="h-2 w-full overflow-hidden rounded-full bg-zinc-800">
                                    <div className="h-full w-[20%] bg-blue-500" />
                                </div>
                            </div>

                            <div className="space-y-2">
                                <div className="flex justify-between text-sm">
                                    <span className="text-zinc-400">Kelly Fraction</span>
                                    <span className="text-zinc-200">0.25x</span>
                                </div>
                                <div className="h-2 w-full overflow-hidden rounded-full bg-zinc-800">
                                    <div className="h-full w-[25%] bg-blue-500" />
                                </div>
                            </div>
                        </div>
                    </div>
                </section>
            </div>
        </div>
    );
}
