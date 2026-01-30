import { MicrostructureMonitor } from "@/components/ops/MicrostructureMonitor";
import { IncidentCenter } from "@/components/health/IncidentLog";
import { PortfolioSummary } from "@/components/dashboard/PortfolioSummary";
import { ActiveOrders } from "@/components/dashboard/ActiveOrders";

export default function Home() {
  return (
    <div className="space-y-6 h-[calc(100vh-theme(spacing.24))] flex flex-col">
      <header className="flex items-center justify-between shrink-0">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white">Live Ops</h1>
          <p className="text-sm text-zinc-500">Real-time market microstructure and execution</p>
        </div>
        <div className="flex items-center gap-2 rounded-full border border-zinc-800 bg-zinc-900 px-3 py-1">
          <div className="h-2 w-2 animate-pulse rounded-full bg-emerald-500" />
          <span className="text-xs font-medium text-zinc-400">SYSTEM ONLINE</span>
        </div>
      </header>

      {/* Top Row: Portfolio Summary */}
      <section className="shrink-0">
        <PortfolioSummary />
      </section>

      {/* Main Grid */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-6 flex-1 min-h-0">

        {/* Left Column: Trading & Execution (Active Orders) */}
        <div className="lg:col-span-5 flex flex-col gap-4 min-h-0">
          <ActiveOrders />

          {/* Future: Positions list could go here */}
        </div>

        {/* Right Column: Monitoring & Health */}
        <div className="lg:col-span-7 flex flex-col gap-4 min-h-0 overflow-y-auto pr-1">
          <MicrostructureMonitor />
          <IncidentCenter />
        </div>
      </section>
    </div>
  );
}
