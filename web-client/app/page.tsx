"use client";

import { useEffect } from "react";
import { useRealtimeStore } from "@/stores/realtimeStore";
import { cn } from "@/lib/utils";
import { PortfolioMetrics } from "@/components/dashboard/PortfolioMetrics";
import { PositionsTable } from "@/components/dashboard/PositionsTable";
import { LiveOrders } from "@/components/dashboard/LiveOrders";
import { PerformanceChart } from "@/components/dashboard/PerformanceChart";
import { MarketMicrostructure } from "@/components/dashboard/MarketMicrostructure";
import { SystemHealth } from "@/components/dashboard/SystemHealth";
import { StrategyStatus } from "@/components/dashboard/StrategyStatus";
import { 
  Activity, 
  RefreshCw, 
  AlertCircle,
  Zap,
  TrendingUp,
  BarChart3
} from "lucide-react";

export default function Dashboard() {
  const { connected, connect, disconnect, botStatus, lastPing } = useRealtimeStore();

  // Initialize connection on mount
  useEffect(() => {
    connect();
    return () => disconnect();
  }, []);

  // Calculate connection age
  const connectionAge = Date.now() - lastPing;
  const isStale = connectionAge > 10000;

  return (
    <div className="min-h-full space-y-6 p-4">
      {/* Header */}
      <header className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-lg border border-blue-500/30">
              <BarChart3 className="h-6 w-6 text-blue-400" />
            </div>
            <div>
              <h1 className="text-2xl font-bold tracking-tight text-white">
                Polymarket Bot
              </h1>
              <p className="text-sm text-zinc-500">
                Real-time trading operations dashboard
              </p>
            </div>
          </div>
        </div>
        
        {/* Status indicators */}
        <div className="flex items-center gap-4">
          {/* Connection status */}
          <div className="flex items-center gap-3 px-4 py-2 rounded-lg border border-zinc-800 bg-zinc-900/50">
            <div className="flex items-center gap-2">
              <span className="relative flex h-2 w-2">
                {connected && !isStale ? (
                  <>
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
                  </>
                ) : (
                  <span className="inline-flex rounded-full h-2 w-2 bg-red-500" />
                )}
              </span>
              <span className={cn(
                "text-xs font-medium",
                connected && !isStale ? "text-emerald-400" : "text-red-400"
              )}>
                {connected && !isStale ? "CONNECTED" : "DISCONNECTED"}
              </span>
            </div>
            
            <div className="w-px h-4 bg-zinc-800" />
            
            <div className="flex items-center gap-2">
              {botStatus.isLive ? (
                <Zap className="h-3.5 w-3.5 text-amber-400" />
              ) : (
                <Activity className="h-3.5 w-3.5 text-blue-400" />
              )}
              <span className={cn(
                "text-xs font-medium",
                botStatus.isLive ? "text-amber-400" : "text-blue-400"
              )}>
                {botStatus.isLive ? "LIVE TRADING" : "PAPER TRADING"}
              </span>
            </div>
            
            {botStatus.isRunning && (
              <>
                <div className="w-px h-4 bg-zinc-800" />
                <div className="flex items-center gap-1.5">
                  <div className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
                  <span className="text-xs font-medium text-emerald-400">BOT ACTIVE</span>
                </div>
              </>
            )}
          </div>

          {/* Refresh button */}
          <button
            onClick={() => window.location.reload()}
            className="p-2 rounded-lg border border-zinc-800 bg-zinc-900/50 text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800 transition-colors"
          >
            <RefreshCw className="h-4 w-4" />
          </button>
        </div>
      </header>

      {/* Warning banner for stale connection */}
      {isStale && (
        <div className="flex items-center gap-3 px-4 py-3 rounded-lg border border-amber-800/50 bg-amber-900/20">
          <AlertCircle className="h-5 w-5 text-amber-400" />
          <span className="text-sm text-amber-200">
            Connection is stale. Last update was {Math.round(connectionAge / 1000)}s ago.
          </span>
        </div>
      )}

      {/* Portfolio Metrics Grid */}
      <section>
        <PortfolioMetrics />
      </section>

      {/* Main Dashboard Grid */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column - Portfolio & Orders */}
        <div className="lg:col-span-5 space-y-6">
          {/* Live Orders */}
          <LiveOrders />
          
          {/* Performance Chart */}
          <PerformanceChart />
        </div>

        {/* Right Column - Positions & Market Data */}
        <div className="lg:col-span-7 space-y-6">
          {/* Active Positions */}
          <PositionsTable />
          
          {/* Market Microstructure */}
          <MarketMicrostructure />
        </div>
      </section>

      {/* Bottom Section - System Health & Strategies */}
      <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* System Health */}
        <div className="lg:col-span-1">
          <SystemHealth />
        </div>
        
        {/* Strategy Status */}
        <div className="lg:col-span-2">
          <StrategyStatus />
        </div>
      </section>
    </div>
  );
}
