"use client";

import { useRealtimeStore } from "@/stores/realtimeStore";
import { MetricCard } from "./MetricCard";
import { 
  DollarSign, 
  TrendingUp, 
  Percent, 
  Target,
  BarChart3,
  Zap,
  Shield,
  Activity
} from "lucide-react";

export function PortfolioMetrics() {
  const { portfolio, performance, botStatus, connected } = useRealtimeStore();
  
  const isLoading = !connected;
  const dailyPnLTrend = portfolio.equity > portfolio.balance ? "up" : 
                        portfolio.equity < portfolio.balance ? "down" : "neutral";

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      {/* Balance */}
      <MetricCard
        title="Balance"
        value={`$${portfolio.balance.toLocaleString("en-US", { 
          minimumFractionDigits: 2, 
          maximumFractionDigits: 2 
        })}`}
        subtitle="Available capital"
        icon={DollarSign}
        variant="default"
        isLoading={isLoading}
      />

      {/* Equity */}
      <MetricCard
        title="Equity"
        value={`$${portfolio.equity.toLocaleString("en-US", { 
          minimumFractionDigits: 2, 
          maximumFractionDigits: 2 
        })}`}
        subtitle="Current value"
        icon={BarChart3}
        variant="primary"
        isLoading={isLoading}
      />

      {/* Daily P&L */}
      <MetricCard
        title="Daily P&L"
        value={`${performance.dailyReturn >= 0 ? "+" : ""}${performance.dailyReturn.toFixed(2)}%`}
        subtitle={`$${performance.dailyPnl.toLocaleString("en-US", { 
          minimumFractionDigits: 2, 
          maximumFractionDigits: 2 
        })}`}
        trend={dailyPnLTrend}
        trendPercent={performance.dailyReturn}
        icon={TrendingUp}
        variant={performance.dailyPnl >= 0 ? "success" : "danger"}
        isLoading={isLoading}
      />

      {/* Win Rate */}
      <MetricCard
        title="Win Rate"
        value={`${(performance.winRate * 100).toFixed(1)}%`}
        subtitle={`${performance.totalTrades} total trades`}
        trend={performance.winRate >= 0.5 ? "up" : "down"}
        trendPercent={(performance.winRate - 0.5) * 100}
        icon={Target}
        variant={performance.winRate >= 0.5 ? "success" : "warning"}
        isLoading={isLoading}
      />

      {/* Exposure */}
      <MetricCard
        title="Exposure"
        value={`${portfolio.marginUsed.toFixed(1)}%`}
        subtitle={`$${portfolio.totalExposure.toLocaleString("en-US", { 
          minimumFractionDigits: 2 
        })} deployed`}
        icon={Activity}
        variant={portfolio.marginUsed > 80 ? "warning" : "default"}
        isLoading={isLoading}
      />

      {/* Profit Factor */}
      <MetricCard
        title="Profit Factor"
        value={performance.profitFactor.toFixed(2)}
        subtitle="Gross profit / Gross loss"
        trend={performance.profitFactor > 1 ? "up" : "down"}
        icon={Zap}
        variant={performance.profitFactor > 1.5 ? "success" : 
                 performance.profitFactor > 1 ? "primary" : "warning"}
        isLoading={isLoading}
      />

      {/* Current Streak */}
      <MetricCard
        title="Streak"
        value={performance.currentStreak >= 0 
          ? `+${performance.currentStreak}` 
          : `${performance.currentStreak}`}
        subtitle={performance.currentStreak > 0 
          ? "Winning streak" 
          : performance.currentStreak < 0 
            ? "Losing streak" 
            : "No active streak"}
        trend={performance.currentStreak > 0 ? "up" : 
               performance.currentStreak < 0 ? "down" : "neutral"}
        icon={Activity}
        variant={performance.currentStreak > 0 ? "success" : 
                 performance.currentStreak < 0 ? "danger" : "default"}
        isLoading={isLoading}
      />

      {/* Trading Mode */}
      <MetricCard
        title="Mode"
        value={botStatus.mode === "live" ? "LIVE" : "PAPER"}
        subtitle={botStatus.isRunning ? "Bot active" : "Bot idle"}
        icon={Shield}
        variant={botStatus.mode === "live" ? "danger" : "primary"}
        isLoading={isLoading}
      />
    </div>
  );
}
