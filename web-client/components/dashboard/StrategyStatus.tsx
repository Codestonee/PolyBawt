"use client";

import { useRealtimeStore } from "@/stores/realtimeStore";
import { 
  Settings, 
  Play, 
  Pause,
  Zap,
  Brain,
  Activity,
  Clock
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useState } from "react";

interface Strategy {
  id: string;
  name: string;
  description: string;
  enabled: boolean;
  weight: number;
  icon: typeof Zap;
  trades: number;
  winRate: number;
  lastTrade?: string;
}

const strategies: Strategy[] = [
  {
    id: "microstructure",
    name: "Microstructure",
    description: "VPIN + OBI based scalping",
    enabled: true,
    weight: 40,
    icon: Activity,
    trades: 47,
    winRate: 0.62,
    lastTrade: "2 min ago",
  },
  {
    id: "momentum",
    name: "Momentum",
    description: "Trend following breakout",
    enabled: true,
    weight: 30,
    icon: Zap,
    trades: 23,
    winRate: 0.58,
    lastTrade: "15 min ago",
  },
  {
    id: "event_driven",
    name: "Event-Driven",
    description: "News catalyst trading",
    enabled: false,
    weight: 20,
    icon: Clock,
    trades: 8,
    winRate: 0.75,
    lastTrade: "2 hours ago",
  },
  {
    id: "ml",
    name: "ML Signal",
    description: "Probabilistic forecasting",
    enabled: true,
    weight: 10,
    icon: Brain,
    trades: 12,
    winRate: 0.67,
    lastTrade: "5 min ago",
  },
];

export function StrategyStatus() {
  const { botStatus, connected } = useRealtimeStore();
  const [activeStrategies, setActiveStrategies] = useState<Record<string, boolean>>(
    Object.fromEntries(strategies.map(s => [s.id, s.enabled]))
  );
  
  const isLoading = !connected;
  const activeCount = Object.values(activeStrategies).filter(Boolean).length;
  
  const toggleStrategy = (id: string) => {
    setActiveStrategies(prev => ({
      ...prev,
      [id]: !prev[id]
    }));
  };

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 flex flex-col h-[400px]">
      {/* Header */}
      <div className="p-4 border-b border-zinc-800 shrink-0">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Settings className="h-5 w-5 text-zinc-400" />
            <h3 className="text-sm font-semibold text-zinc-200">Strategies</h3>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-zinc-500">{activeCount} active</span>
            <span className={cn(
              "text-xs px-1.5 py-0.5 rounded",
              botStatus.isRunning 
                ? "bg-emerald-500/10 text-emerald-400" 
                : "bg-zinc-500/10 text-zinc-400"
            )}>
              {botStatus.isRunning ? "RUNNING" : "PAUSED"}
            </span>
          </div>
        </div>
      </div>

      {/* Strategies list */}
      <div className="overflow-y-auto flex-1">
        {isLoading ? (
          <div className="divide-y divide-zinc-800/50">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="p-4">
                <div className="h-4 w-32 bg-zinc-800 rounded animate-pulse mb-2" />
                <div className="h-3 w-48 bg-zinc-800 rounded animate-pulse" />
              </div>
            ))}
          </div>
        ) : (
          <div className="divide-y divide-zinc-800/50">
            {strategies.map((strategy) => {
              const Icon = strategy.icon;
              const isActive = activeStrategies[strategy.id];
              
              return (
                <div 
                  key={strategy.id}
                  className={cn(
                    "p-4 transition-colors",
                    isActive ? "hover:bg-zinc-800/30" : "opacity-60 hover:bg-zinc-800/20"
                  )}
                >
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <div className={cn(
                        "p-1.5 rounded",
                        isActive ? "bg-zinc-800" : "bg-zinc-800/50"
                      )}>
                        <Icon className={cn(
                          "h-4 w-4",
                          isActive ? "text-zinc-300" : "text-zinc-600"
                        )} />
                      </div>
                      <div>
                        <span className={cn(
                          "text-sm font-medium",
                          isActive ? "text-zinc-200" : "text-zinc-500"
                        )}>
                          {strategy.name}
                        </span>
                        <p className="text-xs text-zinc-500">{strategy.description}</p>
                      </div>
                    </div>
                    <button
                      onClick={() => toggleStrategy(strategy.id)}
                      className={cn(
                        "p-1 rounded transition-colors",
                        isActive 
                          ? "text-emerald-400 hover:bg-emerald-500/10" 
                          : "text-zinc-500 hover:bg-zinc-700/50"
                      )}
                    >
                      {isActive ? <Play className="h-4 w-4" /> : <Pause className="h-4 w-4" />}
                    </button>
                  </div>
                  
                  {/* Stats row */}
                  <div className="flex items-center gap-4 ml-9">
                    <div className="flex items-center gap-1.5">
                      <span className="text-xs text-zinc-500">Weight:</span>
                      <div className="flex items-center gap-1">
                        <div className="w-12 h-1 bg-zinc-800 rounded-full overflow-hidden">
                          <div 
                            className={cn(
                              "h-full rounded-full",
                              isActive ? "bg-blue-500" : "bg-zinc-600"
                            )}
                            style={{ width: `${strategy.weight}%` }}
                          />
                        </div>
                        <span className={cn(
                          "text-xs tabular-nums",
                          isActive ? "text-zinc-400" : "text-zinc-600"
                        )}>
                          {strategy.weight}%
                        </span>
                      </div>
                    </div>
                    
                    <div className="flex items-center gap-1">
                      <span className="text-xs text-zinc-500">Trades:</span>
                      <span className={cn(
                        "text-xs tabular-nums",
                        isActive ? "text-zinc-300" : "text-zinc-600"
                      )}>
                        {strategy.trades}
                      </span>
                    </div>
                    
                    <div className="flex items-center gap-1">
                      <span className="text-xs text-zinc-500">Win:</span>
                      <span className={cn(
                        "text-xs tabular-nums",
                        strategy.winRate >= 0.5 ? "text-emerald-400" : "text-red-400"
                      )}>
                        {(strategy.winRate * 100).toFixed(0)}%
                      </span>
                    </div>
                    
                    {strategy.lastTrade && (
                      <div className="flex items-center gap-1">
                        <span className="text-xs text-zinc-500">Last:</span>
                        <span className={cn(
                          "text-xs text-zinc-500",
                          isActive ? "" : "opacity-50"
                        )}>
                          {strategy.lastTrade}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="p-3 border-t border-zinc-800 bg-zinc-800/20 shrink-0">
        <div className="flex items-center justify-between text-xs">
          <span className="text-zinc-500">
            Combined Strategy Sharpe: <span className="text-zinc-300">1.34</span>
          </span>
          <button className="text-blue-400 hover:text-blue-300 transition-colors">
            Configure →
          </button>
        </div>
      </div>
    </div>
  );
}
