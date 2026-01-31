"use client";

import { useState, useEffect } from "react";
import { useRealtimeStore } from "@/stores/realtimeStore";
import { 
  Activity, 
  Zap, 
  Shield,
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Minus
} from "lucide-react";
import { cn } from "@/lib/utils";

interface MarketMetrics {
  asset: string;
  yesPrice: number;
  noPrice: number;
  spread: number;
  vpin: number;
  obiSignal: number; // -1 to 1
  toxicity: "low" | "moderate" | "high" | "extreme";
  volume24h: number;
  timestamp: string;
}

// Mock market data - in production, fetch from CLOB API
const generateMarketData = (): MarketMetrics[] => [
  {
    asset: "Will BTC hit $100k by Feb?",
    yesPrice: 0.62,
    noPrice: 0.38,
    spread: 0.008,
    vpin: 0.23,
    obiSignal: 0.35,
    toxicity: "low",
    volume24h: 2847500,
    timestamp: new Date().toISOString(),
  },
  {
    asset: "Trump Approval Rating",
    yesPrice: 0.41,
    noPrice: 0.59,
    spread: 0.012,
    vpin: 0.45,
    obiSignal: -0.18,
    toxicity: "moderate",
    volume24h: 1523000,
    timestamp: new Date().toISOString(),
  },
  {
    asset: "ETH ETF Approval",
    yesPrice: 0.78,
    noPrice: 0.22,
    spread: 0.006,
    vpin: 0.19,
    obiSignal: 0.62,
    toxicity: "low",
    volume24h: 984500,
    timestamp: new Date().toISOString(),
  },
  {
    asset: "Fed Rate Cut in March",
    yesPrice: 0.23,
    noPrice: 0.77,
    spread: 0.015,
    vpin: 0.67,
    obiSignal: -0.42,
    toxicity: "high",
    volume24h: 675000,
    timestamp: new Date().toISOString(),
  },
];

const toxicityConfig = {
  low: { color: "text-emerald-400", bg: "bg-emerald-500/10", icon: Shield },
  moderate: { color: "text-amber-400", bg: "bg-amber-500/10", icon: Activity },
  high: { color: "text-orange-400", bg: "bg-orange-500/10", icon: AlertTriangle },
  extreme: { color: "text-red-400", bg: "bg-red-500/10", icon: Zap },
};

function MarketRow({ market }: { market: MarketMetrics }) {
  const config = toxicityConfig[market.toxicity];
  const ToxicityIcon = config.icon;
  
  const obiColor = market.obiSignal > 0.2 
    ? "text-emerald-400" 
    : market.obiSignal < -0.2 
      ? "text-red-400" 
      : "text-zinc-400";
  
  return (
    <div className="grid grid-cols-12 gap-2 items-center py-3 px-4 hover:bg-zinc-800/30 transition-colors">
      {/* Asset name */}
      <div className="col-span-3">
        <span className="text-sm font-medium text-zinc-200 truncate block">
          {market.asset}
        </span>
        <span className="text-xs text-zinc-500">
          Vol: ${(market.volume24h / 1000).toFixed(0)}k
        </span>
      </div>
      
      {/* Price */}
      <div className="col-span-2">
        <div className="flex items-center gap-1">
          <span className="text-sm text-emerald-400 tabular-nums">
            {(market.yesPrice * 100).toFixed(1)}¢
          </span>
          <span className="text-zinc-500">/</span>
          <span className="text-sm text-red-400 tabular-nums">
            {(market.noPrice * 100).toFixed(1)}¢
          </span>
        </div>
        <span className="text-xs text-zinc-500">
          Spread: {(market.spread * 100).toFixed(1)}¢
        </span>
      </div>
      
      {/* VPIN */}
      <div className="col-span-2">
        <div className="flex items-center gap-2">
          <div className="flex-1 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
            <div 
              className={cn(
                "h-full rounded-full",
                market.vpin < 0.3 ? "bg-emerald-500" : 
                market.vpin < 0.5 ? "bg-amber-500" : "bg-red-500"
              )}
              style={{ width: `${Math.min(market.vpin * 100, 100)}%` }}
            />
          </div>
          <span className={cn(
            "text-xs font-medium tabular-nums w-10",
            market.vpin < 0.3 ? "text-emerald-400" : 
            market.vpin < 0.5 ? "text-amber-400" : "text-red-400"
          )}>
            {market.vpin.toFixed(2)}
          </span>
        </div>
        <span className="text-xs text-zinc-500">VPIN</span>
      </div>
      
      {/* OBI Signal */}
      <div className="col-span-2">
        <div className="flex items-center gap-1.5">
          {market.obiSignal > 0.2 ? (
            <TrendingUp className="h-3.5 w-3.5 text-emerald-400" />
          ) : market.obiSignal < -0.2 ? (
            <TrendingDown className="h-3.5 w-3.5 text-red-400" />
          ) : (
            <Minus className="h-3.5 w-3.5 text-zinc-400" />
          )}
          <span className={cn("text-sm font-medium tabular-nums", obiColor)}>
            {market.obiSignal > 0 ? "+" : ""}
            {market.obiSignal.toFixed(2)}
          </span>
        </div>
        <span className="text-xs text-zinc-500">
          {market.obiSignal > 0.2 ? "Bullish" : market.obiSignal < -0.2 ? "Bearish" : "Neutral"}
        </span>
      </div>
      
      {/* Toxicity */}
      <div className="col-span-3">
        <div className={cn(
          "inline-flex items-center gap-1.5 px-2 py-1 rounded text-xs font-medium",
          config.bg,
          config.color
        )}>
          <ToxicityIcon className="h-3 w-3" />
          {market.toxicity.charAt(0).toUpperCase() + market.toxicity.slice(1)} Toxicity
        </div>
      </div>
    </div>
  );
}

export function MarketMicrostructure() {
  const { connected } = useRealtimeStore();
  const [markets, setMarkets] = useState<MarketMetrics[]>([]);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  
  useEffect(() => {
    // Initial load
    setMarkets(generateMarketData());
    
    // Simulate live updates
    const interval = setInterval(() => {
      setMarkets(generateMarketData());
      setLastUpdate(new Date());
    }, 2000);
    
    return () => clearInterval(interval);
  }, []);
  
  const isLoading = !connected;

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/50">
      {/* Header */}
      <div className="p-4 border-b border-zinc-800 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Activity className="h-5 w-5 text-zinc-400" />
          <div>
            <h3 className="text-sm font-semibold text-zinc-200">Market Microstructure</h3>
            <p className="text-xs text-zinc-500">Real-time VPIN, OBI, and toxicity analysis</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-zinc-500">
            Updated: {lastUpdate.toLocaleTimeString()}
          </span>
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
          </span>
        </div>
      </div>

      {/* Column headers */}
      <div className="grid grid-cols-12 gap-2 px-4 py-2 border-b border-zinc-800/50 bg-zinc-800/30">
        <div className="col-span-3 text-xs font-medium text-zinc-500">Market</div>
        <div className="col-span-2 text-xs font-medium text-zinc-500">Price</div>
        <div className="col-span-2 text-xs font-medium text-zinc-500">VPIN</div>
        <div className="col-span-2 text-xs font-medium text-zinc-500">OBI Signal</div>
        <div className="col-span-3 text-xs font-medium text-zinc-500">Toxicity</div>
      </div>

      {/* Market rows */}
      {isLoading ? (
        <div className="divide-y divide-zinc-800/50">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="p-4 flex items-center gap-4">
              <div className="h-4 w-32 bg-zinc-800 rounded animate-pulse" />
              <div className="h-4 w-16 bg-zinc-800 rounded animate-pulse" />
              <div className="h-4 w-20 bg-zinc-800 rounded animate-pulse" />
            </div>
          ))}
        </div>
      ) : (
        <div className="divide-y divide-zinc-800/50">
          {markets.map((market, index) => (
            <MarketRow key={index} market={market} />
          ))}
        </div>
      )}

      {/* Footer */}
      <div className="p-3 border-t border-zinc-800 bg-zinc-800/20">
        <div className="flex items-center justify-between text-xs text-zinc-500">
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-emerald-500" />
              VPIN &lt; 0.3
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-amber-500" />
              VPIN 0.3-0.5
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-red-500" />
              VPIN &gt; 0.5
            </span>
          </div>
          <span>Data from Polymarket CLOB</span>
        </div>
      </div>
    </div>
  );
}
