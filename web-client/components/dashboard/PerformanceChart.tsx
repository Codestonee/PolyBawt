"use client";

import { useState } from "react";
import { useRealtimeStore } from "@/stores/realtimeStore";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { TrendingUp, Calendar } from "lucide-react";
import { cn } from "@/lib/utils";

// Mock data for demonstration - in production, this would come from API
const generateMockData = () => {
  const data = [];
  let equity = 10000;
  const now = new Date();
  
  for (let i = 30; i >= 0; i--) {
    const date = new Date(now);
    date.setDate(date.getDate() - i);
    
    // Random walk
    const change = (Math.random() - 0.48) * 200;
    equity += change;
    
    data.push({
      date: date.toISOString().split("T")[0],
      equity: Math.round(equity * 100) / 100,
      pnl: change,
    });
  }
  
  return data;
};

type TimeRange = "1D" | "1W" | "1M" | "3M" | "ALL";

export function PerformanceChart() {
  const { performance, portfolio, connected } = useRealtimeStore();
  const [timeRange, setTimeRange] = useState<TimeRange>("1W");
  
  // In production, fetch real chart data based on timeRange
  const chartData = generateMockData();
  
  const isLoading = !connected;
  
  const startEquity = chartData[0]?.equity || portfolio.balance;
  const endEquity = chartData[chartData.length - 1]?.equity || portfolio.equity;
  const totalReturn = ((endEquity - startEquity) / startEquity) * 100;

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/50">
      {/* Header */}
      <div className="p-4 border-b border-zinc-800">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <TrendingUp className="h-5 w-5 text-zinc-400" />
            <h3 className="text-sm font-semibold text-zinc-200">Equity Curve</h3>
          </div>
          
          {/* Time range selector */}
          <div className="flex items-center gap-1 bg-zinc-800/50 rounded-lg p-0.5">
            {(["1D", "1W", "1M", "3M", "ALL"] as TimeRange[]).map((range) => (
              <button
                key={range}
                onClick={() => setTimeRange(range)}
                className={cn(
                  "px-2 py-1 text-xs font-medium rounded-md transition-colors",
                  timeRange === range
                    ? "bg-zinc-700 text-zinc-200"
                    : "text-zinc-500 hover:text-zinc-300"
                )}
              >
                {range}
              </button>
            ))}
          </div>
        </div>
        
        {/* Stats */}
        <div className="flex items-center gap-6">
          <div>
            <span className="text-xs text-zinc-500">Total Return</span>
            <div className={cn(
              "text-lg font-bold tabular-nums",
              totalReturn >= 0 ? "text-emerald-400" : "text-red-400"
            )}>
              {totalReturn >= 0 ? "+" : ""}
              {totalReturn.toFixed(2)}%
            </div>
          </div>
          <div>
            <span className="text-xs text-zinc-500">Peak Equity</span>
            <div className="text-lg font-bold text-zinc-200 tabular-nums">
              ${Math.max(...chartData.map(d => d.equity)).toLocaleString()}
            </div>
          </div>
          <div>
            <span className="text-xs text-zinc-500">Data Points</span>
            <div className="text-lg font-bold text-zinc-200 tabular-nums">
              {chartData.length}
            </div>
          </div>
        </div>
      </div>

      {/* Chart */}
      <div className="h-[250px] p-4">
        {isLoading ? (
          <div className="h-full flex items-center justify-center">
            <div className="h-full w-full bg-zinc-800/50 rounded animate-pulse" />
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData}>
              <defs>
                <linearGradient id="colorEquity" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
              <XAxis
                dataKey="date"
                tick={{ fill: "#71717a", fontSize: 10 }}
                tickLine={false}
                axisLine={{ stroke: "#3f3f46" }}
                tickFormatter={(value) => {
                  const date = new Date(value);
                  return `${date.getMonth() + 1}/${date.getDate()}`;
                }}
              />
              <YAxis
                tick={{ fill: "#71717a", fontSize: 10 }}
                tickLine={false}
                axisLine={false}
                tickFormatter={(value) => `$${(value / 1000).toFixed(1)}k`}
                domain={["dataMin - 100", "dataMax + 100"]}
              />
              <Tooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    return (
                      <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-2 shadow-xl">
                        <p className="text-xs text-zinc-500 mb-1">
                          {payload[0].payload.date}
                        </p>
                        <p className="text-sm font-medium text-emerald-400">
                          ${payload[0].value?.toLocaleString()}
                        </p>
                        <p className={cn(
                          "text-xs",
                          payload[0].payload.pnl >= 0 ? "text-emerald-400" : "text-red-400"
                        )}>
                          {payload[0].payload.pnl >= 0 ? "+" : ""}
                          ${payload[0].payload.pnl.toFixed(2)}
                        </p>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <ReferenceLine
                y={startEquity}
                stroke="#3f3f46"
                strokeDasharray="3 3"
              />
              <Area
                type="monotone"
                dataKey="equity"
                stroke="#10b981"
                strokeWidth={2}
                fillOpacity={1}
                fill="url(#colorEquity)"
                animationDuration={1000}
              />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
