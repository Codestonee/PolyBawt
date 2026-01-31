"use client";

import { useState } from "react";
import { useRealtimeStore } from "@/stores/realtimeStore";
import { 
  ChevronUp, 
  ChevronDown, 
  ArrowUpRight, 
  ArrowDownRight,
  Filter,
  X
} from "lucide-react";
import { cn } from "@/lib/utils";

type SortKey = "asset" | "size" | "entryPrice" | "currentPrice" | "pnl" | "pnlPercent";
type SortOrder = "asc" | "desc";

export function PositionsTable() {
  const { positions, connected } = useRealtimeStore();
  const [sortKey, setSortKey] = useState<SortKey>("pnlPercent");
  const [sortOrder, setSortOrder] = useState<SortOrder>("desc");
  const [filterSide, setFilterSide] = useState<"all" | "long_yes" | "long_no">("all");
  
  const isLoading = !connected;

  // Sort handler
  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortOrder(sortOrder === "asc" ? "desc" : "asc");
    } else {
      setSortKey(key);
      setSortOrder("desc");
    }
  };

  // Filter and sort positions
  const filteredPositions = positions.filter(
    p => filterSide === "all" || p.side === filterSide
  );

  const sortedPositions = [...filteredPositions].sort((a, b) => {
    const aVal = a[sortKey];
    const bVal = b[sortKey];
    const multiplier = sortOrder === "asc" ? 1 : -1;
    return (aVal > bVal ? 1 : -1) * multiplier;
  });

  // Column header component
  const SortHeader = ({ label, sortKey: key }: { label: string; sortKey: SortKey }) => (
    <button
      onClick={() => handleSort(key)}
      className="flex items-center gap-1 text-xs font-medium text-zinc-400 hover:text-zinc-200 transition-colors"
    >
      {label}
      {sortKey === key && (
        sortOrder === "asc" ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />
      )}
    </button>
  );

  if (isLoading) {
    return (
      <div className="rounded-lg border border-zinc-800 bg-zinc-900/50">
        <div className="p-4 border-b border-zinc-800">
          <div className="h-5 w-32 bg-zinc-800 rounded animate-pulse" />
        </div>
        <div className="divide-y divide-zinc-800">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="p-4 flex items-center gap-4">
              <div className="h-4 w-24 bg-zinc-800 rounded animate-pulse" />
              <div className="h-4 w-16 bg-zinc-800 rounded animate-pulse" />
              <div className="h-4 w-20 bg-zinc-800 rounded animate-pulse" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-zinc-800 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h3 className="text-sm font-semibold text-zinc-200">Active Positions</h3>
          <span className="text-xs text-zinc-500">
            {sortedPositions.length} open
          </span>
        </div>
        
        {/* Filter buttons */}
        <div className="flex items-center gap-1 bg-zinc-800/50 rounded-lg p-0.5">
          {(["all", "long_yes", "long_no"] as const).map((side) => (
            <button
              key={side}
              onClick={() => setFilterSide(side)}
              className={cn(
                "px-2 py-1 text-xs font-medium rounded-md transition-colors",
                filterSide === side
                  ? "bg-zinc-700 text-zinc-200"
                  : "text-zinc-500 hover:text-zinc-300"
              )}
            >
              {side === "all" ? "All" : side === "long_yes" ? "YES" : "NO"}
            </button>
          ))}
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-zinc-800/50">
              <th className="px-4 py-2 text-left">
                <SortHeader label="Asset" sortKey="asset" />
              </th>
              <th className="px-4 py-2 text-left">
                <SortHeader label="Side" sortKey="size" />
              </th>
              <th className="px-4 py-2 text-right">
                <SortHeader label="Size" sortKey="size" />
              </th>
              <th className="px-4 py-2 text-right">
                <SortHeader label="Entry" sortKey="entryPrice" />
              </th>
              <th className="px-4 py-2 text-right">
                <SortHeader label="Current" sortKey="currentPrice" />
              </th>
              <th className="px-4 py-2 text-right">
                <SortHeader label="P&L" sortKey="pnlPercent" />
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-800/50">
            {sortedPositions.length === 0 ? (
              <tr>
                <td colSpan={6} className="px-4 py-8 text-center text-sm text-zinc-500">
                  No active positions
                </td>
              </tr>
            ) : (
              sortedPositions.map((position) => (
                <tr 
                  key={position.id}
                  className="hover:bg-zinc-800/30 transition-colors"
                >
                  <td className="px-4 py-3">
                    <div className="flex flex-col">
                      <span className="text-sm font-medium text-zinc-200 truncate max-w-[150px]">
                        {position.asset}
                      </span>
                      {position.marketQuestion && (
                        <span className="text-xs text-zinc-500 truncate max-w-[150px]">
                          {position.marketQuestion}
                        </span>
                      )}
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <span className={cn(
                      "inline-flex items-center px-2 py-0.5 rounded text-xs font-medium",
                      position.side === "long_yes" 
                        ? "bg-emerald-500/10 text-emerald-400" 
                        : "bg-red-500/10 text-red-400"
                    )}>
                      {position.side === "long_yes" ? "YES" : "NO"}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-right">
                    <span className="text-sm text-zinc-300 tabular-nums">
                      ${position.size.toLocaleString()}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-right">
                    <span className="text-sm text-zinc-400 tabular-nums">
                      {(position.entryPrice * 100).toFixed(1)}¢
                    </span>
                  </td>
                  <td className="px-4 py-3 text-right">
                    <span className="text-sm text-zinc-300 tabular-nums">
                      {(position.currentPrice * 100).toFixed(1)}¢
                    </span>
                  </td>
                  <td className="px-4 py-3 text-right">
                    <div className="flex flex-col items-end">
                      <span className={cn(
                        "text-sm font-medium tabular-nums flex items-center gap-1",
                        position.pnl >= 0 ? "text-emerald-400" : "text-red-400"
                      )}>
                        {position.pnl >= 0 ? (
                          <ArrowUpRight className="h-3 w-3" />
                        ) : (
                          <ArrowDownRight className="h-3 w-3" />
                        )}
                        ${Math.abs(position.pnl).toFixed(2)}
                      </span>
                      <span className={cn(
                        "text-xs",
                        position.pnlPercent >= 0 ? "text-emerald-500" : "text-red-500"
                      )}>
                        {position.pnlPercent >= 0 ? "+" : ""}
                        {position.pnlPercent.toFixed(2)}%
                      </span>
                    </div>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
