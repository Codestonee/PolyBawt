"use client";

import { useRealtimeStore } from "@/stores/realtimeStore";
import { 
  ArrowUpRight, 
  ArrowDownRight,
  Clock,
  CheckCircle2,
  XCircle,
  Loader2
} from "lucide-react";
import { cn } from "@/lib/utils";
import { formatDistanceToNow } from "date-fns";

const statusConfig = {
  pending: { icon: Loader2, color: "text-amber-400", bg: "bg-amber-500/10", label: "Pending" },
  filled: { icon: CheckCircle2, color: "text-emerald-400", bg: "bg-emerald-500/10", label: "Filled" },
  partial: { icon: Loader2, color: "text-blue-400", bg: "bg-blue-500/10", label: "Partial" },
  cancelled: { icon: XCircle, color: "text-zinc-400", bg: "bg-zinc-500/10", label: "Cancelled" },
  rejected: { icon: XCircle, color: "text-red-400", bg: "bg-red-500/10", label: "Rejected" },
};

export function LiveOrders() {
  const { orders, connected } = useRealtimeStore();
  
  const isLoading = !connected;
  
  // Sort by timestamp (newest first)
  const sortedOrders = [...orders].sort(
    (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
  );

  // Take only the most recent 10
  const recentOrders = sortedOrders.slice(0, 10);

  if (isLoading) {
    return (
      <div className="rounded-lg border border-zinc-800 bg-zinc-900/50">
        <div className="p-4 border-b border-zinc-800">
          <div className="h-5 w-24 bg-zinc-800 rounded animate-pulse" />
        </div>
        <div className="divide-y divide-zinc-800">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="p-3 flex items-center justify-between">
              <div className="h-4 w-20 bg-zinc-800 rounded animate-pulse" />
              <div className="h-4 w-16 bg-zinc-800 rounded animate-pulse" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 flex flex-col h-[350px]">
      {/* Header */}
      <div className="p-4 border-b border-zinc-800 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-3">
          <h3 className="text-sm font-semibold text-zinc-200">Live Orders</h3>
          <span className="text-xs text-zinc-500">
            {orders.filter(o => o.status === "pending" || o.status === "partial").length} active
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="relative flex h-1.5 w-1.5">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
            <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-emerald-500" />
          </span>
          <span className="text-xs text-emerald-400">LIVE</span>
        </div>
      </div>

      {/* Orders list */}
      <div className="overflow-y-auto flex-1">
        {recentOrders.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-zinc-500">
            <Clock className="h-8 w-8 mb-2 opacity-50" />
            <span className="text-sm">No recent orders</span>
          </div>
        ) : (
          <div className="divide-y divide-zinc-800/50">
            {recentOrders.map((order) => {
              const status = statusConfig[order.status] || statusConfig.pending;
              const StatusIcon = status.icon;
              
              return (
                <div 
                  key={order.id}
                  className="p-3 hover:bg-zinc-800/30 transition-colors"
                >
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-2">
                      <span className={cn(
                        "text-xs font-medium uppercase",
                        order.side === "buy" ? "text-emerald-400" : "text-red-400"
                      )}>
                        {order.side === "buy" ? "BUY" : "SELL"}
                      </span>
                      <span className="text-xs text-zinc-500">{order.asset}</span>
                    </div>
                    <span className={cn(
                      "text-[10px] px-1.5 py-0.5 rounded flex items-center gap-1",
                      status.bg,
                      status.color
                    )}>
                      <StatusIcon className="h-3 w-3" />
                      {status.label}
                    </span>
                  </div>
                  
                  <div className="flex items-center justify-between text-xs">
                    <div className="flex items-center gap-3">
                      <span className="text-zinc-300 tabular-nums">
                        ${order.size.toLocaleString()}
                      </span>
                      <span className="text-zinc-500">@</span>
                      <span className="text-zinc-300 tabular-nums">
                        {(order.price * 100).toFixed(1)}¢
                      </span>
                    </div>
                    <span className="text-zinc-500">
                      {formatDistanceToNow(new Date(order.timestamp), { addSuffix: true })}
                    </span>
                  </div>
                  
                  {order.filledSize > 0 && order.filledSize < order.size && (
                    <div className="mt-1.5">
                      <div className="h-1 bg-zinc-800 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-emerald-500 rounded-full"
                          style={{ width: `${(order.filledSize / order.size) * 100}%` }}
                        />
                      </div>
                      <span className="text-[10px] text-zinc-500 mt-0.5">
                        {((order.filledSize / order.size) * 100).toFixed(0)}% filled
                      </span>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Footer */}
      {orders.length > 10 && (
        <div className="p-2 border-t border-zinc-800 text-center">
          <span className="text-xs text-zinc-500">
            Showing 10 of {orders.length} orders
          </span>
        </div>
      )}
    </div>
  );
}
