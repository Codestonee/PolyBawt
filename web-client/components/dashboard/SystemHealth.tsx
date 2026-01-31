"use client";

import { useState, useEffect } from "react";
import { useRealtimeStore } from "@/stores/realtimeStore";
import { 
  Server, 
  Cpu, 
  Database,
  Network,
  CheckCircle2,
  AlertCircle,
  XCircle,
  Activity
} from "lucide-react";
import { cn } from "@/lib/utils";

interface SystemComponent {
  name: string;
  status: "healthy" | "degraded" | "down";
  latency?: number;
  uptime?: string;
  lastCheck: string;
}

interface CircuitBreaker {
  id: string;
  type: string;
  state: "closed" | "soft" | "hard";
  current: number;
  threshold: number;
}

const statusConfig = {
  healthy: { icon: CheckCircle2, color: "text-emerald-400", bg: "bg-emerald-500/10" },
  degraded: { icon: AlertCircle, color: "text-amber-400", bg: "bg-amber-500/10" },
  down: { icon: XCircle, color: "text-red-400", bg: "bg-red-500/10" },
};

const cbStateConfig = {
  closed: { label: "CLOSED", color: "text-emerald-400", bg: "bg-emerald-500/10" },
  soft: { label: "SOFT STOP", color: "text-amber-400", bg: "bg-amber-500/10" },
  hard: { label: "HARD STOP", color: "text-red-400", bg: "bg-red-500/10" },
};

export function SystemHealth() {
  const { botStatus, connected, lastPing } = useRealtimeStore();
  const [systemComponents, setSystemComponents] = useState<SystemComponent[]>([
    { name: "API Server", status: "healthy", latency: 12, uptime: "99.9%", lastCheck: new Date().toISOString() },
    { name: "WebSocket", status: "healthy", latency: 45, lastCheck: new Date().toISOString() },
    { name: "Database", status: "healthy", latency: 8, uptime: "99.99%", lastCheck: new Date().toISOString() },
    { name: "CLOB Connection", status: "healthy", latency: 23, lastCheck: new Date().toISOString() },
  ]);
  
  const [circuitBreakers, setCircuitBreakers] = useState<CircuitBreaker[]>([
    { id: "cb-1", type: "Daily Loss", state: "closed", current: 234, threshold: 500 },
    { id: "cb-2", type: "Single Trade", state: "closed", current: 45, threshold: 100 },
    { id: "cb-3", type: "Position Size", state: "closed", current: 2500, threshold: 5000 },
    { id: "cb-4", type: "Consecutive Loss", state: "closed", current: 2, threshold: 5 },
  ]);
  
  const [connectionQuality, setConnectionQuality] = useState<number>(95);
  
  useEffect(() => {
    // Update connection status based on last ping
    const now = Date.now();
    const pingDiff = now - lastPing;
    
    setSystemComponents(prev => prev.map(comp => {
      if (comp.name === "API Server") {
        return {
          ...comp,
          status: pingDiff > 5000 ? "down" : pingDiff > 2000 ? "degraded" : "healthy",
          latency: Math.max(1, Math.min(100, Math.round(pingDiff / 10))),
          lastCheck: new Date().toISOString(),
        };
      }
      return comp;
    }));
    
    // Update connection quality
    setConnectionQuality(pingDiff > 3000 ? Math.max(0, 95 - pingDiff / 100) : 95);
  }, [lastPing]);
  
  const isLoading = !connected;

  return (
    <div className="space-y-4">
      {/* System Status */}
      <div className="rounded-lg border border-zinc-800 bg-zinc-900/50">
        <div className="p-4 border-b border-zinc-800">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Server className="h-5 w-5 text-zinc-400" />
              <h3 className="text-sm font-semibold text-zinc-200">System Health</h3>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-zinc-500">Connection</span>
              <div className="flex items-center gap-1">
                <div 
                  className={cn(
                    "h-1.5 w-12 rounded-full",
                    connectionQuality > 80 ? "bg-emerald-500" :
                    connectionQuality > 50 ? "bg-amber-500" : "bg-red-500"
                  )}
                  style={{ width: `${connectionQuality}%`, maxWidth: "48px" }}
                />
                <span className="text-xs text-zinc-400">{Math.round(connectionQuality)}%</span>
              </div>
            </div>
          </div>
        </div>
        
        {/* Components list */}
        <div className="divide-y divide-zinc-800/50">
          {isLoading ? (
            [...Array(4)].map((_, i) => (
              <div key={i} className="p-3 flex items-center justify-between">
                <div className="h-4 w-24 bg-zinc-800 rounded animate-pulse" />
                <div className="h-4 w-16 bg-zinc-800 rounded animate-pulse" />
              </div>
            ))
          ) : (
            systemComponents.map((component) => {
              const StatusIcon = statusConfig[component.status].icon;
              
              return (
                <div 
                  key={component.name}
                  className="p-3 flex items-center justify-between hover:bg-zinc-800/30 transition-colors"
                >
                  <div className="flex items-center gap-2">
                    <StatusIcon className={cn(
                      "h-4 w-4",
                      statusConfig[component.status].color
                    )} />
                    <span className="text-sm text-zinc-300">{component.name}</span>
                  </div>
                  <div className="flex items-center gap-3">
                    {component.latency !== undefined && (
                      <span className="text-xs text-zinc-500 tabular-nums">
                        {component.latency}ms
                      </span>
                    )}
                    {component.uptime && (
                      <span className="text-xs text-zinc-500 tabular-nums">
                        {component.uptime}
                      </span>
                    )}
                    <span className={cn(
                      "text-xs px-1.5 py-0.5 rounded",
                      statusConfig[component.status].bg,
                      statusConfig[component.status].color
                    )}>
                      {component.status.toUpperCase()}
                    </span>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>
      
      {/* Circuit Breakers */}
      <div className="rounded-lg border border-zinc-800 bg-zinc-900/50">
        <div className="p-4 border-b border-zinc-800">
          <div className="flex items-center gap-3">
            <Activity className="h-5 w-5 text-zinc-400" />
            <h3 className="text-sm font-semibold text-zinc-200">Circuit Breakers</h3>
          </div>
        </div>
        
        <div className="divide-y divide-zinc-800/50">
          {isLoading ? (
            [...Array(4)].map((_, i) => (
              <div key={i} className="p-3">
                <div className="h-4 w-full bg-zinc-800 rounded animate-pulse" />
              </div>
            ))
          ) : (
            circuitBreakers.map((cb) => {
              const percentage = Math.min(100, (cb.current / cb.threshold) * 100);
              
              return (
                <div key={cb.id} className="p-3">
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="text-sm text-zinc-300">{cb.type}</span>
                    <span className={cn(
                      "text-xs px-1.5 py-0.5 rounded font-medium",
                      cbStateConfig[cb.state].bg,
                      cbStateConfig[cb.state].color
                    )}>
                      {cbStateConfig[cb.state].label}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="flex-1 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                      <div 
                        className={cn(
                          "h-full rounded-full transition-all",
                          percentage < 50 ? "bg-emerald-500" :
                          percentage < 80 ? "bg-amber-500" : "bg-red-500"
                        )}
                        style={{ width: `${percentage}%` }}
                      />
                    </div>
                    <span className="text-xs text-zinc-500 tabular-nums w-16 text-right">
                      {cb.current}/{cb.threshold}
                    </span>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
}
