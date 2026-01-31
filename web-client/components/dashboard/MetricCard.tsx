"use client";

import { TrendingUp, TrendingDown, Minus, type LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  trend?: "up" | "down" | "neutral";
  trendValue?: string;
  trendPercent?: number;
  icon?: LucideIcon;
  variant?: "default" | "primary" | "success" | "warning" | "danger";
  className?: string;
  isLoading?: boolean;
}

export function MetricCard({
  title,
  value,
  subtitle,
  trend,
  trendValue,
  trendPercent,
  icon: Icon,
  variant = "default",
  className,
  isLoading = false,
}: MetricCardProps) {
  const TrendIcon = trend === "up" ? TrendingUp : trend === "down" ? TrendingDown : Minus;
  
  const variantStyles = {
    default: "bg-zinc-900/50 border-zinc-800",
    primary: "bg-blue-900/20 border-blue-800",
    success: "bg-emerald-900/20 border-emerald-800",
    warning: "bg-amber-900/20 border-amber-800",
    danger: "bg-red-900/20 border-red-800",
  };
  
  const trendColor = {
    up: "text-emerald-400",
    down: "text-red-400",
    neutral: "text-zinc-400",
  }[trend || "neutral"];

  return (
    <div
      className={cn(
        "relative overflow-hidden rounded-lg border p-4 transition-all duration-200",
        variantStyles[variant],
        className
      )}
    >
      {/* Background glow for live data */}
      <div className="absolute inset-0 bg-gradient-to-br from-white/[0.02] to-transparent pointer-events-none" />
      
      {/* Pulse indicator for live connection */}
      {!isLoading && (
        <div className="absolute top-3 right-3 flex items-center gap-1.5">
          <span className="relative flex h-1.5 w-1.5">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
            <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-emerald-500" />
          </span>
        </div>
      )}

      <div className="relative">
        {/* Header */}
        <div className="flex items-center gap-2 mb-2">
          {Icon && <Icon className="h-4 w-4 text-zinc-400" />}
          <span className="text-xs font-medium text-zinc-500 uppercase tracking-wider">
            {title}
          </span>
        </div>

        {/* Value */}
        {isLoading ? (
          <div className="h-8 w-24 bg-zinc-800 rounded animate-pulse" />
        ) : (
          <div className="text-2xl font-bold text-zinc-100 tabular-nums">
            {value}
          </div>
        )}

        {/* Trend */}
        {trend && !isLoading && (
          <div className="flex items-center gap-1.5 mt-1.5">
            <TrendIcon className={cn("h-3 w-3", trendColor)} />
            <span className={cn("text-xs font-medium", trendColor)}>
              {trendPercent !== undefined && (
                <>
                  {trendPercent >= 0 ? "+" : ""}
                  {trendPercent.toFixed(2)}%
                </>
              )}
              {trendValue && <span className="ml-1">({trendValue})</span>}
            </span>
          </div>
        )}

        {/* Subtitle */}
        {subtitle && (
          <p className="text-xs text-zinc-500 mt-1.5">{subtitle}</p>
        )}
      </div>
    </div>
  );
}
