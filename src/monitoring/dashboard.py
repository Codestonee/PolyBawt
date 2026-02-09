"""
Dashboard Data Provider for PolyBawt.

Provides real-time aggregated metrics for dashboard consumption:
- PnL waterfall data
- Order book snapshots
- Latency histograms
- Strategy performance breakdowns

Usage:
    provider = DashboardProvider()
    data = provider.get_dashboard_data()
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Deque
from collections import deque
from datetime import datetime, timedelta

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TimeSeriesPoint:
    """A single point in a time series."""
    timestamp: float
    value: float
    label: Optional[str] = None


class RollingStats:
    """Rolling statistics with time window."""
    
    def __init__(self, window_seconds: float = 3600.0, max_points: int = 1000):
        self.window_seconds = window_seconds
        self.max_points = max_points
        self._points: Deque[TimeSeriesPoint] = deque(maxlen=max_points)
    
    def add(self, value: float, label: Optional[str] = None):
        """Add a data point."""
        self._points.append(TimeSeriesPoint(
            timestamp=time.time(),
            value=value,
            label=label,
        ))
    
    def _prune(self):
        """Remove old points."""
        cutoff = time.time() - self.window_seconds
        while self._points and self._points[0].timestamp < cutoff:
            self._points.popleft()
    
    def get_points(self) -> List[Dict]:
        """Get all points as dicts."""
        self._prune()
        return [
            {"t": p.timestamp, "v": p.value, "l": p.label}
            for p in self._points
        ]
    
    @property
    def count(self) -> int:
        self._prune()
        return len(self._points)
    
    @property
    def sum(self) -> float:
        self._prune()
        return sum(p.value for p in self._points)
    
    @property
    def mean(self) -> float:
        self._prune()
        if not self._points:
            return 0.0
        return self.sum / len(self._points)
    
    def percentile(self, pct: float) -> float:
        """Calculate percentile (0-100)."""
        self._prune()
        if not self._points:
            return 0.0
        values = sorted(p.value for p in self._points)
        idx = int(len(values) * pct / 100)
        return values[min(idx, len(values) - 1)]


class DashboardProvider:
    """
    Aggregates and provides real-time dashboard data.
    
    Collects metrics from various sources and provides
    structured data for frontend consumption.
    """
    
    def __init__(self):
        # Time series data
        self.pnl_series = RollingStats(window_seconds=86400)  # 24h
        self.latency_series = RollingStats(window_seconds=3600)  # 1h
        self.fill_series = RollingStats(window_seconds=86400)  # 24h
        self.quote_series = RollingStats(window_seconds=3600)  # 1h
        
        # Snapshot data
        self._last_book_update: Dict[str, Any] = {}
        self._strategy_stats: Dict[str, Dict] = {}
        self._position_snapshot: List[Dict] = []
        
        # Counters
        self._orders_placed = 0
        self._orders_filled = 0
        self._orders_cancelled = 0
        self._start_time = time.time()
    
    # --- Data Recording ---
    
    def record_pnl(self, pnl: float, realized: bool = True):
        """Record PnL change."""
        self.pnl_series.add(pnl, "realized" if realized else "unrealized")
    
    def record_latency(self, latency_ms: float, operation: str = "order"):
        """Record operation latency."""
        self.latency_series.add(latency_ms, operation)
    
    def record_fill(self, fill_size_usd: float, side: str):
        """Record a fill."""
        self.fill_series.add(fill_size_usd, side)
        self._orders_filled += 1
    
    def record_quote(self, spread_bps: float, market_id: str):
        """Record a quote spread."""
        self.quote_series.add(spread_bps, market_id)
    
    def record_order_placed(self):
        """Record order placement."""
        self._orders_placed += 1
    
    def record_order_cancelled(self):
        """Record order cancellation."""
        self._orders_cancelled += 1
    
    def update_book_snapshot(self, market_id: str, snapshot: Dict):
        """Update order book snapshot for a market."""
        self._last_book_update[market_id] = {
            "timestamp": time.time(),
            "snapshot": snapshot,
        }
    
    def update_strategy_stats(self, strategy_name: str, stats: Dict):
        """Update stats for a strategy."""
        self._strategy_stats[strategy_name] = {
            "timestamp": time.time(),
            **stats,
        }
    
    def update_positions(self, positions: List[Dict]):
        """Update position snapshot."""
        self._position_snapshot = positions
    
    # --- Data Retrieval ---
    
    def get_pnl_waterfall(self) -> Dict:
        """Get PnL waterfall data."""
        points = self.pnl_series.get_points()
        
        # Group by hour for waterfall
        hourly: Dict[str, float] = {}
        for p in points:
            hour_key = datetime.fromtimestamp(p["t"]).strftime("%H:00")
            hourly[hour_key] = hourly.get(hour_key, 0) + p["v"]
        
        cumulative = 0.0
        waterfall = []
        for hour, pnl in sorted(hourly.items()):
            cumulative += pnl
            waterfall.append({
                "hour": hour,
                "pnl": round(pnl, 2),
                "cumulative": round(cumulative, 2),
            })
        
        return {
            "waterfall": waterfall,
            "total_pnl": round(cumulative, 2),
            "data_points": len(points),
        }
    
    def get_latency_stats(self) -> Dict:
        """Get latency statistics."""
        return {
            "count": self.latency_series.count,
            "mean_ms": round(self.latency_series.mean, 2),
            "p50_ms": round(self.latency_series.percentile(50), 2),
            "p95_ms": round(self.latency_series.percentile(95), 2),
            "p99_ms": round(self.latency_series.percentile(99), 2),
            "recent_points": self.latency_series.get_points()[-20:],
        }
    
    def get_fill_stats(self) -> Dict:
        """Get fill statistics."""
        fills = self.fill_series.get_points()
        
        buy_fills = [p for p in fills if p["l"] in ("buy", "BUY")]
        sell_fills = [p for p in fills if p["l"] in ("sell", "SELL")]
        
        return {
            "total_fills": len(fills),
            "total_volume_usd": round(self.fill_series.sum, 2),
            "buy_fills": len(buy_fills),
            "sell_fills": len(sell_fills),
            "buy_volume_usd": round(sum(p["v"] for p in buy_fills), 2),
            "sell_volume_usd": round(sum(p["v"] for p in sell_fills), 2),
        }
    
    def get_quote_stats(self) -> Dict:
        """Get quoting statistics."""
        return {
            "quotes_placed": self.quote_series.count,
            "mean_spread_bps": round(self.quote_series.mean, 2),
            "min_spread_bps": round(self.quote_series.percentile(0), 2),
            "max_spread_bps": round(self.quote_series.percentile(100), 2),
        }
    
    def get_order_stats(self) -> Dict:
        """Get order statistics."""
        total = self._orders_placed or 1
        return {
            "orders_placed": self._orders_placed,
            "orders_filled": self._orders_filled,
            "orders_cancelled": self._orders_cancelled,
            "fill_rate_pct": round(100 * self._orders_filled / total, 1),
            "cancel_rate_pct": round(100 * self._orders_cancelled / total, 1),
        }
    
    def get_strategy_breakdown(self) -> Dict:
        """Get per-strategy stats."""
        return self._strategy_stats
    
    def get_positions(self) -> List[Dict]:
        """Get current positions."""
        return self._position_snapshot
    
    def get_uptime(self) -> Dict:
        """Get uptime information."""
        elapsed = time.time() - self._start_time
        return {
            "start_time": datetime.fromtimestamp(self._start_time).isoformat(),
            "uptime_seconds": round(elapsed, 0),
            "uptime_human": str(timedelta(seconds=int(elapsed))),
        }
    
    def get_dashboard_data(self) -> Dict:
        """Get complete dashboard data snapshot."""
        return {
            "timestamp": time.time(),
            "uptime": self.get_uptime(),
            "pnl": self.get_pnl_waterfall(),
            "latency": self.get_latency_stats(),
            "fills": self.get_fill_stats(),
            "quotes": self.get_quote_stats(),
            "orders": self.get_order_stats(),
            "strategies": self.get_strategy_breakdown(),
            "positions": self.get_positions(),
        }


# Global instance
_dashboard_provider: Optional[DashboardProvider] = None


def get_dashboard_provider() -> DashboardProvider:
    """Get the global dashboard provider."""
    global _dashboard_provider
    if _dashboard_provider is None:
        _dashboard_provider = DashboardProvider()
    return _dashboard_provider
