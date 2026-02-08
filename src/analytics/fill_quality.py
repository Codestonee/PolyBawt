"""
Fill Quality Analytics for PolyBawt.

Tracks and analyzes fill quality metrics:
- Realized spread (fill price vs microprice at fill)
- Adverse selection at multiple horizons (+250ms, +1s, +5s)
- Regime tagging (calm, trending, volatile, settlement_approach)
- VPIN and time-to-expiry context

Used for:
- Strategy performance evaluation
- Toxicity model calibration
- PnL attribution
"""

from __future__ import annotations
import time
import asyncio
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable, Any
from enum import Enum
from collections import deque

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class MarketRegime(Enum):
    """Market regime classification."""
    CALM = "calm"
    TRENDING = "trending"
    VOLATILE = "volatile"
    SETTLEMENT_APPROACH = "settlement_approach"


@dataclass
class FillQualityMetrics:
    """Post-fill quality metrics for a single fill."""
    
    # Fill identification
    client_order_id: str
    token_id: str
    strategy: str
    side: str  # "BUY" or "SELL"
    
    # Fill details
    fill_price: float
    fill_size_usd: float
    fill_timestamp: float
    
    # Microprice context
    microprice_at_fill: float
    realized_spread: float  # (fill_price - microprice) * side_sign
    
    # Adverse selection (populated async)
    adverse_250ms: Optional[float] = None
    adverse_1s: Optional[float] = None
    adverse_5s: Optional[float] = None
    
    # Context
    vpin_at_fill: float = 0.0
    time_to_expiry_seconds: float = float('inf')
    regime: MarketRegime = MarketRegime.CALM
    
    # Fee context
    estimated_maker_rebate: float = 0.0
    
    def net_pnl_estimate(self) -> float:
        """Estimate net PnL including spread and rebate."""
        return self.realized_spread + self.estimated_maker_rebate
    
    def is_adverse(self) -> bool:
        """Check if fill was adversely selected based on 1s move."""
        if self.adverse_1s is None:
            return False
        # Adverse if price moved against us by more than realized spread
        return abs(self.adverse_1s) > abs(self.realized_spread)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/persistence."""
        return {
            "client_order_id": self.client_order_id,
            "token_id": self.token_id,
            "strategy": self.strategy,
            "side": self.side,
            "fill_price": self.fill_price,
            "fill_size_usd": self.fill_size_usd,
            "fill_timestamp": self.fill_timestamp,
            "microprice_at_fill": self.microprice_at_fill,
            "realized_spread": self.realized_spread,
            "adverse_250ms": self.adverse_250ms,
            "adverse_1s": self.adverse_1s,
            "adverse_5s": self.adverse_5s,
            "vpin_at_fill": self.vpin_at_fill,
            "time_to_expiry_seconds": self.time_to_expiry_seconds,
            "regime": self.regime.value,
            "estimated_maker_rebate": self.estimated_maker_rebate,
            "net_pnl_estimate": self.net_pnl_estimate(),
            "is_adverse": self.is_adverse(),
        }


class FillQualityTracker:
    """
    Tracks fill quality with delayed adverse selection measurement.
    
    Usage:
        tracker = FillQualityTracker(microprice_provider)
        await tracker.record_fill(fill_data, microprice, context)
        # ... 5 seconds later, adverse selection is computed
        
        stats = tracker.get_summary_stats()
    """
    
    def __init__(
        self,
        microprice_provider: Callable[[str], float],
        max_history: int = 1000,
    ):
        """
        Args:
            microprice_provider: Callable that returns current microprice for token_id
            max_history: Maximum number of fills to retain
        """
        self._microprice_provider = microprice_provider
        self._fills: deque[FillQualityMetrics] = deque(maxlen=max_history)
        self._pending_adverse: Dict[str, FillQualityMetrics] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the adverse selection measurement loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._adverse_selection_loop())
        logger.info("[FillQuality] Started adverse selection tracker")
    
    async def stop(self):
        """Stop the tracker."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("[FillQuality] Stopped")
    
    def record_fill(
        self,
        client_order_id: str,
        token_id: str,
        strategy: str,
        side: str,
        fill_price: float,
        fill_size_usd: float,
        microprice_at_fill: float,
        vpin_at_fill: float = 0.0,
        time_to_expiry_seconds: float = float('inf'),
        regime: MarketRegime = MarketRegime.CALM,
        estimated_maker_rebate: float = 0.0,
    ) -> FillQualityMetrics:
        """
        Record a fill for quality analysis.
        
        Returns the FillQualityMetrics object (adverse selection will be populated later).
        """
        # Calculate realized spread: positive = good for us
        side_sign = 1.0 if side == "SELL" else -1.0
        realized_spread = (fill_price - microprice_at_fill) * side_sign
        
        metrics = FillQualityMetrics(
            client_order_id=client_order_id,
            token_id=token_id,
            strategy=strategy,
            side=side,
            fill_price=fill_price,
            fill_size_usd=fill_size_usd,
            fill_timestamp=time.time(),
            microprice_at_fill=microprice_at_fill,
            realized_spread=realized_spread,
            vpin_at_fill=vpin_at_fill,
            time_to_expiry_seconds=time_to_expiry_seconds,
            regime=regime,
            estimated_maker_rebate=estimated_maker_rebate,
        )
        
        self._fills.append(metrics)
        self._pending_adverse[client_order_id] = metrics
        
        logger.info(
            f"[FillQuality] Recorded: {strategy} {side} {token_id[:8]}... "
            f"price={fill_price:.3f} spread={realized_spread:.4f} "
            f"VPIN={vpin_at_fill:.3f} regime={regime.value}"
        )
        
        return metrics
    
    async def _adverse_selection_loop(self):
        """Background loop to measure adverse selection at delay horizons."""
        horizons = [
            (0.25, "adverse_250ms"),
            (1.0, "adverse_1s"),
            (5.0, "adverse_5s"),
        ]
        
        while self._running:
            now = time.time()
            completed = []
            
            for order_id, metrics in self._pending_adverse.items():
                age = now - metrics.fill_timestamp
                
                # Try to fill in each horizon
                for delay, attr in horizons:
                    if age >= delay and getattr(metrics, attr) is None:
                        try:
                            current_price = self._microprice_provider(metrics.token_id)
                            if current_price is not None:
                                # Adverse selection = price move since fill
                                # Positive = price moved in direction of counterparty
                                side_sign = 1.0 if metrics.side == "SELL" else -1.0
                                adverse = (current_price - metrics.microprice_at_fill) * side_sign
                                setattr(metrics, attr, adverse)
                        except Exception as e:
                            logger.debug(f"[FillQuality] Microprice fetch failed: {e}")
                
                # If all horizons filled or too old, mark complete
                if age > 6.0:  # Allow 1s buffer after 5s horizon
                    completed.append(order_id)
                    self._log_completed_metrics(metrics)
            
            # Clean up completed
            for order_id in completed:
                del self._pending_adverse[order_id]
            
            await asyncio.sleep(0.1)  # Check every 100ms
    
    def _log_completed_metrics(self, metrics: FillQualityMetrics):
        """Log completed fill quality metrics."""
        adverse_str = (
            f"adv250={metrics.adverse_250ms:.4f if metrics.adverse_250ms else 'N/A'} "
            f"adv1s={metrics.adverse_1s:.4f if metrics.adverse_1s else 'N/A'} "
            f"adv5s={metrics.adverse_5s:.4f if metrics.adverse_5s else 'N/A'}"
        )
        logger.info(
            f"[FillQuality] Complete: {metrics.strategy} {metrics.side} "
            f"spread={metrics.realized_spread:.4f} {adverse_str} "
            f"adverse={metrics.is_adverse()}"
        )
    
    def get_recent_fills(self, count: int = 100) -> List[FillQualityMetrics]:
        """Get most recent fills."""
        return list(self._fills)[-count:]
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Compute summary statistics across all recorded fills."""
        if not self._fills:
            return {"count": 0}
        
        fills = list(self._fills)
        spreads = [f.realized_spread for f in fills]
        adverse_1s = [f.adverse_1s for f in fills if f.adverse_1s is not None]
        
        adverse_count = sum(1 for f in fills if f.is_adverse())
        
        return {
            "count": len(fills),
            "mean_realized_spread": sum(spreads) / len(spreads),
            "mean_adverse_1s": sum(adverse_1s) / len(adverse_1s) if adverse_1s else None,
            "adverse_fill_rate": adverse_count / len(fills),
            "by_strategy": self._stats_by_strategy(),
            "by_regime": self._stats_by_regime(),
        }
    
    def _stats_by_strategy(self) -> Dict[str, Dict[str, float]]:
        """Compute stats grouped by strategy."""
        by_strat: Dict[str, List[FillQualityMetrics]] = {}
        for f in self._fills:
            by_strat.setdefault(f.strategy, []).append(f)
        
        result = {}
        for strat, fills in by_strat.items():
            spreads = [f.realized_spread for f in fills]
            result[strat] = {
                "count": len(fills),
                "mean_spread": sum(spreads) / len(spreads),
                "adverse_rate": sum(1 for f in fills if f.is_adverse()) / len(fills),
            }
        return result
    
    def _stats_by_regime(self) -> Dict[str, Dict[str, float]]:
        """Compute stats grouped by market regime."""
        by_regime: Dict[str, List[FillQualityMetrics]] = {}
        for f in self._fills:
            by_regime.setdefault(f.regime.value, []).append(f)
        
        result = {}
        for regime, fills in by_regime.items():
            spreads = [f.realized_spread for f in fills]
            result[regime] = {
                "count": len(fills),
                "mean_spread": sum(spreads) / len(spreads),
                "adverse_rate": sum(1 for f in fills if f.is_adverse()) / len(fills),
            }
        return result


def classify_regime(
    realized_vol_percentile: float,
    vpin: float,
    time_to_expiry_seconds: float,
) -> MarketRegime:
    """
    Classify current market regime based on indicators.
    
    Args:
        realized_vol_percentile: 0-100 percentile of recent volatility
        vpin: Current VPIN value (0-1)
        time_to_expiry_seconds: Seconds until contract expiry
    """
    if time_to_expiry_seconds < 60:
        return MarketRegime.SETTLEMENT_APPROACH
    if vpin > 0.6:
        return MarketRegime.VOLATILE
    if realized_vol_percentile > 70:
        return MarketRegime.VOLATILE
    if realized_vol_percentile > 30:
        return MarketRegime.TRENDING
    return MarketRegime.CALM
