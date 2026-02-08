"""
Toxicity Panel Calibration for PolyBawt.

Multi-signal toxicity assessment with:
- VPIN (Volume-synchronized Probability of Informed Trading)
- Flow imbalance
- Microprice drift
- Realized spread deterioration
- Time-to-expiry decay

Includes calibration tools for VPIN-decile vs post-fill PnL analysis.

Usage:
    panel = ToxicityPanel(config)
    assessment = panel.assess(context)
    if assessment.regime == ToxicityRegime.TOXIC:
        pull_quotes()
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from collections import deque
import statistics

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class ToxicityRegime(Enum):
    """Toxicity regime classification."""
    CALM = "calm"
    TRENDING = "trending"
    VOLATILE = "volatile"
    TOXIC = "toxic"


@dataclass
class ToxicityAssessment:
    """Result of toxicity assessment."""
    regime: ToxicityRegime
    vpin: float
    flow_imbalance: float
    microprice_drift: float  # Absolute drift over window
    realized_spread_health: float  # 1.0 = good, 0.0 = all fills adverse
    time_to_expiry_seconds: float
    
    # Composite score (0-1, higher = more toxic)
    toxicity_score: float
    
    # Recommended actions
    spread_multiplier: float  # 1.0 = normal, >1 = widen
    should_quote: bool
    
    def to_dict(self) -> Dict:
        return {
            "regime": self.regime.value,
            "vpin": round(self.vpin, 3),
            "flow_imbalance": round(self.flow_imbalance, 3),
            "microprice_drift": round(self.microprice_drift, 4),
            "realized_spread_health": round(self.realized_spread_health, 3),
            "time_to_expiry_seconds": round(self.time_to_expiry_seconds, 1),
            "toxicity_score": round(self.toxicity_score, 3),
            "spread_multiplier": round(self.spread_multiplier, 2),
            "should_quote": self.should_quote,
        }


@dataclass
class ToxicityConfig:
    """Configuration for toxicity panel."""
    # VPIN thresholds
    vpin_reduce_threshold: float = 0.4  # Widen spreads
    vpin_halt_threshold: float = 0.6  # Stop quoting
    
    # Flow imbalance (buy_volume - sell_volume / total)
    flow_imbalance_threshold: float = 0.7  # One-sided flow
    
    # Microprice drift (absolute change over window)
    microprice_drift_threshold: float = 0.02  # 2% drift
    
    # Realized spread health (fraction of fills with positive spread)
    spread_health_threshold: float = 0.4  # <40% = bad
    
    # Time to expiry
    settlement_guard_seconds: int = 60  # Pull quotes before expiry
    
    # Spread adjustment
    max_spread_multiplier: float = 3.0  # Max widening


class ToxicityPanel:
    """
    Multi-signal toxicity assessment panel.
    
    Combines multiple signals to determine market regime and
    optimal quoting behavior.
    """
    
    def __init__(self, config: Optional[ToxicityConfig] = None):
        self.config = config or ToxicityConfig()
        
        # Rolling windows for signal computation
        self._microprice_history: deque = deque(maxlen=100)
        self._fill_quality_history: deque = deque(maxlen=50)
        self._vpin_history: deque = deque(maxlen=100)
    
    def assess(
        self,
        vpin: float,
        flow_imbalance: float,
        microprice: Optional[float],
        time_to_expiry_seconds: float,
        recent_fill_spreads: Optional[List[float]] = None,
    ) -> ToxicityAssessment:
        """
        Perform toxicity assessment.
        
        Args:
            vpin: Current VPIN value (0-1)
            flow_imbalance: Flow imbalance (-1 to 1)
            microprice: Current microprice (for drift calculation)
            time_to_expiry_seconds: Time to contract expiry
            recent_fill_spreads: Recent realized spreads (positive = good)
            
        Returns:
            ToxicityAssessment with regime and recommendations
        """
        cfg = self.config
        
        # Update microprice history
        if microprice is not None:
            self._microprice_history.append((time.time(), microprice))
        
        # Calculate microprice drift
        microprice_drift = self._calculate_microprice_drift()
        
        # Calculate realized spread health
        if recent_fill_spreads:
            self._fill_quality_history.extend(recent_fill_spreads)
        spread_health = self._calculate_spread_health()
        
        # Track VPIN
        self._vpin_history.append(vpin)
        
        # === Compute toxicity score ===
        score = 0.0
        weights = {"vpin": 0.35, "flow": 0.2, "drift": 0.2, "spread": 0.15, "time": 0.1}
        
        # VPIN contribution (0-1)
        vpin_contrib = min(1.0, vpin / cfg.vpin_halt_threshold)
        score += weights["vpin"] * vpin_contrib
        
        # Flow imbalance contribution (0-1)
        flow_contrib = abs(flow_imbalance) / cfg.flow_imbalance_threshold
        score += weights["flow"] * min(1.0, flow_contrib)
        
        # Microprice drift contribution (0-1)
        drift_contrib = microprice_drift / cfg.microprice_drift_threshold
        score += weights["drift"] * min(1.0, drift_contrib)
        
        # Spread health contribution (inverted: bad health = high contribution)
        if spread_health is not None:
            spread_contrib = 1.0 - spread_health
            score += weights["spread"] * spread_contrib
        
        # Time-to-expiry contribution
        if time_to_expiry_seconds < cfg.settlement_guard_seconds:
            time_contrib = 1.0 - (time_to_expiry_seconds / cfg.settlement_guard_seconds)
            score += weights["time"] * time_contrib
        
        # === Determine regime ===
        regime = self._classify_regime(
            vpin, flow_imbalance, microprice_drift, spread_health, time_to_expiry_seconds
        )
        
        # === Calculate spread multiplier ===
        spread_multiplier = 1.0
        if score > 0.3:
            spread_multiplier = 1.0 + (score - 0.3) * (cfg.max_spread_multiplier - 1.0) / 0.7
            spread_multiplier = min(cfg.max_spread_multiplier, spread_multiplier)
        
        # === Determine if should quote ===
        should_quote = (
            regime != ToxicityRegime.TOXIC
            and time_to_expiry_seconds >= cfg.settlement_guard_seconds
            and vpin < cfg.vpin_halt_threshold
        )
        
        return ToxicityAssessment(
            regime=regime,
            vpin=vpin,
            flow_imbalance=flow_imbalance,
            microprice_drift=microprice_drift,
            realized_spread_health=spread_health or 1.0,
            time_to_expiry_seconds=time_to_expiry_seconds,
            toxicity_score=score,
            spread_multiplier=spread_multiplier,
            should_quote=should_quote,
        )
    
    def _classify_regime(
        self,
        vpin: float,
        flow_imbalance: float,
        microprice_drift: float,
        spread_health: Optional[float],
        time_to_expiry: float,
    ) -> ToxicityRegime:
        """Classify current market regime."""
        cfg = self.config
        
        # Toxic: any critical threshold breached
        if vpin >= cfg.vpin_halt_threshold:
            return ToxicityRegime.TOXIC
        if time_to_expiry < cfg.settlement_guard_seconds:
            return ToxicityRegime.TOXIC
        if spread_health is not None and spread_health < 0.2:
            return ToxicityRegime.TOXIC
        
        # Volatile: elevated but not toxic
        if vpin >= cfg.vpin_reduce_threshold:
            return ToxicityRegime.VOLATILE
        if microprice_drift >= cfg.microprice_drift_threshold:
            return ToxicityRegime.VOLATILE
        
        # Trending: directional flow
        if abs(flow_imbalance) >= 0.4:
            return ToxicityRegime.TRENDING
        
        return ToxicityRegime.CALM
    
    def _calculate_microprice_drift(self) -> float:
        """Calculate microprice drift over recent window."""
        if len(self._microprice_history) < 2:
            return 0.0
        
        prices = [p for _, p in self._microprice_history]
        if not prices:
            return 0.0
        
        # Absolute drift from start to end
        return abs(prices[-1] - prices[0])
    
    def _calculate_spread_health(self) -> Optional[float]:
        """Calculate fraction of fills with positive realized spread."""
        if not self._fill_quality_history:
            return None
        
        positive_fills = sum(1 for s in self._fill_quality_history if s > 0)
        return positive_fills / len(self._fill_quality_history)


class ToxicityCalibrator:
    """
    Calibration tools for toxicity model.
    
    Analyzes VPIN-decile vs post-fill PnL to validate
    that toxicity signals are predictive.
    """
    
    def __init__(self):
        self._fill_records: List[Dict] = []
    
    def record_fill(
        self,
        vpin_at_fill: float,
        realized_spread: float,
        adverse_1s: Optional[float] = None,
        regime: str = "unknown",
    ):
        """Record a fill for calibration analysis."""
        self._fill_records.append({
            "vpin": vpin_at_fill,
            "spread": realized_spread,
            "adverse_1s": adverse_1s,
            "regime": regime,
            "timestamp": time.time(),
        })
    
    def analyze_vpin_deciles(self) -> Dict[int, Dict[str, float]]:
        """
        Analyze fills by VPIN decile.
        
        Returns dict mapping decile (0-9) to:
        - count: Number of fills
        - mean_spread: Average realized spread
        - adverse_rate: Fraction of adversely selected fills
        """
        if not self._fill_records:
            return {}
        
        # Sort by VPIN and split into deciles
        sorted_records = sorted(self._fill_records, key=lambda x: x["vpin"])
        n = len(sorted_records)
        decile_size = max(1, n // 10)
        
        results = {}
        for decile in range(10):
            start = decile * decile_size
            end = start + decile_size if decile < 9 else n
            records = sorted_records[start:end]
            
            if not records:
                continue
            
            spreads = [r["spread"] for r in records]
            adverse_count = sum(
                1 for r in records
                if r.get("adverse_1s") is not None and r["adverse_1s"] < 0
            )
            
            results[decile] = {
                "count": len(records),
                "vpin_range": (records[0]["vpin"], records[-1]["vpin"]),
                "mean_spread": statistics.mean(spreads) if spreads else 0,
                "std_spread": statistics.stdev(spreads) if len(spreads) > 1 else 0,
                "adverse_rate": adverse_count / len(records),
            }
        
        return results
    
    def is_vpin_predictive(self, min_accuracy: float = 0.55) -> bool:
        """
        Check if VPIN is predictive of fill quality.
        
        Returns True if higher VPIN deciles have worse fill quality.
        """
        deciles = self.analyze_vpin_deciles()
        if len(deciles) < 5:
            return False  # Not enough data
        
        # Compare low deciles (0-2) vs high deciles (7-9)
        low_spreads = [
            deciles[d]["mean_spread"]
            for d in range(3)
            if d in deciles
        ]
        high_spreads = [
            deciles[d]["mean_spread"]
            for d in range(7, 10)
            if d in deciles
        ]
        
        if not low_spreads or not high_spreads:
            return False
        
        # VPIN is predictive if low-VPIN fills have better spreads
        return statistics.mean(low_spreads) > statistics.mean(high_spreads)
    
    def get_calibration_report(self) -> str:
        """Generate calibration report."""
        deciles = self.analyze_vpin_deciles()
        is_predictive = self.is_vpin_predictive()
        
        lines = [
            f"VPIN Calibration Report ({len(self._fill_records)} fills)",
            f"Predictive: {'YES' if is_predictive else 'NO'}",
            "",
            "Decile | Count | VPIN Range | Mean Spread | Adverse Rate",
            "-" * 60,
        ]
        
        for d, stats in sorted(deciles.items()):
            vmin, vmax = stats["vpin_range"]
            lines.append(
                f"  {d}    | {stats['count']:5} | {vmin:.2f}-{vmax:.2f} | "
                f"{stats['mean_spread']:+.4f}   | {stats['adverse_rate']:.1%}"
            )
        
        return "\n".join(lines)
