"""
Analytics modules for PolyBawt.

Contains:
- fill_quality: Post-fill analysis with adverse selection measurement
- latency_slo: Pipeline latency tracking with SLO monitoring
"""

from src.analytics.fill_quality import (
    FillQualityTracker,
    FillQualityMetrics,
    MarketRegime,
    classify_regime,
)

from src.analytics.latency_slo import (
    LatencySLOTracker,
    LatencyStage,
    LatencyStats,
    SLOConfig,
    LatencyTimer,
    create_timer,
)

__all__ = [
    # Fill quality
    "FillQualityTracker",
    "FillQualityMetrics",
    "MarketRegime",
    "classify_regime",
    # Latency SLO
    "LatencySLOTracker",
    "LatencyStage",
    "LatencyStats",
    "SLOConfig",
    "LatencyTimer",
    "create_timer",
]

