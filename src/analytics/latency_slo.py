"""
Latency SLO Tracking for PolyBawt.

Monitors pipeline latency at each stage with SLO enforcement:
- Oracle staleness (Chainlink)
- Local book freshness
- Data → signal latency
- Signal → submit latency
- Submit → ack latency
- Cancel → ack latency
- Fill detection latency

Reports p50, p95, p99 percentiles and triggers alerts on SLO violations.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import deque
from enum import Enum

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class LatencyStage(Enum):
    """Pipeline stages for latency measurement."""
    ORACLE_STALENESS = "oracle_staleness"
    BOOK_FRESHNESS = "book_freshness"
    DATA_TO_SIGNAL = "data_to_signal"
    SIGNAL_TO_SUBMIT = "signal_to_submit"
    SUBMIT_TO_ACK = "submit_to_ack"
    CANCEL_TO_ACK = "cancel_to_ack"
    FILL_DETECTION = "fill_detection"


@dataclass
class SLOConfig:
    """SLO thresholds for a latency stage."""
    p50_ms: float
    p99_ms: float
    violation_action: str = "alert"  # "alert", "disable_quoting", "halt"


# Default SLOs from upgrade plan
DEFAULT_SLOS: Dict[LatencyStage, SLOConfig] = {
    LatencyStage.ORACLE_STALENESS: SLOConfig(p50_ms=250, p99_ms=900, violation_action="disable_quoting"),
    LatencyStage.BOOK_FRESHNESS: SLOConfig(p50_ms=250, p99_ms=1500, violation_action="disable_quoting"),
    LatencyStage.DATA_TO_SIGNAL: SLOConfig(p50_ms=100, p99_ms=500, violation_action="alert"),
    LatencyStage.SIGNAL_TO_SUBMIT: SLOConfig(p50_ms=50, p99_ms=200, violation_action="alert"),
    LatencyStage.SUBMIT_TO_ACK: SLOConfig(p50_ms=300, p99_ms=1500, violation_action="alert"),
    LatencyStage.CANCEL_TO_ACK: SLOConfig(p50_ms=300, p99_ms=1500, violation_action="alert"),
    LatencyStage.FILL_DETECTION: SLOConfig(p50_ms=150, p99_ms=700, violation_action="alert"),
}


@dataclass
class LatencyStats:
    """Computed latency statistics for a stage."""
    stage: LatencyStage
    count: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    mean_ms: float
    slo_p50_ok: bool
    slo_p99_ok: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage.value,
            "count": self.count,
            "p50_ms": round(self.p50_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2),
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "mean_ms": round(self.mean_ms, 2),
            "slo_p50_ok": self.slo_p50_ok,
            "slo_p99_ok": self.slo_p99_ok,
        }


class LatencySLOTracker:
    """
    Tracks pipeline latency with SLO monitoring.
    
    Usage:
        tracker = LatencySLOTracker()
        
        # Record a latency measurement
        tracker.record(LatencyStage.SUBMIT_TO_ACK, latency_ms=45.2)
        
        # Check for violations
        violations = tracker.check_slo_violations()
        
        # Get percentile stats
        stats = tracker.get_stats(LatencyStage.SUBMIT_TO_ACK)
    """
    
    def __init__(
        self,
        slos: Optional[Dict[LatencyStage, SLOConfig]] = None,
        window_size: int = 1000,
        violation_callback: Optional[callable] = None,
    ):
        """
        Args:
            slos: SLO configurations per stage (defaults to upgrade plan values)
            window_size: Number of samples to retain per stage
            violation_callback: Called with (stage, stats) on SLO violation
        """
        self._slos = slos or DEFAULT_SLOS
        self._window_size = window_size
        self._violation_callback = violation_callback
        
        # Latency samples per stage
        self._samples: Dict[LatencyStage, deque] = {
            stage: deque(maxlen=window_size)
            for stage in LatencyStage
        }
        
        # Track consecutive violations for escalation
        self._consecutive_violations: Dict[LatencyStage, int] = {
            stage: 0 for stage in LatencyStage
        }
        
        # Last violation alert time (rate limiting)
        self._last_alert_time: Dict[LatencyStage, float] = {
            stage: 0.0 for stage in LatencyStage
        }
    
    def record(self, stage: LatencyStage, latency_ms: float):
        """Record a latency measurement for a pipeline stage."""
        self._samples[stage].append(latency_ms)
        
        # Check for immediate violation on critical stages
        slo = self._slos.get(stage)
        if slo and latency_ms > slo.p99_ms:
            self._handle_violation(stage, latency_ms)
    
    def record_span(
        self,
        stage: LatencyStage,
        start_time: float,
        end_time: Optional[float] = None,
    ):
        """Record a latency span (end_time defaults to now)."""
        end = end_time or time.time()
        latency_ms = (end - start_time) * 1000
        self.record(stage, latency_ms)
    
    def _handle_violation(self, stage: LatencyStage, latency_ms: float):
        """Handle an SLO violation."""
        self._consecutive_violations[stage] += 1
        now = time.time()
        
        # Rate limit alerts to 1 per 30 seconds per stage
        if now - self._last_alert_time[stage] > 30.0:
            slo = self._slos[stage]
            logger.warning(
                f"[LatencySLO] VIOLATION: {stage.value} latency={latency_ms:.1f}ms "
                f"> p99_slo={slo.p99_ms}ms (consecutive={self._consecutive_violations[stage]})"
            )
            self._last_alert_time[stage] = now
            
            if self._violation_callback:
                stats = self.get_stats(stage)
                self._violation_callback(stage, stats)
    
    def get_stats(self, stage: LatencyStage) -> Optional[LatencyStats]:
        """Compute latency statistics for a stage."""
        samples = list(self._samples[stage])
        if not samples:
            return None
        
        samples_sorted = sorted(samples)
        n = len(samples_sorted)
        
        def percentile(p: float) -> float:
            idx = int(p * (n - 1))
            return samples_sorted[idx]
        
        slo = self._slos.get(stage)
        p50 = percentile(0.50)
        p99 = percentile(0.99)
        
        return LatencyStats(
            stage=stage,
            count=n,
            p50_ms=p50,
            p95_ms=percentile(0.95),
            p99_ms=p99,
            min_ms=samples_sorted[0],
            max_ms=samples_sorted[-1],
            mean_ms=sum(samples) / n,
            slo_p50_ok=p50 <= slo.p50_ms if slo else True,
            slo_p99_ok=p99 <= slo.p99_ms if slo else True,
        )
    
    def get_all_stats(self) -> Dict[str, LatencyStats]:
        """Get statistics for all stages."""
        return {
            stage.value: self.get_stats(stage)
            for stage in LatencyStage
            if self._samples[stage]
        }
    
    def check_slo_violations(self) -> List[LatencyStats]:
        """Check all stages for SLO violations."""
        violations = []
        for stage in LatencyStage:
            stats = self.get_stats(stage)
            if stats and (not stats.slo_p50_ok or not stats.slo_p99_ok):
                violations.append(stats)
        return violations
    
    def reset_violations(self, stage: LatencyStage):
        """Reset consecutive violation counter for a stage."""
        self._consecutive_violations[stage] = 0
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall latency health status."""
        all_stats = self.get_all_stats()
        violations = self.check_slo_violations()
        
        return {
            "healthy": len(violations) == 0,
            "violation_count": len(violations),
            "violations": [v.stage.value for v in violations],
            "consecutive_violations": {
                stage.value: count
                for stage, count in self._consecutive_violations.items()
                if count > 0
            },
            "stats": {k: v.to_dict() for k, v in all_stats.items() if v},
        }


class LatencyTimer:
    """Context manager for timing code blocks."""
    
    def __init__(self, tracker: LatencySLOTracker, stage: LatencyStage):
        self._tracker = tracker
        self._stage = stage
        self._start: float = 0.0
    
    def __enter__(self):
        self._start = time.time()
        return self
    
    def __exit__(self, *args):
        self._tracker.record_span(self._stage, self._start)


def create_timer(tracker: LatencySLOTracker, stage: LatencyStage) -> LatencyTimer:
    """Create a latency timer for a pipeline stage."""
    return LatencyTimer(tracker, stage)
