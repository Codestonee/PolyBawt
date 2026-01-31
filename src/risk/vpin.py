"""VPIN (Volume-Synchronized Probability of Informed Trading).

This module is used both by strategies and by unit tests.

Design (as implied by tests):
- Trades are accumulated into fixed-volume "buckets" (bucket_size).
- For each completed bucket, we track buy_volume and sell_volume.
- VPIN is the rolling average imbalance across the last n_buckets:
    imbalance_i = |buy_i - sell_i| / bucket_size
    VPIN = mean(imbalance_i)

Toxicity thresholds:
- VPIN < 0.3 -> LOW
- 0.3 <= VPIN < 0.5 -> MODERATE
- 0.5 <= VPIN < 0.7 -> HIGH
- VPIN >= 0.7 -> EXTREME

Sizing rules (as implied by tests):
- LOW -> 1.0
- MODERATE -> 0.5
- HIGH/EXTREME -> 0.0 (halt)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from collections import deque
from typing import Deque


class ToxicityLevel(Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class VPINResult:
    vpin: float
    toxicity_level: ToxicityLevel
    n_buckets_filled: int
    is_reliable: bool

    @property
    def should_halt(self) -> bool:
        return self.toxicity_level in (ToxicityLevel.HIGH, ToxicityLevel.EXTREME)

    @property
    def should_reduce_size(self) -> bool:
        return self.toxicity_level == ToxicityLevel.MODERATE

    @property
    def size_multiplier(self) -> float:
        if self.toxicity_level == ToxicityLevel.LOW:
            return 1.0
        if self.toxicity_level == ToxicityLevel.MODERATE:
            return 0.5
        return 0.0


class VPINCalculator:
    """Multi-market VPIN calculator with bucketed volume."""

    def __init__(
        self,
        bucket_size: float = 100.0,
        n_buckets: int = 50,
        min_buckets_reliable: int = 5,
    ):
        self.bucket_size = float(bucket_size)
        self.n_buckets = int(n_buckets)
        self.min_buckets_reliable = int(min_buckets_reliable)

        # market_id -> deque[(buy_vol, sell_vol)] of completed buckets
        self._completed_buckets: dict[str, Deque[tuple[float, float]]] = {}

        # market_id -> current bucket state
        #   {"buy": float, "sell": float, "filled": float}
        self._current_bucket: dict[str, dict[str, float]] = {}

    def _ensure_market(self, market_id: str) -> None:
        if market_id not in self._completed_buckets:
            self._completed_buckets[market_id] = deque(maxlen=self.n_buckets)
        if market_id not in self._current_bucket:
            self._current_bucket[market_id] = {"buy": 0.0, "sell": 0.0, "filled": 0.0}

    def update(self, market_id: str, volume: float, is_buy: bool) -> None:
        """Add a trade volume to a market's VPIN stream.

        If volume exceeds remaining space in the current bucket, it rolls over
        into subsequent buckets.
        """
        self._ensure_market(market_id)

        remaining = float(volume)
        while remaining > 0:
            cur = self._current_bucket[market_id]
            space = self.bucket_size - cur["filled"]
            take = min(space, remaining)

            if is_buy:
                cur["buy"] += take
            else:
                cur["sell"] += take

            cur["filled"] += take
            remaining -= take

            # Bucket completed
            if cur["filled"] >= self.bucket_size - 1e-12:
                self._completed_buckets[market_id].append((cur["buy"], cur["sell"]))
                # reset current bucket
                self._current_bucket[market_id] = {"buy": 0.0, "sell": 0.0, "filled": 0.0}

    def _classify_toxicity(self, vpin: float) -> ToxicityLevel:
        if vpin < 0.3:
            return ToxicityLevel.LOW
        if vpin < 0.5:
            return ToxicityLevel.MODERATE
        if vpin < 0.7:
            return ToxicityLevel.HIGH
        return ToxicityLevel.EXTREME

    def get_vpin(self, market_id: str) -> VPINResult:
        """Compute VPIN for a market."""
        self._ensure_market(market_id)
        buckets = self._completed_buckets.get(market_id)
        n = len(buckets) if buckets is not None else 0

        if n == 0:
            # neutral default (as tests expect)
            return VPINResult(
                vpin=0.5,
                toxicity_level=ToxicityLevel.MODERATE,
                n_buckets_filled=0,
                is_reliable=False,
            )

        # Compute rolling average imbalance
        imbalances = [abs(b - s) / self.bucket_size for (b, s) in buckets]
        vpin = sum(imbalances) / len(imbalances)
        level = self._classify_toxicity(vpin)
        reliable = n >= self.min_buckets_reliable

        # If not reliable, keep neutral-ish classification to avoid overreacting
        if not reliable:
            level = ToxicityLevel.MODERATE

        return VPINResult(
            vpin=float(vpin) if reliable else 0.5,
            toxicity_level=level,
            n_buckets_filled=n,
            is_reliable=reliable,
        )

    def get_all_vpins(self) -> dict[str, VPINResult]:
        return {mid: self.get_vpin(mid) for mid in list(self._completed_buckets.keys())}

    def reset(self, market_id: str | None = None) -> None:
        """Reset one market or all markets."""
        if market_id is None:
            self._completed_buckets.clear()
            self._current_bucket.clear()
            return

        self._completed_buckets.pop(market_id, None)
        self._current_bucket.pop(market_id, None)
