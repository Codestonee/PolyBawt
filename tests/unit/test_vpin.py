"""
Tests for VPIN (Volume-Synchronized Probability of Informed Trading) calculator.
"""

import pytest
from src.risk.vpin import VPINCalculator, ToxicityLevel, VPINResult


class TestVPINCalculator:
    """Test VPIN calculation basics."""

    def test_empty_market_returns_neutral(self):
        """No data should return neutral VPIN."""
        calc = VPINCalculator(bucket_size=100, n_buckets=10)
        result = calc.get_vpin("market_1")

        assert result.vpin == 0.5
        assert result.toxicity_level == ToxicityLevel.MODERATE
        assert not result.is_reliable
        assert result.n_buckets_filled == 0

    def test_balanced_flow_low_vpin(self):
        """Equal buy/sell volume should give low VPIN."""
        calc = VPINCalculator(bucket_size=100, n_buckets=10)

        # Fill buckets with balanced flow
        for _ in range(15):  # More than min_buckets
            calc.update("market_1", volume=50, is_buy=True)
            calc.update("market_1", volume=50, is_buy=False)

        result = calc.get_vpin("market_1")

        assert result.vpin < 0.3  # Low toxicity
        assert result.toxicity_level == ToxicityLevel.LOW
        assert result.is_reliable
        assert not result.should_halt

    def test_imbalanced_buy_flow_high_vpin(self):
        """All buy volume should give high VPIN."""
        calc = VPINCalculator(bucket_size=100, n_buckets=10)

        # Fill buckets with all buys
        for _ in range(15):
            calc.update("market_1", volume=100, is_buy=True)

        result = calc.get_vpin("market_1")

        assert result.vpin > 0.9  # Very high toxicity (all imbalanced)
        assert result.toxicity_level == ToxicityLevel.EXTREME
        assert result.should_halt

    def test_imbalanced_sell_flow_high_vpin(self):
        """All sell volume should give high VPIN."""
        calc = VPINCalculator(bucket_size=100, n_buckets=10)

        # Fill buckets with all sells
        for _ in range(15):
            calc.update("market_1", volume=100, is_buy=False)

        result = calc.get_vpin("market_1")

        assert result.vpin > 0.9
        assert result.toxicity_level == ToxicityLevel.EXTREME

    def test_moderate_imbalance(self):
        """70/30 split should give moderate VPIN."""
        calc = VPINCalculator(bucket_size=100, n_buckets=10)

        # Fill buckets with 70% buy, 30% sell
        for _ in range(15):
            calc.update("market_1", volume=70, is_buy=True)
            calc.update("market_1", volume=30, is_buy=False)

        result = calc.get_vpin("market_1")

        # Imbalance = |70-30| = 40, Total = 100
        # VPIN should be around 0.4
        assert 0.3 <= result.vpin <= 0.5
        assert result.toxicity_level in (ToxicityLevel.MODERATE, ToxicityLevel.HIGH)


class TestVPINBuckets:
    """Test bucket accumulation logic."""

    def test_bucket_overflow(self):
        """Volume exceeding bucket size should roll over."""
        calc = VPINCalculator(bucket_size=100, n_buckets=10)

        # Single large trade that fills multiple buckets
        calc.update("market_1", volume=250, is_buy=True)

        # Should have 2 full buckets + partial
        assert len(calc._completed_buckets["market_1"]) == 2

    def test_small_trades_accumulate(self):
        """Small trades should accumulate into buckets."""
        calc = VPINCalculator(bucket_size=100, n_buckets=10)

        # 10 small trades to fill 1 bucket
        for _ in range(10):
            calc.update("market_1", volume=10, is_buy=True)

        assert len(calc._completed_buckets["market_1"]) == 1

    def test_bucket_maxlen(self):
        """Old buckets should be dropped when limit reached."""
        calc = VPINCalculator(bucket_size=100, n_buckets=5)

        # Fill more than n_buckets
        for _ in range(10):
            calc.update("market_1", volume=100, is_buy=True)

        # Should only keep last n_buckets
        assert len(calc._completed_buckets["market_1"]) == 5


class TestVPINResult:
    """Test VPINResult properties."""

    def test_size_multiplier_low_toxicity(self):
        """Low toxicity should have full size."""
        result = VPINResult(
            vpin=0.2,
            toxicity_level=ToxicityLevel.LOW,
            n_buckets_filled=50,
            is_reliable=True,
        )
        assert result.size_multiplier == 1.0
        assert not result.should_halt

    def test_size_multiplier_moderate_toxicity(self):
        """Moderate toxicity should reduce size."""
        result = VPINResult(
            vpin=0.4,
            toxicity_level=ToxicityLevel.MODERATE,
            n_buckets_filled=50,
            is_reliable=True,
        )
        assert result.size_multiplier == 0.5
        assert not result.should_halt

    def test_size_multiplier_high_toxicity(self):
        """High toxicity should zero size."""
        result = VPINResult(
            vpin=0.6,
            toxicity_level=ToxicityLevel.HIGH,
            n_buckets_filled=50,
            is_reliable=True,
        )
        assert result.size_multiplier == 0.0
        assert result.should_halt

    def test_size_multiplier_extreme_toxicity(self):
        """Extreme toxicity should zero size."""
        result = VPINResult(
            vpin=0.8,
            toxicity_level=ToxicityLevel.EXTREME,
            n_buckets_filled=50,
            is_reliable=True,
        )
        assert result.size_multiplier == 0.0
        assert result.should_halt


class TestVPINMultiMarket:
    """Test VPIN tracks multiple markets independently."""

    def test_separate_markets(self):
        """Each market should have independent VPIN."""
        calc = VPINCalculator(bucket_size=100, n_buckets=10)

        # Market 1: All buys (toxic)
        for _ in range(15):
            calc.update("market_1", volume=100, is_buy=True)

        # Market 2: Balanced (safe)
        for _ in range(15):
            calc.update("market_2", volume=50, is_buy=True)
            calc.update("market_2", volume=50, is_buy=False)

        result_1 = calc.get_vpin("market_1")
        result_2 = calc.get_vpin("market_2")

        assert result_1.vpin > 0.9  # Toxic
        assert result_2.vpin < 0.3  # Safe

    def test_get_all_vpins(self):
        """Should return VPIN for all tracked markets."""
        calc = VPINCalculator(bucket_size=100, n_buckets=10)

        for _ in range(5):
            calc.update("market_1", volume=100, is_buy=True)
            calc.update("market_2", volume=100, is_buy=False)

        all_vpins = calc.get_all_vpins()

        assert "market_1" in all_vpins
        assert "market_2" in all_vpins

    def test_reset_single_market(self):
        """Reset should clear only specified market."""
        calc = VPINCalculator(bucket_size=100, n_buckets=10)

        for _ in range(5):
            calc.update("market_1", volume=100, is_buy=True)
            calc.update("market_2", volume=100, is_buy=True)

        calc.reset("market_1")

        assert calc.get_vpin("market_1").n_buckets_filled == 0
        assert calc.get_vpin("market_2").n_buckets_filled > 0

    def test_reset_all_markets(self):
        """Reset with no arg should clear all."""
        calc = VPINCalculator(bucket_size=100, n_buckets=10)

        for _ in range(5):
            calc.update("market_1", volume=100, is_buy=True)
            calc.update("market_2", volume=100, is_buy=True)

        calc.reset()

        assert len(calc._completed_buckets) == 0
        assert len(calc._current_bucket) == 0


class TestToxicityClassification:
    """Test toxicity level thresholds."""

    def test_low_threshold(self):
        """VPIN < 0.3 is LOW."""
        calc = VPINCalculator()
        assert calc._classify_toxicity(0.0) == ToxicityLevel.LOW
        assert calc._classify_toxicity(0.1) == ToxicityLevel.LOW
        assert calc._classify_toxicity(0.29) == ToxicityLevel.LOW

    def test_moderate_threshold(self):
        """0.3 <= VPIN < 0.5 is MODERATE."""
        calc = VPINCalculator()
        assert calc._classify_toxicity(0.3) == ToxicityLevel.MODERATE
        assert calc._classify_toxicity(0.4) == ToxicityLevel.MODERATE
        assert calc._classify_toxicity(0.49) == ToxicityLevel.MODERATE

    def test_high_threshold(self):
        """0.5 <= VPIN < 0.7 is HIGH."""
        calc = VPINCalculator()
        assert calc._classify_toxicity(0.5) == ToxicityLevel.HIGH
        assert calc._classify_toxicity(0.6) == ToxicityLevel.HIGH
        assert calc._classify_toxicity(0.69) == ToxicityLevel.HIGH

    def test_extreme_threshold(self):
        """VPIN >= 0.7 is EXTREME."""
        calc = VPINCalculator()
        assert calc._classify_toxicity(0.7) == ToxicityLevel.EXTREME
        assert calc._classify_toxicity(0.9) == ToxicityLevel.EXTREME
        assert calc._classify_toxicity(1.0) == ToxicityLevel.EXTREME
