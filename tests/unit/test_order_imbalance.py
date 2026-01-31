"""
Tests for Order Book Imbalance (OBI) signal.
"""

import pytest
from src.strategy.features.order_imbalance import (
    OrderBookImbalance,
    OBISignal,
    OBIResult,
    calculate_obi,
)


class TestOrderBookImbalance:
    """Test OBI calculation basics."""

    def test_empty_book_returns_neutral(self):
        """Empty order book should return neutral."""
        obi = OrderBookImbalance()
        result = obi.calculate(bids=[], asks=[])

        assert result.obi == 0.0
        assert result.signal == OBISignal.NEUTRAL
        assert not result.is_reliable

    def test_balanced_book_neutral(self):
        """Equal bid/ask depth should be neutral."""
        obi = OrderBookImbalance()
        bids = [(0.55, 100), (0.54, 100)]
        asks = [(0.56, 100), (0.57, 100)]

        result = obi.calculate(bids=bids, asks=asks)

        assert -0.1 <= result.obi <= 0.1  # Near zero
        assert result.signal == OBISignal.NEUTRAL

    def test_heavy_bids_buy_signal(self):
        """Much more bids than asks should give buy signal."""
        obi = OrderBookImbalance()
        bids = [(0.55, 400), (0.54, 400)]  # 800 total
        asks = [(0.56, 100), (0.57, 100)]   # 200 total

        result = obi.calculate(bids=bids, asks=asks)

        assert result.obi > 0.2
        assert result.signal in (OBISignal.BUY, OBISignal.STRONG_BUY)
        assert result.is_actionable

    def test_heavy_asks_sell_signal(self):
        """Much more asks than bids should give sell signal."""
        obi = OrderBookImbalance()
        bids = [(0.55, 100), (0.54, 100)]   # 200 total
        asks = [(0.56, 400), (0.57, 400)]  # 800 total

        result = obi.calculate(bids=bids, asks=asks)

        assert result.obi < -0.2
        assert result.signal in (OBISignal.SELL, OBISignal.STRONG_SELL)
        assert result.is_actionable

    def test_strong_buy_threshold(self):
        """Very heavy bids should give STRONG_BUY."""
        obi = OrderBookImbalance(strong_threshold=0.4)
        # 90% bids, 10% asks -> OBI = (90-10)/(90+10) = 0.8
        bids = [(0.55, 90)]
        asks = [(0.56, 10)]

        result = obi.calculate(bids=bids, asks=asks)

        assert result.obi >= 0.4
        assert result.signal == OBISignal.STRONG_BUY

    def test_strong_sell_threshold(self):
        """Very heavy asks should give STRONG_SELL."""
        obi = OrderBookImbalance(strong_threshold=0.4)
        bids = [(0.55, 10)]
        asks = [(0.56, 90)]

        result = obi.calculate(bids=bids, asks=asks)

        assert result.obi <= -0.4
        assert result.signal == OBISignal.STRONG_SELL


class TestOBILevels:
    """Test level-based analysis."""

    def test_respects_level_limit(self):
        """Should only analyze specified number of levels."""
        obi = OrderBookImbalance(levels=2)
        bids = [
            (0.55, 100),  # Level 1 - included
            (0.54, 100),  # Level 2 - included
            (0.53, 1000), # Level 3 - excluded
        ]
        asks = [(0.56, 100), (0.57, 100)]

        result = obi.calculate(bids=bids, asks=asks)

        # If level 3 was included, bids would be 1200 vs 200 asks
        # But with only 2 levels, it's 200 vs 200
        assert result.levels_analyzed == 2

    def test_fewer_levels_than_requested(self):
        """Should handle books with fewer levels than requested."""
        obi = OrderBookImbalance(levels=10)
        bids = [(0.55, 100)]
        asks = [(0.56, 100)]

        result = obi.calculate(bids=bids, asks=asks)

        assert result.levels_analyzed == 1


class TestOBIWeighting:
    """Test price-weighted volume calculation."""

    def test_weighted_favors_closer_orders(self):
        """Orders closer to mid should have more weight."""
        obi_weighted = OrderBookImbalance(weighted=True, levels=3)
        obi_unweighted = OrderBookImbalance(weighted=False, levels=3)

        # Bids: small close, large far
        bids = [(0.50, 50), (0.40, 200)]
        # Asks: large close, small far
        asks = [(0.51, 200), (0.60, 50)]

        result_w = obi_weighted.calculate(bids=bids, asks=asks)
        result_uw = obi_unweighted.calculate(bids=bids, asks=asks)

        # Weighted should favor asks (large volume close to mid)
        # Unweighted: bids=250, asks=250, OBI=0
        assert result_uw.obi == 0  # Unweighted is balanced
        # Weighted should be negative (more ask weight)
        assert result_w.obi < result_uw.obi


class TestOBIReliability:
    """Test reliability based on volume."""

    def test_low_volume_unreliable(self):
        """Low total volume should be unreliable."""
        obi = OrderBookImbalance(min_volume_usd=1000)
        bids = [(0.55, 10)]
        asks = [(0.56, 10)]

        result = obi.calculate(bids=bids, asks=asks)

        assert not result.is_reliable

    def test_high_volume_reliable(self):
        """High total volume should be reliable."""
        obi = OrderBookImbalance(min_volume_usd=100)
        bids = [(0.55, 500)]
        asks = [(0.56, 500)]

        result = obi.calculate(bids=bids, asks=asks)

        assert result.is_reliable


class TestOBIResult:
    """Test OBIResult properties."""

    def test_signal_strength(self):
        """Signal strength should be absolute OBI."""
        result = OBIResult(
            obi=-0.6,
            signal=OBISignal.STRONG_SELL,
            bid_volume=200,
            ask_volume=800,
            levels_analyzed=5,
            is_reliable=True,
        )
        assert result.signal_strength == 0.6

    def test_actionable_signals(self):
        """BUY/SELL signals should be actionable."""
        for signal in [OBISignal.STRONG_BUY, OBISignal.BUY, OBISignal.SELL, OBISignal.STRONG_SELL]:
            result = OBIResult(
                obi=0.3,
                signal=signal,
                bid_volume=100,
                ask_volume=100,
                levels_analyzed=5,
                is_reliable=True,
            )
            assert result.is_actionable

    def test_neutral_not_actionable(self):
        """NEUTRAL should not be actionable."""
        result = OBIResult(
            obi=0.0,
            signal=OBISignal.NEUTRAL,
            bid_volume=100,
            ask_volume=100,
            levels_analyzed=5,
            is_reliable=True,
        )
        assert not result.is_actionable


class TestSimpleOBI:
    """Test the simple calculate_obi function."""

    def test_simple_obi_calculation(self):
        """Simple OBI function should work."""
        bids = [(0.55, 300)]
        asks = [(0.56, 100)]

        obi = calculate_obi(bids=bids, asks=asks)

        # (300 - 100) / (300 + 100) = 0.5
        assert obi == 0.5

    def test_simple_obi_empty(self):
        """Empty book should return 0."""
        assert calculate_obi(bids=[], asks=[]) == 0.0

    def test_simple_obi_levels(self):
        """Should respect level limit."""
        bids = [(0.55, 100), (0.54, 1000)]  # 1100 if all, 100 if 1 level
        asks = [(0.56, 100)]

        obi_1 = calculate_obi(bids=bids, asks=asks, levels=1)
        obi_2 = calculate_obi(bids=bids, asks=asks, levels=2)

        # 1 level: (100 - 100) / 200 = 0
        # 2 levels: (1100 - 100) / 1200 = 0.833
        assert obi_1 == 0.0
        assert obi_2 > 0.8
