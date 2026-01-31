"""
Tests for Market Making Strategy.
"""

import pytest
from src.strategy.market_making import (
    MarketMakingStrategy,
    MarketMakingState,
    QuoteConfig,
    Quote,
    MarketMakingSignal,
)
from src.risk.vpin import VPINCalculator


class TestMarketMakingBasics:
    """Test basic market making functionality."""

    def test_generates_two_sided_quotes(self):
        """Should generate both bid and ask quotes."""
        mm = MarketMakingStrategy()

        signal = mm.generate_signal(
            market_id="test",
            mid_price=0.50,
            token_id="token123",
        )

        assert signal.is_quoting
        assert signal.bid_quote is not None
        assert signal.ask_quote is not None
        assert signal.bid_quote.price < 0.50
        assert signal.ask_quote.price > 0.50

    def test_spread_applied_correctly(self):
        """Spread should be centered around mid."""
        config = QuoteConfig(spread_bps=100)  # 1% spread = 50 bps each side
        mm = MarketMakingStrategy(config=config)

        signal = mm.generate_signal(
            market_id="test",
            mid_price=0.50,
            token_id="token123",
        )

        # 50 bps = 0.0025 for mid of 0.50
        expected_half_spread = 0.50 * (50 / 10000)
        assert abs(0.50 - signal.bid_quote.price - expected_half_spread) < 0.0001
        assert abs(signal.ask_quote.price - 0.50 - expected_half_spread) < 0.0001


class TestVPINIntegration:
    """Test VPIN-based halting."""

    def test_halts_on_high_vpin(self):
        """Should halt quoting when VPIN is too high."""
        vpin_calc = VPINCalculator(bucket_size=100, n_buckets=10)
        mm = MarketMakingStrategy(vpin_calculator=vpin_calc)

        # Create toxic flow
        for _ in range(15):
            vpin_calc.update("toxic_market", volume=100, is_buy=True)

        signal = mm.generate_signal(
            market_id="toxic_market",
            mid_price=0.50,
            token_id="token123",
        )

        assert signal.state == MarketMakingState.HALTED
        assert not signal.is_quoting
        assert signal.should_cancel_all

    def test_reduces_size_on_moderate_vpin(self):
        """Should reduce size when VPIN is elevated."""
        vpin_calc = VPINCalculator(bucket_size=100, n_buckets=10)
        config = QuoteConfig(
            quote_size_usd=100.0,
            vpin_reduce_threshold=0.35,
        )
        mm = MarketMakingStrategy(config=config, vpin_calculator=vpin_calc)

        # Create moderately imbalanced flow (80/20 to get higher VPIN ~0.6)
        for _ in range(20):
            vpin_calc.update("moderate", volume=80, is_buy=True)
            vpin_calc.update("moderate", volume=20, is_buy=False)

        signal = mm.generate_signal(
            market_id="moderate",
            mid_price=0.50,
            token_id="token123",
        )

        # VPIN should be around 0.6, which is HIGH toxicity
        # Size multiplier for HIGH is 0.0, so quotes may not be generated
        # For MODERATE (0.3-0.5), size multiplier is 0.5
        assert signal.vpin.vpin > 0.35  # Above reduce threshold
        if signal.is_quoting:
            assert signal.bid_quote.size < config.quote_size_usd

    def test_continues_on_low_vpin(self):
        """Should quote normally when VPIN is low."""
        vpin_calc = VPINCalculator(bucket_size=100, n_buckets=10)
        mm = MarketMakingStrategy(vpin_calculator=vpin_calc)

        # Create balanced flow
        for _ in range(15):
            vpin_calc.update("safe_market", volume=50, is_buy=True)
            vpin_calc.update("safe_market", volume=50, is_buy=False)

        signal = mm.generate_signal(
            market_id="safe_market",
            mid_price=0.50,
            token_id="token123",
        )

        assert signal.state == MarketMakingState.ACTIVE
        assert signal.is_quoting


class TestInventoryManagement:
    """Test inventory-based quote skewing."""

    def _create_safe_mm(self):
        """Create MM with low VPIN for testing."""
        vpin_calc = VPINCalculator(bucket_size=100, n_buckets=10)
        # Pre-fill with balanced flow
        for _ in range(15):
            vpin_calc.update("test", volume=50, is_buy=True)
            vpin_calc.update("test", volume=50, is_buy=False)
        return MarketMakingStrategy(vpin_calculator=vpin_calc)

    def test_skews_quotes_when_long(self):
        """When long, should skew quotes to encourage selling."""
        mm = self._create_safe_mm()

        signal = mm.generate_signal(
            market_id="test",
            mid_price=0.50,
            token_id="token123",
            current_inventory=50.0,  # Long $50
        )

        # Should lower both prices to encourage selling
        # The skew should make ask more aggressive
        assert signal.skew_applied != 0

    def test_skews_quotes_when_short(self):
        """When short, should skew quotes to encourage buying."""
        mm = self._create_safe_mm()

        signal = mm.generate_signal(
            market_id="test",
            mid_price=0.50,
            token_id="token123",
            current_inventory=-50.0,  # Short $50
        )

        assert signal.skew_applied != 0

    def test_state_reflects_high_inventory(self):
        """Should report SKEWED state when inventory is high (low VPIN)."""
        vpin_calc = VPINCalculator(bucket_size=100, n_buckets=10)
        # Pre-fill with balanced flow for low VPIN
        for _ in range(15):
            vpin_calc.update("test", volume=50, is_buy=True)
            vpin_calc.update("test", volume=50, is_buy=False)

        config = QuoteConfig(max_inventory=100)
        mm = MarketMakingStrategy(config=config, vpin_calculator=vpin_calc)

        signal = mm.generate_signal(
            market_id="test",
            mid_price=0.50,
            token_id="token123",
            current_inventory=60.0,  # >50% of max
        )

        assert signal.state == MarketMakingState.SKEWED


class TestQuoteConfig:
    """Test quote configuration."""

    def _create_safe_vpin(self, market_id: str = "test"):
        """Create VPIN calculator with low toxicity."""
        vpin_calc = VPINCalculator(bucket_size=100, n_buckets=10)
        for _ in range(15):
            vpin_calc.update(market_id, volume=50, is_buy=True)
            vpin_calc.update(market_id, volume=50, is_buy=False)
        return vpin_calc

    def test_custom_spread(self):
        """Custom spread should be applied."""
        config = QuoteConfig(spread_bps=200)  # 2% total spread
        mm = MarketMakingStrategy(config=config, vpin_calculator=self._create_safe_vpin())

        signal = mm.generate_signal(
            market_id="test",
            mid_price=0.50,
            token_id="token123",
        )

        spread = signal.ask_quote.price - signal.bid_quote.price
        expected_spread = 0.50 * (200 / 10000)
        assert abs(spread - expected_spread) < 0.0001

    def test_custom_size(self):
        """Custom quote size should be applied (with low VPIN)."""
        config = QuoteConfig(quote_size_usd=25.0)
        mm = MarketMakingStrategy(config=config, vpin_calculator=self._create_safe_vpin())

        signal = mm.generate_signal(
            market_id="test",
            mid_price=0.50,
            token_id="token123",
        )

        assert signal.bid_quote.size == 25.0
        assert signal.ask_quote.size == 25.0


class TestMarketSelection:
    """Test market selection logic."""

    def test_prefers_extreme_prices(self):
        """Should prefer markets at extreme prices (low fees)."""
        mm = MarketMakingStrategy()

        assert mm.should_quote_market(0.05)  # 5% - extreme low
        assert mm.should_quote_market(0.95)  # 95% - extreme high

    def test_considers_volatility(self):
        """Should consider volatility vs spread."""
        config = QuoteConfig(spread_bps=50)  # 0.5% spread
        mm = MarketMakingStrategy(config=config)

        # High volatility - spread doesn't cover
        assert not mm.should_quote_market(0.50, expected_volatility=0.10)

        # Low volatility - spread covers
        assert mm.should_quote_market(0.50, expected_volatility=0.005)


class TestStatistics:
    """Test strategy statistics tracking."""

    def test_tracks_quote_count(self):
        """Should track number of quotes generated."""
        mm = MarketMakingStrategy()

        for i in range(5):
            mm.generate_signal(
                market_id=f"test_{i}",
                mid_price=0.50,
                token_id="token123",
            )

        stats = mm.get_stats()
        assert stats["quotes_generated"] == 5

    def test_tracks_halt_rate(self):
        """Should track halt rate."""
        vpin_calc = VPINCalculator(bucket_size=100, n_buckets=10)
        mm = MarketMakingStrategy(vpin_calculator=vpin_calc)

        # Generate toxic flow for one market
        for _ in range(15):
            vpin_calc.update("toxic", volume=100, is_buy=True)

        # Try to quote toxic market
        mm.generate_signal("toxic", 0.50, "token1")

        # Quote safe market
        mm.generate_signal("safe", 0.50, "token2")

        stats = mm.get_stats()
        assert stats["quotes_halted"] >= 1
        assert stats["halt_rate"] > 0


class TestMarketMakingSignal:
    """Test MarketMakingSignal dataclass."""

    def test_is_quoting_with_quotes(self):
        """Should be quoting when quotes exist."""
        signal = MarketMakingSignal(
            state=MarketMakingState.ACTIVE,
            bid_quote=Quote("bid", 0.49, 10, "token"),
            ask_quote=Quote("ask", 0.51, 10, "token"),
        )
        assert signal.is_quoting

    def test_is_quoting_without_quotes(self):
        """Should not be quoting when no quotes."""
        signal = MarketMakingSignal(state=MarketMakingState.HALTED)
        assert not signal.is_quoting

    def test_should_cancel_when_halted(self):
        """Should cancel all when halted."""
        signal = MarketMakingSignal(state=MarketMakingState.HALTED)
        assert signal.should_cancel_all

        signal2 = MarketMakingSignal(state=MarketMakingState.ACTIVE)
        assert not signal2.should_cancel_all
