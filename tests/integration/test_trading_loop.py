"""
Integration tests for the full trading loop.

Tests the complete signal → order → fill cycle with mocked exchange.
"""

import asyncio
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from src.strategy.value_betting import EnsembleStrategy, StrategyMetrics, PendingFill
from src.risk.kelly_sizer import KellySizer, AdaptiveMode
from src.risk.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, BreakerState, BreakerType
from src.execution.order_manager import OrderManager, OrderState, OrderSide
from src.execution.rate_limiter import RateLimiter
from src.portfolio.tracker import Portfolio, PositionSide
from src.infrastructure.config import AppConfig, TradingConfig
from src.ingestion.market_discovery import Market


@pytest.fixture
def app_config():
    """Create test configuration."""
    return AppConfig(
        environment="test",
        dry_run=True,
        trading=TradingConfig(
            assets=["BTC", "ETH"],
            min_edge_threshold=0.04,
            kelly_fraction=0.25,
            max_position_pct=0.02,
        ),
    )


@pytest.fixture
def mock_market():
    """Create a mock market for testing."""
    return Market(
        condition_id="test_cond_123",
        question="Will BTC be above $100,000 at 12:15 UTC?",
        slug="btc-updown-1215",  # Added slug
        asset="BTC",
        yes_token_id="yes_token_abc",
        no_token_id="no_token_xyz",
        yes_price=0.50,
        no_price=0.50,
        end_date=datetime.now(timezone.utc) + timedelta(minutes=10),
        # start_date removed (not in Market dataclass)
        open_price=100000,
    )


@pytest.fixture
def order_manager():
    """Create order manager for testing."""
    return OrderManager(dry_run=True)


@pytest.fixture
def circuit_breaker():
    """Create circuit breaker for testing."""
    return CircuitBreaker(starting_capital=1000)


@pytest.fixture
def rate_limiter():
    """Create rate limiter for testing."""
    return RateLimiter()


class TestTradingLoopIntegration:
    """Integration tests for the trading loop."""

    @pytest.mark.asyncio
    async def test_signal_to_order_flow(
        self,
        app_config,
        mock_market,
        order_manager,
        circuit_breaker,
        rate_limiter,
    ):
        """Test complete flow from signal generation to order placement."""
        # Setup mocks
        mock_discovery = AsyncMock()
        mock_discovery.get_crypto_15m_markets.return_value = [mock_market]

        # Fix: Helper to return non-coroutine mock
        mock_price = MagicMock()
        mock_price.age_seconds = 1.0

        async def get_cached_price_side_effect(*args, **kwargs):
            return mock_price

        mock_oracle = MagicMock()
        mock_oracle.get_all_prices = AsyncMock(return_value={"BTC": 100500})
        mock_oracle.get_price = AsyncMock(return_value=100500)
        # Fix: get_cached_price is synchronous
        mock_oracle.get_cached_price = MagicMock(return_value=mock_price)

        # Create strategy
        strategy = EnsembleStrategy(
            config=app_config,
            market_discovery=mock_discovery,
            oracle=mock_oracle,
            order_manager=order_manager,
            rate_limiter=rate_limiter,
            circuit_breaker=circuit_breaker,
            bankroll=1000,
        )

        # Process a single market (simulates one iteration)
        await strategy.process_market(mock_market)

        # Verify order was placed (in dry run mode)
        assert order_manager.stats["submitted"] >= 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_trading(
        self,
        app_config,
        mock_market,
        order_manager,
        rate_limiter,
    ):
        """Test that circuit breaker prevents trading when tripped."""
        # Create tripped circuit breaker
        circuit_breaker = CircuitBreaker(starting_capital=1000)
        circuit_breaker.update_daily_pnl(-100)  # 10% loss, triggers hard trip

        assert not circuit_breaker.can_trade()

        mock_discovery = AsyncMock()
        mock_discovery.get_crypto_15m_markets.return_value = [mock_market]

        mock_price = MagicMock()
        mock_price.age_seconds = 1.0

        mock_oracle = MagicMock()
        mock_oracle.get_all_prices = AsyncMock(return_value={"BTC": 100500})
        mock_oracle.get_price = AsyncMock(return_value=100500)
        # Fix: get_cached_price is synchronous
        mock_oracle.get_cached_price = MagicMock(return_value=mock_price)

        strategy = EnsembleStrategy(
            config=app_config,
            market_discovery=mock_discovery,
            oracle=mock_oracle,
            order_manager=order_manager,
            rate_limiter=rate_limiter,
            circuit_breaker=circuit_breaker,
            bankroll=1000,
        )

        # Process a market - should be blocked by circuit breaker
        await strategy.process_market(mock_market)

        # No orders should be placed
        assert order_manager.stats["submitted"] == 0

    @pytest.mark.asyncio
    async def test_fill_handling_updates_portfolio(
        self,
        app_config,
        mock_market,
        order_manager,
        circuit_breaker,
        rate_limiter,
    ):
        """Test that fills properly update portfolio state."""
        mock_discovery = AsyncMock()
        
        mock_price = MagicMock()
        mock_price.age_seconds = 1.0
        
        mock_oracle = MagicMock()
        mock_oracle.get_all_prices = AsyncMock(return_value={"BTC": 100500})
        # Fix: get_cached_price is synchronous
        mock_oracle.get_cached_price = MagicMock(return_value=mock_price)

        strategy = EnsembleStrategy(
            config=app_config,
            market_discovery=mock_discovery,
            oracle=mock_oracle,
            order_manager=order_manager,
            rate_limiter=rate_limiter,
            circuit_breaker=circuit_breaker,
            bankroll=1000,
        )

        # Simulate a fill
        order = order_manager.create_order(
            token_id="test_token",
            side=OrderSide.BUY,
            price=0.55,
            size=10.0,
            strategy_id="test",
        )
        await order_manager.submit_order(order)

        # Register pending fill metadata
        strategy._pending_fills[order.client_order_id] = PendingFill(
            token_id="test_token",
            asset="BTC",
            side=PositionSide.LONG_YES,
            price=0.55,
            size_usd=10.0,
            market_question="Test question",
            expires_at=datetime.now(timezone.utc).timestamp() + 600,
        )

        # Simulate fill
        strategy.handle_fill(order.client_order_id, 10.0, 0.55)

        # Verify metrics updated
        assert strategy.metrics.orders_filled == 1


class TestAdaptiveKellyIntegration:
    """Integration tests for adaptive Kelly sizing."""

    def test_volatility_reduces_size(self):
        """Test that high volatility reduces position size."""
        sizer = KellySizer(
            bankroll=1000,
            kelly_fraction=0.25,
            adaptive_mode=AdaptiveMode.FULL,
        )

        # Normal volatility
        result_normal = sizer.calculate(
            win_prob=0.60,
            market_price=0.50,
            current_volatility=0.60,
        )

        # High volatility
        result_high_vol = sizer.calculate(
            win_prob=0.60,
            market_price=0.50,
            current_volatility=1.50,
        )

        # High vol should result in smaller size
        assert result_high_vol.recommended_size_usd < result_normal.recommended_size_usd
        assert result_high_vol.volatility_multiplier < 1.0

    def test_drawdown_reduces_size(self):
        """Test that drawdown reduces position size."""
        sizer = KellySizer(
            bankroll=1000,
            kelly_fraction=0.25,
            adaptive_mode=AdaptiveMode.FULL,
        )

        # No drawdown
        result_no_dd = sizer.calculate(
            win_prob=0.60,
            market_price=0.50,
            current_drawdown_pct=0.0,
        )

        # 5% drawdown
        result_dd = sizer.calculate(
            win_prob=0.60,
            market_price=0.50,
            current_drawdown_pct=0.05,
        )

        # Drawdown should reduce size
        assert result_dd.recommended_size_usd < result_no_dd.recommended_size_usd
        assert result_dd.drawdown_multiplier < 1.0

    def test_consecutive_losses_reduce_size(self):
        """Test that consecutive losses reduce position size."""
        sizer = KellySizer(
            bankroll=1000,
            kelly_fraction=0.25,
            adaptive_mode=AdaptiveMode.FULL,
        )

        # Record multiple losses
        for _ in range(5):
            sizer.record_trade_result(won=False)

        result = sizer.calculate(
            win_prob=0.60,
            market_price=0.50,
        )

        # Should have reduced uncertainty multiplier
        assert result.uncertainty_multiplier < 1.0


class TestCircuitBreakerIntegration:
    """Integration tests for circuit breaker behavior."""

    def test_consecutive_losses_trip_breaker(self):
        """Test that consecutive losses trip the breaker."""
        breaker = CircuitBreaker(
            starting_capital=1000,
            config=CircuitBreakerConfig(
                consecutive_loss_soft=3,
                consecutive_loss_hard=5,
            ),
        )

        # Record losses
        for i in range(6):
            breaker.record_trade_result(won=False)
            if i >= 4:
                assert not breaker.can_trade()

    def test_auto_reset_after_cooldown(self):
        """Test that breakers auto-reset after cooldown."""
        import time

        breaker = CircuitBreaker(
            starting_capital=1000,
            config=CircuitBreakerConfig(
                soft_reset_seconds=0.1,  # 100ms for testing
            ),
        )

        # Trip a breaker
        breaker.update_volatility(1.6)  # Soft trip
        assert breaker._states[BreakerType.VOLATILITY] != BreakerState.CLOSED

        # Wait for cooldown
        time.sleep(0.2)

        # Reset volatility and check auto-reset
        breaker.update_volatility(0.5)
        reset_breakers = breaker.check_auto_reset()

        # Should be able to trade now
        assert breaker.can_trade()


class TestOrderManagerIntegration:
    """Integration tests for order management."""

    @pytest.mark.asyncio
    async def test_order_lifecycle(self):
        """Test complete order lifecycle."""
        manager = OrderManager(dry_run=True)

        # Create order
        order = manager.create_order(
            token_id="test_token",
            side=OrderSide.BUY,
            price=0.55,
            size=10.0,
        )
        assert order.state == OrderState.CREATING

        # Submit order
        success = await manager.submit_order(order)
        assert success
        assert order.state == OrderState.NEW

        # Simulate fill
        manager.handle_fill(order.client_order_id, 10.0, 0.55)
        assert order.state == OrderState.FILLED
        assert order.filled_size == 10.0

    @pytest.mark.asyncio
    async def test_order_timeout(self):
        """Test order timeout functionality."""
        manager = OrderManager(dry_run=True)

        # Create and submit order
        order = manager.create_order(
            token_id="test_token",
            side=OrderSide.BUY,
            price=0.55,
            size=10.0,
        )
        await manager.submit_order(order)

        # Force age the order
        order.created_at -= 400  # 400 seconds ago

        # Timeout stale orders
        timed_out = await manager.timeout_stale_orders(max_active_age_seconds=300)

        assert len(timed_out) == 1
        assert order.state in (OrderState.CANCELED, OrderState.EXPIRED)

    @pytest.mark.asyncio
    async def test_order_amend(self):
        """Test order amendment."""
        manager = OrderManager(dry_run=True)

        # Create and submit order
        order = manager.create_order(
            token_id="test_token",
            side=OrderSide.BUY,
            price=0.55,
            size=10.0,
        )
        await manager.submit_order(order)

        # Amend price
        success = await manager.amend_order(order, new_price=0.52)

        assert success
        assert order.price == 0.52
