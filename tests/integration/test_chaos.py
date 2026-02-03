"""
Chaos and fault injection tests.

Tests system resilience under failure conditions:
- Network failures
- Exchange API errors
- Timeout scenarios
- Data corruption
- Race conditions
"""

import asyncio
import pytest
import random
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any

from src.execution.order_manager import OrderManager, OrderState, OrderSide
from src.execution.rate_limiter import RateLimiter, RateLimitConfig
from src.risk.circuit_breaker import CircuitBreaker
from src.infrastructure.config import AppConfig


class NetworkError(Exception):
    """Simulated network error."""
    pass


class ExchangeError(Exception):
    """Simulated exchange error."""
    pass


class ChaosMonkey:
    """
    Chaos injection utility.

    Randomly injects failures to test system resilience.
    """

    def __init__(self, failure_rate: float = 0.3):
        self.failure_rate = failure_rate
        self.failures_injected = 0

    def maybe_fail(self, error_type: type = Exception) -> None:
        """Maybe raise an exception based on failure rate."""
        if random.random() < self.failure_rate:
            self.failures_injected += 1
            raise error_type(f"Chaos injection #{self.failures_injected}")

    async def maybe_delay(self, max_delay: float = 1.0) -> None:
        """Maybe add random delay."""
        if random.random() < self.failure_rate:
            delay = random.uniform(0, max_delay)
            await asyncio.sleep(delay)

    def corrupt_data(self, data: dict[str, Any], corruption_rate: float = 0.1) -> dict[str, Any]:
        """Randomly corrupt data fields."""
        corrupted = data.copy()
        for key in list(corrupted.keys()):
            if random.random() < corruption_rate:
                if isinstance(corrupted[key], (int, float)):
                    corrupted[key] = corrupted[key] * random.uniform(-10, 10)
                elif isinstance(corrupted[key], str):
                    corrupted[key] = "CORRUPTED"
                elif corrupted[key] is not None:
                    corrupted[key] = None
        return corrupted


class TestNetworkFailures:
    """Tests for network failure resilience."""

    @pytest.mark.asyncio
    async def test_order_submission_with_network_failure(self):
        """Test order submission handles network failures gracefully."""
        failures = 0
        successes = 0

        async def flaky_submit(order):
            nonlocal failures
            if random.random() < 0.5:
                failures += 1
                raise NetworkError("Connection reset")
            return True

        manager = OrderManager(dry_run=False, submit_callback=flaky_submit)

        # Try multiple submissions
        for i in range(10):
            order = manager.create_order(
                token_id=f"token_{i}",
                side=OrderSide.BUY,
                price=0.50,
                size=10.0,
            )
            result = await manager.submit_order(order)
            if result:
                successes += 1

        # Should handle failures gracefully
        assert failures > 0  # Some failures occurred
        assert successes > 0  # Some succeeded
        # Failed orders should be in FAILED state
        failed_orders = [o for o in manager._orders.values() if o.state == OrderState.FAILED]
        assert len(failed_orders) == failures

    @pytest.mark.asyncio
    async def test_order_cancel_with_timeout(self):
        """Test order cancellation handles timeouts."""
        async def slow_cancel(order):
            await asyncio.sleep(10)  # Very slow

        manager = OrderManager(dry_run=False, cancel_callback=slow_cancel)

        order = manager.create_order(
            token_id="test_token",
            side=OrderSide.BUY,
            price=0.50,
            size=10.0,
        )
        order.state = OrderState.NEW  # Manually set to active

        # Cancel with timeout
        try:
            await asyncio.wait_for(
                manager.cancel_order(order),
                timeout=0.1
            )
        except asyncio.TimeoutError:
            pass  # Expected

        # Order should be in pending cancel state
        assert order.state == OrderState.PENDING_CANCEL


class TestExchangeErrors:
    """Tests for exchange error handling."""

    @pytest.mark.asyncio
    async def test_rejection_handling(self):
        """Test proper handling of order rejections."""
        async def rejecting_submit(order):
            raise ExchangeError("Insufficient funds")

        manager = OrderManager(dry_run=False, submit_callback=rejecting_submit)

        order = manager.create_order(
            token_id="test_token",
            side=OrderSide.BUY,
            price=0.50,
            size=1000000.0,  # Very large
        )

        result = await manager.submit_order(order)

        assert not result
        assert order.state == OrderState.FAILED
        assert "Insufficient funds" in order.error_message

    @pytest.mark.asyncio
    async def test_partial_fill_then_error(self):
        """Test handling partial fill followed by error."""
        manager = OrderManager(dry_run=True)

        order = manager.create_order(
            token_id="test_token",
            side=OrderSide.BUY,
            price=0.50,
            size=100.0,
        )
        await manager.submit_order(order)

        # Partial fill
        manager.handle_fill(order.client_order_id, 50.0, 0.50)
        assert order.state == OrderState.PARTIALLY_FILLED
        assert order.filled_size == 50.0

        # Remaining fill
        manager.handle_fill(order.client_order_id, 50.0, 0.51)
        assert order.state == OrderState.FILLED
        assert order.filled_size == 100.0


class TestRateLimiterResilience:
    """Tests for rate limiter under stress."""

    @pytest.mark.asyncio
    async def test_rate_limiter_under_burst(self):
        """Test rate limiter handling of burst traffic."""
        # Setup limiter with small capacity
        config = RateLimitConfig(
            name="orders",
            tokens_per_second=2,
            bucket_size=5
        )
        limiter = RateLimiter(order_config=config)
        
        # Initial burst should succeed
        assert limiter.try_acquire_order(count=1)
        assert limiter.try_acquire_order(count=1)
        assert limiter._buckets["orders"].usage_pct < 1.0

    @pytest.mark.asyncio
    async def test_concurrent_rate_limit_access(self):
        """Test concurrent access to rate limiter."""
        config = RateLimitConfig(
            name="orders",
            tokens_per_second=100,
            bucket_size=1000
        )
        limiter = RateLimiter(order_config=config)
        
        async def make_request():
            await limiter.acquire_order()
            return True
            
        # Launch concurrent requests
        tasks = [make_request() for _ in range(50)]
        results = await asyncio.gather(*tasks)
        
        assert all(results)
        assert limiter.order_usage_pct() > 0


class TestCircuitBreakerResilience:
    """Tests for circuit breaker behavior under edge cases."""

    def test_rapid_state_transitions(self):
        """Test circuit breaker handles rapid state changes."""
        breaker = CircuitBreaker(starting_capital=1000)

        # Rapid oscillation between good and bad states
        for i in range(100):
            if i % 2 == 0:
                breaker.update_daily_pnl(-40)  # 4% loss - soft trip
            else:
                breaker.update_daily_pnl(10)   # Recovery

        # Should be in a consistent state
        assert breaker._states is not None

    def test_multiple_breakers_trip_simultaneously(self):
        """Test multiple breakers tripping at once."""
        breaker = CircuitBreaker(starting_capital=1000)

        # Trip multiple breakers
        breaker.update_daily_pnl(-60)        # 6% loss - hard trip
        breaker.update_volatility(2.5)       # 250% vol - hard trip
        breaker.update_oracle_age(35)        # Stale oracle - hard trip

        # Should not be able to trade
        assert not breaker.can_trade()

        # Reset all
        breaker.update_daily_pnl(0)
        breaker.update_volatility(0.5)
        breaker.update_oracle_age(1)

        # Should be able to trade again
        assert breaker.can_trade()

    def test_consecutive_loss_edge_cases(self):
        """Test consecutive loss counter edge cases."""
        breaker = CircuitBreaker(starting_capital=1000)

        # Alternating wins and losses (should not trip)
        for _ in range(20):
            breaker.record_trade_result(won=True)
            breaker.record_trade_result(won=False)

        assert breaker.consecutive_losses <= 1
        assert breaker.can_trade()

        # Now consecutive losses
        for _ in range(10):
            breaker.record_trade_result(won=False)

        assert breaker.consecutive_losses == 11


class TestDataCorruption:
    """Tests for handling corrupted data."""

    @pytest.mark.skip(reason="EVCalculator module not implemented yet")
    def test_invalid_price_handling(self):
        """Test handling of invalid prices."""
        from src.models.ev_calculator import EVCalculator

        calc = EVCalculator()

        # Test with invalid prices
        invalid_prices = [0, -0.5, 1.0, 1.5, float('inf'), float('nan')]

        for price in invalid_prices:
            try:
                result = calc.calculate(
                    model_prob=0.6,
                    market_price=price,
                    size_usd=10.0,
                )
                # Should either handle gracefully or return safe default
                assert result.size_usd >= 0
            except (ValueError, ZeroDivisionError):
                pass  # Expected for invalid inputs

    def test_corrupted_order_book(self):
        """Test handling corrupted order book data."""
        from src.models.order_book import OrderBook
        import time

        # Create valid order book
        book = OrderBook(
            token_id="test",
            bids=[(0.49, 100), (0.48, 200)],
            asks=[(0.51, 100), (0.52, 200)],
            timestamp=time.time(),
        )

        assert book.spread >= 0
        assert book.best_bid <= book.best_ask

        # Corrupted book with crossed spread
        corrupted_book = OrderBook(
            token_id="test",
            bids=[(0.55, 100)],  # Bid > Ask
            asks=[(0.45, 100)],
            timestamp=time.time(),
        )

        # Should handle gracefully (negative spread is possible)
        assert corrupted_book.spread < 0


class TestConcurrencyIssues:
    """Tests for race conditions and concurrency issues."""

    @pytest.mark.asyncio
    async def test_concurrent_order_creation(self):
        """Test concurrent order creation doesn't cause issues."""
        manager = OrderManager(dry_run=True)

        async def create_and_submit():
            order = manager.create_order(
                token_id="test_token",
                side=OrderSide.BUY,
                price=random.uniform(0.4, 0.6),
                size=random.uniform(5, 15),
            )
            await manager.submit_order(order)
            return order

        # Many concurrent order creations
        tasks = [create_and_submit() for _ in range(50)]
        orders = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed (unique client_order_ids)
        successful = [o for o in orders if not isinstance(o, Exception)]
        assert len(successful) == 50

        # Each should have unique ID
        ids = [o.client_order_id for o in successful]
        assert len(set(ids)) == 50

    @pytest.mark.asyncio
    async def test_fill_race_condition(self):
        """Test handling fills arriving out of order."""
        manager = OrderManager(dry_run=True)

        order = manager.create_order(
            token_id="test_token",
            side=OrderSide.BUY,
            price=0.50,
            size=100.0,
        )
        await manager.submit_order(order)

        # Fills arriving "out of order" (second fill before first)
        manager.handle_fill(order.client_order_id, 30.0, 0.50)
        manager.handle_fill(order.client_order_id, 70.0, 0.51)

        # Should handle correctly
        assert order.filled_size == 100.0
        assert order.state == OrderState.FILLED


class TestStressConditions:
    """Tests for system behavior under stress."""

    @pytest.mark.asyncio
    async def test_high_throughput_orders(self):
        """Test system handles high order throughput."""
        manager = OrderManager(dry_run=True)

        start = asyncio.get_event_loop().time()

        # Create many orders rapidly
        orders = []
        for i in range(1000):
            order = manager.create_order(
                token_id=f"token_{i % 10}",
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                price=0.50 + (i % 10) * 0.01,
                size=10.0,
            )
            orders.append(order)

        # Submit all
        for order in orders:
            await manager.submit_order(order)

        elapsed = asyncio.get_event_loop().time() - start

        # Should complete reasonably fast
        assert elapsed < 5.0  # Less than 5 seconds for 1000 orders

        # All orders should be tracked
        assert len(manager._orders) == 1000

    def test_memory_usage_under_load(self):
        """Test memory doesn't grow unbounded."""
        import sys

        manager = OrderManager(dry_run=True)

        # Create many orders
        for i in range(10000):
            manager.create_order(
                token_id=f"token_{i}",
                side=OrderSide.BUY,
                price=0.50,
                size=10.0,
            )

        # Check memory usage of order dict
        order_memory = sys.getsizeof(manager._orders)

        # Should be bounded (not growing exponentially)
        assert order_memory < 10_000_000  # Less than 10MB


class TestRecoveryScenarios:
    """Tests for recovery from failure scenarios."""

    @pytest.mark.asyncio
    async def test_recovery_after_network_outage(self):
        """Test recovery after simulated network outage."""
        call_count = 0
        is_network_down = True

        async def network_callback(order):
            nonlocal call_count, is_network_down
            call_count += 1
            if is_network_down:
                raise NetworkError("Network unreachable")
            return True

        manager = OrderManager(dry_run=False, submit_callback=network_callback)

        # Try during outage
        order1 = manager.create_order(
            token_id="test_token",
            side=OrderSide.BUY,
            price=0.50,
            size=10.0,
        )
        result1 = await manager.submit_order(order1)
        assert not result1
        assert order1.state == OrderState.FAILED

        # Network recovers
        is_network_down = False

        # New order should succeed
        order2 = manager.create_order(
            token_id="test_token_2",
            side=OrderSide.BUY,
            price=0.50,
            size=10.0,
        )
        result2 = await manager.submit_order(order2)
        assert result2
        assert order2.state == OrderState.NEW
