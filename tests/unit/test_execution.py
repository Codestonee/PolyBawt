"""Tests for execution layer."""

import asyncio
import pytest

from src.execution.order_manager import (
    OrderManager,
    OrderState,
    OrderType,
    OrderSide,
    Order,
    generate_client_order_id,
)
from src.execution.rate_limiter import TokenBucket, RateLimiter


class TestClientOrderId:
    """Tests for client order ID generation."""
    
    def test_format(self):
        """ID should follow expected format."""
        cid = generate_client_order_id("strategy1", "abc123def456")
        parts = cid.split("_")
        
        assert len(parts) == 4
        assert parts[0] == "strategy1"
        assert parts[1] == "abc123de"  # 8 char prefix
        assert parts[2].isdigit()  # timestamp
        assert len(parts[3]) == 6  # nonce
    
    def test_uniqueness(self):
        """Generated IDs should be unique."""
        ids = [generate_client_order_id("test", "token123") for _ in range(100)]
        assert len(set(ids)) == 100


class TestOrderLifecycle:
    """Tests for order state machine."""
    
    @pytest.fixture
    def manager(self):
        return OrderManager(dry_run=True)
    
    def test_create_order(self, manager):
        """Order should start in CREATING state."""
        order = manager.create_order(
            token_id="test_token",
            side=OrderSide.BUY,
            price=0.55,
            size=10.0,
        )
        
        assert order.state == OrderState.CREATING
        assert order.token_id == "test_token"
        assert order.side == OrderSide.BUY
        assert order.price == 0.55
        assert order.size == 10.0
    
    @pytest.mark.asyncio
    async def test_submit_order_dry_run(self, manager):
        """Dry run should simulate submission."""
        order = manager.create_order(
            token_id="test_token",
            side=OrderSide.BUY,
            price=0.55,
            size=10.0,
        )
        
        success = await manager.submit_order(order)
        
        assert success
        assert order.state == OrderState.NEW
        assert order.exchange_order_id is not None
    
    def test_duplicate_order_id(self, manager):
        """Duplicate client_order_id should return existing order."""
        order1 = manager.create_order(
            token_id="test_token",
            side=OrderSide.BUY,
            price=0.55,
            size=10.0,
            client_order_id="custom_id_123",
        )
        
        order2 = manager.create_order(
            token_id="different_token",
            side=OrderSide.SELL,
            price=0.45,
            size=5.0,
            client_order_id="custom_id_123",  # Same ID
        )
        
        # Should return the existing order
        assert order1 is order2
    
    def test_fill_handling(self, manager):
        """Fills should update order state correctly."""
        order = manager.create_order(
            token_id="test_token",
            side=OrderSide.BUY,
            price=0.55,
            size=10.0,
        )
        order.update_state(OrderState.NEW)
        
        # Partial fill
        manager.handle_fill(order.client_order_id, 5.0, 0.54)
        assert order.state == OrderState.PARTIALLY_FILLED
        assert order.filled_size == 5.0
        assert order.remaining_size == 5.0
        
        # Complete fill
        manager.handle_fill(order.client_order_id, 5.0, 0.55)
        assert order.state == OrderState.FILLED
        assert order.filled_size == 10.0
        assert order.remaining_size == 0.0
    
    def test_average_fill_price(self, manager):
        """Average fill price should be weighted correctly."""
        order = manager.create_order(
            token_id="test_token",
            side=OrderSide.BUY,
            price=0.55,
            size=10.0,
        )
        order.update_state(OrderState.NEW)
        
        # Two fills at different prices
        order.add_fill(6.0, 0.50)  # 6 @ 0.50
        order.add_fill(4.0, 0.60)  # 4 @ 0.60
        
        # Weighted average: (6*0.50 + 4*0.60) / 10 = 5.4/10 = 0.54
        assert abs(order.average_fill_price - 0.54) < 0.001
    
    @pytest.mark.asyncio
    async def test_cancel_order(self, manager):
        """Cancel should work on active orders."""
        order = manager.create_order(
            token_id="test_token",
            side=OrderSide.BUY,
            price=0.55,
            size=10.0,
        )
        await manager.submit_order(order)
        
        success = await manager.cancel_order(order)
        
        assert success
        assert order.state == OrderState.CANCELED
    
    def test_stats(self, manager):
        """Stats should track orders correctly."""
        stats = manager.stats
        assert stats["total_orders"] == 0
        
        manager.create_order(
            token_id="test",
            side=OrderSide.BUY,
            price=0.5,
            size=10,
        )
        
        assert manager.stats["total_orders"] == 1


class TestTokenBucket:
    """Tests for token bucket rate limiter."""
    
    def test_initial_full(self):
        """Bucket should start full."""
        bucket = TokenBucket(tokens_per_second=10, bucket_size=100)
        assert bucket.available_tokens == 100
    
    def test_try_acquire_success(self):
        """Should acquire when tokens available."""
        bucket = TokenBucket(tokens_per_second=10, bucket_size=100)
        assert bucket.try_acquire(10)
        assert bucket.available_tokens == pytest.approx(90, abs=1)
    
    def test_try_acquire_fail(self):
        """Should fail when insufficient tokens."""
        bucket = TokenBucket(tokens_per_second=10, bucket_size=100)
        bucket.try_acquire(100)  # Drain bucket
        assert not bucket.try_acquire(10)
    
    @pytest.mark.asyncio
    async def test_acquire_waits(self):
        """Should wait when tokens unavailable."""
        bucket = TokenBucket(tokens_per_second=100, bucket_size=10)
        bucket.try_acquire(10)  # Drain
        
        # Should wait ~0.1s for 10 tokens at 100/s
        start = asyncio.get_event_loop().time()
        await bucket.acquire(10)
        elapsed = asyncio.get_event_loop().time() - start
        
        assert elapsed >= 0.09  # Allow some tolerance
    
    def test_usage_pct(self):
        """Usage should be calculated correctly."""
        bucket = TokenBucket(tokens_per_second=10, bucket_size=100)
        assert bucket.usage_pct == pytest.approx(0.0, abs=0.01)
        
        bucket.try_acquire(50)
        assert bucket.usage_pct == pytest.approx(0.5, abs=0.01)


class TestOrderSizeConversion:
    """Tests for USD to shares conversion."""

    def test_usd_to_shares_conversion(self):
        """
        Verify that size conversion from USD to shares is correct.

        For Polymarket binary options (prices 0-1):
        shares = usd_amount / price

        Examples:
        - $10 at price 0.5 = 10 / 0.5 = 20 shares
        - $10 at price 0.2 = 10 / 0.2 = 50 shares
        - $10 at price 0.8 = 10 / 0.8 = 12.5 shares
        """
        # Test case 1: $10 at 0.5 = 20 shares
        usd = 10.0
        price = 0.5
        expected_shares = 20.0
        actual_shares = usd / price
        assert abs(actual_shares - expected_shares) < 0.01

        # Test case 2: $10 at 0.2 = 50 shares
        usd = 10.0
        price = 0.2
        expected_shares = 50.0
        actual_shares = usd / price
        assert abs(actual_shares - expected_shares) < 0.01

        # Test case 3: $10 at 0.8 = 12.5 shares
        usd = 10.0
        price = 0.8
        expected_shares = 12.5
        actual_shares = usd / price
        assert abs(actual_shares - expected_shares) < 0.01

        # Test case 4: $100 at 0.55 = ~181.82 shares
        usd = 100.0
        price = 0.55
        expected_shares = 181.82
        actual_shares = usd / price
        assert abs(actual_shares - expected_shares) < 0.1


class TestRateLimiter:
    """Tests for multi-bucket rate limiter."""

    def test_separate_buckets(self):
        """Different endpoints should have separate limits."""
        limiter = RateLimiter()

        # Drain order bucket
        for _ in range(450):
            limiter.try_acquire_order()

        # Order bucket should be nearly empty
        assert limiter.order_usage_pct() > 0.9

        # But query bucket should still be full
        query_bucket = limiter._buckets["queries"]
        assert query_bucket.usage_pct < 0.1

    def test_is_critical(self):
        """Should detect critical rate limit state."""
        limiter = RateLimiter()
        assert not limiter.is_critical()

        # Drain a bucket
        for _ in range(460):
            limiter.try_acquire_order()

        assert limiter.is_critical(threshold=0.9)
