"""
Token bucket rate limiter for Polymarket API.

Implements rate limiting to stay under API limits:
- POST /order: 500 burst per 10s, 3000 per 10min
- Other endpoints: Similar constraints

Uses token bucket algorithm for smooth rate limiting.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limit configuration for an endpoint."""
    
    name: str
    tokens_per_second: float  # Refill rate
    bucket_size: int  # Max burst capacity
    
    @classmethod
    def for_orders(cls) -> "RateLimitConfig":
        """
        Rate limit for POST /order.
        
        Polymarket limit: 500 per 10s = 50/s, with bursting.
        We use 90% of this to be safe.
        """
        return cls(
            name="orders",
            tokens_per_second=45,  # 90% of 50/s
            bucket_size=450,  # 90% of 500
        )
    
    @classmethod
    def for_websocket(cls) -> "RateLimitConfig":
        """Rate limit for WebSocket messages."""
        return cls(
            name="websocket",
            tokens_per_second=10,
            bucket_size=50,
        )
    
    @classmethod
    def for_queries(cls) -> "RateLimitConfig":
        """Rate limit for read-only queries."""
        return cls(
            name="queries",
            tokens_per_second=20,
            bucket_size=100,
        )


class TokenBucket:
    """
    Token bucket rate limiter.
    
    Tokens are added at a constant rate up to a maximum bucket size.
    Requests consume tokens. If no tokens available, request must wait.
    
    Usage:
        bucket = TokenBucket(tokens_per_second=50, bucket_size=500)
        
        # Non-blocking check
        if bucket.try_acquire():
            await make_request()
        
        # Blocking wait
        await bucket.acquire()
        await make_request()
    """
    
    def __init__(
        self,
        tokens_per_second: float,
        bucket_size: int,
    ):
        self.tokens_per_second = tokens_per_second
        self.bucket_size = bucket_size
        
        self._tokens = float(bucket_size)  # Start full
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()
        
        # Metrics
        self._total_acquired = 0
        self._total_waited = 0
        self._total_wait_time = 0.0
    
    def _refill(self) -> None:
        """Add tokens based on time elapsed."""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._last_update = now
        
        # Add tokens
        tokens_to_add = elapsed * self.tokens_per_second
        self._tokens = min(self.bucket_size, self._tokens + tokens_to_add)
    
    @property
    def available_tokens(self) -> float:
        """Current token count (approximate)."""
        elapsed = time.monotonic() - self._last_update
        tokens_added = elapsed * self.tokens_per_second
        return min(self.bucket_size, self._tokens + tokens_added)
    
    @property
    def usage_pct(self) -> float:
        """Current usage as percentage of capacity."""
        return 1.0 - (self.available_tokens / self.bucket_size)
    
    def try_acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens without waiting.
        
        Args:
            tokens: Number of tokens to acquire
        
        Returns:
            True if tokens acquired, False if would need to wait
        """
        self._refill()
        
        if self._tokens >= tokens:
            self._tokens -= tokens
            self._total_acquired += tokens
            return True
        
        return False
    
    async def acquire(self, tokens: int = 1) -> float:
        """
        Acquire tokens, waiting if necessary.
        
        Args:
            tokens: Number of tokens to acquire
        
        Returns:
            Time waited in seconds
        """
        async with self._lock:
            self._refill()
            
            if self._tokens >= tokens:
                self._tokens -= tokens
                self._total_acquired += tokens
                return 0.0
            
            # Calculate wait time
            tokens_needed = tokens - self._tokens
            wait_time = tokens_needed / self.tokens_per_second
            
            # Wait
            self._total_waited += 1
            self._total_wait_time += wait_time
            
            logger.debug(
                "Rate limit wait",
                tokens_needed=tokens_needed,
                wait_seconds=wait_time,
            )
            
            await asyncio.sleep(wait_time)
            
            # Take tokens after wait
            self._refill()
            self._tokens -= tokens
            self._total_acquired += tokens
            
            return wait_time
    
    def time_until_available(self, tokens: int = 1) -> float:
        """Calculate time until tokens will be available."""
        self._refill()
        
        if self._tokens >= tokens:
            return 0.0
        
        tokens_needed = tokens - self._tokens
        return tokens_needed / self.tokens_per_second
    
    @property
    def stats(self) -> dict[str, Any]:
        """Get rate limiter statistics."""
        return {
            "available_tokens": round(self.available_tokens, 1),
            "bucket_size": self.bucket_size,
            "usage_pct": round(self.usage_pct * 100, 1),
            "total_acquired": self._total_acquired,
            "total_waited": self._total_waited,
            "avg_wait_time": (
                self._total_wait_time / self._total_waited 
                if self._total_waited > 0 else 0
            ),
        }


class RateLimiter:
    """
    Multi-bucket rate limiter for different endpoint types.
    
    Usage:
        limiter = RateLimiter()
        
        # Before making an order request
        await limiter.acquire_order()
        
        # Check if we're near limits
        if limiter.is_critical():
            pause_trading()
    """
    
    def __init__(
        self,
        order_config: RateLimitConfig | None = None,
        query_config: RateLimitConfig | None = None,
        ws_config: RateLimitConfig | None = None,
    ):
        self._buckets: dict[str, TokenBucket] = {}
        
        # Initialize buckets
        order_cfg = order_config or RateLimitConfig.for_orders()
        query_cfg = query_config or RateLimitConfig.for_queries()
        ws_cfg = ws_config or RateLimitConfig.for_websocket()
        
        self._buckets["orders"] = TokenBucket(
            tokens_per_second=order_cfg.tokens_per_second,
            bucket_size=order_cfg.bucket_size,
        )
        self._buckets["queries"] = TokenBucket(
            tokens_per_second=query_cfg.tokens_per_second,
            bucket_size=query_cfg.bucket_size,
        )
        self._buckets["websocket"] = TokenBucket(
            tokens_per_second=ws_cfg.tokens_per_second,
            bucket_size=ws_cfg.bucket_size,
        )
    
    async def acquire_order(self, count: int = 1) -> float:
        """Acquire tokens for order operations."""
        return await self._buckets["orders"].acquire(count)
    
    async def acquire_query(self, count: int = 1) -> float:
        """Acquire tokens for query operations."""
        return await self._buckets["queries"].acquire(count)
    
    async def acquire_ws(self, count: int = 1) -> float:
        """Acquire tokens for WebSocket operations."""
        return await self._buckets["websocket"].acquire(count)
    
    def try_acquire_order(self, count: int = 1) -> bool:
        """Try to acquire order tokens without waiting."""
        return self._buckets["orders"].try_acquire(count)
    
    def order_usage_pct(self) -> float:
        """Get order rate limit usage percentage."""
        return self._buckets["orders"].usage_pct
    
    def is_critical(self, threshold: float = 0.90) -> bool:
        """Check if any bucket is critically depleted."""
        return any(
            bucket.usage_pct > threshold 
            for bucket in self._buckets.values()
        )
    
    def stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all buckets."""
        return {
            name: bucket.stats 
            for name, bucket in self._buckets.items()
        }


# Pre-instantiated rate limiter
rate_limiter = RateLimiter()
