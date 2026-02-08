"""
Circuit Breaker for PolyBawt.

Implements the circuit breaker pattern with:
- CLOSED: Normal operation
- OPEN: Failing, reject all requests
- HALF_OPEN: Testing recovery with limited requests

Usage:
    breaker = CircuitBreaker("clob_api")
    async with breaker:
        await make_api_call()
"""

from __future__ import annotations
import asyncio
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Any, Dict
from enum import Enum
from functools import wraps

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5        # Failures before opening
    success_threshold: int = 3        # Successes to close from half-open
    timeout_seconds: float = 30.0     # Time before half-open attempt
    half_open_max_calls: int = 3      # Max concurrent in half-open


@dataclass
class CircuitStats:
    """Statistics for a circuit breaker."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    state_changes: int = 0
    
    @property
    def failure_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.failed_calls / self.total_calls


class CircuitOpenError(Exception):
    """Raised when circuit is open and rejecting requests."""
    pass


class CircuitBreaker:
    """
    Circuit breaker for resilient API calls.
    
    Prevents cascade failures by stopping calls to failing services.
    Automatically recovers when service becomes healthy.
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        on_state_change: Optional[Callable[[CircuitState, CircuitState], None]] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.on_state_change = on_state_change
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        self._lock = asyncio.Lock()
        
        self.stats = CircuitStats()
        
        logger.info(f"[CircuitBreaker:{name}] Initialized in CLOSED state")
    
    @property
    def state(self) -> CircuitState:
        return self._state
    
    @property
    def is_closed(self) -> bool:
        return self._state == CircuitState.CLOSED
    
    @property
    def is_open(self) -> bool:
        return self._state == CircuitState.OPEN
    
    @property
    def is_half_open(self) -> bool:
        return self._state == CircuitState.HALF_OPEN
    
    async def _transition_to(self, new_state: CircuitState):
        """Transition to a new state."""
        if self._state == new_state:
            return
        
        old_state = self._state
        self._state = new_state
        self.stats.state_changes += 1
        
        # Reset counters based on new state
        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._success_count = 0
        
        logger.warning(
            f"[CircuitBreaker:{self.name}] State change: {old_state.value} → {new_state.value}"
        )
        
        if self.on_state_change:
            try:
                self.on_state_change(old_state, new_state)
            except Exception as e:
                logger.error(f"[CircuitBreaker:{self.name}] Callback error: {e}")
    
    async def _check_open_timeout(self):
        """Check if timeout has passed for half-open transition."""
        if self._state != CircuitState.OPEN:
            return
        
        elapsed = time.time() - self._last_failure_time
        if elapsed >= self.config.timeout_seconds:
            await self._transition_to(CircuitState.HALF_OPEN)
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call a function through the circuit breaker.
        
        Raises:
            CircuitOpenError: If circuit is open and rejecting calls
        """
        async with self._lock:
            await self._check_open_timeout()
            
            # Reject if open
            if self._state == CircuitState.OPEN:
                self.stats.rejected_calls += 1
                raise CircuitOpenError(
                    f"Circuit {self.name} is OPEN, rejecting call"
                )
            
            # Limit calls in half-open
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    self.stats.rejected_calls += 1
                    raise CircuitOpenError(
                        f"Circuit {self.name} is HALF_OPEN, max calls reached"
                    )
                self._half_open_calls += 1
        
        # Execute the call
        self.stats.total_calls += 1
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            await self._record_success()
            return result
            
        except Exception as e:
            await self._record_failure(e)
            raise
    
    async def _record_success(self):
        """Record a successful call."""
        async with self._lock:
            self.stats.successful_calls += 1
            self.stats.last_success_time = time.time()
            self._success_count += 1
            
            # In half-open, enough successes → close
            if self._state == CircuitState.HALF_OPEN:
                if self._success_count >= self.config.success_threshold:
                    await self._transition_to(CircuitState.CLOSED)
    
    async def _record_failure(self, error: Exception):
        """Record a failed call."""
        async with self._lock:
            self.stats.failed_calls += 1
            self.stats.last_failure_time = time.time()
            self._last_failure_time = time.time()
            self._failure_count += 1
            
            logger.warning(
                f"[CircuitBreaker:{self.name}] Failure #{self._failure_count}: {error}"
            )
            
            # In half-open, any failure → open
            if self._state == CircuitState.HALF_OPEN:
                await self._transition_to(CircuitState.OPEN)
                return
            
            # In closed, too many failures → open
            if self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    await self._transition_to(CircuitState.OPEN)
    
    async def force_open(self):
        """Force the circuit to open (emergency stop)."""
        async with self._lock:
            await self._transition_to(CircuitState.OPEN)
            self._last_failure_time = time.time()
    
    async def force_close(self):
        """Force the circuit to close (manual recovery)."""
        async with self._lock:
            await self._transition_to(CircuitState.CLOSED)
    
    async def reset(self):
        """Reset the circuit breaker to initial state."""
        async with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
            self.stats = CircuitStats()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "stats": {
                "total_calls": self.stats.total_calls,
                "successful_calls": self.stats.successful_calls,
                "failed_calls": self.stats.failed_calls,
                "rejected_calls": self.stats.rejected_calls,
                "failure_rate": round(self.stats.failure_rate, 3),
                "state_changes": self.stats.state_changes,
            },
        }
    
    async def __aenter__(self):
        """Context manager entry - check if can proceed."""
        async with self._lock:
            await self._check_open_timeout()
            
            if self._state == CircuitState.OPEN:
                raise CircuitOpenError(f"Circuit {self.name} is OPEN")
            
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    raise CircuitOpenError(f"Circuit {self.name} HALF_OPEN max reached")
                self._half_open_calls += 1
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - record result."""
        self.stats.total_calls += 1
        
        if exc_type is None:
            await self._record_success()
        else:
            await self._record_failure(exc_val or Exception("Unknown error"))
        
        return False  # Don't suppress exceptions


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.
    
    Usage:
        registry = CircuitBreakerRegistry()
        clob_breaker = registry.get("clob_api")
        ws_breaker = registry.get("websocket")
    """
    
    def __init__(self, default_config: Optional[CircuitBreakerConfig] = None):
        self.default_config = default_config or CircuitBreakerConfig()
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()
    
    def get(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> CircuitBreaker:
        """Get or create a circuit breaker by name."""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                name,
                config or self.default_config,
            )
        return self._breakers[name]
    
    async def open_all(self):
        """Open all circuit breakers (emergency stop)."""
        for breaker in self._breakers.values():
            await breaker.force_open()
        logger.warning("[CircuitBreakerRegistry] All circuits OPENED")
    
    async def close_all(self):
        """Close all circuit breakers (global reset)."""
        for breaker in self._breakers.values():
            await breaker.force_close()
        logger.info("[CircuitBreakerRegistry] All circuits CLOSED")
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get stats for all circuit breakers."""
        return {
            name: breaker.get_stats()
            for name, breaker in self._breakers.items()
        }
    
    def any_open(self) -> bool:
        """Check if any circuit is open."""
        return any(b.is_open for b in self._breakers.values())


# Global registry
_global_registry = CircuitBreakerRegistry()


def get_circuit_breaker(name: str) -> CircuitBreaker:
    """Get a circuit breaker from the global registry."""
    return _global_registry.get(name)


def circuit_protected(name: str):
    """
    Decorator to wrap a function with circuit breaker protection.
    
    Usage:
        @circuit_protected("clob_api")
        async def place_order(...):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            breaker = get_circuit_breaker(name)
            return await breaker.call(func, *args, **kwargs)
        return wrapper
    return decorator
