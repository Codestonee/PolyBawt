"""
WebSocket Health Monitor for PolyBawt.

Monitors WebSocket connection health with:
- Heartbeat tracking
- Reconnection logic with exponential backoff
- Staleness detection
- Circuit breaker integration

Usage:
    monitor = WebSocketHealthMonitor("clob_ws")
    await monitor.on_message_received()
    if monitor.is_stale:
        await reconnect()
"""

from __future__ import annotations
import asyncio
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Any, Awaitable
from enum import Enum

from src.infrastructure.logging import get_logger
from src.infrastructure.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

logger = get_logger(__name__)


class ConnectionState(Enum):
    """WebSocket connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


@dataclass
class HealthConfig:
    """Configuration for WebSocket health monitoring."""
    heartbeat_interval_seconds: float = 30.0
    stale_threshold_seconds: float = 60.0
    max_reconnect_attempts: int = 10
    initial_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 60.0
    backoff_multiplier: float = 2.0


@dataclass
class HealthStats:
    """Statistics for WebSocket health."""
    messages_received: int = 0
    reconnect_attempts: int = 0
    successful_reconnects: int = 0
    failed_reconnects: int = 0
    total_downtime_seconds: float = 0.0
    last_message_time: float = 0.0
    connection_start_time: float = 0.0


class WebSocketHealthMonitor:
    """
    Health monitor for WebSocket connections.
    
    Tracks message receipt, detects staleness, and manages
    reconnection with exponential backoff.
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[HealthConfig] = None,
        on_reconnect: Optional[Callable[[], Awaitable[bool]]] = None,
        on_stale: Optional[Callable[[], Awaitable[None]]] = None,
    ):
        self.name = name
        self.config = config or HealthConfig()
        self.on_reconnect = on_reconnect
        self.on_stale = on_stale
        
        self._state = ConnectionState.DISCONNECTED
        self._last_message_ns = 0
        self._last_heartbeat_ns = 0
        self._reconnect_count = 0
        self._current_backoff = self.config.initial_backoff_seconds
        self._disconnect_time = 0.0
        
        self.stats = HealthStats()
        self._circuit_breaker = CircuitBreaker(
            f"ws_{name}",
            CircuitBreakerConfig(
                failure_threshold=3,
                timeout_seconds=30.0,
            )
        )
        
        self._monitor_task: Optional[asyncio.Task] = None
    
    @property
    def state(self) -> ConnectionState:
        return self._state
    
    @property
    def is_connected(self) -> bool:
        return self._state == ConnectionState.CONNECTED
    
    @property
    def is_stale(self) -> bool:
        """Check if connection is stale (no recent messages)."""
        if self._state != ConnectionState.CONNECTED:
            return True
        
        age_seconds = (time.time_ns() - self._last_message_ns) / 1e9
        return age_seconds > self.config.stale_threshold_seconds
    
    @property
    def age_seconds(self) -> float:
        """Seconds since last message."""
        if self._last_message_ns == 0:
            return float('inf')
        return (time.time_ns() - self._last_message_ns) / 1e9
    
    async def mark_connected(self):
        """Mark connection as established."""
        if self._disconnect_time > 0:
            downtime = time.time() - self._disconnect_time
            self.stats.total_downtime_seconds += downtime
        
        self._state = ConnectionState.CONNECTED
        self._last_message_ns = time.time_ns()
        self._reconnect_count = 0
        self._current_backoff = self.config.initial_backoff_seconds
        self.stats.connection_start_time = time.time()
        
        logger.info(f"[WSHealth:{self.name}] Connected")
    
    async def mark_disconnected(self):
        """Mark connection as lost."""
        self._state = ConnectionState.DISCONNECTED
        self._disconnect_time = time.time()
        
        logger.warning(f"[WSHealth:{self.name}] Disconnected")
    
    async def on_message_received(self):
        """Record message receipt (call on every message)."""
        self._last_message_ns = time.time_ns()
        self.stats.messages_received += 1
        self.stats.last_message_time = time.time()
    
    async def on_heartbeat_received(self):
        """Record heartbeat receipt."""
        self._last_heartbeat_ns = time.time_ns()
        await self.on_message_received()
    
    async def attempt_reconnect(self) -> bool:
        """
        Attempt to reconnect with exponential backoff.
        
        Returns:
            True if reconnection succeeded
        """
        if not self.on_reconnect:
            logger.error(f"[WSHealth:{self.name}] No reconnect handler configured")
            return False
        
        if self._reconnect_count >= self.config.max_reconnect_attempts:
            self._state = ConnectionState.FAILED
            self.stats.failed_reconnects += 1
            await self._circuit_breaker.force_open()
            logger.error(f"[WSHealth:{self.name}] Max reconnect attempts reached")
            return False
        
        self._state = ConnectionState.RECONNECTING
        self._reconnect_count += 1
        self.stats.reconnect_attempts += 1
        
        logger.info(
            f"[WSHealth:{self.name}] Reconnect attempt {self._reconnect_count} "
            f"(backoff: {self._current_backoff:.1f}s)"
        )
        
        # Wait with backoff
        await asyncio.sleep(self._current_backoff)
        
        try:
            async with self._circuit_breaker:
                success = await self.on_reconnect()
            
            if success:
                await self.mark_connected()
                self.stats.successful_reconnects += 1
                return True
            else:
                self._current_backoff = min(
                    self._current_backoff * self.config.backoff_multiplier,
                    self.config.max_backoff_seconds,
                )
                return False
                
        except Exception as e:
            logger.error(f"[WSHealth:{self.name}] Reconnect failed: {e}")
            self._current_backoff = min(
                self._current_backoff * self.config.backoff_multiplier,
                self.config.max_backoff_seconds,
            )
            return False
    
    async def _monitor_loop(self):
        """Background monitoring loop."""
        while True:
            await asyncio.sleep(self.config.heartbeat_interval_seconds / 2)
            
            if self._state == ConnectionState.CONNECTED and self.is_stale:
                logger.warning(
                    f"[WSHealth:{self.name}] Connection stale "
                    f"(age: {self.age_seconds:.1f}s)"
                )
                
                if self.on_stale:
                    await self.on_stale()
                
                await self.mark_disconnected()
                await self.attempt_reconnect()
    
    async def start_monitoring(self):
        """Start the background health monitor."""
        if self._monitor_task is not None:
            return
        
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info(f"[WSHealth:{self.name}] Started monitoring")
    
    async def stop_monitoring(self):
        """Stop the background health monitor."""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        
        logger.info(f"[WSHealth:{self.name}] Stopped monitoring")
    
    def get_stats(self) -> dict:
        """Get health statistics."""
        return {
            "name": self.name,
            "state": self._state.value,
            "is_stale": self.is_stale,
            "age_seconds": round(self.age_seconds, 2),
            "messages_received": self.stats.messages_received,
            "reconnect_attempts": self.stats.reconnect_attempts,
            "successful_reconnects": self.stats.successful_reconnects,
            "failed_reconnects": self.stats.failed_reconnects,
            "total_downtime_seconds": round(self.stats.total_downtime_seconds, 2),
            "circuit_breaker": self._circuit_breaker.get_stats(),
        }
