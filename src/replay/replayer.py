"""
Deterministic Replayer for PolyBawt.

Replays NDJSON event logs through the system for validation:
- Deterministic event ordering
- Configurable speed (real-time, accelerated, AFAP)
- Latency and jitter injection
- Output comparison for reproducibility testing

Usage:
    replayer = Replayer(event_logger_path="logs/events")
    results = await replayer.replay_date("2026-02-08", speed=float('inf'))
"""

from __future__ import annotations
import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, AsyncIterator
from enum import Enum

from src.replay.event_logger import EventReader, Event, EventType
from src.replay.exchange_sim import ExchangeSimulator, OrderSide
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ReplayConfig:
    """Configuration for replay session."""
    speed: float = 1.0  # 1.0 = real-time, inf = AFAP
    ack_delay_ms: float = 50.0  # Simulated ack delay
    wss_jitter_ms: float = 10.0  # WebSocket jitter
    snapshot_gap_prob: float = 0.0  # Probability of missing a snapshot
    start_ts_ns: Optional[int] = None  # Skip events before this
    end_ts_ns: Optional[int] = None  # Stop at this timestamp


@dataclass
class ReplayStats:
    """Statistics from a replay session."""
    events_processed: int = 0
    orders_placed: int = 0
    fills_generated: int = 0
    cancels_processed: int = 0
    total_pnl: float = 0.0
    total_fees: float = 0.0
    total_rebates: float = 0.0
    duration_seconds: float = 0.0
    
    # By event type
    events_by_type: Dict[str, int] = field(default_factory=dict)
    
    def summary(self) -> str:
        return (
            f"Replayed {self.events_processed} events in {self.duration_seconds:.2f}s: "
            f"{self.orders_placed} orders, {self.fills_generated} fills, "
            f"PnL=${self.total_pnl:.2f} (fees=${self.total_fees:.2f}, rebates=${self.total_rebates:.2f})"
        )


class Replayer:
    """
    Deterministic event replayer for backtesting and validation.
    
    Reads events from NDJSON logs and replays them through the system,
    optionally using an exchange simulator for fill generation.
    """
    
    def __init__(
        self,
        event_log_dir: str,
        exchange_sim: Optional[ExchangeSimulator] = None,
    ):
        """
        Args:
            event_log_dir: Directory containing NDJSON event logs
            exchange_sim: Optional exchange simulator for order matching
        """
        self.reader = EventReader(event_log_dir)
        self.exchange_sim = exchange_sim or ExchangeSimulator()
        
        self._handlers: Dict[EventType, List[Callable]] = {}
        self._stats = ReplayStats()
    
    def on_event(self, event_type: EventType, handler: Callable[[Event], Any]):
        """Register a handler for an event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
    
    async def replay_date(
        self,
        date: str,
        config: Optional[ReplayConfig] = None,
    ) -> ReplayStats:
        """
        Replay all events for a specific date.
        
        Args:
            date: Date string in YYYY-MM-DD format
            config: Replay configuration (defaults to AFAP)
            
        Returns:
            ReplayStats with results
        """
        config = config or ReplayConfig(speed=float('inf'))
        self._stats = ReplayStats()
        start_time = time.time()
        
        prev_ts_ns = None
        
        for event in self.reader.read_date(date):
            # Skip events outside range
            if config.start_ts_ns and event.ts_recv_ns < config.start_ts_ns:
                continue
            if config.end_ts_ns and event.ts_recv_ns > config.end_ts_ns:
                break
            
            # Real-time pacing
            if config.speed != float('inf') and prev_ts_ns:
                delay_ns = (event.ts_recv_ns - prev_ts_ns) / config.speed
                await asyncio.sleep(delay_ns / 1e9)
            prev_ts_ns = event.ts_recv_ns
            
            # Process event
            await self._process_event(event, config)
            
            self._stats.events_processed += 1
            self._stats.events_by_type[event.event_type.value] = (
                self._stats.events_by_type.get(event.event_type.value, 0) + 1
            )
        
        self._stats.duration_seconds = time.time() - start_time
        
        logger.info(f"[Replayer] {self._stats.summary()}")
        return self._stats
    
    async def replay_range(
        self,
        start_date: str,
        end_date: str,
        config: Optional[ReplayConfig] = None,
    ) -> ReplayStats:
        """Replay events across a date range."""
        config = config or ReplayConfig(speed=float('inf'))
        total_stats = ReplayStats()
        start_time = time.time()
        
        prev_ts_ns = None
        
        for event in self.reader.read_range(start_date, end_date):
            if config.start_ts_ns and event.ts_recv_ns < config.start_ts_ns:
                continue
            if config.end_ts_ns and event.ts_recv_ns > config.end_ts_ns:
                break
            
            if config.speed != float('inf') and prev_ts_ns:
                delay_ns = (event.ts_recv_ns - prev_ts_ns) / config.speed
                await asyncio.sleep(delay_ns / 1e9)
            prev_ts_ns = event.ts_recv_ns
            
            await self._process_event(event, config)
            
            total_stats.events_processed += 1
        
        total_stats.duration_seconds = time.time() - start_time
        return total_stats
    
    async def _process_event(self, event: Event, config: ReplayConfig):
        """Process a single event."""
        # Call registered handlers
        handlers = self._handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"[Replayer] Handler error: {e}")
        
        # Built-in processing for key events
        if event.event_type == EventType.ORDER_SUBMIT:
            await self._handle_order_submit(event, config)
        elif event.event_type == EventType.CANCEL_SUBMIT:
            await self._handle_cancel_submit(event, config)
        elif event.event_type == EventType.FILL:
            self._stats.fills_generated += 1
            self._stats.total_fees += event.data.get("fee_paid", 0)
            self._stats.total_rebates += event.data.get("rebate_est", 0)
    
    async def _handle_order_submit(self, event: Event, config: ReplayConfig):
        """Handle order submission through simulator."""
        data = event.data
        
        # Simulate ack delay
        if config.ack_delay_ms > 0:
            jitter = (config.wss_jitter_ms * (hash(event.ts_recv_ns) % 1000) / 1000)
            await asyncio.sleep((config.ack_delay_ms + jitter) / 1000)
        
        order, fills = self.exchange_sim.place_order(
            order_id=data.get("client_order_id", str(event.ts_recv_ns)),
            token_id=event.token_id or "unknown",
            side=data.get("side", "BUY"),
            price=data.get("price", 0.5),
            size=data.get("size", 1.0),
            order_type=data.get("order_type", "LIMIT"),
        )
        
        self._stats.orders_placed += 1
        self._stats.fills_generated += len(fills)
        
        for fill in fills:
            self._stats.total_fees += fill.fee_paid
            self._stats.total_rebates += fill.rebate_est
    
    async def _handle_cancel_submit(self, event: Event, config: ReplayConfig):
        """Handle cancel through simulator."""
        data = event.data
        
        if config.ack_delay_ms > 0:
            await asyncio.sleep(config.ack_delay_ms / 1000)
        
        self.exchange_sim.cancel_order(
            token_id=event.token_id or "unknown",
            order_id=data.get("order_id", ""),
        )
        self._stats.cancels_processed += 1
    
    def get_stats(self) -> ReplayStats:
        """Get current replay statistics."""
        return self._stats
    
    def reset(self):
        """Reset replayer state."""
        self._stats = ReplayStats()
        self.exchange_sim.reset()


async def run_replay_test(
    event_log_dir: str,
    date: str,
    speed: float = float('inf'),
) -> ReplayStats:
    """
    Convenience function to run a quick replay test.
    
    Usage:
        stats = await run_replay_test("logs/events", "2026-02-08")
    """
    replayer = Replayer(event_log_dir)
    config = ReplayConfig(speed=speed)
    return await replayer.replay_date(date, config)
