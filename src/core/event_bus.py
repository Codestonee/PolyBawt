"""
Priority Event Queue for PolyBawt.

Implements backpressure-aware event dispatching with priority levels:
- P0: Critical (fills, cancels, kill-switch) - always processed immediately
- P1: High (order acks, active book updates) - processed with minimal delay
- P2: Background (snapshots, diagnostics) - can be delayed under load

Features:
- Priority queue with P0 > P1 > P2 ordering
- Backpressure detection (queue depth monitoring)
- Coalescing of book deltas by token_id under load
- Async-safe processing
"""

from __future__ import annotations
import asyncio
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, Optional, List, Awaitable
from collections import defaultdict
import heapq

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class EventPriority(IntEnum):
    """Event priority levels (lower = higher priority)."""
    P0_CRITICAL = 0   # fills, cancels, kill-switch
    P1_HIGH = 1       # order acks, active book updates
    P2_BACKGROUND = 2  # snapshots, diagnostics


@dataclass(order=True)
class PrioritizedEvent:
    """Event wrapper with priority ordering."""
    priority: int
    timestamp: float = field(compare=True)
    event_type: str = field(compare=False)
    payload: Any = field(compare=False)
    token_id: Optional[str] = field(compare=False, default=None)
    _counter: int = field(compare=True, default=0)  # Tie-breaker


class BackpressureState:
    """Tracks backpressure conditions."""
    
    def __init__(
        self,
        warning_threshold: int = 100,
        critical_threshold: int = 500,
    ):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.current_depth = 0
        self.total_processed = 0
        self.total_dropped = 0
        self.last_warning_time = 0.0
    
    @property
    def is_warning(self) -> bool:
        return self.current_depth >= self.warning_threshold
    
    @property
    def is_critical(self) -> bool:
        return self.current_depth >= self.critical_threshold
    
    def update_depth(self, depth: int):
        self.current_depth = depth
        now = time.time()
        
        if self.is_critical and now - self.last_warning_time > 5.0:
            logger.warning(
                f"[EventBus] CRITICAL backpressure: depth={depth}, "
                f"dropped={self.total_dropped}"
            )
            self.last_warning_time = now
        elif self.is_warning and now - self.last_warning_time > 10.0:
            logger.info(
                f"[EventBus] Backpressure warning: depth={depth}"
            )
            self.last_warning_time = now


class PriorityEventBus:
    """
    Priority-based async event bus with backpressure handling.
    
    Usage:
        bus = PriorityEventBus()
        bus.subscribe("fill", handle_fill)
        await bus.publish("fill", fill_data, priority=EventPriority.P0_CRITICAL)
        await bus.start()  # Starts processing loop
    """
    
    def __init__(
        self,
        max_queue_size: int = 1000,
        coalesce_book_deltas: bool = True,
    ):
        self._queue: List[PrioritizedEvent] = []
        self._counter = 0
        self._handlers: Dict[str, List[Callable]] = defaultdict(list)
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        
        self.max_queue_size = max_queue_size
        self.coalesce_book_deltas = coalesce_book_deltas
        self.backpressure = BackpressureState()
        
        # For coalescing: track latest book delta per token
        self._pending_book_deltas: Dict[str, PrioritizedEvent] = {}
    
    def subscribe(
        self,
        event_type: str,
        handler: Callable[[Any], Awaitable[None]],
    ):
        """Subscribe a handler to an event type."""
        self._handlers[event_type].append(handler)
        logger.debug(f"[EventBus] Subscribed handler to '{event_type}'")
    
    async def publish(
        self,
        event_type: str,
        payload: Any,
        priority: EventPriority = EventPriority.P1_HIGH,
        token_id: Optional[str] = None,
    ):
        """
        Publish an event to the bus.
        
        P0 events are never dropped.
        P1/P2 events may be coalesced or dropped under backpressure.
        """
        async with self._lock:
            # Check queue capacity
            self.backpressure.update_depth(len(self._queue))
            
            # P0 events always go through
            if priority == EventPriority.P0_CRITICAL:
                self._enqueue(event_type, payload, priority, token_id)
                return
            
            # Under critical backpressure, drop P2 events
            if self.backpressure.is_critical and priority == EventPriority.P2_BACKGROUND:
                self.backpressure.total_dropped += 1
                return
            
            # Coalesce book deltas by token_id
            if (
                self.coalesce_book_deltas
                and event_type == "book_delta"
                and token_id
                and self.backpressure.is_warning
            ):
                # Replace pending delta for this token
                self._pending_book_deltas[token_id] = PrioritizedEvent(
                    priority=priority,
                    timestamp=time.time(),
                    event_type=event_type,
                    payload=payload,
                    token_id=token_id,
                    _counter=self._counter,
                )
                self._counter += 1
                return
            
            # Normal enqueue
            if len(self._queue) < self.max_queue_size:
                self._enqueue(event_type, payload, priority, token_id)
            else:
                self.backpressure.total_dropped += 1
    
    def _enqueue(
        self,
        event_type: str,
        payload: Any,
        priority: EventPriority,
        token_id: Optional[str],
    ):
        """Add event to priority queue."""
        event = PrioritizedEvent(
            priority=priority,
            timestamp=time.time(),
            event_type=event_type,
            payload=payload,
            token_id=token_id,
            _counter=self._counter,
        )
        self._counter += 1
        heapq.heappush(self._queue, event)
    
    async def _flush_coalesced(self):
        """Flush coalesced book deltas to main queue."""
        async with self._lock:
            for token_id, event in self._pending_book_deltas.items():
                heapq.heappush(self._queue, event)
            self._pending_book_deltas.clear()
    
    async def _process_event(self, event: PrioritizedEvent):
        """Process a single event."""
        handlers = self._handlers.get(event.event_type, [])
        
        for handler in handlers:
            try:
                await handler(event.payload)
            except Exception as e:
                logger.error(
                    f"[EventBus] Handler error for '{event.event_type}': {e}"
                )
        
        self.backpressure.total_processed += 1
    
    async def _run_loop(self):
        """Main event processing loop."""
        while self._running:
            # Periodically flush coalesced events
            await self._flush_coalesced()
            
            async with self._lock:
                if not self._queue:
                    pass  # Will sleep below
                else:
                    event = heapq.heappop(self._queue)
            
            if event := (event if 'event' in dir() else None):
                await self._process_event(event)
                event = None
            else:
                # No events, sleep briefly
                await asyncio.sleep(0.001)
    
    async def start(self):
        """Start the event processing loop."""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("[EventBus] Started priority event bus")
    
    async def stop(self):
        """Stop the event processing loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info(
            f"[EventBus] Stopped. Processed={self.backpressure.total_processed}, "
            f"Dropped={self.backpressure.total_dropped}"
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Return queue statistics."""
        return {
            "queue_depth": len(self._queue),
            "total_processed": self.backpressure.total_processed,
            "total_dropped": self.backpressure.total_dropped,
            "is_warning": self.backpressure.is_warning,
            "is_critical": self.backpressure.is_critical,
            "pending_coalesced": len(self._pending_book_deltas),
        }


# Convenience functions for common event types
def p0_event(event_type: str, payload: Any, token_id: Optional[str] = None):
    """Create a P0 (critical) event tuple."""
    return (event_type, payload, EventPriority.P0_CRITICAL, token_id)


def p1_event(event_type: str, payload: Any, token_id: Optional[str] = None):
    """Create a P1 (high) event tuple."""
    return (event_type, payload, EventPriority.P1_HIGH, token_id)


def p2_event(event_type: str, payload: Any, token_id: Optional[str] = None):
    """Create a P2 (background) event tuple."""
    return (event_type, payload, EventPriority.P2_BACKGROUND, token_id)
