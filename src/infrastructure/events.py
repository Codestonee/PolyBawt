"""
Domain Events System for decoupled component communication.

Implements a simple pub/sub event system for:
- Order lifecycle events
- Trade execution events
- Risk alerts
- Market data updates

This enables loose coupling between components and easier testing.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine, TypeVar
from collections import defaultdict

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class EventType(Enum):
    """Types of domain events."""

    # Order events
    ORDER_CREATED = "order.created"
    ORDER_SUBMITTED = "order.submitted"
    ORDER_ACKNOWLEDGED = "order.acknowledged"
    ORDER_PARTIALLY_FILLED = "order.partially_filled"
    ORDER_FILLED = "order.filled"
    ORDER_CANCELED = "order.canceled"
    ORDER_REJECTED = "order.rejected"
    ORDER_EXPIRED = "order.expired"
    ORDER_FAILED = "order.failed"

    # Trade events
    TRADE_SIGNAL = "trade.signal"
    TRADE_EXECUTED = "trade.executed"
    TRADE_WON = "trade.won"
    TRADE_LOST = "trade.lost"

    # Position events
    POSITION_OPENED = "position.opened"
    POSITION_CLOSED = "position.closed"
    POSITION_UPDATED = "position.updated"

    # Risk events
    CIRCUIT_BREAKER_TRIPPED = "risk.circuit_breaker_tripped"
    CIRCUIT_BREAKER_RESET = "risk.circuit_breaker_reset"
    DAILY_LOSS_WARNING = "risk.daily_loss_warning"
    DRAWDOWN_WARNING = "risk.drawdown_warning"
    VOLATILITY_SPIKE = "risk.volatility_spike"

    # Market events
    MARKET_DISCOVERED = "market.discovered"
    MARKET_EXPIRED = "market.expired"
    MARKET_RESOLVED = "market.resolved"
    PRICE_UPDATE = "market.price_update"
    ARBITRAGE_DETECTED = "market.arbitrage_detected"

    # System events
    STRATEGY_STARTED = "system.strategy_started"
    STRATEGY_STOPPED = "system.strategy_stopped"
    CONFIG_RELOADED = "system.config_reloaded"
    ERROR_OCCURRED = "system.error"


@dataclass
class DomainEvent:
    """Base class for all domain events."""

    event_type: EventType
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = ""
    correlation_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def age_ms(self) -> float:
        """Age of event in milliseconds."""
        return (datetime.now(timezone.utc) - self.timestamp).total_seconds() * 1000


# Specific event types for type safety

@dataclass
class OrderEvent(DomainEvent):
    """Order-related event."""
    client_order_id: str = ""
    exchange_order_id: str = ""
    token_id: str = ""
    side: str = ""
    price: float = 0.0
    size: float = 0.0
    filled_size: float = 0.0
    error_message: str = ""


@dataclass
class TradeEvent(DomainEvent):
    """Trade execution event."""
    trade_id: str = ""
    asset: str = ""
    side: str = ""
    price: float = 0.0
    size_usd: float = 0.0
    model_prob: float = 0.0
    market_price: float = 0.0
    edge: float = 0.0
    pnl: float = 0.0


@dataclass
class RiskEvent(DomainEvent):
    """Risk alert event."""
    breaker_type: str = ""
    current_value: float = 0.0
    threshold: float = 0.0
    severity: str = "warning"  # warning, critical


@dataclass
class MarketEvent(DomainEvent):
    """Market data event."""
    market_id: str = ""
    asset: str = ""
    yes_price: float = 0.0
    no_price: float = 0.0
    volume_usd: float = 0.0


# Type for event handlers
EventHandler = Callable[[DomainEvent], Coroutine[Any, Any, None]]
SyncEventHandler = Callable[[DomainEvent], None]


class EventBus:
    """
    Central event bus for publishing and subscribing to domain events.

    Supports both sync and async handlers.

    Usage:
        bus = EventBus()

        # Subscribe
        async def on_order_filled(event: OrderEvent):
            print(f"Order filled: {event.client_order_id}")

        bus.subscribe(EventType.ORDER_FILLED, on_order_filled)

        # Publish
        await bus.publish(OrderEvent(
            event_type=EventType.ORDER_FILLED,
            client_order_id="abc123",
        ))
    """

    def __init__(self, max_queue_size: int = 1000):
        self._handlers: dict[EventType, list[EventHandler | SyncEventHandler]] = defaultdict(list)
        self._all_handlers: list[EventHandler | SyncEventHandler] = []
        self._queue: asyncio.Queue[DomainEvent] = asyncio.Queue(maxsize=max_queue_size)
        self._running = False
        self._processed_count = 0
        self._error_count = 0
        self._dropped_count = 0

    def subscribe(
        self,
        event_type: EventType,
        handler: EventHandler | SyncEventHandler,
    ) -> None:
        """
        Subscribe to a specific event type.

        Args:
            event_type: Type of event to subscribe to
            handler: Async or sync callback function
        """
        self._handlers[event_type].append(handler)
        logger.debug("Event handler subscribed", event_type=event_type.value)

    def subscribe_all(self, handler: EventHandler | SyncEventHandler) -> None:
        """Subscribe to all events."""
        self._all_handlers.append(handler)

    def unsubscribe(
        self,
        event_type: EventType,
        handler: EventHandler | SyncEventHandler,
    ) -> None:
        """Unsubscribe from an event type."""
        if handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)

    async def publish(self, event: DomainEvent) -> None:
        """
        Publish an event to all subscribers.

        Args:
            event: Event to publish
        """
        # Get handlers for this event type
        handlers = list(self._handlers[event.event_type]) + list(self._all_handlers)

        if not handlers:
            return

        # Execute handlers
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                self._error_count += 1
                logger.error(
                    "Event handler error",
                    event_type=event.event_type.value,
                    error=str(e),
                )

        self._processed_count += 1

    def publish_sync(self, event: DomainEvent) -> None:
        """
        Queue an event for async processing.

        Use this when publishing from sync context.
        """
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            self._dropped_count += 1
            logger.warning("Event queue full, dropping event", event_type=event.event_type.value)

    async def process_queue(self) -> None:
        """Process queued events. Call this in an async context."""
        while not self._queue.empty():
            try:
                event = self._queue.get_nowait()
                await self.publish(event)
            except asyncio.QueueEmpty:
                break

    async def run(self) -> None:
        """Run event processing loop."""
        self._running = True
        logger.info("Event bus started")

        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                await self.publish(event)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error("Event bus error", error=str(e))

    def stop(self) -> None:
        """Stop the event bus."""
        self._running = False

    @property
    def stats(self) -> dict[str, int]:
        """Get event bus statistics."""
        return {
            "processed": self._processed_count,
            "errors": self._error_count,
            "dropped": self._dropped_count,
            "queue_size": self._queue.qsize(),
            "handlers": sum(len(h) for h in self._handlers.values()) + len(self._all_handlers),
        }


class EventLogger:
    """
    Logs all events for audit trail.

    Can be attached to EventBus to log all events.
    """

    def __init__(self, log_level: str = "debug"):
        self.log_level = log_level
        self._event_counts: dict[EventType, int] = defaultdict(int)

    async def handle(self, event: DomainEvent) -> None:
        """Handle an event by logging it."""
        self._event_counts[event.event_type] += 1

        log_fn = getattr(logger, self.log_level, logger.debug)

        log_fn(
            "Domain event",
            event_type=event.event_type.value,
            source=event.source,
            correlation_id=event.correlation_id,
            payload=event.payload,
        )

    def get_counts(self) -> dict[str, int]:
        """Get event counts by type."""
        return {k.value: v for k, v in self._event_counts.items()}


# Global event bus instance
event_bus = EventBus()
event_logger = EventLogger()


# Helper functions for common events

async def emit_order_filled(
    client_order_id: str,
    token_id: str,
    side: str,
    price: float,
    size: float,
    source: str = "order_manager",
) -> None:
    """Emit an order filled event."""
    await event_bus.publish(OrderEvent(
        event_type=EventType.ORDER_FILLED,
        source=source,
        client_order_id=client_order_id,
        token_id=token_id,
        side=side,
        price=price,
        size=size,
        filled_size=size,
    ))


async def emit_circuit_breaker_tripped(
    breaker_type: str,
    current_value: float,
    threshold: float,
    severity: str = "warning",
) -> None:
    """Emit a circuit breaker event."""
    await event_bus.publish(RiskEvent(
        event_type=EventType.CIRCUIT_BREAKER_TRIPPED,
        source="circuit_breaker",
        breaker_type=breaker_type,
        current_value=current_value,
        threshold=threshold,
        severity=severity,
    ))


async def emit_arbitrage_detected(
    market_id: str,
    asset: str,
    profit_pct: float,
    action: str,
) -> None:
    """Emit an arbitrage detection event."""
    await event_bus.publish(MarketEvent(
        event_type=EventType.ARBITRAGE_DETECTED,
        source="arbitrage_detector",
        market_id=market_id,
        asset=asset,
        payload={"profit_pct": profit_pct, "action": action},
    ))
