"""
Core infrastructure modules for PolyBawt.

Contains:
- event_bus: Priority-based async event dispatching with backpressure
- idempotency: Order deduplication and COID generation
"""

from src.core.event_bus import (
    EventPriority,
    PriorityEventBus,
    BackpressureState,
    p0_event,
    p1_event,
    p2_event,
)

from src.core.idempotency import (
    IdempotencyRegistry,
    IntentRecord,
)

__all__ = [
    # Event bus
    "EventPriority",
    "PriorityEventBus",
    "BackpressureState",
    "p0_event",
    "p1_event",
    "p2_event",
    # Idempotency
    "IdempotencyRegistry",
    "IntentRecord",
]
