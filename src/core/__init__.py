"""
Core infrastructure modules for PolyBawt.

Contains:
- event_bus: Priority-based async event dispatching with backpressure
"""

from src.core.event_bus import (
    EventPriority,
    PriorityEventBus,
    BackpressureState,
    p0_event,
    p1_event,
    p2_event,
)

__all__ = [
    "EventPriority",
    "PriorityEventBus",
    "BackpressureState",
    "p0_event",
    "p1_event",
    "p2_event",
]
