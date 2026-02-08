"""Tests for event bus behavior."""

from src.infrastructure.events import EventBus, DomainEvent, EventType


def test_event_bus_tracks_dropped_events():
    bus = EventBus(max_queue_size=1)
    e1 = DomainEvent(event_type=EventType.ERROR_OCCURRED, source="test")
    e2 = DomainEvent(event_type=EventType.ERROR_OCCURRED, source="test")

    bus.publish_sync(e1)
    bus.publish_sync(e2)

    stats = bus.stats
    assert stats["queue_size"] == 1
    assert stats["dropped"] == 1
