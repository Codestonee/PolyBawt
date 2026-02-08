"""
Replay module for PolyBawt.

Components:
- event_logger: NDJSON event persistence for replay-grade logging
- exchange_sim: Simulated exchange with price-time priority
- replayer: Deterministic event replay engine
"""

from src.replay.event_logger import (
    EventLogger,
    EventReader,
    Event,
    EventType,
)

from src.replay.exchange_sim import (
    ExchangeSimulator,
    OrderBook,
    SimOrder,
    SimFill,
    OrderSide,
    OrderType,
    OrderStatus,
)

from src.replay.replayer import (
    Replayer,
    ReplayConfig,
    ReplayStats,
    run_replay_test,
)

__all__ = [
    # Event logger
    "EventLogger",
    "EventReader",
    "Event",
    "EventType",
    # Exchange sim
    "ExchangeSimulator",
    "OrderBook",
    "SimOrder",
    "SimFill",
    "OrderSide",
    "OrderType",
    "OrderStatus",
    # Replayer
    "Replayer",
    "ReplayConfig",
    "ReplayStats",
    "run_replay_test",
]
