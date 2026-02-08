from dataclasses import dataclass, field
from typing import Any, List

from src.portfolio.tracker import Portfolio


@dataclass
class BotState:
    """Shared state between trading bot and API."""
    portfolio: Portfolio | None = None
    order_manager: Any | None = None  # OrderManager for kill switch access
    active_orders: List[Any] = field(default_factory=list)
    recent_signals: List[Any] = field(default_factory=list)
    is_running: bool = False
    is_live: bool = False  # True if sending real orders
    active_strategies: dict[str, bool] = field(default_factory=dict)
    last_update: float = 0


# Global singleton instance
_state = BotState()


def get_state() -> BotState:
    """Get the global bot state."""
    return _state
