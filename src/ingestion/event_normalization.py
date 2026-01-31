"""Event market normalization helpers.

Purpose: keep categorization and tradeability heuristics out of the network client.
"""

from __future__ import annotations

from datetime import datetime, timezone


def is_tradeable_event_market(
    *,
    volume_24h: float,
    liquidity: float,
    min_volume_24h: float,
    min_liquidity: float,
) -> bool:
    return float(volume_24h) >= float(min_volume_24h) and float(liquidity) >= float(min_liquidity)


def hours_to_resolution(end_date) -> float | None:
    if end_date is None:
        return None
    delta = end_date - datetime.now(timezone.utc)
    return delta.total_seconds() / 3600

# NOTE: keep this module import-free of event_market_discovery to avoid circular imports.
