"""
Polymarket fee helpers.

Fee model reference:
    fee = size * 0.25 * (p * (1-p))^2
"""

from __future__ import annotations


def polymarket_fee_rate(price: float) -> float:
    """
    Return estimated fee rate for a leg at probability `price`.

    The return value is a fraction of notional size.
    """
    p = max(0.001, min(0.999, float(price)))
    return 0.25 * (p * (1.0 - p)) ** 2


def estimated_leg_fee_usd(price: float, notional_usd: float) -> float:
    """Estimate fee in USD for one leg."""
    return max(0.0, float(notional_usd)) * polymarket_fee_rate(price)
