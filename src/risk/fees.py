"""
Polymarket Dynamic Fee Model.

Fee model reference (January 2026 regime):
    taker_fee = size * 0.25 * (p * (1-p))^2
    
    At p=0.50: taker_fee ≈ 1.56% (peak)
    At p=0.10: taker_fee ≈ 0.20%
    At p=0.90: taker_fee ≈ 0.20%
    
Maker rebates are distributed daily from 100% of taker fees,
weighted by maker volume near p=0.50 (where fees are highest).
"""

from __future__ import annotations
from enum import Enum
from dataclasses import dataclass


class OrderRole(Enum):
    """Order execution role."""
    MAKER = "maker"
    TAKER = "taker"


@dataclass
class FeeEstimate:
    """Fee/rebate estimate for a trade."""
    gross_fee: float  # Taker fee if applicable
    estimated_rebate: float  # Maker rebate estimate
    net_cost: float  # Net fee after rebate (negative = net income)
    probability: float
    role: OrderRole


def taker_fee_rate(probability: float) -> float:
    """
    Return taker fee rate for a leg at given probability.
    
    Formula: 0.25 * (p * (1-p))^2
    
    The return value is a fraction of notional size.
    Peak fee (~1.56%) occurs at p=0.50.
    """
    p = max(0.001, min(0.999, float(probability)))
    return 0.25 * (p * (1.0 - p)) ** 2


def taker_fee_usd(probability: float, size_usd: float) -> float:
    """Calculate taker fee in USD for a given trade."""
    return max(0.0, float(size_usd)) * taker_fee_rate(probability)


def estimated_maker_rebate_rate(probability: float) -> float:
    """
    Estimate maker rebate rate based on probability.
    
    Rebates are distributed proportionally to liquidity provided
    near p=0.50 where taker fees are highest. This is an estimate
    based on the fee curve shape - actual rebates depend on daily
    pool distribution.
    
    Conservative estimate: ~60% of equivalent taker fee at same probability.
    """
    # Rebate tracks fee curve but at reduced rate due to pool dilution
    base_rate = taker_fee_rate(probability)
    rebate_efficiency = 0.60  # Conservative estimate of rebate capture
    return base_rate * rebate_efficiency


def estimated_maker_rebate_usd(probability: float, size_usd: float) -> float:
    """Estimate maker rebate in USD for a given trade."""
    return max(0.0, float(size_usd)) * estimated_maker_rebate_rate(probability)


def net_edge_after_fees(
    gross_edge: float,
    probability: float,
    is_maker: bool,
    size_usd: float = 1.0,
) -> float:
    """
    Calculate net edge after accounting for fees/rebates.
    
    Args:
        gross_edge: Raw edge before fees (as fraction, e.g., 0.02 = 2%)
        probability: Contract probability (0-1)
        is_maker: True if maker order, False if taker
        size_usd: Trade size in USD (default 1.0 for rate calculation)
        
    Returns:
        Net edge after fees/rebates (as fraction)
    """
    if is_maker:
        # Makers pay no fee and receive rebate
        rebate = estimated_maker_rebate_rate(probability)
        return gross_edge + rebate
    else:
        # Takers pay fee
        fee = taker_fee_rate(probability)
        return gross_edge - fee


def estimate_trade_fees(
    probability: float,
    size_usd: float,
    role: OrderRole,
) -> FeeEstimate:
    """
    Comprehensive fee estimation for a trade.
    
    Args:
        probability: Contract probability (0-1)
        size_usd: Trade size in USD
        role: MAKER or TAKER
        
    Returns:
        FeeEstimate with all fee/rebate details
    """
    if role == OrderRole.TAKER:
        gross_fee = taker_fee_usd(probability, size_usd)
        rebate = 0.0
        net_cost = gross_fee
    else:
        gross_fee = 0.0
        rebate = estimated_maker_rebate_usd(probability, size_usd)
        net_cost = -rebate  # Negative = income
    
    return FeeEstimate(
        gross_fee=gross_fee,
        estimated_rebate=rebate,
        net_cost=net_cost,
        probability=probability,
        role=role,
    )


def is_taker_ev_positive(
    gross_edge: float,
    probability: float,
    confidence: float = 0.8,
) -> bool:
    """
    Check if a taker trade has positive expected value after fees.
    
    Args:
        gross_edge: Raw edge before fees (as fraction)
        probability: Contract probability (0-1)
        confidence: Required confidence multiplier (default 0.8 for safety)
        
    Returns:
        True if net edge is positive with safety buffer
    """
    net = net_edge_after_fees(gross_edge, probability, is_maker=False)
    return net > 0 and net >= gross_edge * (1 - confidence)


# Backward compatibility aliases
polymarket_fee_rate = taker_fee_rate
estimated_leg_fee_usd = taker_fee_usd
