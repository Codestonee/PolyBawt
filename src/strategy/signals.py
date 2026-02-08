"""
Signal and Intent models for strategy decoupling.

Implements a clean separation between:
- Signal: Raw strategy output (opportunity detected)
- Intent: Validated, risk-checked signal ready for execution

This decouples strategy logic from execution/risk logic.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import time


class SignalType(Enum):
    """Type of trading signal."""
    
    ENTRY = "entry"        # New position
    EXIT = "exit"          # Close position
    ADJUST = "adjust"      # Modify existing position
    HEDGE = "hedge"        # Hedge existing exposure


class SignalSide(Enum):
    """Side of the signal."""
    
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Signal:
    """
    Raw trading signal from a strategy.
    
    A Signal represents an *opportunity* that a strategy has detected.
    It has NOT yet been validated for risk limits or position constraints.
    
    Usage:
        signal = Signal(
            strategy_id="arb_taker",
            signal_type=SignalType.ENTRY,
            side=SignalSide.BUY,
            token_id="0x123...",
            asset="BTC",
            price=0.55,
            size_usd=5.0,
            edge=0.05,  # 5% expected edge
            confidence=0.8,
        )
    """
    
    # Strategy identification
    strategy_id: str
    
    # Signal details
    signal_type: SignalType
    side: SignalSide
    token_id: str
    asset: str
    
    # Pricing
    price: float  # Limit price
    size_usd: float  # Requested size in USD
    
    # Edge/confidence (from strategy analysis)
    edge: float = 0.0  # Expected edge (after fees)
    confidence: float = 0.5  # Strategy confidence (0-1)
    
    # Metadata
    market_question: str = ""
    reason: str = ""  # Human-readable reason for signal
    created_at: float = field(default_factory=time.time)
    
    # Optional order params
    order_type: str = "GTC"
    
    @property
    def age_seconds(self) -> float:
        """Age of this signal in seconds."""
        return time.time() - self.created_at
    
    def to_order_params(self) -> dict[str, Any]:
        """Convert to order params dict for compatibility."""
        return {
            "side": self.side.value,
            "token_id": self.token_id,
            "price": self.price,
            "size": self.size_usd,
            "order_type": self.order_type,
            "strategy_id": self.strategy_id,
        }


@dataclass
class Intent:
    """
    Validated intent ready for execution.
    
    An Intent is a Signal that has passed through the RiskGate.
    It has been validated for:
    - Position limits
    - Exposure caps
    - Circuit breaker status
    - Rate limit availability
    
    The size may have been adjusted from the original Signal.
    """
    
    # Original signal (for audit trail)
    signal: Signal
    
    # Validated execution params (may differ from signal)
    validated_size_usd: float
    validated_price: float
    
    # Risk gate results
    passed_risk_check: bool = True
    risk_adjustments: list[str] = field(default_factory=list)
    rejection_reason: str = ""
    
    # Timestamps
    validated_at: float = field(default_factory=time.time)
    
    @property
    def strategy_id(self) -> str:
        return self.signal.strategy_id
    
    @property
    def token_id(self) -> str:
        return self.signal.token_id
    
    @property
    def side(self) -> SignalSide:
        return self.signal.side
    
    @property
    def was_size_reduced(self) -> bool:
        """Check if size was reduced by risk gate."""
        return self.validated_size_usd < self.signal.size_usd
    
    def to_order_params(self) -> dict[str, Any]:
        """Convert to order params dict for execution."""
        return {
            "side": self.signal.side.value,
            "token_id": self.signal.token_id,
            "price": self.validated_price,
            "size": self.validated_size_usd,
            "order_type": self.signal.order_type,
            "strategy_id": self.signal.strategy_id,
        }


def signal_from_order_params(params: dict, strategy_id: str = "legacy") -> Signal:
    """
    Convert legacy order params dict to Signal.
    
    For backward compatibility with existing strategies that return dicts.
    """
    return Signal(
        strategy_id=strategy_id,
        signal_type=SignalType.ENTRY,
        side=SignalSide.BUY if params.get("side", "BUY").upper() == "BUY" else SignalSide.SELL,
        token_id=params.get("token_id", ""),
        asset=params.get("asset", ""),
        price=params.get("price", 0.0),
        size_usd=params.get("size", 0.0),
        edge=params.get("edge", 0.0),
        confidence=params.get("confidence", 0.5),
        market_question=params.get("market_question", ""),
        order_type=params.get("order_type", "GTC"),
    )
