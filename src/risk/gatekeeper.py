"""
RiskGate - Centralized risk validation for trading signals.

Validates signals against:
- Position limits
- Exposure caps
- Circuit breaker status
- Rate limit availability
- Kill switch state

Converts Signal -> Intent (or rejects).
"""

from dataclasses import dataclass
from typing import Protocol

from src.infrastructure.logging import get_logger
from src.strategy.signals import Signal, Intent
from src.risk.circuit_breaker import CircuitBreaker
from src.execution.rate_limiter import RateLimiter
from src.portfolio.tracker import Portfolio

logger = get_logger(__name__)


class RiskGateConfig(Protocol):
    """Configuration interface for risk gate."""
    
    @property
    def max_trade_usd(self) -> float:
        """Max USD per single trade."""
        ...
    
    @property
    def max_total_exposure_usd(self) -> float:
        """Max total portfolio exposure."""
        ...
    
    @property
    def min_order_size_usd(self) -> float:
        """Minimum viable order size."""
        ...


@dataclass
class RiskGateResult:
    """Result of risk gate validation."""
    
    passed: bool
    intent: Intent | None
    rejection_reason: str = ""
    adjustments: list[str] | None = None


class RiskGate:
    """
    Centralized risk validation gate.
    
    All signals must pass through the RiskGate before execution.
    The gate checks all risk constraints and either:
    - Approves (returns Intent with possibly adjusted size)
    - Rejects (returns rejected Intent with reason)
    
    Usage:
        gate = RiskGate(
            circuit_breaker=cb,
            rate_limiter=rl,
            portfolio=portfolio,
            max_trade_usd=5.0,
            max_total_exposure_usd=20.0,
            min_order_size_usd=1.0,
        )
        
        result = gate.validate(signal)
        if result.passed:
            execute_order(result.intent)
        else:
            logger.info(f"Rejected: {result.rejection_reason}")
    """
    
    def __init__(
        self,
        circuit_breaker: CircuitBreaker,
        rate_limiter: RateLimiter,
        portfolio: Portfolio,
        max_trade_usd: float = 5.0,
        max_total_exposure_usd: float = 20.0,
        min_order_size_usd: float = 1.0,
        max_per_token_usd: float = 10.0,  # Per-token concentration limit
        pending_exposure_getter: callable = None,  # Function to get pending order exposure
    ):
        self.circuit_breaker = circuit_breaker
        self.rate_limiter = rate_limiter
        self.portfolio = portfolio
        self.max_trade_usd = max_trade_usd
        self.max_total_exposure_usd = max_total_exposure_usd
        self.min_order_size_usd = min_order_size_usd
        self.max_per_token_usd = max_per_token_usd
        self._get_pending_exposure = pending_exposure_getter or (lambda: 0.0)
    
    def validate(self, signal: Signal) -> RiskGateResult:
        """
        Validate a signal against all risk constraints.
        
        Returns RiskGateResult with:
        - passed=True: Intent ready for execution
        - passed=False: Rejection reason provided
        """
        adjustments = []
        
        # 1. Check circuit breaker
        if not self.circuit_breaker.can_trade():
            return self._reject(signal, "Circuit breaker active - trading halted")
        
        if not self.circuit_breaker.can_enter_new_position():
            return self._reject(signal, "Circuit breaker soft-tripped - no new entries")
        
        # 2. Check rate limit
        if self.rate_limiter.is_critical():
            return self._reject(signal, "Rate limit critical - backing off")
        
        # 3. Validate and adjust size
        validated_size = signal.size_usd
        
        # Cap at max trade size
        if validated_size > self.max_trade_usd:
            validated_size = self.max_trade_usd
            adjustments.append(f"Size capped to ${self.max_trade_usd} max trade")
        
        # Check total exposure cap
        current_exposure = self.portfolio.total_exposure
        pending_exposure = self._get_pending_exposure()
        available_exposure = self.max_total_exposure_usd - current_exposure - pending_exposure
        
        if available_exposure <= 0:
            return self._reject(
                signal, 
                f"Max exposure ${self.max_total_exposure_usd} reached "
                f"(current=${current_exposure:.2f}, pending=${pending_exposure:.2f})"
            )
        
        if validated_size > available_exposure:
            validated_size = available_exposure
            adjustments.append(f"Size reduced to ${validated_size:.2f} for exposure cap")
        
        # 4. Check per-token concentration limit
        token_id = getattr(signal, 'token_id', None)
        if token_id and self.max_per_token_usd > 0:
            current_token_exposure = sum(
                pos.size_usd for pos in self.portfolio.get_open_positions()
                if pos.token_id == token_id
            )
            available_token_exposure = self.max_per_token_usd - current_token_exposure
            if available_token_exposure <= 0:
                return self._reject(
                    signal,
                    f"Per-token limit ${self.max_per_token_usd} reached for token (current=${current_token_exposure:.2f})"
                )
            if validated_size > available_token_exposure:
                validated_size = available_token_exposure
                adjustments.append(f"Size reduced to ${validated_size:.2f} for per-token limit")
        
        # Check minimum viable size
        if validated_size < self.min_order_size_usd:
            return self._reject(
                signal,
                f"Size ${validated_size:.2f} below minimum ${self.min_order_size_usd}"
            )
        
        # 4. Create approved intent
        intent = Intent(
            signal=signal,
            validated_size_usd=validated_size,
            validated_price=signal.price,
            passed_risk_check=True,
            risk_adjustments=adjustments,
        )
        
        logger.debug(
            "Signal approved by RiskGate",
            strategy=signal.strategy_id,
            original_size=signal.size_usd,
            validated_size=validated_size,
            adjustments=adjustments,
        )
        
        return RiskGateResult(
            passed=True,
            intent=intent,
            adjustments=adjustments,
        )
    
    def _reject(self, signal: Signal, reason: str) -> RiskGateResult:
        """Create rejection result."""
        logger.debug(
            "Signal rejected by RiskGate",
            strategy=signal.strategy_id,
            reason=reason,
        )
        
        intent = Intent(
            signal=signal,
            validated_size_usd=0.0,
            validated_price=signal.price,
            passed_risk_check=False,
            rejection_reason=reason,
        )
        
        return RiskGateResult(
            passed=False,
            intent=intent,
            rejection_reason=reason,
        )
