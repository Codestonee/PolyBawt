"""
NO-TRADE gate - rejection logic for unsafe or unprofitable trades.

Implements a comprehensive checklist of conditions that must pass
before any trade is executed.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from src.infrastructure.logging import get_logger
from src.models.ev_calculator import EVResult

logger = get_logger(__name__)


class RejectionReason(Enum):
    """Categorized rejection reasons."""
    
    # Edge-related
    NEGATIVE_EV = "negative_ev"
    EDGE_BELOW_FEE_MULTIPLE = "edge_below_fee_multiple"
    NO_DIRECTIONAL_CONVICTION = "no_directional_conviction"
    
    # Temporal
    TOO_CLOSE_TO_EXPIRY = "too_close_to_expiry"
    TOO_EARLY_AFTER_OPEN = "too_early_after_open"
    
    # Risk
    DAILY_LOSS_EXCEEDED = "daily_loss_exceeded"
    DRAWDOWN_EXCEEDED = "drawdown_exceeded"
    CORRELATION_TOO_HIGH = "correlation_too_high"
    VOLATILITY_TOO_HIGH = "volatility_too_high"
    POSITION_LIMIT_EXCEEDED = "position_limit_exceeded"
    
    # Operational
    ORACLE_STALE = "oracle_stale"
    RATE_LIMIT_CRITICAL = "rate_limit_critical"
    SPREAD_TOO_WIDE = "spread_too_wide"
    LIQUIDITY_TOO_LOW = "liquidity_too_low"
    
    # System
    TRADING_HALTED = "trading_halted"
    RISK_CHECK_FAILED = "risk_check_failed"


@dataclass
class GateResult:
    """Result of NO-TRADE gate evaluation."""
    
    passed: bool
    rejection_reason: RejectionReason | None = None
    rejection_message: str = ""
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)
    
    def __bool__(self) -> bool:
        return self.passed


@dataclass
class TradeContext:
    """Context for evaluating a potential trade."""

    # Trade details
    ev_result: EVResult
    asset: str
    token_id: str

    # Timing
    seconds_to_expiry: float
    seconds_since_open: float = 0

    # Market quality
    spread: float = 0  # Bid-ask spread
    book_depth_usd: float = 1000  # Depth on our side
    oracle_age_seconds: float | None = None  # None means missing data (fail-safe)

    # Portfolio state
    current_position_usd: float = 0
    correlation_with_portfolio: float = 0
    realized_vol_15m: float = 0.6  # Annualized

    # System state
    daily_pnl_pct: float = 0
    current_drawdown_pct: float = 0
    rate_limit_usage_pct: float = 0
    trading_halted: bool = False

    # Dynamic overrides (e.g., for extreme probability markets)
    override_min_edge: float | None = None  # If set, overrides config.min_edge_threshold


@dataclass
class GateConfig:
    """Configuration for NO-TRADE gate thresholds."""
    
    # Edge thresholds
    min_edge_threshold: float = 0.04  # 4% minimum edge
    min_edge_to_fee_ratio: float = 1.5  # Edge must be 1.5x fees
    
    # Temporal thresholds
    min_seconds_to_expiry: int = 60
    min_seconds_after_open: int = 30
    
    # Risk thresholds
    daily_loss_soft_limit_pct: float = 0.03
    daily_loss_hard_limit_pct: float = 0.05
    max_drawdown_pct: float = 0.10
    max_correlation: float = 0.7
    max_volatility_annualized: float = 2.0  # 200%
    max_position_pct: float = 0.02
    
    # Operational thresholds
    max_oracle_age_seconds: float = 10
    max_rate_limit_usage_pct: float = 0.90
    max_spread: float = 0.10  # 10 cents
    min_book_depth_usd: float = 100


class NoTradeGate:
    """
    Gate that evaluates whether a trade should be rejected.
    
    Implements a comprehensive NO-TRADE checklist based on:
    - EV/edge quality
    - Timing constraints
    - Risk limits
    - Operational health
    
    Usage:
        gate = NoTradeGate(config)
        context = TradeContext(ev_result=ev, ...)
        result = gate.evaluate(context)
        
        if not result.passed:
            logger.info(f"Trade rejected: {result.rejection_reason}")
    """
    
    def __init__(self, config: GateConfig | None = None):
        self.config = config or GateConfig()
    
    def evaluate(self, context: TradeContext) -> GateResult:
        """
        Evaluate all gate conditions.
        
        Returns as soon as first failure is found (fail-fast).
        
        Args:
            context: Trade context with all relevant state
        
        Returns:
            GateResult indicating pass/fail with reason
        """
        checks_passed = []
        checks_failed = []
        
        # Run all checks
        checks = [
            ("trading_not_halted", self._check_trading_not_halted),
            ("positive_ev", self._check_positive_ev),
            ("edge_above_threshold", self._check_edge_threshold),
            ("edge_to_fee_ratio", self._check_edge_to_fee_ratio),
            ("time_to_expiry", self._check_time_to_expiry),
            ("time_since_open", self._check_time_since_open),
            ("daily_loss_limit", self._check_daily_loss),
            ("drawdown_limit", self._check_drawdown),
            ("volatility_limit", self._check_volatility),
            ("correlation_limit", self._check_correlation),
            ("oracle_freshness", self._check_oracle_freshness),
            ("rate_limit_headroom", self._check_rate_limit),
            ("spread_acceptable", self._check_spread),
            ("liquidity_sufficient", self._check_liquidity),
        ]
        
        for check_name, check_fn in checks:
            reason, message = check_fn(context)
            
            if reason is not None:
                checks_failed.append(check_name)
                logger.debug(
                    "Trade gate failed",
                    check=check_name,
                    reason=reason.value,
                    message=message,
                )
                return GateResult(
                    passed=False,
                    rejection_reason=reason,
                    rejection_message=message,
                    checks_passed=checks_passed,
                    checks_failed=checks_failed,
                )
            else:
                checks_passed.append(check_name)
        
        # All checks passed
        return GateResult(
            passed=True,
            checks_passed=checks_passed,
            checks_failed=[],
        )
    
    def _check_trading_not_halted(
        self, ctx: TradeContext
    ) -> tuple[RejectionReason | None, str]:
        if ctx.trading_halted:
            return RejectionReason.TRADING_HALTED, "Trading is halted"
        return None, ""
    
    def _check_positive_ev(
        self, ctx: TradeContext
    ) -> tuple[RejectionReason | None, str]:
        if not ctx.ev_result.is_positive_ev:
            return RejectionReason.NEGATIVE_EV, f"Net EV = ${ctx.ev_result.net_ev:.4f}"
        return None, ""
    
    def _check_edge_threshold(
        self, ctx: TradeContext
    ) -> tuple[RejectionReason | None, str]:
        # Use override if provided (e.g., for extreme probability markets with lower fees)
        min_edge = ctx.override_min_edge if ctx.override_min_edge is not None else self.config.min_edge_threshold
        if abs(ctx.ev_result.gross_edge) < min_edge:
            return (
                RejectionReason.NO_DIRECTIONAL_CONVICTION,
                f"Edge {ctx.ev_result.gross_edge:.3f} < {min_edge}"
            )
        return None, ""
    
    def _check_edge_to_fee_ratio(
        self, ctx: TradeContext
    ) -> tuple[RejectionReason | None, str]:
        if ctx.ev_result.edge_to_fee_ratio < self.config.min_edge_to_fee_ratio:
            return (
                RejectionReason.EDGE_BELOW_FEE_MULTIPLE,
                f"Edge/Fee = {ctx.ev_result.edge_to_fee_ratio:.2f}x < {self.config.min_edge_to_fee_ratio}x"
            )
        return None, ""
    
    def _check_time_to_expiry(
        self, ctx: TradeContext
    ) -> tuple[RejectionReason | None, str]:
        if ctx.seconds_to_expiry < self.config.min_seconds_to_expiry:
            return (
                RejectionReason.TOO_CLOSE_TO_EXPIRY,
                f"{ctx.seconds_to_expiry:.0f}s < {self.config.min_seconds_to_expiry}s"
            )
        return None, ""
    
    def _check_time_since_open(
        self, ctx: TradeContext
    ) -> tuple[RejectionReason | None, str]:
        if ctx.seconds_since_open < self.config.min_seconds_after_open:
            return (
                RejectionReason.TOO_EARLY_AFTER_OPEN,
                f"{ctx.seconds_since_open:.0f}s < {self.config.min_seconds_after_open}s since open"
            )
        return None, ""
    
    def _check_daily_loss(
        self, ctx: TradeContext
    ) -> tuple[RejectionReason | None, str]:
        if ctx.daily_pnl_pct < -self.config.daily_loss_hard_limit_pct:
            return (
                RejectionReason.DAILY_LOSS_EXCEEDED,
                f"Daily loss {ctx.daily_pnl_pct:.1%} exceeds hard limit"
            )
        if ctx.daily_pnl_pct < -self.config.daily_loss_soft_limit_pct:
            return (
                RejectionReason.DAILY_LOSS_EXCEEDED,
                f"Daily loss {ctx.daily_pnl_pct:.1%} exceeds soft limit"
            )
        return None, ""
    
    def _check_drawdown(
        self, ctx: TradeContext
    ) -> tuple[RejectionReason | None, str]:
        if abs(ctx.current_drawdown_pct) > self.config.max_drawdown_pct:
            return (
                RejectionReason.DRAWDOWN_EXCEEDED,
                f"Drawdown {ctx.current_drawdown_pct:.1%} > {self.config.max_drawdown_pct:.1%}"
            )
        return None, ""
    
    def _check_volatility(
        self, ctx: TradeContext
    ) -> tuple[RejectionReason | None, str]:
        if ctx.realized_vol_15m > self.config.max_volatility_annualized:
            return (
                RejectionReason.VOLATILITY_TOO_HIGH,
                f"Vol {ctx.realized_vol_15m:.0%} > {self.config.max_volatility_annualized:.0%}"
            )
        return None, ""
    
    def _check_correlation(
        self, ctx: TradeContext
    ) -> tuple[RejectionReason | None, str]:
        if ctx.correlation_with_portfolio > self.config.max_correlation:
            return (
                RejectionReason.CORRELATION_TOO_HIGH,
                f"Correlation {ctx.correlation_with_portfolio:.2f} > {self.config.max_correlation}"
            )
        return None, ""
    
    def _check_oracle_freshness(
        self, ctx: TradeContext
    ) -> tuple[RejectionReason | None, str]:
        # FIX: Treat missing oracle data as stale
        # If oracle_age_seconds is None, 0, or very small, data may be missing
        if ctx.oracle_age_seconds is None:
            return (
                RejectionReason.ORACLE_STALE,
                "Oracle data missing (age is None)"
            )
        
        # Age of 0 likely means no timestamp was set - treat as stale
        if ctx.oracle_age_seconds <= 0:
            return (
                RejectionReason.ORACLE_STALE,
                f"Oracle age suspicious ({ctx.oracle_age_seconds}s) - may be missing"
            )
        
        if ctx.oracle_age_seconds > self.config.max_oracle_age_seconds:
            return (
                RejectionReason.ORACLE_STALE,
                f"Oracle age {ctx.oracle_age_seconds:.1f}s > {self.config.max_oracle_age_seconds}s"
            )
        return None, ""
    
    def _check_rate_limit(
        self, ctx: TradeContext
    ) -> tuple[RejectionReason | None, str]:
        if ctx.rate_limit_usage_pct > self.config.max_rate_limit_usage_pct:
            return (
                RejectionReason.RATE_LIMIT_CRITICAL,
                f"Rate limit usage {ctx.rate_limit_usage_pct:.0%}"
            )
        return None, ""
    
    def _check_spread(
        self, ctx: TradeContext
    ) -> tuple[RejectionReason | None, str]:
        if ctx.spread > self.config.max_spread:
            return (
                RejectionReason.SPREAD_TOO_WIDE,
                f"Spread ${ctx.spread:.2f} > ${self.config.max_spread:.2f}"
            )
        return None, ""
    
    def _check_liquidity(
        self, ctx: TradeContext
    ) -> tuple[RejectionReason | None, str]:
        if ctx.book_depth_usd < self.config.min_book_depth_usd:
            return (
                RejectionReason.LIQUIDITY_TOO_LOW,
                f"Depth ${ctx.book_depth_usd:.0f} < ${self.config.min_book_depth_usd:.0f}"
            )
        return None, ""


# Pre-instantiated gate with defaults
no_trade_gate = NoTradeGate()
