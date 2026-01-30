"""
Circuit breakers for risk management.

Implements automatic trading halts when risk thresholds are breached:
- Daily loss limits
- Max drawdown
- Volatility spikes
- Oracle staleness
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Any

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class BreakerState(Enum):
    """Circuit breaker state."""
    
    CLOSED = "closed"  # Normal operation
    SOFT_TRIPPED = "soft_tripped"  # Warning, restrictions apply
    HARD_TRIPPED = "hard_tripped"  # Trading halted


class BreakerType(Enum):
    """Types of circuit breakers."""

    DAILY_LOSS = "daily_loss"
    DRAWDOWN = "drawdown"
    VOLATILITY = "volatility"
    ORACLE = "oracle"
    RATE_LIMIT = "rate_limit"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    MANUAL = "manual"


@dataclass
class BreakerStatus:
    """Status of a single circuit breaker."""
    
    breaker_type: BreakerType
    state: BreakerState
    current_value: float
    soft_threshold: float
    hard_threshold: float
    message: str = ""
    tripped_at: float | None = None
    
    @property
    def is_tripped(self) -> bool:
        return self.state != BreakerState.CLOSED


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breakers."""

    # Daily loss limits (as % of starting capital)
    daily_loss_soft_pct: float = 0.03  # -3%
    daily_loss_hard_pct: float = 0.05  # -5%

    # Drawdown limits
    max_drawdown_soft_pct: float = 0.07  # -7%
    max_drawdown_hard_pct: float = 0.10  # -10%

    # Volatility thresholds (annualized)
    volatility_soft: float = 1.5  # 150%
    volatility_hard: float = 2.0  # 200%

    # Oracle staleness (seconds)
    oracle_stale_soft: float = 15.0
    oracle_stale_hard: float = 30.0

    # Rate limit usage
    rate_limit_soft_pct: float = 0.80
    rate_limit_hard_pct: float = 0.95

    # Consecutive loss limits
    consecutive_loss_soft: int = 5   # Soft trip after 5 consecutive losses
    consecutive_loss_hard: int = 8   # Hard trip after 8 consecutive losses

    # Auto-reset after time (0 = no auto-reset)
    soft_reset_seconds: float = 300   # 5 minutes for soft trips
    hard_reset_seconds: float = 1800  # 30 minutes for hard trips (0 = never)

    # Cooldown periods by breaker type (seconds)
    breaker_cooldowns: dict[str, float] = field(default_factory=lambda: {
        "daily_loss": 0,           # Never auto-reset daily loss
        "drawdown": 0,             # Never auto-reset drawdown
        "volatility": 300,         # 5 min cooldown
        "oracle": 60,              # 1 min cooldown
        "rate_limit": 120,         # 2 min cooldown
        "consecutive_losses": 600, # 10 min cooldown
    })


class CircuitBreaker:
    """
    Manages trading circuit breakers.
    
    Breakers can be in three states:
    - CLOSED: Normal operation
    - SOFT_TRIPPED: Warning level, new entries blocked
    - HARD_TRIPPED: Emergency, all trading halted
    
    Usage:
        breakers = CircuitBreaker(starting_capital=1000)
        
        # Update with current values
        breakers.update_daily_pnl(-40)  # $40 loss
        breakers.update_volatility(1.8)
        
        # Check before trading
        if breakers.can_trade():
            execute_trade()
        
        # Get status
        for status in breakers.get_all_status():
            if status.is_tripped:
                logger.warning(f"{status.breaker_type}: {status.message}")
    """
    
    def __init__(
        self,
        starting_capital: float,
        config: CircuitBreakerConfig | None = None,
    ):
        self.starting_capital = starting_capital
        self.config = config or CircuitBreakerConfig()

        # Current values
        self._daily_pnl = 0.0
        self._peak_capital = starting_capital
        self._current_capital = starting_capital
        self._current_volatility = 0.6
        self._oracle_age = 0.0
        self._rate_limit_usage = 0.0
        self._consecutive_losses = 0
        self._last_trade_won = True

        # Breaker states
        self._states: dict[BreakerType, BreakerState] = {
            bt: BreakerState.CLOSED for bt in BreakerType
        }
        self._trip_times: dict[BreakerType, float] = {}

        # Manual override
        self._manual_halt = False
    
    def update_daily_pnl(self, pnl: float) -> None:
        """Update daily PnL and check breaker."""
        self._daily_pnl = pnl
        self._current_capital = self.starting_capital + pnl
        self._peak_capital = max(self._peak_capital, self._current_capital)
        
        pnl_pct = abs(pnl) / self.starting_capital if self.starting_capital > 0 else 0
        
        if pnl < 0:
            if pnl_pct >= self.config.daily_loss_hard_pct:
                self._trip(BreakerType.DAILY_LOSS, BreakerState.HARD_TRIPPED)
            elif pnl_pct >= self.config.daily_loss_soft_pct:
                self._trip(BreakerType.DAILY_LOSS, BreakerState.SOFT_TRIPPED)
            else:
                self._reset(BreakerType.DAILY_LOSS)
        else:
            self._reset(BreakerType.DAILY_LOSS)
    
    def update_drawdown(self, current_capital: float) -> None:
        """Update capital and check drawdown breaker."""
        self._current_capital = current_capital
        self._peak_capital = max(self._peak_capital, current_capital)
        
        if self._peak_capital > 0:
            drawdown = (self._peak_capital - current_capital) / self._peak_capital
        else:
            drawdown = 0
        
        if drawdown >= self.config.max_drawdown_hard_pct:
            self._trip(BreakerType.DRAWDOWN, BreakerState.HARD_TRIPPED)
        elif drawdown >= self.config.max_drawdown_soft_pct:
            self._trip(BreakerType.DRAWDOWN, BreakerState.SOFT_TRIPPED)
        else:
            self._reset(BreakerType.DRAWDOWN)
    
    def update_volatility(self, annualized_vol: float) -> None:
        """Update volatility and check breaker."""
        self._current_volatility = annualized_vol
        
        if annualized_vol >= self.config.volatility_hard:
            self._trip(BreakerType.VOLATILITY, BreakerState.HARD_TRIPPED)
        elif annualized_vol >= self.config.volatility_soft:
            self._trip(BreakerType.VOLATILITY, BreakerState.SOFT_TRIPPED)
        else:
            self._reset(BreakerType.VOLATILITY)
    
    def update_oracle_age(self, age_seconds: float) -> None:
        """Update oracle staleness and check breaker."""
        self._oracle_age = age_seconds
        
        if age_seconds >= self.config.oracle_stale_hard:
            self._trip(BreakerType.ORACLE, BreakerState.HARD_TRIPPED)
        elif age_seconds >= self.config.oracle_stale_soft:
            self._trip(BreakerType.ORACLE, BreakerState.SOFT_TRIPPED)
        else:
            self._reset(BreakerType.ORACLE)
    
    def update_rate_limit_usage(self, usage_pct: float) -> None:
        """Update rate limit usage and check breaker."""
        self._rate_limit_usage = usage_pct

        if usage_pct >= self.config.rate_limit_hard_pct:
            self._trip(BreakerType.RATE_LIMIT, BreakerState.HARD_TRIPPED)
        elif usage_pct >= self.config.rate_limit_soft_pct:
            self._trip(BreakerType.RATE_LIMIT, BreakerState.SOFT_TRIPPED)
        else:
            self._reset(BreakerType.RATE_LIMIT)

    def record_trade_result(self, won: bool) -> None:
        """
        Record a trade result to track consecutive losses.

        Args:
            won: True if the trade was profitable
        """
        self._last_trade_won = won

        if won:
            self._consecutive_losses = 0
            self._reset(BreakerType.CONSECUTIVE_LOSSES)
        else:
            self._consecutive_losses += 1
            self._check_consecutive_losses()

    def _check_consecutive_losses(self) -> None:
        """Check and update consecutive loss breaker."""
        if self._consecutive_losses >= self.config.consecutive_loss_hard:
            self._trip(BreakerType.CONSECUTIVE_LOSSES, BreakerState.HARD_TRIPPED)
            logger.warning(
                "Consecutive loss limit hit (hard)",
                losses=self._consecutive_losses,
                limit=self.config.consecutive_loss_hard,
            )
        elif self._consecutive_losses >= self.config.consecutive_loss_soft:
            self._trip(BreakerType.CONSECUTIVE_LOSSES, BreakerState.SOFT_TRIPPED)
            logger.warning(
                "Consecutive loss limit hit (soft)",
                losses=self._consecutive_losses,
                limit=self.config.consecutive_loss_soft,
            )

    def check_auto_reset(self) -> list[BreakerType]:
        """
        Check for breakers that should auto-reset based on cooldown.

        Call this periodically (e.g., each iteration) to enable auto-reset.

        Returns:
            List of breakers that were auto-reset
        """
        reset_breakers = []
        now = time.time()

        for breaker, trip_time in list(self._trip_times.items()):
            if breaker == BreakerType.MANUAL:
                continue  # Manual breaker never auto-resets

            state = self._states.get(breaker, BreakerState.CLOSED)
            if state == BreakerState.CLOSED:
                continue

            # Get cooldown for this breaker
            cooldown = self.config.breaker_cooldowns.get(breaker.value, 0)
            if cooldown <= 0:
                # Also check global settings
                if state == BreakerState.SOFT_TRIPPED:
                    cooldown = self.config.soft_reset_seconds
                else:
                    cooldown = self.config.hard_reset_seconds

            if cooldown <= 0:
                continue  # No auto-reset configured

            elapsed = now - trip_time
            if elapsed >= cooldown:
                logger.info(
                    "Auto-resetting circuit breaker",
                    breaker=breaker.value,
                    elapsed_seconds=f"{elapsed:.0f}",
                    cooldown_seconds=f"{cooldown:.0f}",
                )
                self._reset(breaker)
                reset_breakers.append(breaker)

                # Reset related counters
                if breaker == BreakerType.CONSECUTIVE_LOSSES:
                    self._consecutive_losses = 0

        return reset_breakers

    @property
    def consecutive_losses(self) -> int:
        """Get current consecutive loss count."""
        return self._consecutive_losses
    
    def manual_halt(self) -> None:
        """Manually halt all trading."""
        self._manual_halt = True
        self._trip(BreakerType.MANUAL, BreakerState.HARD_TRIPPED)
        logger.warning("Manual trading halt activated")
    
    def manual_resume(self) -> None:
        """Manually resume trading."""
        self._manual_halt = False
        self._reset(BreakerType.MANUAL)
        logger.info("Manual trading resumed")
    
    def _trip(self, breaker: BreakerType, state: BreakerState) -> None:
        """Trip a breaker."""
        old_state = self._states[breaker]
        self._states[breaker] = state
        
        if old_state != state:
            self._trip_times[breaker] = time.time()
            logger.warning(
                "Circuit breaker tripped",
                breaker=breaker.value,
                state=state.value,
            )
    
    def _reset(self, breaker: BreakerType) -> None:
        """Reset a breaker to closed."""
        if self._states[breaker] != BreakerState.CLOSED:
            self._states[breaker] = BreakerState.CLOSED
            self._trip_times.pop(breaker, None)
    
    def can_trade(self) -> bool:
        """Check if new trades are allowed."""
        # Hard trip on any breaker = no trading
        for state in self._states.values():
            if state == BreakerState.HARD_TRIPPED:
                return False
        return True
    
    def can_enter_new_position(self) -> bool:
        """Check if new position entries are allowed."""
        # Any trip = no new entries
        for state in self._states.values():
            if state != BreakerState.CLOSED:
                return False
        return True
    
    def should_flatten(self) -> bool:
        """Check if positions should be flattened."""
        # Hard trip on daily loss or drawdown = flatten
        for breaker in [BreakerType.DAILY_LOSS, BreakerType.DRAWDOWN]:
            if self._states[breaker] == BreakerState.HARD_TRIPPED:
                return True
        return self._manual_halt
    
    def get_status(self, breaker: BreakerType) -> BreakerStatus:
        """Get status of a specific breaker."""
        state = self._states[breaker]
        trip_time = self._trip_times.get(breaker)
        
        if breaker == BreakerType.DAILY_LOSS:
            value = abs(self._daily_pnl) / self.starting_capital if self.starting_capital > 0 else 0
            soft = self.config.daily_loss_soft_pct
            hard = self.config.daily_loss_hard_pct
            msg = f"Daily PnL: ${self._daily_pnl:.2f} ({value:.1%})"
        elif breaker == BreakerType.DRAWDOWN:
            value = (self._peak_capital - self._current_capital) / self._peak_capital if self._peak_capital > 0 else 0
            soft = self.config.max_drawdown_soft_pct
            hard = self.config.max_drawdown_hard_pct
            msg = f"Drawdown: {value:.1%} from peak ${self._peak_capital:.2f}"
        elif breaker == BreakerType.VOLATILITY:
            value = self._current_volatility
            soft = self.config.volatility_soft
            hard = self.config.volatility_hard
            msg = f"Volatility: {value:.0%} annualized"
        elif breaker == BreakerType.ORACLE:
            value = self._oracle_age
            soft = self.config.oracle_stale_soft
            hard = self.config.oracle_stale_hard
            msg = f"Oracle age: {value:.1f}s"
        elif breaker == BreakerType.RATE_LIMIT:
            value = self._rate_limit_usage
            soft = self.config.rate_limit_soft_pct
            hard = self.config.rate_limit_hard_pct
            msg = f"Rate limit usage: {value:.0%}"
        elif breaker == BreakerType.CONSECUTIVE_LOSSES:
            value = float(self._consecutive_losses)
            soft = float(self.config.consecutive_loss_soft)
            hard = float(self.config.consecutive_loss_hard)
            msg = f"Consecutive losses: {self._consecutive_losses}"
        else:
            value = 1.0 if self._manual_halt else 0.0
            soft = hard = 0.5
            msg = "Manual halt" if self._manual_halt else "OK"
        
        return BreakerStatus(
            breaker_type=breaker,
            state=state,
            current_value=value,
            soft_threshold=soft,
            hard_threshold=hard,
            message=msg,
            tripped_at=trip_time,
        )
    
    def get_all_status(self) -> list[BreakerStatus]:
        """Get status of all breakers."""
        return [self.get_status(bt) for bt in BreakerType]
    
    def reset_daily(self) -> None:
        """Reset daily counters (call at start of new day)."""
        self._daily_pnl = 0.0
        self._reset(BreakerType.DAILY_LOSS)
        logger.info("Daily breakers reset")
