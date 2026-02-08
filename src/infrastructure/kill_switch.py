"""
Kill Switch for PolyBawt.

Global emergency stop mechanism that:
- Triggers on critical conditions
- Cancels all open orders
- Disables all strategies
- Persists kill state across restarts

Conditions:
- Manual trigger
- Drawdown limit breach
- Circuit breaker cascade
- API rate limit exhaustion
- Stale data detection

Usage:
    kill_switch = KillSwitch()
    kill_switch.register_strategy(strategy)
    
    if critical_condition:
        await kill_switch.trigger("rate_limit_breach")
"""

from __future__ import annotations
import asyncio
import time
import json
from dataclasses import dataclass, field
from typing import Callable, Optional, Any, List, Awaitable, Dict
from enum import Enum
from pathlib import Path

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class KillReason(Enum):
    """Reasons for kill switch activation."""
    MANUAL = "manual"
    DRAWDOWN = "drawdown"
    CIRCUIT_BREAKER = "circuit_breaker"
    RATE_LIMIT = "rate_limit"
    STALE_DATA = "stale_data"
    API_ERROR = "api_error"
    UNKNOWN = "unknown"


@dataclass
class KillEvent:
    """Record of a kill switch event."""
    reason: KillReason
    message: str
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)


class KillSwitch:
    """
    Global emergency stop mechanism.
    
    When triggered:
    1. Sets global killed flag
    2. Notifies all registered strategies
    3. Cancels all open orders via registered handlers
    4. Persists kill state
    5. Logs the event
    """
    
    def __init__(
        self,
        persist_path: Optional[str] = None,
        auto_recover_after_seconds: Optional[float] = None,
    ):
        self.persist_path = Path(persist_path) if persist_path else None
        self.auto_recover_after = auto_recover_after_seconds
        
        self._killed = False
        self._kill_time = 0.0
        self._kill_reason: Optional[KillReason] = None
        self._kill_message = ""
        
        self._strategies: List[Any] = []  # Objects with stop() method
        self._order_cancel_handlers: List[Callable[[], Awaitable[None]]] = []
        self._event_handlers: List[Callable[[KillEvent], Awaitable[None]]] = []
        
        self._history: List[KillEvent] = []
        self._lock = asyncio.Lock()
        
        # Load persisted state
        if self.persist_path and self.persist_path.exists():
            self._load_state()
    
    @property
    def is_killed(self) -> bool:
        return self._killed
    
    @property
    def kill_reason(self) -> Optional[KillReason]:
        return self._kill_reason
    
    @property
    def time_since_kill(self) -> float:
        if not self._killed:
            return 0.0
        return time.time() - self._kill_time
    
    def register_strategy(self, strategy: Any):
        """Register a strategy to be stopped on kill."""
        self._strategies.append(strategy)
    
    def register_order_cancel_handler(self, handler: Callable[[], Awaitable[None]]):
        """Register a handler to cancel all open orders."""
        self._order_cancel_handlers.append(handler)
    
    def register_event_handler(self, handler: Callable[[KillEvent], Awaitable[None]]):
        """Register a handler for kill events."""
        self._event_handlers.append(handler)
    
    async def trigger(
        self,
        reason: str | KillReason,
        message: str = "",
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Trigger the kill switch.
        
        Args:
            reason: Reason for kill (string or KillReason)
            message: Human-readable message
            details: Additional context
        """
        async with self._lock:
            if self._killed:
                logger.warning(f"[KillSwitch] Already killed, ignoring trigger")
                return
            
            # Parse reason
            if isinstance(reason, str):
                try:
                    kill_reason = KillReason(reason)
                except ValueError:
                    kill_reason = KillReason.UNKNOWN
            else:
                kill_reason = reason
            
            self._killed = True
            self._kill_time = time.time()
            self._kill_reason = kill_reason
            self._kill_message = message
            
            event = KillEvent(
                reason=kill_reason,
                message=message,
                timestamp=self._kill_time,
                details=details or {},
            )
            self._history.append(event)
            
            logger.critical(
                f"[KillSwitch] TRIGGERED: {kill_reason.value} - {message}"
            )
        
        # Execute kill sequence
        await self._execute_kill_sequence(event)
        
        # Persist state
        self._save_state()
    
    async def _execute_kill_sequence(self, event: KillEvent):
        """Execute the kill sequence."""
        # 1. Stop all strategies
        for strategy in self._strategies:
            try:
                if hasattr(strategy, 'stop') and callable(strategy.stop):
                    if asyncio.iscoroutinefunction(strategy.stop):
                        await strategy.stop()
                    else:
                        strategy.stop()
                logger.info(f"[KillSwitch] Stopped strategy: {type(strategy).__name__}")
            except Exception as e:
                logger.error(f"[KillSwitch] Failed to stop strategy: {e}")
        
        # 2. Cancel all orders
        for handler in self._order_cancel_handlers:
            try:
                await handler()
            except Exception as e:
                logger.error(f"[KillSwitch] Order cancel failed: {e}")
        
        # 3. Notify event handlers
        for handler in self._event_handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"[KillSwitch] Event handler failed: {e}")
    
    async def recover(self, force: bool = False) -> bool:
        """
        Attempt to recover from kill state.
        
        Args:
            force: If True, recover regardless of time
            
        Returns:
            True if recovered
        """
        async with self._lock:
            if not self._killed:
                return True
            
            # Check auto-recovery time
            if not force and self.auto_recover_after:
                if self.time_since_kill < self.auto_recover_after:
                    logger.warning(
                        f"[KillSwitch] Cannot recover yet "
                        f"({self.time_since_kill:.0f}s / {self.auto_recover_after:.0f}s)"
                    )
                    return False
            
            self._killed = False
            self._kill_reason = None
            self._kill_message = ""
            
            logger.info("[KillSwitch] Recovered from kill state")
        
        self._save_state()
        return True
    
    def check_and_raise(self):
        """Check if killed and raise exception if so."""
        if self._killed:
            raise SystemExit(
                f"Kill switch active: {self._kill_reason.value if self._kill_reason else 'unknown'}"
            )
    
    async def check_drawdown(
        self,
        current_pnl: float,
        max_drawdown: float,
        threshold: float = -0.1,  # -10%
    ) -> bool:
        """
        Check drawdown and trigger kill if breached.
        
        Args:
            current_pnl: Current PnL
            max_drawdown: Maximum drawdown threshold
            threshold: Fraction of max drawdown to trigger
            
        Returns:
            True if kill was triggered
        """
        if current_pnl <= max_drawdown * threshold:
            await self.trigger(
                KillReason.DRAWDOWN,
                f"Drawdown limit breached: PnL={current_pnl:.2f}",
                {"current_pnl": current_pnl, "threshold": threshold},
            )
            return True
        return False
    
    def _save_state(self):
        """Persist kill state to disk."""
        if not self.persist_path:
            return
        
        state = {
            "killed": self._killed,
            "kill_time": self._kill_time,
            "kill_reason": self._kill_reason.value if self._kill_reason else None,
            "kill_message": self._kill_message,
            "history": [
                {
                    "reason": e.reason.value,
                    "message": e.message,
                    "timestamp": e.timestamp,
                }
                for e in self._history[-10:]  # Keep last 10
            ],
        }
        
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.persist_path, "w") as f:
            json.dump(state, f, indent=2)
    
    def _load_state(self):
        """Load kill state from disk."""
        try:
            with open(self.persist_path, "r") as f:
                state = json.load(f)
            
            if state.get("killed"):
                self._killed = True
                self._kill_time = state.get("kill_time", time.time())
                reason_str = state.get("kill_reason")
                if reason_str:
                    try:
                        self._kill_reason = KillReason(reason_str)
                    except ValueError:
                        self._kill_reason = KillReason.UNKNOWN
                self._kill_message = state.get("kill_message", "")
                
                logger.warning(
                    f"[KillSwitch] Loaded persisted kill state: {self._kill_reason}"
                )
        except Exception as e:
            logger.warning(f"[KillSwitch] Failed to load state: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current kill switch status."""
        return {
            "killed": self._killed,
            "kill_reason": self._kill_reason.value if self._kill_reason else None,
            "kill_message": self._kill_message,
            "time_since_kill": round(self.time_since_kill, 2) if self._killed else 0,
            "history_count": len(self._history),
            "registered_strategies": len(self._strategies),
            "registered_cancel_handlers": len(self._order_cancel_handlers),
        }


# Global kill switch instance
_global_kill_switch: Optional[KillSwitch] = None


def get_kill_switch() -> KillSwitch:
    """Get the global kill switch instance."""
    global _global_kill_switch
    if _global_kill_switch is None:
        _global_kill_switch = KillSwitch()
    return _global_kill_switch


def init_kill_switch(persist_path: Optional[str] = None) -> KillSwitch:
    """Initialize the global kill switch."""
    global _global_kill_switch
    _global_kill_switch = KillSwitch(persist_path=persist_path)
    return _global_kill_switch


async def emergency_stop(reason: str, message: str = ""):
    """Trigger emergency stop on the global kill switch."""
    kill_switch = get_kill_switch()
    await kill_switch.trigger(reason, message)
