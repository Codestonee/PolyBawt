"""
Emergency kill switch for immediate trading halt.

Monitors for external signals to immediately stop all trading activity.
Can be triggered by:
- File-based kill switch (/tmp/kill_trading, .kill_trading)
- API endpoint (via admin API)
- Telegram command
- Automated risk triggers

Usage:
    # In main strategy loop
    if await kill_switch.is_triggered():
        await emergency_halt()
    
    # Trigger manually
    await kill_switch.trigger("Manual halt by operator")
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Any

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class KillSwitchType(Enum):
    """Types of kill switch triggers."""
    FILE = "file"           # File-based trigger
    API = "api"             # API-triggered
    TELEGRAM = "telegram"   # Telegram command
    AUTO = "auto"           # Automated risk trigger
    MANUAL = "manual"       # Manual code trigger


@dataclass
class KillSwitchEvent:
    """Record of a kill switch trigger."""
    triggered_at: datetime
    switch_type: KillSwitchType
    reason: str
    triggered_by: str = "system"
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "triggered_at": self.triggered_at.isoformat(),
            "type": self.switch_type.value,
            "reason": self.reason,
            "triggered_by": self.triggered_by,
        }


class KillSwitch:
    """
    Emergency kill switch for trading halt.
    
    Monitors multiple trigger sources and provides immediate
    halt capability with audit logging.
    
    Usage:
        kill_switch = KillSwitch()
        
        # Start monitoring
        await kill_switch.start_monitoring()
        
        # In trading loop
        if await kill_switch.check():
            logger.critical("Kill switch active - halting")
            break
        
        # Trigger manually
        await kill_switch.trigger("Risk limit exceeded", KillSwitchType.AUTO)
    """
    
    # File paths that trigger kill switch
    KILL_FILES = [
        ".kill_trading",
        "/tmp/kill_trading",
        "data/kill_trading",
    ]
    
    def __init__(
        self,
        check_interval_seconds: float = 5.0,
        auto_trigger_on_error: bool = True,
    ):
        self.check_interval = check_interval_seconds
        self.auto_trigger_on_error = auto_trigger_on_error
        
        self._triggered = False
        self._trigger_event: KillSwitchEvent | None = None
        self._history: list[KillSwitchEvent] = []
        self._callbacks: list[Callable[[KillSwitchEvent], Any]] = []
        self._monitoring = False
        self._monitor_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
    
    async def start_monitoring(self) -> None:
        """Start background monitoring for kill signals."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Kill switch monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Kill switch monitoring stopped")
    
    async def _monitor_loop(self) -> None:
        """Background loop to check for kill signals."""
        while self._monitoring:
            try:
                await self._check_file_triggers()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error("Error in kill switch monitor", error=str(e))
                if self.auto_trigger_on_error:
                    await self.trigger(
                        f"Monitor error: {e}",
                        KillSwitchType.AUTO,
                    )
                await asyncio.sleep(1.0)
    
    async def _check_file_triggers(self) -> None:
        """Check for file-based kill triggers."""
        for filepath in self.KILL_FILES:
            path = Path(filepath)
            if path.exists():
                # Read reason from file if present
                try:
                    reason = path.read_text().strip() or "File-based kill switch"
                except Exception:
                    reason = "File-based kill switch"
                
                await self.trigger(reason, KillSwitchType.FILE)
                
                # Remove file to prevent re-trigger
                try:
                    path.unlink()
                except Exception:
                    pass
                break
    
    async def check(self) -> bool:
        """
        Check if kill switch is triggered.
        
        Returns:
            True if kill switch is active
        """
        # Also check files synchronously
        await self._check_file_triggers()
        return self._triggered
    
    async def trigger(
        self,
        reason: str,
        switch_type: KillSwitchType = KillSwitchType.MANUAL,
        triggered_by: str = "system",
    ) -> None:
        """
        Trigger the kill switch.
        
        Args:
            reason: Why the kill switch was triggered
            switch_type: Type of trigger
            triggered_by: Who/what triggered it
        """
        async with self._lock:
            if self._triggered:
                return  # Already triggered
            
            self._triggered = True
            self._trigger_event = KillSwitchEvent(
                triggered_at=datetime.now(timezone.utc),
                switch_type=switch_type,
                reason=reason,
                triggered_by=triggered_by,
            )
            self._history.append(self._trigger_event)
        
        logger.critical(
            "🔴 KILL SWITCH TRIGGERED",
            reason=reason,
            type=switch_type.value,
            triggered_by=triggered_by,
        )
        
        # Execute callbacks
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(self._trigger_event)
                else:
                    callback(self._trigger_event)
            except Exception as e:
                logger.error("Kill switch callback failed", error=str(e))
    
    def register_callback(self, callback: Callable[[KillSwitchEvent], Any]) -> None:
        """
        Register a callback to execute when kill switch triggers.
        
        Args:
            callback: Function to call on trigger
        """
        self._callbacks.append(callback)
    
    async def reset(self, reset_by: str = "system") -> bool:
        """
        Reset the kill switch (requires explicit authorization).
        
        Args:
            reset_by: Who is resetting the switch
            
        Returns:
            True if reset was successful
        """
        async with self._lock:
            if not self._triggered:
                return False
            
            logger.warning(
                "🟢 Kill switch reset",
                previously_triggered=self._trigger_event.reason if self._trigger_event else None,
                reset_by=reset_by,
            )
            
            self._triggered = False
            self._trigger_event = None
            return True
    
    @property
    def is_triggered(self) -> bool:
        """Whether kill switch is currently triggered."""
        return self._triggered
    
    @property
    def trigger_reason(self) -> str | None:
        """Reason for current trigger, if any."""
        return self._trigger_event.reason if self._trigger_event else None
    
    def get_history(self, limit: int = 10) -> list[KillSwitchEvent]:
        """Get history of kill switch triggers."""
        return self._history[-limit:]


# Global instance
kill_switch = KillSwitch()
