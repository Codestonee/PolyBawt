"""
Telegram alerts for critical events.

Sends notifications for:
- Circuit breaker trips
- Large losses
- System errors
- Daily summaries
"""

import asyncio
from dataclasses import dataclass
from enum import Enum

import aiohttp

from src.infrastructure.logging import get_logger
from src.infrastructure.config import get_secrets

logger = get_logger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "ℹ️"
    WARNING = "⚠️"
    CRITICAL = "🚨"
    SUCCESS = "✅"


@dataclass
class Alert:
    """Alert message."""
    level: AlertLevel
    title: str
    message: str
    
    def format(self) -> str:
        return f"{self.level.value} *{self.title}*\n\n{self.message}"


class TelegramAlerter:
    """
    Sends alerts to Telegram.
    
    Usage:
        alerter = TelegramAlerter()
        await alerter.send(Alert(
            level=AlertLevel.CRITICAL,
            title="Circuit Breaker Tripped",
            message="Daily loss limit exceeded: -5.2%"
        ))
    """
    
    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._session: aiohttp.ClientSession | None = None
    
    def _load_config(self) -> bool:
        """Load config from secrets if not provided."""
        if self._bot_token and self._chat_id:
            return True
        
        try:
            secrets = get_secrets()
            self._bot_token = secrets.telegram_bot_token
            self._chat_id = secrets.telegram_chat_id
            return bool(self._bot_token and self._chat_id)
        except Exception:
            return False
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def send(self, alert: Alert) -> bool:
        """
        Send an alert to Telegram.
        
        Args:
            alert: Alert to send
        
        Returns:
            True if sent successfully
        """
        if not self.enabled:
            logger.debug("Telegram alerts disabled", title=alert.title)
            return False
        
        if not self._load_config():
            logger.debug("Telegram not configured")
            return False
        
        try:
            session = await self._get_session()
            url = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
            
            payload = {
                "chat_id": self._chat_id,
                "text": alert.format(),
                "parse_mode": "Markdown",
            }
            
            async with session.post(url, json=payload) as resp:
                if resp.status == 200:
                    logger.debug("Telegram alert sent", title=alert.title)
                    return True
                else:
                    logger.warning(
                        "Telegram send failed",
                        status=resp.status,
                    )
                    return False
                    
        except Exception as e:
            logger.error("Telegram error", error=str(e))
            return False
    
    async def send_circuit_breaker_alert(
        self,
        breaker_type: str,
        state: str,
        value: float,
        threshold: float,
    ) -> None:
        """Send circuit breaker alert."""
        await self.send(Alert(
            level=AlertLevel.CRITICAL if state == "hard" else AlertLevel.WARNING,
            title=f"Circuit Breaker: {breaker_type}",
            message=(
                f"State: {state.upper()}\n"
                f"Current: {value:.2%}\n"
                f"Threshold: {threshold:.2%}"
            ),
        ))
    
    async def send_daily_summary(
        self,
        pnl: float,
        trades: int,
        win_rate: float,
        drawdown: float,
    ) -> None:
        """Send daily performance summary."""
        level = AlertLevel.SUCCESS if pnl >= 0 else AlertLevel.WARNING
        
        await self.send(Alert(
            level=level,
            title="Daily Summary",
            message=(
                f"PnL: ${pnl:+.2f}\n"
                f"Trades: {trades}\n"
                f"Win Rate: {win_rate:.0%}\n"
                f"Drawdown: {drawdown:.1%}"
            ),
        ))
    
    async def send_error(self, error: str, context: str = "") -> None:
        """Send error alert."""
        await self.send(Alert(
            level=AlertLevel.CRITICAL,
            title="System Error",
            message=f"{context}\n\n`{error}`" if context else f"`{error}`",
        ))


# Pre-instantiated alerter
alerter = TelegramAlerter()
