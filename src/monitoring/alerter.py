"""
Alerting System for PolyBawt.

Multi-channel alerting with:
- Telegram bot integration
- Discord webhook support
- Rate limiting to prevent spam
- Priority-based routing
- Alert history and deduplication

Usage:
    alerter = Alerter()
    await alerter.send_alert(
        AlertLevel.WARNING,
        "Drawdown approaching limit",
        details={"current_pnl": -150, "limit": -200}
    )
"""

from __future__ import annotations
import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Awaitable
from enum import Enum
from datetime import datetime, timedelta
import aiohttp

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Available alert channels."""
    TELEGRAM = "telegram"
    DISCORD = "discord"
    CONSOLE = "console"


@dataclass
class AlertConfig:
    """Configuration for alerting system."""
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    discord_webhook_url: Optional[str] = None
    
    # Rate limiting
    min_interval_seconds: float = 60.0  # Min time between same alerts
    max_alerts_per_minute: int = 10
    
    # Routing: which levels go to which channels
    level_routing: Dict[AlertLevel, List[AlertChannel]] = field(default_factory=lambda: {
        AlertLevel.DEBUG: [AlertChannel.CONSOLE],
        AlertLevel.INFO: [AlertChannel.CONSOLE],
        AlertLevel.WARNING: [AlertChannel.CONSOLE, AlertChannel.TELEGRAM],
        AlertLevel.ERROR: [AlertChannel.CONSOLE, AlertChannel.TELEGRAM, AlertChannel.DISCORD],
        AlertLevel.CRITICAL: [AlertChannel.CONSOLE, AlertChannel.TELEGRAM, AlertChannel.DISCORD],
    })


@dataclass
class Alert:
    """An alert to be sent."""
    level: AlertLevel
    title: str
    message: str
    details: Dict[str, Any]
    timestamp: float
    alert_id: str
    
    def to_telegram_message(self) -> str:
        """Format for Telegram."""
        emoji = {
            AlertLevel.DEBUG: "🔍",
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "❌",
            AlertLevel.CRITICAL: "🚨",
        }.get(self.level, "📢")
        
        lines = [
            f"{emoji} *{self.level.value.upper()}*: {self.title}",
            "",
            self.message,
        ]
        
        if self.details:
            lines.append("")
            lines.append("```")
            for k, v in self.details.items():
                lines.append(f"{k}: {v}")
            lines.append("```")
        
        lines.append(f"\n_Time: {datetime.fromtimestamp(self.timestamp).strftime('%H:%M:%S')}_")
        return "\n".join(lines)
    
    def to_discord_embed(self) -> Dict:
        """Format for Discord webhook."""
        color = {
            AlertLevel.DEBUG: 0x808080,
            AlertLevel.INFO: 0x3498db,
            AlertLevel.WARNING: 0xf39c12,
            AlertLevel.ERROR: 0xe74c3c,
            AlertLevel.CRITICAL: 0x9b59b6,
        }.get(self.level, 0x95a5a6)
        
        embed = {
            "title": f"{self.level.value.upper()}: {self.title}",
            "description": self.message,
            "color": color,
            "timestamp": datetime.fromtimestamp(self.timestamp).isoformat(),
            "footer": {"text": "PolyBawt Alert System"},
        }
        
        if self.details:
            embed["fields"] = [
                {"name": k, "value": str(v), "inline": True}
                for k, v in list(self.details.items())[:10]  # Discord limit
            ]
        
        return embed


class Alerter:
    """
    Multi-channel alerting system.
    
    Sends alerts to configured channels with rate limiting
    and deduplication.
    """
    
    def __init__(self, config: Optional[AlertConfig] = None):
        self.config = config or AlertConfig()
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Rate limiting
        self._alert_history: Dict[str, float] = {}  # alert_id -> last_sent_time
        self._minute_window: List[float] = []  # timestamps of recent alerts
        
        # Stats
        self._total_sent = 0
        self._total_suppressed = 0
        self._by_level: Dict[AlertLevel, int] = {level: 0 for level in AlertLevel}
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    def _generate_alert_id(self, level: AlertLevel, title: str, message: str) -> str:
        """Generate unique ID for deduplication."""
        content = f"{level.value}:{title}:{message}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _should_send(self, alert_id: str) -> bool:
        """Check rate limits and deduplication."""
        now = time.time()
        
        # Check per-alert rate limit
        if alert_id in self._alert_history:
            elapsed = now - self._alert_history[alert_id]
            if elapsed < self.config.min_interval_seconds:
                return False
        
        # Check global rate limit
        cutoff = now - 60.0
        self._minute_window = [t for t in self._minute_window if t > cutoff]
        if len(self._minute_window) >= self.config.max_alerts_per_minute:
            return False
        
        return True
    
    async def send_alert(
        self,
        level: AlertLevel,
        title: str,
        message: str = "",
        details: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> bool:
        """
        Send an alert to configured channels.
        
        Args:
            level: Alert severity
            title: Short title
            message: Detailed message
            details: Additional structured data
            force: Bypass rate limiting
            
        Returns:
            True if alert was sent
        """
        alert_id = self._generate_alert_id(level, title, message)
        
        # Check rate limits
        if not force and not self._should_send(alert_id):
            self._total_suppressed += 1
            return False
        
        # Create alert
        alert = Alert(
            level=level,
            title=title,
            message=message,
            details=details or {},
            timestamp=time.time(),
            alert_id=alert_id,
        )
        
        # Get channels for this level
        channels = self.config.level_routing.get(level, [AlertChannel.CONSOLE])
        
        # Send to each channel
        sent_any = False
        for channel in channels:
            try:
                if channel == AlertChannel.CONSOLE:
                    self._send_console(alert)
                    sent_any = True
                elif channel == AlertChannel.TELEGRAM:
                    if await self._send_telegram(alert):
                        sent_any = True
                elif channel == AlertChannel.DISCORD:
                    if await self._send_discord(alert):
                        sent_any = True
            except Exception as e:
                logger.error(f"[Alerter] Failed to send to {channel.value}: {e}")
        
        if sent_any:
            self._alert_history[alert_id] = time.time()
            self._minute_window.append(time.time())
            self._total_sent += 1
            self._by_level[level] += 1
        
        return sent_any
    
    def _send_console(self, alert: Alert):
        """Send to console/log."""
        log_method = {
            AlertLevel.DEBUG: logger.debug,
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.ERROR: logger.error,
            AlertLevel.CRITICAL: logger.critical,
        }.get(alert.level, logger.info)
        
        log_method(f"[ALERT] {alert.title}: {alert.message}", **alert.details)
    
    async def _send_telegram(self, alert: Alert) -> bool:
        """Send to Telegram."""
        if not self.config.telegram_bot_token or not self.config.telegram_chat_id:
            return False
        
        session = await self._get_session()
        url = f"https://api.telegram.org/bot{self.config.telegram_bot_token}/sendMessage"
        
        try:
            async with session.post(url, json={
                "chat_id": self.config.telegram_chat_id,
                "text": alert.to_telegram_message(),
                "parse_mode": "Markdown",
            }) as resp:
                if resp.status == 200:
                    return True
                else:
                    logger.error(f"[Alerter] Telegram error: {resp.status}")
                    return False
        except Exception as e:
            logger.error(f"[Alerter] Telegram exception: {e}")
            return False
    
    async def _send_discord(self, alert: Alert) -> bool:
        """Send to Discord webhook."""
        if not self.config.discord_webhook_url:
            return False
        
        session = await self._get_session()
        
        try:
            async with session.post(
                self.config.discord_webhook_url,
                json={"embeds": [alert.to_discord_embed()]},
            ) as resp:
                if resp.status in (200, 204):
                    return True
                else:
                    logger.error(f"[Alerter] Discord error: {resp.status}")
                    return False
        except Exception as e:
            logger.error(f"[Alerter] Discord exception: {e}")
            return False
    
    async def close(self):
        """Close HTTP session."""
        if self._session:
            await self._session.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get alerting statistics."""
        return {
            "total_sent": self._total_sent,
            "total_suppressed": self._total_suppressed,
            "by_level": {level.value: count for level, count in self._by_level.items()},
            "alerts_in_last_minute": len(self._minute_window),
        }


# Convenience functions
_global_alerter: Optional[Alerter] = None


def init_alerter(
    telegram_token: Optional[str] = None,
    telegram_chat_id: Optional[str] = None,
    discord_webhook: Optional[str] = None,
) -> Alerter:
    """Initialize the global alerter."""
    global _global_alerter
    config = AlertConfig(
        telegram_bot_token=telegram_token,
        telegram_chat_id=telegram_chat_id,
        discord_webhook_url=discord_webhook,
    )
    _global_alerter = Alerter(config)
    return _global_alerter


def get_alerter() -> Alerter:
    """Get the global alerter."""
    global _global_alerter
    if _global_alerter is None:
        _global_alerter = Alerter()
    return _global_alerter


async def alert(
    level: AlertLevel | str,
    title: str,
    message: str = "",
    **details
):
    """Send an alert using the global alerter."""
    if isinstance(level, str):
        level = AlertLevel(level)
    alerter = get_alerter()
    await alerter.send_alert(level, title, message, details)


# Pre-defined alert functions
async def alert_info(title: str, message: str = "", **details):
    await alert(AlertLevel.INFO, title, message, **details)

async def alert_warning(title: str, message: str = "", **details):
    await alert(AlertLevel.WARNING, title, message, **details)

async def alert_error(title: str, message: str = "", **details):
    await alert(AlertLevel.ERROR, title, message, **details)

async def alert_critical(title: str, message: str = "", **details):
    await alert(AlertLevel.CRITICAL, title, message, **details)
