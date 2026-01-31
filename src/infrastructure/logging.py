"""
Structured logging configuration.

Provides:
- JSON logging for production (easy to aggregate)
- Text logging for development (human readable)
- Context injection for tracing
"""

import logging
import sys
from datetime import datetime, timezone
from typing import Any

import structlog
from structlog.types import EventDict, Processor
from structlog import DropEvent

from .config import get_config


# EVENTS TO SILENCE IN CLEAN MODE
NOISE_EVENTS = [
    "Found 15m market via slug", 
    "Binance price",
    "Using spot price as initial",
    "Order book analyzed",
    "Imbalance below threshold",
    "Regime detected",
    "Ensemble probability calculated",
    "Ensemble model used",
    "Sentiment confidence too low",
    "Position sizing",
    "Signal rejected",
    "Found 15m crypto markets",
    "Adaptive Kelly reduction applied",
    "Order book signal",
    "iteration",
    "Rapid price movement detected",
    "Toxic order imbalance detected",
    "Wide spread detected",
    "Thin order book detected",
    "High VPIN detected",
]


def filter_noise(
    logger: logging.Logger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Filter out noise events."""
    if method_name == "debug":
        raise DropEvent

    event = event_dict.get("event", "")
    for noise in NOISE_EVENTS:
        if noise in event:
            raise DropEvent
    return event_dict


def add_timestamp(
    logger: logging.Logger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Add ISO timestamp to log entry."""
    event_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
    return event_dict


def add_log_level(
    logger: logging.Logger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Add log level to event dict."""
    event_dict["level"] = method_name.upper()
    return event_dict


def censor_secrets(
    logger: logging.Logger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Remove sensitive data from logs."""
    sensitive_keys = {
        "private_key", "api_key", "api_secret", "passphrase", 
        "password", "token", "secret"
    }
    
    def _censor(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {
                k: "[REDACTED]" if any(s in k.lower() for s in sensitive_keys) else _censor(v)
                for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [_censor(item) for item in obj]
        return obj
    
    return _censor(event_dict)


def configure_logging(log_level: str = "INFO", log_format: str = "json") -> None:
    """
    Configure structured logging.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Output format ("json", "text", or "clean")
    """
    # Shared processors
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        add_timestamp,
        add_log_level,
        censor_secrets,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    if log_format == "json":
        # JSON format for production
        processors = shared_processors + [
            structlog.processors.JSONRenderer()
        ]
    elif log_format == "clean":
        # Clean format for terminal - minimal, readable output
        processors = shared_processors + [
            filter_noise, 
            CleanConsoleRenderer()
        ]
    else:
        # Text format for development (verbose)
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True)
        ]
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(log_level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )
    
    # Also configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.getLevelName(log_level),
    )
    
    # Silence noisy HTTP libraries
    for noisy_logger in [
        "httpx", "httpcore", "httpcore.http2", "httpcore.http11",
        "hpack", "hpack.hpack", "urllib3", "aiohttp", 
        "aiohttp.client", "websockets", "h2",
    ]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    
    # In clean mode, also silence verbose internal loggers
    if log_format == "clean":
        for verbose_logger in [
            "src.models.regime_detector",
            "src.models.advanced_pricing",
            "src.ingestion.oracle_feed",
        ]:
            logging.getLogger(verbose_logger).setLevel(logging.WARNING)


class CleanConsoleRenderer:
    """
    Ultra-simplified console renderer for human readability.
    
    Transforms technical log events into friendly, emoji-coded one-liners.
    Aggressively filters noise.
    """
    
    # Emoji mappings for specific events
    EMOJIS = {
        "startup": "🚀",
        "shutdown": "🛑",
        "market_found": "🔎", # Usually suppressed unless new
        "opportunity": "🎯",
        "trade_placed": "💸",
        "trade_filled": "✅",
        "trade_skipped": "🛡️",  # Skipped due to gate/risk
        "trade_error": "❌",
        "arbitrage": "⚡",
        "arb_executed": "💎",  # Arbitrage trade executed
        "favorite": "🎲",  # Favorite fallback bet
        "toxicity": "☣️",
        "profit": "💰",
        "loss": "📉",
        "api": "🌐",
    }
    
    def __call__(
        self,
        logger: logging.Logger,
        method_name: str,
        event_dict: EventDict,
    ) -> str:
        """Render log entry as simplified human text."""
        level = event_dict.get("level", "INFO").upper()
        event = event_dict.get("event", "")
        
        # 3. Transform specific technical messages into friendly ones
        message = ""
        emoji = "ℹ️" # Default
        
        # --- Market / Opportunity ---
        if "Trade opportunity" in event:
            emoji = self.EMOJIS["opportunity"]
            asset = event_dict.get("asset", "Unknown")
            side = event_dict.get("side", "").replace("buy_", "").upper()
            edge = event_dict.get("edge", 0)
            if isinstance(edge, (float, str)):
                 try:
                     edge_str = f"{float(edge):.1%}"
                 except:
                     edge_str = str(edge)
            else:
                edge_str = "?"
            
            size = event_dict.get("kelly_size", 0)
            has_size = False
            try:
                if size and float(str(size).replace("$", "")) > 0:
                    has_size = True
            except (ValueError, TypeError):
                pass
            
            if has_size:
                message = f"{asset} {side} opportunity found! Edge: {edge_str} | Bet Size: ${size}"
            else:
                message = f"{asset} {side} opportunity found! Edge: {edge_str}"

        elif "Order placed" in event:
            emoji = self.EMOJIS["trade_placed"]
            asset = event_dict.get("asset", "")
            side = event_dict.get("side", "").upper()
            size = event_dict.get("size", "")
            price = event_dict.get("price", "")
            message = f"Placed order: {asset} {side} | ${size} @ {price}"

        elif "Fill processed" in event:
            emoji = self.EMOJIS["trade_filled"]
            asset = event_dict.get("asset", "")
            profit = event_dict.get("realized_pnl", 0)
            message = f"Trade Filled! {asset}"

        elif "Strategy starting" in event or "Trading Bot starting" in event:
            emoji = self.EMOJIS["startup"]
            message = "Bot is starting up... (Waiting for markets)"

        elif "API Server starting" in event:
            emoji = self.EMOJIS["api"]
            message = "Dashboard API is online"

        elif "ARBITRAGE" in event:
            emoji = self.EMOJIS["arbitrage"]
            asset = event_dict.get("asset", "")
            profit = event_dict.get("profit_pct", "")
            message = f"Arbitrage detected on {asset}! Risk-free profit: {profit}"

        elif "Arbitrage executed" in event:
            emoji = self.EMOJIS["arb_executed"]
            asset = event_dict.get("asset", "")
            profit = event_dict.get("profit_pct", "")
            size = event_dict.get("size", "")
            message = f"Arbitrage EXECUTED on {asset}! Profit: {profit} | Size: {size}"

        elif "Favorite fallback" in event:
            emoji = self.EMOJIS["favorite"]
            asset = event_dict.get("asset", "")
            side = event_dict.get("side", "")
            size = event_dict.get("size", "")
            message = f"Favorite bet placed: {asset} {side} | {size}"

        elif "Toxic" in event or "toxicity" in event:
            emoji = self.EMOJIS["toxicity"]
            reason = event_dict.get("reason", "")
            asset = event_dict.get("asset", "")
            
            if "Reduced" in event or "reducing" in event:
                 action = "Reducing size on"
            else:
                 action = "Skipping"

            if reason:
                message = f"{action} {asset} trade due to toxic flow ({reason})"
            else:
                message = f"{action} {asset} trade due to toxic flow"

        elif "Skipping trade" in event:
             emoji = self.EMOJIS["trade_skipped"]
             reason = event_dict.get("reason", "")
             asset = event_dict.get("asset", "")
             message = f"Skipping {asset}: {reason}"

        elif level == "ERROR" or "error" in event.lower():
            emoji = self.EMOJIS["trade_error"]
            error_msg = event_dict.get("error", event)
            message = f"Error: {error_msg}"
        
        # If we didn't match a specific friendly format, but it's important (INFO/WARN), show it simply
        elif not message:
             # Fallback for generic info
             emoji = "📝" if level == "INFO" else "⚠️"
             message = event

        # Timestamp (just HH:MM:SS)
        time_str = datetime.now().strftime("%H:%M:%S")
        
        # Final Format: [12:00:00] 🚀 Message
        return f"\033[90m[{time_str}]\033[0m {emoji} {message}"


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Bound logger instance
    """
    return structlog.get_logger(name)


class LogContext:
    """Context manager for adding temporary context to logs."""
    
    def __init__(self, **context: Any):
        self.context = context
        self._token = None
    
    def __enter__(self) -> "LogContext":
        self._token = structlog.contextvars.bind_contextvars(**self.context)
        return self
    
    def __exit__(self, *args: Any) -> None:
        if self._token:
            structlog.contextvars.unbind_contextvars(*self.context.keys())


def bind_context(**context: Any) -> None:
    """Bind context variables for the current async context."""
    structlog.contextvars.bind_contextvars(**context)


def unbind_context(*keys: str) -> None:
    """Unbind context variables."""
    structlog.contextvars.unbind_contextvars(*keys)


def clear_context() -> None:
    """Clear all context variables."""
    structlog.contextvars.clear_contextvars()
