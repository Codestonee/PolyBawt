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

from .config import get_config


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
        log_format: Output format ("json" or "text")
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
    else:
        # Text format for development
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
