"""Market normalization helpers.

Keep parsing/business rules out of discovery/networking code.

These functions are pure (no I/O), easy to unit test, and can be reused by
multiple discovery modules.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


def detect_asset(question: str) -> str:
    """Detect which crypto asset the market is for."""
    q = (question or "").upper()

    if "BTC" in q or "BITCOIN" in q:
        return "BTC"
    if "ETH" in q or "ETHEREUM" in q:
        return "ETH"
    if "SOL" in q or "SOLANA" in q:
        return "SOL"
    if "XRP" in q or "RIPPLE" in q:
        return "XRP"

    return ""


def parse_reference_price_from_question(question: str) -> float | None:
    """Extract reference price from market question text.

    Examples:
      - "Will the price of BTC be 103,500 USDT or higher..." -> 103500.0
      - "Will ETH be $3,250 or higher at 2:30 PM?" -> 3250.0
      - "Will SOL price be 175.50 or above..." -> 175.50
    """
    if not question:
        return None

    patterns = [
        r"be\s+\$?([\d]+(?:,\d{3})*(?:\.\d+)?)\s*(?:USDT|USD|or higher|or above)",
        r"price\s+(?:of\s+)?(?:be\s+)?\$?([\d]+(?:,\d{3})*(?:\.\d+)?)",
        r"(?:above|at|over)\s+\$?([\d]+(?:,\d{3})*(?:\.\d+)?)",
        r"\$([\d]+(?:,\d{3})*(?:\.\d+)?)",
        r"be\s+([\d,]+(?:\.\d+)?)\s",
    ]

    for pattern in patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            try:
                price_str = match.group(1).replace(",", "")
                price = float(price_str)
                if price > 0:
                    logger.debug(
                        "Parsed reference price from question",
                        price=price,
                        question=question[:60],
                    )
                    return price
            except (ValueError, IndexError):
                continue

    return None


def is_15m_market(
    question: str,
    raw: dict[str, Any],
    created_at: datetime | None,
    end_date: datetime,
) -> tuple[bool, int | None]:
    """Determine if market is a 15-minute market with strict validation."""

    question_lower = (question or "").lower()
    has_time_pattern = bool(re.search(r"\d{1,2}:\d{2}", question_lower))
    has_direction_word = any(
        word in question_lower
        for word in ["up", "down", "above", "below", "higher", "lower"]
    )

    interval_minutes: int | None = None
    if created_at:
        delta = end_date - created_at
        interval_minutes = int(delta.total_seconds() / 60)

        if 14 <= interval_minutes <= 16:
            if has_time_pattern and has_direction_word:
                logger.debug(
                    "Identified 15m market (strict validation)",
                    question=question[:50] if question else "",
                    interval_minutes=interval_minutes,
                )
                return (True, interval_minutes)

    if "15 min" in question_lower or "15min" in question_lower:
        if created_at is None:
            logger.warning(
                "15m market detected by text only (no created_at)",
                question=question[:50] if question else "",
            )
            return (True, 15)
        elif interval_minutes and not (14 <= interval_minutes <= 16):
            logger.warning(
                "15m text in question but interval mismatch",
                question=question[:50] if question else "",
                interval_minutes=interval_minutes,
            )
            return (False, interval_minutes)

    now = datetime.now(timezone.utc)
    delta_to_expiry = end_date - now
    minutes_to_expiry = delta_to_expiry.total_seconds() / 60

    if 0 < minutes_to_expiry <= 30 and created_at is None:
        logger.warning(
            "15m market detected by expiry heuristic (unreliable)",
            question=question[:50] if question else "",
            minutes_to_expiry=int(minutes_to_expiry),
        )
        return (True, None)

    return (False, interval_minutes)
