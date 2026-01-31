import pytest
from datetime import datetime, timezone

from src.ingestion.market_normalization import (
    detect_asset,
    parse_reference_price_from_question,
    is_15m_market,
)


def test_detect_asset_basic():
    assert detect_asset("Will BTC be up?") == "BTC"
    assert detect_asset("Ethereum price") == "ETH"
    assert detect_asset("Solana") == "SOL"
    assert detect_asset("Ripple XRP") == "XRP"
    assert detect_asset("Unknown") == ""


@pytest.mark.parametrize(
    "q,expected",
    [
        ("Will the price of BTC be 103,500 USDT or higher?", 103500.0),
        ("Will ETH be $3,250 or higher at 2:30 PM?", 3250.0),
        ("Will SOL price be 175.50 or above?", 175.50),
        ("Will BTC be above $103,500 at 12:00?", 103500.0),
    ],
)
def test_parse_reference_price_from_question(q, expected):
    assert parse_reference_price_from_question(q) == expected


def test_is_15m_market_strict_interval():
    # created_at -> end_date gives 15m interval and question has time + direction
    created_at = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    end_date = datetime(2026, 1, 1, 12, 15, 0, tzinfo=timezone.utc)
    q = "Will BTC be up at 12:15 UTC?"

    ok, interval = is_15m_market(q, {}, created_at, end_date)
    assert ok is True
    assert interval in (15, 14, 16)


def test_is_15m_market_text_only_without_created_at():
    created_at = None
    end_date = datetime(2026, 1, 1, 12, 15, 0, tzinfo=timezone.utc)
    q = "BTC 15min market"

    ok, interval = is_15m_market(q, {}, created_at, end_date)
    assert ok is True
    assert interval == 15
