from datetime import datetime, timezone, timedelta

from src.ingestion.event_normalization import is_tradeable_event_market, hours_to_resolution


def test_is_tradeable_event_market_thresholds():
    assert is_tradeable_event_market(
        volume_24h=1000,
        liquidity=500,
        min_volume_24h=1000,
        min_liquidity=500,
    )
    assert not is_tradeable_event_market(
        volume_24h=999,
        liquidity=500,
        min_volume_24h=1000,
        min_liquidity=500,
    )
    assert not is_tradeable_event_market(
        volume_24h=1000,
        liquidity=499,
        min_volume_24h=1000,
        min_liquidity=500,
    )


def test_hours_to_resolution():
    now = datetime.now(timezone.utc)

    # None -> None
    assert hours_to_resolution(None) is None

    end = now + timedelta(hours=2)
    h = hours_to_resolution(end)
    assert h is not None
    assert 1.9 <= h <= 2.1
