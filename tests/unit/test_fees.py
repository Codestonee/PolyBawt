from src.risk.fees import polymarket_fee_rate, estimated_leg_fee_usd


def test_fee_rate_bounds_and_shape():
    assert polymarket_fee_rate(0.001) > 0
    assert polymarket_fee_rate(0.5) > polymarket_fee_rate(0.1)
    assert abs(polymarket_fee_rate(0.2) - polymarket_fee_rate(0.8)) < 1e-12


def test_estimated_leg_fee_usd_non_negative():
    assert estimated_leg_fee_usd(0.5, 10.0) >= 0.0
    assert estimated_leg_fee_usd(0.5, -5.0) == 0.0
