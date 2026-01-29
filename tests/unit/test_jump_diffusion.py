"""Tests for Jump-Diffusion pricing model."""

import math
import pytest

from src.models.jump_diffusion import (
    JumpDiffusionModel,
    JumpDiffusionParams,
    BlackScholesModel,
    norm_cdf,
    seconds_to_years,
    minutes_to_years,
)


class TestNormCDF:
    """Tests for normal CDF."""
    
    def test_cdf_at_zero(self):
        """CDF at 0 should be 0.5."""
        assert abs(norm_cdf(0) - 0.5) < 1e-10
    
    def test_cdf_symmetry(self):
        """CDF(-x) + CDF(x) should equal 1."""
        for x in [0.5, 1.0, 2.0, 3.0]:
            assert abs(norm_cdf(-x) + norm_cdf(x) - 1.0) < 1e-10
    
    def test_cdf_extreme_values(self):
        """CDF should approach 0 and 1 at extremes."""
        assert norm_cdf(-5) < 0.001
        assert norm_cdf(5) > 0.999


class TestJumpDiffusionModel:
    """Tests for Jump-Diffusion model."""
    
    @pytest.fixture
    def model(self):
        return JumpDiffusionModel()
    
    @pytest.fixture
    def params(self):
        return JumpDiffusionParams()
    
    def test_at_money_is_near_half(self, model, params):
        """ATM option should have ~50% probability."""
        prob = model.prob_up(
            spot=100000,
            initial=100000,  # ATM
            time_years=minutes_to_years(15),
            params=params,
        )
        # Should be close to 0.5, but slightly less due to drift adjustment
        assert 0.40 < prob < 0.60
    
    def test_deep_itm_near_one(self, model, params):
        """Deep ITM should have probability near 1."""
        prob = model.prob_up(
            spot=110000,  # 10% above strike
            initial=100000,
            time_years=minutes_to_years(15),
            params=params,
        )
        assert prob > 0.90
    
    def test_deep_otm_near_zero(self, model, params):
        """Deep OTM should have probability near 0."""
        prob = model.prob_up(
            spot=90000,  # 10% below strike
            initial=100000,
            time_years=minutes_to_years(15),
            params=params,
        )
        assert prob < 0.10
    
    def test_zero_time_deterministic(self, model, params):
        """At expiry, outcome is deterministic."""
        # At expiry, above strike = 100%
        assert model.prob_up(spot=100001, initial=100000, time_years=0, params=params) == 1.0
        # At expiry, at strike = 100%
        assert model.prob_up(spot=100000, initial=100000, time_years=0, params=params) == 1.0
        # At expiry, below strike = 0%
        assert model.prob_up(spot=99999, initial=100000, time_years=0, params=params) == 0.0
    
    def test_prob_down_complement(self, model, params):
        """P(down) = 1 - P(up)."""
        prob_up = model.prob_up(
            spot=100000,
            initial=99500,
            time_years=minutes_to_years(15),
            params=params,
        )
        prob_down = model.prob_down(
            spot=100000,
            initial=99500,
            time_years=minutes_to_years(15),
            params=params,
        )
        assert abs(prob_up + prob_down - 1.0) < 1e-10
    
    def test_higher_vol_widens_distribution(self, model):
        """Higher volatility should pull probability toward 0.5."""
        low_vol = JumpDiffusionParams(sigma=0.30)
        high_vol = JumpDiffusionParams(sigma=0.90)
        
        # For an ITM option, higher vol should lower probability
        prob_low = model.prob_up(
            spot=102000,
            initial=100000,
            time_years=minutes_to_years(15),
            params=low_vol,
        )
        prob_high = model.prob_up(
            spot=102000,
            initial=100000,
            time_years=minutes_to_years(15),
            params=high_vol,
        )
        
        # Higher vol should pull probability toward 0.5
        assert abs(prob_high - 0.5) < abs(prob_low - 0.5)
    
    def test_asset_specific_params(self):
        """Asset-specific parameters should have different volatilities."""
        btc_params = JumpDiffusionParams.for_asset("BTC")
        sol_params = JumpDiffusionParams.for_asset("SOL")
        
        assert btc_params.sigma == 0.60
        assert sol_params.sigma == 0.85


class TestBlackScholesModel:
    """Tests for Black-Scholes fallback model."""
    
    @pytest.fixture
    def model(self):
        return BlackScholesModel()
    
    def test_atm_near_half(self, model):
        """ATM should be near 0.5."""
        prob = model.prob_up(
            spot=100000,
            initial=100000,
            time_years=minutes_to_years(15),
            sigma=0.60,
        )
        assert 0.40 < prob < 0.60
    
    def test_monotonic_with_spot(self, model):
        """Probability should increase monotonically with spot price."""
        probs = []
        for spot in [95000, 97500, 100000, 102500, 105000]:
            prob = model.prob_up(
                spot=spot,
                initial=100000,
                time_years=minutes_to_years(15),
                sigma=0.60,
            )
            probs.append(prob)
        
        # Should be strictly increasing
        for i in range(len(probs) - 1):
            assert probs[i] < probs[i + 1]


class TestTimeConversions:
    """Tests for time conversion utilities."""
    
    def test_minutes_to_years(self):
        """15 minutes should convert correctly."""
        t = minutes_to_years(15)
        # 15 / (365 * 24 * 60) ≈ 2.85e-5
        expected = 15 / (365 * 24 * 60)
        assert abs(t - expected) < 1e-10
    
    def test_seconds_to_years(self):
        """900 seconds = 15 minutes."""
        t = seconds_to_years(900)
        expected = 900 / (365 * 24 * 60 * 60)
        assert abs(t - expected) < 1e-10
