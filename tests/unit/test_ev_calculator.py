"""Tests for EV calculator and NO-TRADE gate."""

import pytest

from src.models.ev_calculator import (
    EVCalculator,
    TradeSide,
    EVResult,
)
from src.models.no_trade_gate import (
    NoTradeGate,
    GateConfig,
    TradeContext,
    RejectionReason,
)


class TestFeeCalculation:
    """Tests for dynamic fee calculation."""
    
    @pytest.fixture
    def calc(self):
        return EVCalculator()
    
    def test_fee_peaks_at_50_percent(self, calc):
        """Fee should be highest at 50% probability."""
        fee_50 = calc.calculate_taker_fee_rate(0.50)
        fee_30 = calc.calculate_taker_fee_rate(0.30)
        fee_70 = calc.calculate_taker_fee_rate(0.70)
        
        # 50% should have highest fee
        assert fee_50 > fee_30
        assert fee_50 > fee_70
        
        # Fee at 50% should be approximately 1.56% (0.25 * 0.0625)
        # Actually: 0.25 * (0.5 * 0.5)^2 = 0.25 * 0.0625 = 0.015625
        assert abs(fee_50 - 0.015625) < 0.001
    
    def test_fee_symmetric(self, calc):
        """Fee should be symmetric around 50%."""
        fee_30 = calc.calculate_taker_fee_rate(0.30)
        fee_70 = calc.calculate_taker_fee_rate(0.70)
        
        assert abs(fee_30 - fee_70) < 0.0001
    
    def test_fee_near_zero_at_extremes(self, calc):
        """Fee should approach 0 at extreme probabilities."""
        fee_05 = calc.calculate_taker_fee_rate(0.05)
        fee_95 = calc.calculate_taker_fee_rate(0.95)
        
        assert fee_05 < 0.001  # Less than 0.1%
        assert fee_95 < 0.001


class TestEVCalculation:
    """Tests for Expected Value calculation."""
    
    @pytest.fixture
    def calc(self):
        return EVCalculator(slippage_rate=0.001, adverse_selection_rate=0.005)
    
    def test_positive_ev_buy_yes(self, calc):
        """When model probability > market price, buy YES."""
        result = calc.calculate(
            model_prob=0.60,
            market_price=0.50,
            size_usd=10.0,
        )
        
        assert result.side == TradeSide.BUY_YES
        assert result.gross_edge == pytest.approx(0.10, abs=0.001)
    
    def test_positive_ev_buy_no(self, calc):
        """When model probability < market price, buy NO."""
        result = calc.calculate(
            model_prob=0.40,
            market_price=0.50,
            size_usd=10.0,
        )
        
        assert result.side == TradeSide.BUY_NO
        # Edge on the NO side
        assert result.gross_edge == pytest.approx(0.10, abs=0.001)
    
    def test_no_edge_low_net_ev(self, calc):
        """With no edge, net EV should be negative due to fees."""
        result = calc.calculate(
            model_prob=0.50,
            market_price=0.50,
            size_usd=10.0,
        )
        
        # Gross edge is 0, but fees make net EV negative
        assert result.gross_edge == pytest.approx(0.0, abs=0.001)
        assert result.net_ev < 0
        assert not result.is_positive_ev
    
    def test_minimum_edge_for_profit(self, calc):
        """Should correctly calculate break-even edge."""
        min_edge = calc.minimum_edge_for_profit(0.50)
        
        # Should be fee_rate + slippage + adverse_selection
        fee_at_50 = calc.calculate_taker_fee_rate(0.50)
        expected = fee_at_50 + 0.001 + 0.005
        
        assert min_edge == pytest.approx(expected, abs=0.0001)


class TestNoTradeGate:
    """Tests for NO-TRADE gate."""
    
    @pytest.fixture
    def gate(self):
        config = GateConfig(
            min_edge_threshold=0.04,
            min_edge_to_fee_ratio=1.5,
            min_seconds_to_expiry=60,
            daily_loss_soft_limit_pct=0.03,
        )
        return NoTradeGate(config)
    
    @pytest.fixture
    def calc(self):
        return EVCalculator()
    
    def create_context(
        self,
        calc: EVCalculator,
        model_prob: float = 0.60,
        market_price: float = 0.50,
        seconds_to_expiry: float = 300,
        **kwargs
    ) -> TradeContext:
        """Helper to create trade contexts."""
        ev_result = calc.calculate(
            model_prob=model_prob,
            market_price=market_price,
            size_usd=10.0,
        )
        
        defaults = {
            "ev_result": ev_result,
            "asset": "BTC",
            "token_id": "test_token",
            "seconds_to_expiry": seconds_to_expiry,
            "seconds_since_open": 60,
            "spread": 0.02,
            "book_depth_usd": 500,
            "oracle_age_seconds": 1.0,
            "daily_pnl_pct": 0.0,
            "current_drawdown_pct": 0.0,
            "rate_limit_usage_pct": 0.5,
            "trading_halted": False,
        }
        defaults.update(kwargs)
        return TradeContext(**defaults)
    
    def test_pass_all_checks(self, gate, calc):
        """Should pass when all conditions are met."""
        ctx = self.create_context(calc, model_prob=0.60, market_price=0.50)
        result = gate.evaluate(ctx)
        
        assert result.passed
        assert result.rejection_reason is None
        assert len(result.checks_passed) > 0
    
    def test_reject_negative_ev(self, gate, calc):
        """Should reject negative EV trades."""
        ctx = self.create_context(calc, model_prob=0.50, market_price=0.50)
        result = gate.evaluate(ctx)
        
        assert not result.passed
        assert result.rejection_reason == RejectionReason.NEGATIVE_EV
    
    def test_reject_too_close_to_expiry(self, gate, calc):
        """Should reject trades too close to expiry."""
        ctx = self.create_context(calc, seconds_to_expiry=30)
        result = gate.evaluate(ctx)
        
        assert not result.passed
        assert result.rejection_reason == RejectionReason.TOO_CLOSE_TO_EXPIRY
    
    def test_reject_daily_loss_exceeded(self, gate, calc):
        """Should reject when daily loss limit exceeded."""
        ctx = self.create_context(calc, daily_pnl_pct=-0.05)
        result = gate.evaluate(ctx)
        
        assert not result.passed
        assert result.rejection_reason == RejectionReason.DAILY_LOSS_EXCEEDED
    
    def test_reject_stale_oracle(self, gate, calc):
        """Should reject when oracle is stale."""
        ctx = self.create_context(calc, oracle_age_seconds=15.0)
        result = gate.evaluate(ctx)
        
        assert not result.passed
        assert result.rejection_reason == RejectionReason.ORACLE_STALE
    
    def test_reject_trading_halted(self, gate, calc):
        """Should reject when trading is halted."""
        ctx = self.create_context(calc, trading_halted=True)
        result = gate.evaluate(ctx)
        
        assert not result.passed
        assert result.rejection_reason == RejectionReason.TRADING_HALTED
    
    def test_gate_result_bool(self, gate, calc):
        """GateResult should be truthy when passed."""
        ctx_pass = self.create_context(calc, model_prob=0.60)
        ctx_fail = self.create_context(calc, trading_halted=True)
        
        assert gate.evaluate(ctx_pass)  # Truthy
        assert not gate.evaluate(ctx_fail)  # Falsy
