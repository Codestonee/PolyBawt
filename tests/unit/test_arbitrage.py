"""
Unit tests for Arbitrage Detection module.
Tests the research-based risk-free profit detection from arXiv:2510.15205.
"""

import pytest
from src.strategy.arbitrage_detector import (
    ArbitrageDetector,
    ArbitrageType,
    ArbitrageOpportunity,
)


class TestArbitrageDetector:
    """Test arbitrage detection logic."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = ArbitrageDetector(
            fee_rate=0.02,  # 2%
            min_profit_threshold=0.005,  # 0.5%
            slippage_buffer=0.002,  # 0.2%
        )
    
    def test_no_arbitrage_balanced_market(self):
        """Test: YES=0.50, NO=0.50 -> no arbitrage."""
        result = self.detector.check_binary_market(0.50, 0.50)
        
        assert result.arb_type == ArbitrageType.NONE
        assert result.is_profitable is False
        assert result.total == 1.0
    
    def test_no_arbitrage_within_fees(self):
        """Test: Small deviation eaten by fees -> no arbitrage."""
        # YES=0.48, NO=0.50 -> total=0.98, gross profit=2%
        # But fees (4%) + slippage (0.2%) = 4.2% > 2%
        result = self.detector.check_binary_market(0.48, 0.50)
        
        assert result.arb_type == ArbitrageType.NONE
        assert result.is_profitable is False
    
    def test_long_arbitrage_detected(self):
        """Test: YES + NO < 1 (minus fees) -> long arbitrage."""
        # YES=0.45, NO=0.45 -> total=0.90, gross profit=10%
        # Net profit = 10% - 4.2% = 5.8%
        result = self.detector.check_binary_market(0.45, 0.45)
        
        assert result.arb_type == ArbitrageType.LONG_REBALANCING
        assert result.is_profitable is True
        assert result.gross_profit_pct == pytest.approx(0.10, abs=0.001)
        assert result.net_profit_pct > 0.05  # After fees
        assert result.recommended_action == "BUY_YES_AND_NO"
    
    def test_short_arbitrage_detected(self):
        """Test: YES + NO > 1 (plus fees) -> short arbitrage."""
        # YES=0.55, NO=0.55 -> total=1.10, gross profit=10%
        # Net profit = 10% - 4.2% = 5.8%
        result = self.detector.check_binary_market(0.55, 0.55)
        
        assert result.arb_type == ArbitrageType.SHORT_REBALANCING
        assert result.is_profitable is True
        assert result.gross_profit_pct == pytest.approx(0.10, abs=0.001)
        assert result.recommended_action == "SPLIT_AND_SELL_YES"
    
    def test_edge_case_exactly_one(self):
        """Test: Exactly 1.00 -> no arbitrage."""
        result = self.detector.check_binary_market(0.60, 0.40)
        
        assert result.arb_type == ArbitrageType.NONE
        assert result.total == 1.0
    
    def test_size_multiplier_scales_with_profit(self):
        """Test: Larger profit -> larger size multiplier (arb is risk-free)."""
        # 5% profit after fees
        result_small = self.detector.check_binary_market(0.45, 0.45)
        
        # 15% profit after fees
        result_large = self.detector.check_binary_market(0.40, 0.40)
        
        assert result_large.size_multiplier > result_small.size_multiplier
        assert result_large.size_multiplier <= 5.0  # Capped at 5x
    
    def test_unprofitable_has_zero_multiplier(self):
        """Test: Unprofitable opportunity has 0 size multiplier."""
        result = self.detector.check_binary_market(0.50, 0.50)
        
        assert result.size_multiplier == 0


class TestArbitrageWithFees:
    """Test fee impact on arbitrage detection."""
    
    def test_high_fee_eliminates_arbitrage(self):
        """Test: High fees can eliminate otherwise profitable arb."""
        # With 5% fees, need larger deviation
        detector = ArbitrageDetector(fee_rate=0.05)
        
        # Small deviation that would be profitable at 2% fee
        result = detector.check_binary_market(0.46, 0.46)
        
        assert not result.is_profitable
    
    def test_zero_fee_makes_small_arb_profitable(self):
        """Test: Zero fee mode (for testing) finds smaller arbs."""
        detector = ArbitrageDetector(
            fee_rate=0.0,
            min_profit_threshold=0.001,
            slippage_buffer=0.0,
        )
        
        # 1% profit with no fees
        result = detector.check_binary_market(0.495, 0.495)
        
        assert result.is_profitable
        assert result.net_profit_pct == pytest.approx(0.01, abs=0.001)


class TestArbitrageOpportunity:
    """Test ArbitrageOpportunity dataclass."""
    
    def test_is_profitable_positive_net(self):
        """Test: is_profitable is True when net_profit > 0."""
        opp = ArbitrageOpportunity(
            arb_type=ArbitrageType.LONG_REBALANCING,
            yes_price=0.45,
            no_price=0.45,
            total=0.90,
            gross_profit_pct=0.10,
            net_profit_pct=0.05,
            recommended_action="BUY_YES_AND_NO",
        )
        
        assert opp.is_profitable is True
    
    def test_is_profitable_negative_net(self):
        """Test: is_profitable is False when net_profit <= 0."""
        opp = ArbitrageOpportunity(
            arb_type=ArbitrageType.LONG_REBALANCING,
            yes_price=0.48,
            no_price=0.48,
            total=0.96,
            gross_profit_pct=0.04,
            net_profit_pct=-0.002,  # Fees eat the profit
            recommended_action="BUY_YES_AND_NO",
        )
        
        assert opp.is_profitable is False
