"""
Unit tests for Toxicity Filter module.
Tests the adverse selection detection from arXiv:2510.15205.
"""

import pytest
from src.strategy.toxicity_filter import (
    ToxicityFilter,
    ToxicityReason,
    ToxicityResult,
)
from src.models.order_book import OrderBook


class MockOrderBook:
    """Mock order book for testing."""
    
    def __init__(
        self,
        bids: list[tuple[float, float]] = None,
        asks: list[tuple[float, float]] = None,
    ):
        self.bids = bids or [(0.48, 100), (0.47, 200)]
        self.asks = asks or [(0.52, 100), (0.53, 200)]
    
    @property
    def best_bid(self) -> float | None:
        return self.bids[0][0] if self.bids else None
    
    @property
    def best_ask(self) -> float | None:
        return self.asks[0][0] if self.asks else None
    
    @property
    def spread(self) -> float | None:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None
    
    def depth_usd(self, levels: int = 3) -> float:
        bid_depth = sum(p * s for p, s in self.bids[:levels])
        ask_depth = sum(p * s for p, s in self.asks[:levels])
        return bid_depth + ask_depth


class TestToxicityFilter:
    """Test toxicity detection logic."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.filter = ToxicityFilter(
            imbalance_threshold=0.7,
            spread_threshold=0.08,
            min_depth_usd=50.0,
            price_move_threshold=0.05,
        )
    
    def test_balanced_market_not_toxic(self):
        """Test: Balanced order book is not toxic."""
        book = MockOrderBook(
            bids=[(0.49, 100), (0.48, 100)],
            asks=[(0.51, 100), (0.52, 100)],
        )
        
        result = self.filter.analyze(book)
        
        assert result.is_toxic is False
        assert result.reason == ToxicityReason.NONE
    
    def test_imbalanced_bids_toxic(self):
        """Test: Heavy bid imbalance triggers toxicity."""
        # Bids have 10x the size of asks
        book = MockOrderBook(
            bids=[(0.49, 1000), (0.48, 1000)],
            asks=[(0.51, 50), (0.52, 50)],
        )
        
        result = self.filter.analyze(book)
        
        assert result.is_toxic is True
        assert result.reason == ToxicityReason.ORDER_IMBALANCE
        assert result.should_reduce_size is True
    
    def test_imbalanced_asks_toxic(self):
        """Test: Heavy ask imbalance triggers toxicity."""
        book = MockOrderBook(
            bids=[(0.49, 50), (0.48, 50)],
            asks=[(0.51, 1000), (0.52, 1000)],
        )
        
        result = self.filter.analyze(book)
        
        assert result.is_toxic is True
        assert result.reason == ToxicityReason.ORDER_IMBALANCE
    
    def test_wide_spread_toxic(self):
        """Test: Wide spread indicates informed traders."""
        book = MockOrderBook(
            bids=[(0.40, 100)],  # 10% spread
            asks=[(0.50, 100)],
        )
        
        result = self.filter.analyze(book)
        
        assert result.is_toxic is True
        assert result.reason == ToxicityReason.WIDE_SPREAD
    
    def test_thin_book_toxic(self):
        """Test: Thin order book is toxic."""
        # Very small sizes -> low depth
        book = MockOrderBook(
            bids=[(0.49, 1)],  # ~$0.50 depth per side
            asks=[(0.51, 1)],
        )
        
        result = self.filter.analyze(book)
        
        assert result.is_toxic is True
        assert result.reason == ToxicityReason.THIN_BOOK
    
    def test_rapid_price_move_toxic(self):
        """Test: Large price movement suggests informed flow."""
        book = MockOrderBook()
        
        # 10% price move
        result = self.filter.analyze(
            book,
            recent_price=0.50,
            current_price=0.55,
        )
        
        assert result.is_toxic is True
        assert result.reason == ToxicityReason.RAPID_PRICE_MOVE


class TestToxicityResult:
    """Test ToxicityResult behavior."""
    
    def test_size_multiplier_pause(self):
        """Test: should_pause -> 0 size multiplier."""
        result = ToxicityResult(
            is_toxic=True,
            reason=ToxicityReason.THIN_BOOK,
            severity=1.0,
            should_pause=True,
        )
        
        assert result.size_multiplier == 0
    
    def test_size_multiplier_reduce(self):
        """Test: should_reduce_size scales with severity."""
        result = ToxicityResult(
            is_toxic=True,
            reason=ToxicityReason.ORDER_IMBALANCE,
            severity=0.5,
            should_reduce_size=True,
        )
        
        # Size multiplier = 1 - 0.5 = 0.5
        assert result.size_multiplier == 0.5
    
    def test_size_multiplier_normal(self):
        """Test: Non-toxic has 1.0 multiplier."""
        result = ToxicityResult(
            is_toxic=False,
            reason=ToxicityReason.NONE,
            severity=0,
        )
        
        assert result.size_multiplier == 1.0


class TestToxicityFilterPriceHistory:
    """Test price momentum tracking."""
    
    def setup_method(self):
        self.filter = ToxicityFilter()
    
    def test_price_history_tracking(self):
        """Test: Prices are tracked for momentum analysis."""
        self.filter.update_price_history(0.50)
        self.filter.update_price_history(0.51)
        self.filter.update_price_history(0.52)
        
        # Should return 3 prices ago
        recent = self.filter.get_recent_price()
        assert recent == 0.50
    
    def test_price_history_none_when_empty(self):
        """Test: Returns None when not enough history."""
        recent = self.filter.get_recent_price()
        assert recent is None
