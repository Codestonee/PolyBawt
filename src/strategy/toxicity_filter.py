"""
Toxicity Filter for Order Flow Analysis.

Detects conditions where adverse selection is high and trading
should be paused to avoid being "picked off" by informed traders.

Reference: arXiv:2510.15205 Section 4.2 "Execution hygiene (anti pick-off)"
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.infrastructure.logging import get_logger
from src.models.order_book import OrderBook

logger = get_logger(__name__)


class ToxicityReason(str, Enum):
    """Reasons for detecting toxic flow."""
    ORDER_IMBALANCE = "order_imbalance"
    WIDE_SPREAD = "wide_spread"
    THIN_BOOK = "thin_book"
    RAPID_PRICE_MOVE = "rapid_price_move"
    VPIN_HIGH = "vpin_high"  # NEW: Research-backed (Claude/Perplexity/Gemini)
    NONE = "none"


@dataclass
class ToxicityResult:
    """Result of toxicity analysis."""
    
    is_toxic: bool
    reason: ToxicityReason
    severity: float  # 0-1, higher = more toxic
    message: str = ""
    
    # Recommendations
    should_widen_spread: bool = False
    should_reduce_size: bool = False
    should_pause: bool = False
    
    @property
    def size_multiplier(self) -> float:
        """Recommended size multiplier based on toxicity."""
        if self.should_pause:
            return 0
        if self.should_reduce_size:
            return max(0.25, 1 - self.severity)
        return 1.0


class ToxicityFilter:
    """
    Detects toxic order flow conditions.
    
    Implements the paper's "anti pick-off" safeguards:
    1. Order book imbalance detection
    2. Spread widening detection
    3. Thin book detection
    4. Rapid price movement detection
    
    Usage:
        filter = ToxicityFilter()
        result = filter.analyze(order_book, recent_price=0.5, current_price=0.55)
        if result.is_toxic:
            skip_trade()
    """
    
    def __init__(
        self,
        imbalance_threshold: float = 0.7,   # 70% imbalance triggers alert
        spread_threshold: float = 0.08,      # 8% spread is wide
        min_depth_usd: float = 50.0,         # Minimum $50 depth
        price_move_threshold: float = 0.05,  # 5% rapid move
        vpin_threshold: float = 0.7,         # VPIN > 0.7 = extremely toxic (research)
    ):
        """
        Initialize toxicity filter.
        
        Args:
            imbalance_threshold: Max bid/ask imbalance ratio (0-1)
            spread_threshold: Max spread before toxic
            min_depth_usd: Minimum book depth in USD
            price_move_threshold: Max price move to be considered stable
        """
        self.imbalance_threshold = imbalance_threshold
        self.spread_threshold = spread_threshold
        self.min_depth_usd = min_depth_usd
        self.price_move_threshold = price_move_threshold
        self.vpin_threshold = vpin_threshold  # NEW
        
        # Track recent prices for momentum detection
        self._recent_prices: list[float] = []
        self._max_history = 10
    
    def analyze(
        self,
        order_book: OrderBook,
        recent_price: Optional[float] = None,
        current_price: Optional[float] = None,
        vpin: Optional[float] = None,  # NEW: VPIN from WebSocket client
    ) -> ToxicityResult:
        """
        Analyze order book and price action for toxicity.
        
        Args:
            order_book: Current order book
            recent_price: Price from recent past (optional)
            current_price: Current mid price (optional)
        
        Returns:
            ToxicityResult with analysis
        """
        # 1. Check order book imbalance
        imbalance_result = self._check_imbalance(order_book)
        if imbalance_result.is_toxic:
            return imbalance_result
        
        # 2. Check spread
        spread_result = self._check_spread(order_book)
        if spread_result.is_toxic:
            return spread_result
        
        # 3. Check book depth
        depth_result = self._check_depth(order_book)
        if depth_result.is_toxic:
            return depth_result
        
        # 4. Check price momentum
        if recent_price is not None and current_price is not None:
            momentum_result = self._check_momentum(recent_price, current_price)
            if momentum_result.is_toxic:
                return momentum_result
        
        # 5. Check VPIN (research-backed: Claude/Perplexity/Gemini)
        if vpin is not None:
            vpin_result = self._check_vpin(vpin)
            if vpin_result.is_toxic:
                return vpin_result
        
        # All clear
        return ToxicityResult(
            is_toxic=False,
            reason=ToxicityReason.NONE,
            severity=0,
            message="Order flow appears clean",
        )
    
    def _check_imbalance(self, order_book: OrderBook) -> ToxicityResult:
        """Check for bid/ask imbalance."""
        # Sum depth on each side (top 5 levels)
        bid_depth = sum(size for _, size in order_book.bids[:5]) if order_book.bids else 0
        ask_depth = sum(size for _, size in order_book.asks[:5]) if order_book.asks else 0
        
        total_depth = bid_depth + ask_depth
        if total_depth < 1:  # Avoid division by zero
            return ToxicityResult(
                is_toxic=True,
                reason=ToxicityReason.THIN_BOOK,
                severity=1.0,
                message="No liquidity on either side",
                should_pause=True,
            )
        
        # Imbalance ratio: 0 = balanced, 1 = completely one-sided
        imbalance = abs(bid_depth - ask_depth) / total_depth
        
        if imbalance > self.imbalance_threshold:
            side = "bids" if bid_depth > ask_depth else "asks"
            severity = min(1.0, imbalance)
            
            logger.warning(
                "Toxic order imbalance detected",
                imbalance=f"{imbalance:.2%}",
                heavy_side=side,
                bid_depth=bid_depth,
                ask_depth=ask_depth,
            )
            
            return ToxicityResult(
                is_toxic=True,
                reason=ToxicityReason.ORDER_IMBALANCE,
                severity=severity,
                message=f"Order book heavily skewed toward {side} ({imbalance:.0%})",
                should_reduce_size=True,
                should_widen_spread=True,
            )
        
        return ToxicityResult(is_toxic=False, reason=ToxicityReason.NONE, severity=0)
    
    def _check_spread(self, order_book: OrderBook) -> ToxicityResult:
        """Check for wide spreads (informed trader activity)."""
        spread = order_book.spread
        
        if spread is None:
            return ToxicityResult(
                is_toxic=True,
                reason=ToxicityReason.THIN_BOOK,
                severity=1.0,
                message="Cannot calculate spread - missing prices",
                should_pause=True,
            )
        
        if spread > self.spread_threshold:
            severity = min(1.0, spread / self.spread_threshold - 0.5)
            
            logger.warning(
                "Wide spread detected",
                spread=f"{spread:.2%}",
                threshold=f"{self.spread_threshold:.2%}",
            )
            
            return ToxicityResult(
                is_toxic=True,
                reason=ToxicityReason.WIDE_SPREAD,
                severity=severity,
                message=f"Spread {spread:.1%} exceeds threshold {self.spread_threshold:.1%}",
                should_reduce_size=True,
            )
        
        return ToxicityResult(is_toxic=False, reason=ToxicityReason.NONE, severity=0)
    
    def _check_depth(self, order_book: OrderBook) -> ToxicityResult:
        """Check for thin books."""
        depth = order_book.depth_usd(levels=3)
        
        if depth < self.min_depth_usd:
            severity = 1 - (depth / self.min_depth_usd)
            
            logger.warning(
                "Thin order book detected",
                depth_usd=depth,
                min_required=self.min_depth_usd,
            )
            
            return ToxicityResult(
                is_toxic=True,
                reason=ToxicityReason.THIN_BOOK,
                severity=min(1.0, severity),
                message=f"Book depth ${depth:.2f} below minimum ${self.min_depth_usd}",
                should_reduce_size=True,
                should_pause=depth < self.min_depth_usd * 0.2,
            )
        
        return ToxicityResult(is_toxic=False, reason=ToxicityReason.NONE, severity=0)
    
    def _check_momentum(
        self,
        recent_price: float | None,
        current_price: float,
    ) -> ToxicityResult:
        """Check for rapid price movements."""
        if recent_price is None or recent_price <= 0:
            return ToxicityResult(is_toxic=False, reason=ToxicityReason.NONE, severity=0)
        
        price_move = abs(current_price - recent_price) / recent_price
        
        if price_move > self.price_move_threshold:
            direction = "up" if current_price > recent_price else "down"
            severity = min(1.0, price_move / self.price_move_threshold - 0.5)
            
            logger.warning(
                "Rapid price movement detected",
                move=f"{price_move:.2%}",
                direction=direction,
            )
            
            return ToxicityResult(
                is_toxic=True,
                reason=ToxicityReason.RAPID_PRICE_MOVE,
                severity=severity,
                message=f"Price moved {price_move:.1%} {direction} - possible informed flow",
                should_reduce_size=True,
                should_widen_spread=True,
            )
        
        return ToxicityResult(is_toxic=False, reason=ToxicityReason.NONE, severity=0)
    
    def update_price_history(self, price: float) -> None:
        """Track price for momentum analysis."""
        self._recent_prices.append(price)
        if len(self._recent_prices) > self._max_history:
            self._recent_prices.pop(0)
    
    def get_recent_price(self) -> Optional[float]:
        """Get price from a few ticks ago."""
        if len(self._recent_prices) >= 3:
            return self._recent_prices[-3]
        return None
    
    def _check_vpin(self, vpin: float) -> ToxicityResult:
        """
        Check for high VPIN (Volume-Synchronized Probability of Informed Trading).
        
        Research finding (Claude/Perplexity/Gemini):
        - VPIN > 0.7: Extremely toxic (informed trading, stop quoting)
        - VPIN > 0.5: Moderately toxic
        - VPIN < 0.3: Normal uninformed flow (safe to market make)
        
        "VPIN was able to foresee the flash crash and predict short-term
        volatility... Order flow is regarded as toxic when it adversely
        selects market makers."
        """
        if vpin > self.vpin_threshold:
            severity = min(1.0, (vpin - self.vpin_threshold) / (1 - self.vpin_threshold))
            
            logger.warning(
                "High VPIN detected (informed trading)",
                vpin=f"{vpin:.2f}",
                threshold=f"{self.vpin_threshold:.2f}",
            )
            
            return ToxicityResult(
                is_toxic=True,
                reason=ToxicityReason.VPIN_HIGH,
                severity=severity,
                message=f"VPIN {vpin:.2f} exceeds threshold {self.vpin_threshold:.2f} - likely informed flow",
                should_reduce_size=True,
                should_pause=vpin > 0.85,  # Extreme toxicity
            )
        
        return ToxicityResult(is_toxic=False, reason=ToxicityReason.NONE, severity=0)


# Pre-instantiated filter with default settings
toxicity_filter = ToxicityFilter()
