"""
Arbitrage Detection for Polymarket Binary Markets.

Detects risk-free profit opportunities when market prices deviate
from theoretical constraints.

Types:
1. Market Rebalancing Arbitrage: YES + NO ≠ 1
2. Combinatorial Arbitrage: Cross-market price inconsistencies (future)

Reference: arXiv:2510.15205 Section 3.2.1
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.infrastructure.logging import get_logger
from src.models.order_book import OrderBook

logger = get_logger(__name__)


class ArbitrageType(str, Enum):
    """Type of arbitrage opportunity."""
    LONG_REBALANCING = "long_rebalancing"  # YES + NO < 1
    SHORT_REBALANCING = "short_rebalancing"  # YES + NO > 1
    NONE = "none"


@dataclass
class ArbitrageOpportunity:
    """A detected arbitrage opportunity."""
    
    arb_type: ArbitrageType
    yes_price: float
    no_price: float
    total: float
    gross_profit_pct: float  # Before fees
    net_profit_pct: float    # After fees
    recommended_action: str
    confidence: float = 1.0  # Arbitrage is theoretically risk-free
    
    @property
    def is_profitable(self) -> bool:
        """Check if profitable after fees."""
        return self.net_profit_pct > 0
    
    @property
    def size_multiplier(self) -> float:
        """
        How much to scale position size.
        Arbitrage is risk-free, so we can size up.
        """
        if not self.is_profitable:
            return 0
        # Scale with confidence and profit margin
        return min(5.0, 1.0 + self.net_profit_pct * 20)


class ArbitrageDetector:
    """
    Detects arbitrage opportunities in binary prediction markets.
    
    For binary YES/NO markets:
    - Theoretical constraint: P(YES) + P(NO) = 1
    - Long arb: If actual < 1, buy both → guaranteed profit
    - Short arb: If actual > 1, NO tokens undervalued
    
    Usage:
        detector = ArbitrageDetector(fee_rate=0.02)
        opportunity = detector.check_binary_market(yes_price=0.48, no_price=0.49)
        if opportunity.is_profitable:
            execute_arbitrage(opportunity)
    """
    
    # Polymarket fee structure
    DEFAULT_FEE_RATE = 0.02  # 2% per trade
    
    def __init__(
        self,
        fee_rate: float = DEFAULT_FEE_RATE,
        min_profit_threshold: float = 0.005,  # 0.5% minimum profit
        slippage_buffer: float = 0.002,  # 0.2% slippage estimate
    ):
        """
        Initialize arbitrage detector.
        
        Args:
            fee_rate: Trading fee per transaction (default 2%)
            min_profit_threshold: Minimum profit to consider (default 0.5%)
            slippage_buffer: Expected slippage (default 0.2%)
        """
        self.fee_rate = fee_rate
        self.min_profit_threshold = min_profit_threshold
        self.slippage_buffer = slippage_buffer
    
    def check_binary_market(
        self,
        yes_price: float,
        no_price: float,
    ) -> ArbitrageOpportunity:
        """
        Check for arbitrage in a binary YES/NO market.
        
        Args:
            yes_price: Current YES token price (0-1)
            no_price: Current NO token price (0-1)
        
        Returns:
            ArbitrageOpportunity with type and profitability
        """
        total = yes_price + no_price
        
        # Calculate fees for round-trip
        # Long arb: buy YES + buy NO (2 buys)
        # Short arb: more complex, involves split
        round_trip_fees = self.fee_rate * 2
        
        # Add slippage buffer
        total_costs = round_trip_fees + self.slippage_buffer
        
        if total < 1.0:
            # Long Rebalancing Arbitrage
            # Buy 1 unit of YES and 1 unit of NO
            # Cost: yes_price + no_price = total < 1
            # Payout: exactly 1 (one will pay out)
            # Profit: 1 - total
            gross_profit = 1 - total
            net_profit = gross_profit - total_costs
            
            if net_profit > self.min_profit_threshold:
                logger.info(
                    "Long arbitrage detected",
                    yes_price=yes_price,
                    no_price=no_price,
                    total=total,
                    gross_profit_pct=f"{gross_profit:.2%}",
                    net_profit_pct=f"{net_profit:.2%}",
                )
                
                return ArbitrageOpportunity(
                    arb_type=ArbitrageType.LONG_REBALANCING,
                    yes_price=yes_price,
                    no_price=no_price,
                    total=total,
                    gross_profit_pct=gross_profit,
                    net_profit_pct=net_profit,
                    recommended_action="BUY_YES_AND_NO",
                )
        
        elif total > 1.0:
            # Short Rebalancing Arbitrage
            # NO tokens are undervalued relative to YES
            # Strategy: Create a "split" (buy 1 USDC of both outcomes)
            # Then sell the overvalued YES positions
            gross_profit = total - 1
            net_profit = gross_profit - total_costs
            
            if net_profit > self.min_profit_threshold:
                logger.info(
                    "Short arbitrage detected",
                    yes_price=yes_price,
                    no_price=no_price,
                    total=total,
                    gross_profit_pct=f"{gross_profit:.2%}",
                    net_profit_pct=f"{net_profit:.2%}",
                )
                
                return ArbitrageOpportunity(
                    arb_type=ArbitrageType.SHORT_REBALANCING,
                    yes_price=yes_price,
                    no_price=no_price,
                    total=total,
                    gross_profit_pct=gross_profit,
                    net_profit_pct=net_profit,
                    recommended_action="SPLIT_AND_SELL_YES",
                )
        
        # No arbitrage opportunity
        return ArbitrageOpportunity(
            arb_type=ArbitrageType.NONE,
            yes_price=yes_price,
            no_price=no_price,
            total=total,
            gross_profit_pct=0,
            net_profit_pct=0,
            recommended_action="NONE",
        )
    
    def check_from_order_books(
        self,
        yes_book: OrderBook,
        no_book: OrderBook,
        size_usd: float = 10.0,
    ) -> ArbitrageOpportunity:
        """
        Check for arbitrage using order book depth.
        
        More accurate as it considers executable prices at given size.
        
        Args:
            yes_book: Order book for YES token
            no_book: Order book for NO token
            size_usd: Size to check in USD
        
        Returns:
            ArbitrageOpportunity accounting for liquidity
        """
        # Get executable prices at this size
        yes_ask = yes_book.best_ask or 0.5
        no_ask = no_book.best_ask or 0.5
        
        # Check if we can actually fill at these prices
        yes_depth = yes_book.depth_usd(levels=3)
        no_depth = no_book.depth_usd(levels=3)
        
        # Need enough liquidity on both sides
        if yes_depth < size_usd or no_depth < size_usd:
            return ArbitrageOpportunity(
                arb_type=ArbitrageType.NONE,
                yes_price=yes_ask,
                no_price=no_ask,
                total=yes_ask + no_ask,
                gross_profit_pct=0,
                net_profit_pct=0,
                recommended_action="INSUFFICIENT_LIQUIDITY",
            )
        
        # Use ask prices (what we'd actually pay)
        return self.check_binary_market(yes_ask, no_ask)
    
    def estimate_max_size(
        self,
        opportunity: ArbitrageOpportunity,
        yes_book: OrderBook,
        no_book: OrderBook,
    ) -> float:
        """
        Estimate maximum size for arbitrage trade.
        
        Limited by:
        1. Order book depth on both sides
        2. Maintaining profitability as we walk the book
        
        Args:
            opportunity: Detected opportunity
            yes_book: YES order book
            no_book: NO order book
        
        Returns:
            Maximum size in USD that maintains profitability
        """
        if not opportunity.is_profitable:
            return 0
        
        # Conservative: use minimum of top 3 levels depth
        yes_depth = yes_book.depth_usd(levels=3)
        no_depth = no_book.depth_usd(levels=3)
        
        max_size = min(yes_depth, no_depth)
        
        # Scale down to maintain profit margin
        # As size increases, slippage eats into profit
        profit_cushion = opportunity.net_profit_pct / self.slippage_buffer
        scaled_size = max_size * min(1.0, profit_cushion / 3)
        
        return max(0, scaled_size)


# Pre-instantiated detector with default settings
arbitrage_detector = ArbitrageDetector()
