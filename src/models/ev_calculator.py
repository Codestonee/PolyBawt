"""
Expected Value (EV) calculator for Polymarket trades.

Calculates net EV after accounting for:
- Dynamic taker fees (0-3.15%)
- Slippage
- Adverse selection
"""

from dataclasses import dataclass
from enum import Enum

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class TradeSide(Enum):
    """Which side of the market to trade."""
    BUY_YES = "buy_yes"
    BUY_NO = "buy_no"


@dataclass
class FeeCalculation:
    """Breakdown of fees and costs."""
    
    taker_fee_rate: float  # As decimal (0.0315 = 3.15%)
    taker_fee_usd: float   # Absolute fee amount
    slippage_usd: float
    adverse_selection_usd: float
    total_cost_usd: float
    
    @property
    def total_cost_rate(self) -> float:
        """Total cost as a rate of position size."""
        if self.total_cost_usd == 0:
            return 0
        # Approximate based on typical position
        return self.taker_fee_rate + 0.001 + 0.005  # fee + slip + adverse


@dataclass
class EVResult:
    """Complete EV calculation result."""
    
    # Trade details
    side: TradeSide
    model_prob: float      # Our model's probability
    market_price: float    # Current market price
    size_usd: float        # Position size in USD
    
    # Edge analysis
    gross_edge: float      # model_prob - market_price (for YES)
    gross_ev: float        # Expected profit before costs
    
    # Costs
    fees: FeeCalculation
    
    # Net results
    net_ev: float          # Expected profit after all costs
    net_edge: float        # Edge after costs as percentage
    
    # Signals
    is_positive_ev: bool
    edge_to_fee_ratio: float  # gross_edge / fee_rate
    
    def __repr__(self) -> str:
        return (
            f"EVResult(side={self.side.value}, "
            f"model={self.model_prob:.3f}, mkt={self.market_price:.3f}, "
            f"edge={self.gross_edge:.3f}, net_ev=${self.net_ev:.4f})"
        )


class EVCalculator:
    """
    Calculate Expected Value for binary prediction market trades.
    
    Polymarket fee formula for 15m crypto markets:
        fee = shares × 0.25 × (p × (1-p))²
    
    This peaks at ~3.15% when p = 0.50 and approaches 0% at extremes.
    
    Usage:
        calc = EVCalculator()
        result = calc.calculate(
            model_prob=0.55,
            market_price=0.50,
            size_usd=10.0
        )
        if result.is_positive_ev:
            execute_trade(result.side)
    """
    
    # Polymarket fee coefficient for 15m crypto markets
    FEE_COEFFICIENT = 0.25
    
    # Default cost assumptions
    DEFAULT_SLIPPAGE_RATE = 0.001   # 0.1%
    DEFAULT_ADVERSE_SELECTION = 0.005  # 0.5%
    
    def __init__(
        self,
        slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
        adverse_selection_rate: float = DEFAULT_ADVERSE_SELECTION,
    ):
        self.slippage_rate = slippage_rate
        self.adverse_selection_rate = adverse_selection_rate
    
    def calculate_taker_fee_rate(self, price: float) -> float:
        """
        Calculate dynamic taker fee rate.
        
        Formula: fee_rate = 0.25 × (p × (1-p))²
        
        This is approximately:
        - 3.15% at p = 0.50
        - 1.6% at p = 0.30 or 0.70
        - 0.2% at p = 0.10 or 0.90
        - ~0% at p < 0.04 or p > 0.96
        
        Args:
            price: Market price (0 to 1)
        
        Returns:
            Fee rate as decimal
        """
        # Clamp to valid range
        p = max(0.001, min(0.999, price))
        
        # Polymarket formula
        fee_rate = self.FEE_COEFFICIENT * (p * (1 - p)) ** 2
        
        return fee_rate
    
    def calculate_fees(
        self,
        market_price: float,
        size_usd: float,
    ) -> FeeCalculation:
        """
        Calculate all trading costs.
        
        Args:
            market_price: Current market price
            size_usd: Position size in USD
        
        Returns:
            FeeCalculation breakdown
        """
        # Dynamic taker fee
        taker_fee_rate = self.calculate_taker_fee_rate(market_price)
        taker_fee_usd = size_usd * taker_fee_rate
        
        # Slippage (conservative estimate)
        slippage_usd = size_usd * self.slippage_rate
        
        # Adverse selection (toxicity cost)
        adverse_selection_usd = size_usd * self.adverse_selection_rate
        
        # Total
        total_cost_usd = taker_fee_usd + slippage_usd + adverse_selection_usd
        
        return FeeCalculation(
            taker_fee_rate=taker_fee_rate,
            taker_fee_usd=taker_fee_usd,
            slippage_usd=slippage_usd,
            adverse_selection_usd=adverse_selection_usd,
            total_cost_usd=total_cost_usd,
        )
    
    def calculate(
        self,
        model_prob: float,
        market_price: float,
        size_usd: float,
    ) -> EVResult:
        """
        Calculate Expected Value for a potential trade.
        
        For a binary bet:
        - If we BUY YES at price p and it wins, we get (1-p)/p return
        - If we BUY NO at price (1-p) and it wins, we get p/(1-p) return
        
        Args:
            model_prob: Our model's probability of UP (0 to 1)
            market_price: Current YES price (0 to 1)
            size_usd: Position size in USD
        
        Returns:
            EVResult with complete analysis
        """
        # Determine which side to trade
        if model_prob > market_price:
            side = TradeSide.BUY_YES
            gross_edge = model_prob - market_price
            
            # EV for buying YES
            # Win: profit = size * (1/price - 1) = size * (1-price)/price
            # Lose: loss = size
            win_profit = size_usd * (1 - market_price) / market_price
            lose_loss = size_usd
            
            gross_ev = model_prob * win_profit - (1 - model_prob) * lose_loss
            
        else:
            side = TradeSide.BUY_NO
            no_price = 1 - market_price
            gross_edge = (1 - model_prob) - no_price  # Edge on NO side
            
            # EV for buying NO
            win_profit = size_usd * market_price / no_price
            lose_loss = size_usd
            
            gross_ev = (1 - model_prob) * win_profit - model_prob * lose_loss
        
        # Calculate fees using the price we'd be trading at
        trade_price = market_price if side == TradeSide.BUY_YES else (1 - market_price)
        fees = self.calculate_fees(trade_price, size_usd)
        
        # Net EV
        net_ev = gross_ev - fees.total_cost_usd
        
        # Net edge as percentage of size
        net_edge = net_ev / size_usd if size_usd > 0 else 0
        
        # Quality metrics
        is_positive_ev = net_ev > 0
        edge_to_fee_ratio = (
            abs(gross_edge) / fees.taker_fee_rate 
            if fees.taker_fee_rate > 0 else float('inf')
        )
        
        return EVResult(
            side=side,
            model_prob=model_prob,
            market_price=market_price,
            size_usd=size_usd,
            gross_edge=gross_edge,
            gross_ev=gross_ev,
            fees=fees,
            net_ev=net_ev,
            net_edge=net_edge,
            is_positive_ev=is_positive_ev,
            edge_to_fee_ratio=edge_to_fee_ratio,
        )
    
    def minimum_edge_for_profit(self, market_price: float) -> float:
        """
        Calculate minimum edge needed to break even at a given price.
        
        Args:
            market_price: Current market price
        
        Returns:
            Minimum gross edge needed
        """
        fee_rate = self.calculate_taker_fee_rate(market_price)
        total_cost_rate = fee_rate + self.slippage_rate + self.adverse_selection_rate
        return total_cost_rate


# Pre-instantiated calculator
ev_calculator = EVCalculator()
