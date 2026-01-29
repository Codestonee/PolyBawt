"""
Kelly Criterion position sizer for binary bets.

Implements fractional Kelly sizing for optimal bet sizing
with risk management controls.
"""

from dataclasses import dataclass

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


@dataclass
class KellySizeResult:
    """Result of Kelly sizing calculation."""
    
    # Recommended size
    kelly_fraction: float  # Full Kelly as fraction of bankroll
    recommended_fraction: float  # After applying fraction
    recommended_size_usd: float  # Absolute size
    
    # Limits applied
    capped_by_max_position: bool = False
    capped_by_max_asset: bool = False
    capped_by_kelly_cap: bool = False
    
    # Context
    edge: float = 0
    win_prob: float = 0
    odds: float = 0


class KellySizer:
    """
    Kelly Criterion position sizer for binary options.
    
    For a binary bet with:
    - p = probability of winning
    - b = odds (net payout if win, e.g., if you bet $1 and win $2, b = 1)
    
    Kelly fraction = (p * b - (1-p)) / b = (p * (b + 1) - 1) / b
    
    For binary options at price P:
    - Win probability = our model's p
    - Odds b = (1-P)/P for YES, or P/(1-P) for NO
    
    We use fractional Kelly (typically 1/4) for safety.
    
    Usage:
        sizer = KellySizer(
            bankroll=1000,
            kelly_fraction=0.25,
            max_position_pct=0.02,
        )
        result = sizer.calculate(
            win_prob=0.60,
            market_price=0.50,
            side="YES",
        )
    """
    
    def __init__(
        self,
        bankroll: float,
        kelly_fraction: float = 0.25,  # 1/4 Kelly
        max_position_pct: float = 0.02,  # 2% per trade
        max_asset_exposure_pct: float = 0.05,  # 5% per asset
        kelly_cap: float = 0.25,  # Never recommend > 25% Kelly
    ):
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction
        self.max_position_pct = max_position_pct
        self.max_asset_exposure_pct = max_asset_exposure_pct
        self.kelly_cap = kelly_cap
    
    def update_bankroll(self, new_bankroll: float) -> None:
        """Update bankroll (e.g., after PnL changes)."""
        self.bankroll = new_bankroll
    
    def calculate(
        self,
        win_prob: float,
        market_price: float,
        side: str = "YES",
        current_asset_exposure: float = 0,
    ) -> KellySizeResult:
        """
        Calculate optimal position size.
        
        Args:
            win_prob: Our model's probability of the YES outcome
            market_price: Current YES price
            side: "YES" or "NO"
            current_asset_exposure: Current exposure to this asset in USD
        
        Returns:
            KellySizeResult with recommended size
        """
        # Calculate odds based on side
        if side == "YES":
            p = win_prob
            price = market_price
        else:
            p = 1 - win_prob
            price = 1 - market_price
        
        # Odds: if we buy at price P, we win (1-P)/P if correct
        if price <= 0 or price >= 1:
            return KellySizeResult(
                kelly_fraction=0,
                recommended_fraction=0,
                recommended_size_usd=0,
            )
        
        b = (1 - price) / price  # Net odds
        
        # Kelly fraction: f* = (p*b - q) / b where q = 1-p
        # = (p*b - (1-p)) / b
        # = (p*(b+1) - 1) / b
        kelly = (p * (b + 1) - 1) / b
        
        # Edge calculation
        edge = p - price
        
        # Apply limits
        capped_max_pos = False
        capped_max_asset = False
        capped_kelly = False
        
        # No bet if Kelly is negative
        if kelly <= 0:
            return KellySizeResult(
                kelly_fraction=kelly,
                recommended_fraction=0,
                recommended_size_usd=0,
                edge=edge,
                win_prob=p,
                odds=b,
            )
        
        # Cap full Kelly at maximum
        if kelly > self.kelly_cap:
            kelly = self.kelly_cap
            capped_kelly = True
        
        # Apply fractional Kelly
        recommended = kelly * self.kelly_fraction
        
        # Cap by max position size
        if recommended > self.max_position_pct:
            recommended = self.max_position_pct
            capped_max_pos = True
        
        # Calculate absolute size
        size_usd = recommended * self.bankroll
        
        # Cap by max asset exposure
        max_additional = (self.max_asset_exposure_pct * self.bankroll) - current_asset_exposure
        if size_usd > max_additional:
            size_usd = max(0, max_additional)
            recommended = size_usd / self.bankroll if self.bankroll > 0 else 0
            capped_max_asset = True
        
        return KellySizeResult(
            kelly_fraction=kelly,
            recommended_fraction=recommended,
            recommended_size_usd=max(0, size_usd),
            capped_by_max_position=capped_max_pos,
            capped_by_max_asset=capped_max_asset,
            capped_by_kelly_cap=capped_kelly,
            edge=edge,
            win_prob=p,
            odds=b,
        )
    
    def kelly_for_edge(self, edge: float, price: float = 0.5) -> float:
        """
        Quick calculation of Kelly fraction for given edge at a price.
        
        Useful for threshold checks.
        """
        if edge <= 0:
            return 0
        
        p = price + edge  # Win probability = market + edge
        p = max(0.01, min(0.99, p))
        
        b = (1 - price) / price
        kelly = (p * (b + 1) - 1) / b
        
        return max(0, kelly)


@dataclass  
class CorrelationEntry:
    """Correlation between two assets."""
    asset1: str
    asset2: str
    correlation: float


class CorrelationMatrix:
    """
    Manages correlation between crypto assets.
    
    Used to reduce position sizes when assets are correlated.
    """
    
    # Default correlations based on recent data
    DEFAULT_CORRELATIONS = [
        CorrelationEntry("BTC", "ETH", 0.89),
        CorrelationEntry("BTC", "SOL", 0.99),
        CorrelationEntry("BTC", "XRP", 0.86),
        CorrelationEntry("ETH", "SOL", 0.85),
        CorrelationEntry("ETH", "XRP", 0.82),
        CorrelationEntry("SOL", "XRP", 0.80),
    ]
    
    def __init__(self):
        self._correlations: dict[tuple[str, str], float] = {}
        
        # Load defaults
        for entry in self.DEFAULT_CORRELATIONS:
            self.set_correlation(entry.asset1, entry.asset2, entry.correlation)
    
    def _key(self, asset1: str, asset2: str) -> tuple[str, str]:
        """Normalize key order."""
        return (min(asset1, asset2), max(asset1, asset2))
    
    def set_correlation(self, asset1: str, asset2: str, corr: float) -> None:
        """Set correlation between two assets."""
        self._correlations[self._key(asset1, asset2)] = corr
    
    def get_correlation(self, asset1: str, asset2: str) -> float:
        """Get correlation between two assets (1.0 if same asset)."""
        if asset1 == asset2:
            return 1.0
        return self._correlations.get(self._key(asset1, asset2), 0.5)
    
    def max_correlation_with(
        self,
        asset: str,
        open_positions: dict[str, float],
    ) -> float:
        """
        Get maximum correlation with any open position.
        
        Args:
            asset: Asset to check
            open_positions: Dict of asset -> position size
        
        Returns:
            Maximum correlation coefficient
        """
        if not open_positions:
            return 0.0
        
        max_corr = 0.0
        for pos_asset in open_positions:
            if pos_asset != asset:
                corr = self.get_correlation(asset, pos_asset)
                max_corr = max(max_corr, corr)
        
        return max_corr
    
    def correlation_adjustment(
        self,
        asset: str,
        open_positions: dict[str, float],
        threshold: float = 0.7,
    ) -> float:
        """
        Calculate size multiplier based on correlation.
        
        If correlation > threshold, reduce position size.
        
        Returns:
            Multiplier between 0 and 1
        """
        max_corr = self.max_correlation_with(asset, open_positions)
        
        if max_corr <= threshold:
            return 1.0
        
        # Linear reduction from threshold to 1.0
        # At threshold: 1.0, at 1.0 correlation: 0.25
        reduction = (max_corr - threshold) / (1.0 - threshold)
        return max(0.25, 1.0 - reduction * 0.75)


# Pre-instantiated instances
correlation_matrix = CorrelationMatrix()
