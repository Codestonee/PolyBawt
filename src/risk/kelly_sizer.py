"""
Kelly Criterion position sizer for binary bets.

Implements fractional Kelly sizing for optimal bet sizing
with risk management controls.

Enhanced with:
- Adaptive Kelly based on volatility (Kelly-VIX hybrid approach)
- Drawdown-adjusted sizing
- Edge uncertainty discounting
- Multi-asset portfolio considerations

Based on research:
- arXiv:2508.16598 - Sizing the Risk: Kelly, VIX, and Hybrid Approaches
- Alpha Theory - Kelly Criterion in Practice
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class AdaptiveMode(Enum):
    """Adaptive Kelly adjustment modes."""
    NONE = "none"                    # Standard fractional Kelly
    VOLATILITY = "volatility"        # Reduce on high vol
    DRAWDOWN = "drawdown"           # Reduce during drawdowns
    FULL = "full"                   # Both adjustments


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
    capped_by_absolute: bool = False  # Absolute USD cap

    # Adaptive adjustments applied
    volatility_multiplier: float = 1.0
    drawdown_multiplier: float = 1.0
    uncertainty_multiplier: float = 1.0

    # Context
    edge: float = 0
    win_prob: float = 0
    odds: float = 0


@dataclass
class AdaptiveKellyConfig:
    """Configuration for adaptive Kelly adjustments."""

    # Volatility-based adjustment
    baseline_volatility: float = 0.60  # 60% annualized baseline
    vol_reduction_threshold: float = 1.2  # Start reducing at 120% of baseline
    vol_max_reduction: float = 0.5  # Reduce to 50% at extreme vol

    # Drawdown-based adjustment
    drawdown_reduction_start: float = 0.03  # Start reducing at 3% drawdown
    drawdown_reduction_max: float = 0.10    # Maximum reduction at 10% drawdown
    drawdown_min_multiplier: float = 0.25   # Never go below 25% of normal size

    # Edge uncertainty adjustment
    min_observations: int = 20  # Minimum trades for full confidence
    uncertainty_discount: float = 0.5  # Discount for uncertain edges

    # Consecutive loss adjustment
    loss_streak_threshold: int = 3  # Start reducing after 3 losses
    loss_streak_reduction_per: float = 0.1  # 10% reduction per additional loss


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

    Enhanced with adaptive adjustments:
    - Volatility-based: Reduce size when market vol is elevated
    - Drawdown-based: Reduce size during drawdown periods
    - Uncertainty-based: Discount edge when confidence is low

    Usage:
        sizer = KellySizer(
            bankroll=1000,
            kelly_fraction=0.25,
            max_position_pct=0.02,
            adaptive_mode=AdaptiveMode.FULL,
        )
        result = sizer.calculate(
            win_prob=0.60,
            market_price=0.50,
            side="YES",
            current_volatility=0.80,
            current_drawdown_pct=0.05,
        )
    """

    def __init__(
        self,
        bankroll: float,
        kelly_fraction: float = 0.25,  # 1/4 Kelly
        max_position_pct: float = 0.02,  # 2% per trade
        max_asset_exposure_pct: float = 0.05,  # 5% per asset
        kelly_cap: float = 0.25,  # Never recommend > 25% Kelly
        max_position_usd: float = 100.0,  # Absolute max $100 per trade
        adaptive_mode: AdaptiveMode = AdaptiveMode.FULL,
        adaptive_config: AdaptiveKellyConfig | None = None,
    ):
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction
        self.max_position_pct = max_position_pct
        self.max_asset_exposure_pct = max_asset_exposure_pct
        self.kelly_cap = kelly_cap
        self.max_position_usd = max_position_usd  # Absolute safety cap
        self.adaptive_mode = adaptive_mode
        self.adaptive_config = adaptive_config or AdaptiveKellyConfig()

        # Tracking for adaptive adjustments
        self._peak_bankroll = bankroll
        self._consecutive_losses = 0
        self._total_trades = 0
        self._winning_trades = 0

    def update_bankroll(self, new_bankroll: float) -> None:
        """Update bankroll (e.g., after PnL changes)."""
        self.bankroll = new_bankroll
        self._peak_bankroll = max(self._peak_bankroll, new_bankroll)

    def record_trade_result(self, won: bool) -> None:
        """Record trade result for adaptive adjustments."""
        self._total_trades += 1
        if won:
            self._winning_trades += 1
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1

    def reset_tracking(self) -> None:
        """Reset tracking counters (e.g., daily reset)."""
        self._consecutive_losses = 0
        self._peak_bankroll = self.bankroll

    def _calculate_volatility_multiplier(
        self,
        current_volatility: float | None,
    ) -> float:
        """
        Calculate position size multiplier based on current volatility.

        Based on Kelly-VIX hybrid approach from arXiv:2508.16598.
        Systematically reduces exposure when market uncertainty is high.

        Returns:
            Multiplier between vol_max_reduction and 1.0
        """
        if current_volatility is None:
            return 1.0

        cfg = self.adaptive_config
        vol_ratio = current_volatility / cfg.baseline_volatility

        if vol_ratio <= cfg.vol_reduction_threshold:
            return 1.0

        # Linear reduction from threshold to 2x baseline
        # At 2x baseline, we're at max reduction
        max_ratio = 2.0
        if vol_ratio >= max_ratio:
            return cfg.vol_max_reduction

        # Interpolate
        reduction_range = 1.0 - cfg.vol_max_reduction
        progress = (vol_ratio - cfg.vol_reduction_threshold) / (max_ratio - cfg.vol_reduction_threshold)
        multiplier = 1.0 - (progress * reduction_range)

        return max(cfg.vol_max_reduction, multiplier)

    def _calculate_drawdown_multiplier(
        self,
        current_drawdown_pct: float | None,
    ) -> float:
        """
        Calculate position size multiplier based on current drawdown.

        Reduces position sizes proportionally during drawdowns to
        preserve capital and reduce emotional stress.

        Returns:
            Multiplier between drawdown_min_multiplier and 1.0
        """
        if current_drawdown_pct is None or current_drawdown_pct <= 0:
            return 1.0

        cfg = self.adaptive_config

        if current_drawdown_pct < cfg.drawdown_reduction_start:
            return 1.0

        if current_drawdown_pct >= cfg.drawdown_reduction_max:
            return cfg.drawdown_min_multiplier

        # Linear interpolation
        reduction_range = 1.0 - cfg.drawdown_min_multiplier
        progress = (current_drawdown_pct - cfg.drawdown_reduction_start) / \
                   (cfg.drawdown_reduction_max - cfg.drawdown_reduction_start)
        multiplier = 1.0 - (progress * reduction_range)

        return max(cfg.drawdown_min_multiplier, multiplier)

    def _calculate_uncertainty_multiplier(
        self,
        edge_confidence: float | None,
    ) -> float:
        """
        Calculate position size multiplier based on edge uncertainty.

        When edge estimates are uncertain (low sample size, model disagreement),
        we should bet less aggressively.

        Args:
            edge_confidence: Confidence in edge estimate (0-1)

        Returns:
            Multiplier between uncertainty_discount and 1.0
        """
        if edge_confidence is None:
            # Use trade count as proxy for confidence
            if self._total_trades < self.adaptive_config.min_observations:
                progress = self._total_trades / self.adaptive_config.min_observations
                return self.adaptive_config.uncertainty_discount + \
                       (1.0 - self.adaptive_config.uncertainty_discount) * progress
            return 1.0

        # Direct confidence mapping
        return self.adaptive_config.uncertainty_discount + \
               edge_confidence * (1.0 - self.adaptive_config.uncertainty_discount)

    def _calculate_loss_streak_multiplier(self) -> float:
        """
        Calculate multiplier based on consecutive losses.

        Helps prevent tilt-driven overtrading during losing streaks.
        """
        cfg = self.adaptive_config

        if self._consecutive_losses < cfg.loss_streak_threshold:
            return 1.0

        extra_losses = self._consecutive_losses - cfg.loss_streak_threshold
        reduction = extra_losses * cfg.loss_streak_reduction_per

        return max(0.25, 1.0 - reduction)
    
    def calculate(
        self,
        win_prob: float,
        market_price: float,
        side: str = "YES",
        current_asset_exposure: float = 0,
        current_volatility: float | None = None,
        current_drawdown_pct: float | None = None,
        edge_confidence: float | None = None,
    ) -> KellySizeResult:
        """
        Calculate optimal position size with adaptive adjustments.

        Args:
            win_prob: Our model's probability of the YES outcome
            market_price: Current YES price
            side: "YES" or "NO"
            current_asset_exposure: Current exposure to this asset in USD
            current_volatility: Current annualized volatility (for adaptive mode)
            current_drawdown_pct: Current drawdown percentage (for adaptive mode)
            edge_confidence: Confidence in edge estimate 0-1 (for adaptive mode)

        Returns:
            KellySizeResult with recommended size and adjustment details
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

        # Calculate adaptive multipliers
        vol_mult = 1.0
        dd_mult = 1.0
        unc_mult = 1.0

        if self.adaptive_mode in (AdaptiveMode.VOLATILITY, AdaptiveMode.FULL):
            vol_mult = self._calculate_volatility_multiplier(current_volatility)

        if self.adaptive_mode in (AdaptiveMode.DRAWDOWN, AdaptiveMode.FULL):
            dd_mult = self._calculate_drawdown_multiplier(current_drawdown_pct)

        if self.adaptive_mode == AdaptiveMode.FULL:
            unc_mult = self._calculate_uncertainty_multiplier(edge_confidence)
            # Also apply loss streak adjustment
            loss_mult = self._calculate_loss_streak_multiplier()
            unc_mult *= loss_mult

        # Apply adaptive adjustments with floor to prevent near-zero sizing
        total_adaptive_mult = max(0.25, vol_mult * dd_mult * unc_mult)
        recommended *= total_adaptive_mult

        # Log significant reductions
        if total_adaptive_mult < 0.8:
            logger.info(
                "Adaptive Kelly reduction applied",
                vol_mult=f"{vol_mult:.2f}",
                dd_mult=f"{dd_mult:.2f}",
                unc_mult=f"{unc_mult:.2f}",
                total_mult=f"{total_adaptive_mult:.2f}",
            )

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
        
        # Absolute safety cap in USD (anti-whale protection)
        capped_absolute = False
        if size_usd > self.max_position_usd:
            size_usd = self.max_position_usd
            recommended = size_usd / self.bankroll if self.bankroll > 0 else 0
            capped_absolute = True
            logger.info(
                "Position capped by absolute limit",
                max_position_usd=self.max_position_usd,
                requested_size=size_usd,
            )

        return KellySizeResult(
            kelly_fraction=kelly,
            recommended_fraction=recommended,
            recommended_size_usd=max(0, size_usd),
            capped_by_max_position=capped_max_pos,
            capped_by_max_asset=capped_max_asset,
            capped_by_kelly_cap=capped_kelly,
            capped_by_absolute=capped_absolute,
            volatility_multiplier=vol_mult,
            drawdown_multiplier=dd_mult,
            uncertainty_multiplier=unc_mult,
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

        price = max(0.01, min(0.99, price))
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
        return self._correlations.get(self._key(asset1, asset2), 0.85)
    
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
