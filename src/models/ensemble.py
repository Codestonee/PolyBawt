"""
Research-Backed Ensemble Model.

Combines multiple pricing models with order book microstructure signals
using dynamic weighting based on market conditions.

Research findings:
- "Order book features often beat academic models" (65-75% vs 55-65% accuracy)
- "Ensemble approach with dynamic weighting outperforms single models"
- "Bates SVJ achieves lowest MAPE for ETH; Kou lowest RMSE for BTC"

CRITICAL - BACKTESTING REQUIREMENTS (Walk-Forward Validation):
=============================================================
NEVER use random k-fold cross-validation for this model. Research (claude.md):
"Walk-forward validation preserves temporal causality: training data MUST
precede test data. Random k-fold gives optimistically biased estimates due
to information leakage."

When backtesting, you MUST use walk-forward (rolling/expanding window) validation:

1. EXPANDING WINDOW:
   - Train on [0, T], test on [T+1, T+gap]
   - Train on [0, T+gap], test on [T+gap+1, T+2*gap]
   - Continue expanding training window

2. ROLLING WINDOW (fixed size):
   - Train on [0, window], test on [window+1, window+gap]
   - Train on [gap, window+gap], test on [window+gap+1, window+2*gap]
   - Slide window forward

3. REQUIRED GAP:
   - Always include a gap between train and test (e.g., 1 hour) to prevent
     look-ahead bias from correlated observations

4. MINIMUM SAMPLES:
   - Minimum 500-1000 resolved markets for statistical significance
   - Must span bull, bear, and sideways regimes

Example implementation:
```python
def walk_forward_backtest(data, train_size=1000, test_size=100, gap=10):
    results = []
    for start in range(0, len(data) - train_size - test_size - gap, test_size):
        train = data[start:start + train_size]
        test = data[start + train_size + gap:start + train_size + gap + test_size]

        model.fit(train)
        predictions = model.predict(test)
        results.append(evaluate(predictions, test.outcomes))
    return aggregate(results)
```
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.infrastructure.logging import get_logger
from src.models.advanced_pricing import (
    KouModel, KouParams,
    HestonModel, HestonParams,
    BatesModel, BatesParams,
    VolatilityCalibrator,
    PricingModel,
)
from src.models.jump_diffusion import JumpDiffusionModel, JumpDiffusionParams
from src.models.orderbook_signal import OrderBookSignalModel, OrderBookFeatures
from src.models.regime_detector import (
    KMeansRegimeDetector, 
    RegimeFeatures, 
    RegimeResult,
    MarketRegime as KMeansRegime,
)

logger = get_logger(__name__)


class MarketRegime(str, Enum):
    """Market regime for dynamic weight adjustment."""
    LOW_VOL = "low_volatility"
    NORMAL = "normal"
    HIGH_VOL = "high_volatility"
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"


@dataclass
class ModelWeight:
    """Weight configuration for a pricing model."""
    base_weight: float
    regime_adjustments: dict[MarketRegime, float] = field(default_factory=dict)
    
    def get_weight(self, regime: MarketRegime) -> float:
        """Get adjusted weight for current regime."""
        adjustment = self.regime_adjustments.get(regime, 1.0)
        return self.base_weight * adjustment


@dataclass
class EnsembleResult:
    """Result from ensemble model calculation."""
    
    # Final probability
    probability: float
    
    # Component contributions
    jd_prob: float
    jd_weight: float
    
    kou_prob: float
    kou_weight: float
    
    bates_prob: float
    bates_weight: float
    
    orderbook_adjustment: float
    orderbook_weight: float
    
    # Metadata
    regime: MarketRegime
    confidence: float  # 0-1, based on agreement between models
    
    @property
    def model_agreement(self) -> float:
        """Measure of agreement between models (0 = full disagreement, 1 = perfect agreement)."""
        probs = [self.jd_prob, self.kou_prob, self.bates_prob]
        if not probs:
            return 0.5
        avg = sum(probs) / len(probs)
        variance = sum((p - avg) ** 2 for p in probs) / len(probs)
        # Convert variance to agreement score (higher variance = lower agreement)
        # Max theoretical variance is 0.25 (when probs are 0 and 1)
        return max(0.0, 1.0 - variance / 0.25)


class ResearchEnsembleModel:
    """
    Research-backed ensemble model combining:
    1. Jump-Diffusion (Merton) - baseline
    2. Kou Double-Exponential - asymmetric jumps
    3. Bates SVJ - stochastic vol + jumps
    4. Order book imbalance signal - microstructure alpha
    
    Dynamic weighting based on:
    - Asset type (BTC prefers Kou, ETH prefers Bates)
    - Market regime (high vol = trust models less, order book more)
    - Model agreement (high agreement = higher confidence)
    
    Research (Perplexity): "Ensemble approach with dynamic weighting
    outperforms single models for 15-minute binary options."
    
    Usage:
        model = ResearchEnsembleModel()
        result = model.prob_up(
            spot=100000,
            initial=99500,
            time_years=15/525600,
            asset="BTC",
            orderbook_features=features,
        )
        print(f"Probability: {result.probability:.2%}")
        print(f"Confidence: {result.confidence:.2%}")
    """
    
    # Base weights per asset (from research: Kou for BTC, Bates for ETH)
    ASSET_WEIGHTS = {
        "BTC": {
            "jd": 0.15,
            "kou": 0.45,    # Research: lowest RMSE for BTC
            "bates": 0.20,
            "orderbook": 0.20,
        },
        "ETH": {
            "jd": 0.15,
            "kou": 0.20,
            "bates": 0.45,  # Research: lowest MAPE for ETH
            "orderbook": 0.20,
        },
        "SOL": {
            "jd": 0.15,
            "kou": 0.25,
            "bates": 0.35,
            "orderbook": 0.25,  # Higher for more volatile assets
        },
        "XRP": {
            "jd": 0.15,
            "kou": 0.25,
            "bates": 0.35,
            "orderbook": 0.25,
        },
    }
    
    # Default weights for unknown assets
    DEFAULT_WEIGHTS = {
        "jd": 0.20,
        "kou": 0.25,
        "bates": 0.30,
        "orderbook": 0.25,
    }
    
    # Regime-based weight adjustments
    REGIME_ADJUSTMENTS = {
        MarketRegime.HIGH_VOL: {
            # In high vol, models less reliable, order book more important
            "jd": 0.8,
            "kou": 0.9,
            "bates": 0.9,
            "orderbook": 1.5,
        },
        MarketRegime.LOW_VOL: {
            # In low vol, models more reliable
            "jd": 1.2,
            "kou": 1.1,
            "bates": 1.1,
            "orderbook": 0.8,
        },
        MarketRegime.TRENDING: {
            # In trending markets, order book imbalance matters more
            "jd": 0.9,
            "kou": 1.0,
            "bates": 1.0,
            "orderbook": 1.3,
        },
    }
    
    def __init__(
        self,
        orderbook_signal_weight: float = 0.20,
        min_model_weight: float = 0.05,
    ):
        """
        Initialize ensemble model.
        
        Args:
            orderbook_signal_weight: Base weight for order book signals
            min_model_weight: Minimum weight for any model (prevents zero weight)
        """
        # Pricing models
        self._jd = JumpDiffusionModel()
        self._kou = KouModel()
        self._bates = BatesModel()
        
        # Order book signal model
        self._orderbook = OrderBookSignalModel()
        
        # Volatility calibrator for regime detection
        self._vol_calibrator = VolatilityCalibrator()
        
        # Config
        self.orderbook_signal_weight = orderbook_signal_weight
        self.min_model_weight = min_model_weight
        
        # Performance tracking for adaptive weighting
        self._model_scores: dict[str, float] = {
            "jd": 1.0,
            "kou": 1.0,
            "bates": 1.0,
            "orderbook": 1.0,
        }
        
        # K-means regime detector for advanced regime identification
        self._regime_detector = KMeansRegimeDetector()
        self._last_regime_result: RegimeResult | None = None
    
    def _detect_regime(
        self,
        asset: str,
        orderbook_features: OrderBookFeatures | None = None,
        vpin: float | None = None,
    ) -> MarketRegime:
        """
        Detect current market regime using K-means clustering.
        
        Uses:
        - Realized volatility
        - Order book spread and depth
        - Imbalance and VPIN
        """
        # Get calibrated volatility
        vol_estimate = self._vol_calibrator.get_best_estimate(asset)
        vol = vol_estimate.annualized_vol
        
        # Build regime features
        features = RegimeFeatures(
            realized_vol=vol,
            spread_bps=orderbook_features.spread_bps if orderbook_features else 50.0,
            depth_usd=orderbook_features.total_depth_100bp if orderbook_features else 1000.0,
            imbalance=orderbook_features.imbalance_100bp if orderbook_features else 0.0,
            vpin=getattr(vpin, 'vpin', 0.0) if hasattr(vpin, 'vpin') else (vpin or 0.0),
        )
        
        # Run K-means regime detection
        regime_result = self._regime_detector.detect(features)
        self._last_regime_result = regime_result
        
        # Map K-means regime to ensemble MarketRegime enum
        regime_mapping = {
            KMeansRegime.LOW_VOL: MarketRegime.LOW_VOL,
            KMeansRegime.NORMAL: MarketRegime.NORMAL,
            KMeansRegime.HIGH_VOL: MarketRegime.HIGH_VOL,
            KMeansRegime.TRENDING: MarketRegime.TRENDING,
            KMeansRegime.THIN_LIQUIDITY: MarketRegime.NORMAL,  # Map thin liquidity to normal
            KMeansRegime.UNKNOWN: MarketRegime.NORMAL,
        }
        
        return regime_mapping.get(regime_result.regime, MarketRegime.NORMAL)
    
    def _get_weights(
        self,
        asset: str,
        regime: MarketRegime,
    ) -> dict[str, float]:
        """Get adjusted weights for asset and regime."""
        # Start with asset-specific or default weights
        base_weights = self.ASSET_WEIGHTS.get(asset, self.DEFAULT_WEIGHTS).copy()
        
        # Apply regime adjustments
        regime_adj = self.REGIME_ADJUSTMENTS.get(regime, {})
        for model, adjustment in regime_adj.items():
            if model in base_weights:
                base_weights[model] *= adjustment
        
        # Apply performance-based adjustments
        for model in base_weights:
            base_weights[model] *= self._model_scores.get(model, 1.0)
        
        # Enforce minimum weights
        for model in base_weights:
            base_weights[model] = max(self.min_model_weight, base_weights[model])
        
        # Normalize to sum to 1
        total = sum(base_weights.values())
        if total > 0:
            for model in base_weights:
                base_weights[model] /= total
        
        return base_weights
    
    def prob_up(
        self,
        spot: float,
        initial: float,
        time_years: float,
        asset: str = "BTC",
        orderbook_features: OrderBookFeatures | None = None,
        vpin: float | None = None,
    ) -> EnsembleResult:
        """
        Calculate probability using ensemble of models.
        
        Args:
            spot: Current spot price
            initial: Strike/reference price
            time_years: Time to expiry in years
            asset: Asset symbol
            orderbook_features: Features from order book (optional)
            vpin: VPIN toxicity metric (optional)
        
        Returns:
            EnsembleResult with probability and metadata
        """
        # Edge cases
        if time_years <= 0:
            base_prob = 1.0 if spot >= initial else 0.0
            return EnsembleResult(
                probability=base_prob,
                jd_prob=base_prob, jd_weight=0.25,
                kou_prob=base_prob, kou_weight=0.25,
                bates_prob=base_prob, bates_weight=0.25,
                orderbook_adjustment=0.0, orderbook_weight=0.25,
                regime=MarketRegime.NORMAL,
                confidence=1.0,
            )
        
        # Calculate probability from each model
        # 1. Jump-Diffusion (classic)
        jd_vol = self._vol_calibrator.get_best_estimate(asset).annualized_vol
        jd_params = JumpDiffusionParams(sigma=jd_vol)
        
        jd_prob = self._jd.prob_up(
            spot=spot,
            initial=initial,
            time_years=time_years,
            params=jd_params,
        )
        
        kou_params = KouParams.for_asset(asset)
        kou_prob = self._kou.prob_up(spot, initial, time_years, kou_params)
        
        bates_params = BatesParams.for_asset(asset)
        bates_prob = self._bates.prob_up(spot, initial, time_years, bates_params)
        
        # Order book signal (if available)
        orderbook_adjustment = 0.0
        if orderbook_features:
            # Add VPIN to features if available
            if vpin is not None:
                # vpin might be a VPINResult object or a float
                vpin_value = getattr(vpin, 'vpin', vpin) if hasattr(vpin, 'vpin') else vpin
                orderbook_features.vpin = vpin_value
            orderbook_adjustment = self._orderbook.get_probability_adjustment(orderbook_features)
        
        # Detect regime using K-means and get weights
        regime = self._detect_regime(asset, orderbook_features, vpin)
        weights = self._get_weights(asset, regime)
        
        # Weighted combination of model probabilities
        model_prob = (
            weights["jd"] * jd_prob +
            weights["kou"] * kou_prob +
            weights["bates"] * bates_prob
        )
        
        # Add order book adjustment (proportionally weighted)
        # Order book is an ADJUSTMENT, not a probability itself
        orderbook_weight = weights.get("orderbook", 0.2)
        final_prob = model_prob + orderbook_adjustment * orderbook_weight
        
        # Clamp to valid range
        final_prob = max(0.01, min(0.99, final_prob))
        
        # Calculate confidence based on model agreement
        probs = [jd_prob, kou_prob, bates_prob]
        avg = sum(probs) / len(probs)
        variance = sum((p - avg) ** 2 for p in probs) / len(probs)
        agreement = max(0.0, 1.0 - variance / 0.25)  # 0-1
        
        # Reduce confidence if VPIN is high (toxic flow)
        vpin_val = getattr(vpin, 'vpin', vpin) if hasattr(vpin, 'vpin') else vpin
        if vpin_val and vpin_val > 0.5:
            agreement *= (1.0 - (vpin_val - 0.5))
        
        logger.debug(
            "Ensemble probability calculated",
            asset=asset,
            spot=spot,
            initial=initial,
            jd_prob=f"{jd_prob:.3f}",
            kou_prob=f"{kou_prob:.3f}",
            bates_prob=f"{bates_prob:.3f}",
            orderbook_adj=f"{orderbook_adjustment:.3f}",
            final_prob=f"{final_prob:.3f}",
            regime=regime.value,
            agreement=f"{agreement:.2f}",
        )
        
        return EnsembleResult(
            probability=final_prob,
            jd_prob=jd_prob,
            jd_weight=weights["jd"],
            kou_prob=kou_prob,
            kou_weight=weights["kou"],
            bates_prob=bates_prob,
            bates_weight=weights["bates"],
            orderbook_adjustment=orderbook_adjustment,
            orderbook_weight=orderbook_weight,
            regime=regime,
            confidence=agreement,
        )
    
    def prob_down(
        self,
        spot: float,
        initial: float,
        time_years: float,
        asset: str = "BTC",
        orderbook_features: OrderBookFeatures | None = None,
        vpin: float | None = None,
    ) -> EnsembleResult:
        """Calculate probability of DOWN outcome."""
        result = self.prob_up(spot, initial, time_years, asset, orderbook_features, vpin)
        # Invert probability
        return EnsembleResult(
            probability=1.0 - result.probability,
            jd_prob=1.0 - result.jd_prob,
            jd_weight=result.jd_weight,
            kou_prob=1.0 - result.kou_prob,
            kou_weight=result.kou_weight,
            bates_prob=1.0 - result.bates_prob,
            bates_weight=result.bates_weight,
            orderbook_adjustment=-result.orderbook_adjustment,
            orderbook_weight=result.orderbook_weight,
            regime=result.regime,
            confidence=result.confidence,
        )
    
    def update_calibration(self, asset: str, price: float) -> None:
        """Update volatility calibration with new price."""
        self._vol_calibrator.update_price(asset, price)
    
    def record_outcome(
        self,
        asset: str,
        predicted_probs: dict[str, float],
        actual_outcome: bool,
    ) -> None:
        """
        Record outcome for adaptive weight adjustment.
        
        Args:
            asset: Asset symbol
            predicted_probs: Dict of model -> predicted probability
            actual_outcome: True if UP, False if DOWN
        """
        target = 1.0 if actual_outcome else 0.0
        
        for model, prob in predicted_probs.items():
            if model in self._model_scores:
                # Brier-like score update (lower error = higher score)
                error = (prob - target) ** 2
                # Exponential moving average of inverse error
                self._model_scores[model] = (
                    0.95 * self._model_scores[model] + 0.05 * (1.0 - error)
                )
        
        logger.debug(
            "Model scores updated",
            scores=self._model_scores,
        )


# Factory function
def create_research_ensemble() -> ResearchEnsembleModel:
    """Create a research-backed ensemble model."""
    return ResearchEnsembleModel()


# Pre-instantiated model
research_ensemble = ResearchEnsembleModel()
