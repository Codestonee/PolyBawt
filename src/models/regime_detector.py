"""
Market Regime Detection using K-Means Clustering.

Research finding (Perplexity/Grok):
"Implement K-means clustering on volatility, spread, and liquidity
to identify market regimes for dynamic model/strategy selection."

Detects regimes:
- LOW_VOL: Calm markets, models reliable
- NORMAL: Standard conditions
- HIGH_VOL: Volatile, reduce confidence in models
- TRENDING: Strong directional bias
- THIN_LIQUIDITY: Low depth, increase position caution
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class MarketRegime(str, Enum):
    """Detected market regime."""
    LOW_VOL = "low_volatility"
    NORMAL = "normal"
    HIGH_VOL = "high_volatility"
    TRENDING = "trending"
    THIN_LIQUIDITY = "thin_liquidity"
    UNKNOWN = "unknown"


@dataclass
class RegimeFeatures:
    """Features used for regime detection."""
    
    # Volatility features
    realized_vol: float = 0.60      # Annualized realized volatility
    vol_of_vol: float = 0.10        # Volatility of volatility
    vol_regime_z: float = 0.0       # Z-score vs long-term average
    
    # Spread features
    spread_bps: float = 50.0        # Current spread in basis points
    spread_avg: float = 50.0        # Average spread
    spread_z: float = 0.0           # Z-score
    
    # Liquidity features
    depth_usd: float = 1000.0       # Total depth in USD
    depth_avg: float = 1000.0       # Average depth
    depth_z: float = 0.0            # Z-score
    
    # Order flow features
    imbalance: float = 0.0          # Order book imbalance [-1, 1]
    vpin: float = 0.0               # VPIN toxicity [0, 1]
    
    # Price momentum
    price_momentum: float = 0.0     # Recent price change %


@dataclass
class RegimeResult:
    """Result of regime detection."""
    
    regime: MarketRegime
    confidence: float               # 0-1, how confident in this regime
    features: RegimeFeatures
    
    # Strategy adjustments based on regime
    model_weight_adj: float = 1.0   # Multiply model weights by this
    orderbook_weight_adj: float = 1.0  # Multiply order book weight
    position_size_adj: float = 1.0  # Multiply position size
    
    # Cluster info (for debugging)
    cluster_id: int = -1
    distance_to_centroid: float = 0.0


class KMeansRegimeDetector:
    """
    K-Means based regime detection.
    
    Uses online K-Means to cluster market conditions into regimes.
    Updates centroids incrementally as new data arrives.
    
    Features used:
    - Volatility z-score
    - Spread z-score
    - Depth z-score
    - Order imbalance
    - VPIN
    
    Research (Perplexity): "K-means on vol/spread/liquidity provides
    intuitive regime classification for strategy adaptation."
    
    Usage:
        detector = KMeansRegimeDetector()
        result = detector.detect(features)
        print(f"Regime: {result.regime}")
        print(f"Position size adj: {result.position_size_adj}")
    """
    
    # Number of clusters (regimes)
    N_CLUSTERS = 5
    
    # Feature weights for distance calculation
    FEATURE_WEIGHTS = {
        "vol_z": 2.0,       # Volatility most important
        "spread_z": 1.5,    # Spread important
        "depth_z": 1.0,     # Depth matters
        "imbalance": 1.2,   # Imbalance for trend
        "vpin": 1.5,        # VPIN for toxicity
    }
    
    # Initial centroids (hand-tuned based on research)
    INITIAL_CENTROIDS = {
        MarketRegime.LOW_VOL: {
            "vol_z": -1.5,
            "spread_z": -0.5,
            "depth_z": 0.5,
            "imbalance": 0.0,
            "vpin": 0.2,
        },
        MarketRegime.NORMAL: {
            "vol_z": 0.0,
            "spread_z": 0.0,
            "depth_z": 0.0,
            "imbalance": 0.0,
            "vpin": 0.3,
        },
        MarketRegime.HIGH_VOL: {
            "vol_z": 2.0,
            "spread_z": 1.0,
            "depth_z": -0.5,
            "imbalance": 0.0,
            "vpin": 0.5,
        },
        MarketRegime.TRENDING: {
            "vol_z": 0.5,
            "spread_z": 0.2,
            "depth_z": 0.0,
            "imbalance": 0.6,  # High imbalance = trending
            "vpin": 0.4,
        },
        MarketRegime.THIN_LIQUIDITY: {
            "vol_z": 0.3,
            "spread_z": 2.0,   # Wide spreads
            "depth_z": -2.0,   # Low depth
            "imbalance": 0.0,
            "vpin": 0.4,
        },
    }
    
    # Regime -> strategy adjustments
    REGIME_ADJUSTMENTS = {
        MarketRegime.LOW_VOL: {
            "model_weight_adj": 1.2,      # Trust models more
            "orderbook_weight_adj": 0.8,  # Order book less important
            "position_size_adj": 1.1,     # Can size up slightly
        },
        MarketRegime.NORMAL: {
            "model_weight_adj": 1.0,
            "orderbook_weight_adj": 1.0,
            "position_size_adj": 1.0,
        },
        MarketRegime.HIGH_VOL: {
            "model_weight_adj": 0.7,      # Models less reliable
            "orderbook_weight_adj": 1.3,  # Order book more important
            "position_size_adj": 0.7,     # Reduce size
        },
        MarketRegime.TRENDING: {
            "model_weight_adj": 0.9,
            "orderbook_weight_adj": 1.2,  # Imbalance matters
            "position_size_adj": 1.0,
        },
        MarketRegime.THIN_LIQUIDITY: {
            "model_weight_adj": 0.8,
            "orderbook_weight_adj": 0.9,
            "position_size_adj": 0.5,     # Significantly reduce size
        },
    }
    
    def __init__(
        self,
        learning_rate: float = 0.05,
        min_observations: int = 10,
    ):
        """
        Initialize regime detector.
        
        Args:
            learning_rate: How fast to update centroids (0-1)
            min_observations: Min observations before confident detection
        """
        self.learning_rate = learning_rate
        self.min_observations = min_observations
        
        # Initialize centroids from hand-tuned values
        self._centroids = {
            regime: centroid.copy()
            for regime, centroid in self.INITIAL_CENTROIDS.items()
        }
        
        # Running statistics for z-score calculation
        self._vol_stats = RunningStats()
        self._spread_stats = RunningStats()
        self._depth_stats = RunningStats()
        
        # Observation count
        self._n_observations = 0
    
    def _extract_feature_vector(self, features: RegimeFeatures) -> dict[str, float]:
        """Extract normalized feature vector for clustering."""
        # Update running stats
        self._vol_stats.update(features.realized_vol)
        self._spread_stats.update(features.spread_bps)
        self._depth_stats.update(features.depth_usd)
        
        # Calculate z-scores
        vol_z = self._vol_stats.z_score(features.realized_vol)
        spread_z = self._spread_stats.z_score(features.spread_bps)
        depth_z = self._depth_stats.z_score(features.depth_usd)
        
        return {
            "vol_z": vol_z,
            "spread_z": spread_z,
            "depth_z": depth_z,
            "imbalance": features.imbalance,
            "vpin": features.vpin,
        }
    
    def _weighted_distance(
        self,
        vec: dict[str, float],
        centroid: dict[str, float],
    ) -> float:
        """Calculate weighted Euclidean distance."""
        total = 0.0
        for key, weight in self.FEATURE_WEIGHTS.items():
            diff = vec.get(key, 0.0) - centroid.get(key, 0.0)
            total += weight * diff ** 2
        return math.sqrt(total)
    
    def _update_centroid(
        self,
        regime: MarketRegime,
        vec: dict[str, float],
    ) -> None:
        """Online K-Means centroid update."""
        centroid = self._centroids[regime]
        for key in centroid:
            old_val = centroid[key]
            new_val = vec.get(key, old_val)
            centroid[key] = old_val + self.learning_rate * (new_val - old_val)
    
    def detect(self, features: RegimeFeatures) -> RegimeResult:
        """
        Detect current market regime.
        
        Args:
            features: Current market features
            
        Returns:
            RegimeResult with regime and strategy adjustments
        """
        self._n_observations += 1
        
        # Extract feature vector
        vec = self._extract_feature_vector(features)
        
        # Find nearest centroid
        best_regime = MarketRegime.NORMAL
        best_distance = float("inf")
        
        for regime, centroid in self._centroids.items():
            dist = self._weighted_distance(vec, centroid)
            if dist < best_distance:
                best_distance = dist
                best_regime = regime
        
        # Update centroid (online learning)
        self._update_centroid(best_regime, vec)
        
        # Calculate confidence based on distance and observation count
        # Lower distance = higher confidence
        # More observations = higher confidence
        distance_conf = max(0.0, 1.0 - best_distance / 5.0)  # Scale distance
        obs_conf = min(1.0, self._n_observations / self.min_observations)
        confidence = distance_conf * obs_conf
        
        # Get strategy adjustments
        adj = self.REGIME_ADJUSTMENTS.get(best_regime, self.REGIME_ADJUSTMENTS[MarketRegime.NORMAL])
        
        result = RegimeResult(
            regime=best_regime,
            confidence=confidence,
            features=features,
            model_weight_adj=adj["model_weight_adj"],
            orderbook_weight_adj=adj["orderbook_weight_adj"],
            position_size_adj=adj["position_size_adj"],
            cluster_id=list(self._centroids.keys()).index(best_regime),
            distance_to_centroid=best_distance,
        )
        
        logger.debug(
            "Regime detected",
            regime=best_regime.value,
            confidence=f"{confidence:.2f}",
            distance=f"{best_distance:.2f}",
            vol_z=f"{vec.get('vol_z', 0):.2f}",
            spread_z=f"{vec.get('spread_z', 0):.2f}",
            depth_z=f"{vec.get('depth_z', 0):.2f}",
            imbalance=f"{vec.get('imbalance', 0):.2f}",
            vpin=f"{vec.get('vpin', 0):.2f}",
        )
        
        return result
    
    def get_centroids(self) -> dict[MarketRegime, dict[str, float]]:
        """Get current centroid values (for debugging)."""
        return {regime: dict(centroid) for regime, centroid in self._centroids.items()}


class RunningStats:
    """
    Welford's online algorithm for running mean and variance.
    
    More numerically stable than naive approaches.
    """
    
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0  # Sum of squared differences from mean
    
    def update(self, x: float) -> None:
        """Add a new observation."""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
    
    @property
    def variance(self) -> float:
        """Population variance."""
        if self.n < 2:
            return 1.0  # Default to 1 to avoid division by zero
        return self.M2 / self.n
    
    @property
    def std(self) -> float:
        """Standard deviation."""
        return math.sqrt(self.variance)
    
    def z_score(self, x: float) -> float:
        """Calculate z-score for a value."""
        if self.std < 1e-10:
            return 0.0
        return (x - self.mean) / self.std


# Factory function
def create_regime_detector() -> KMeansRegimeDetector:
    """Create a regime detector with default settings."""
    return KMeansRegimeDetector()


# Pre-instantiated detector
regime_detector = KMeansRegimeDetector()
