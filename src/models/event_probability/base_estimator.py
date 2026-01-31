"""Abstract base class for category-specific probability estimators."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from src.ingestion.event_market_discovery import EventMarket, MarketCategory
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class ConfidenceLevel(Enum):
    """Confidence level in probability estimate."""
    VERY_LOW = "very_low"    # < 50% confidence
    LOW = "low"              # 50-65%
    MEDIUM = "medium"        # 65-80%
    HIGH = "high"            # 80-90%
    VERY_HIGH = "very_high"  # > 90%


@dataclass
class ProbabilityResult:
    """Result of probability estimation."""
    
    # Core probability
    probability: float  # 0-1, P(YES outcome)
    
    # Confidence metrics
    confidence: float  # 0-1, how confident in this estimate
    confidence_level: ConfidenceLevel
    
    # Uncertainty bounds
    lower_bound: float  # 5th percentile
    upper_bound: float  # 95th percentile
    
    # Model metadata
    model_name: str
    category: MarketCategory
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Source contributions (for ensemble analysis)
    source_breakdown: dict[str, float] = field(default_factory=dict)
    
    # Feature values (for debugging/analysis)
    features: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Ensure probability is bounded
        self.probability = max(0.01, min(0.99, self.probability))
        self.lower_bound = max(0.0, min(self.probability, self.lower_bound))
        self.upper_bound = min(1.0, max(self.probability, self.upper_bound))
    
    @property
    def edge_vs_market(self, market_price: float = 0.5) -> float:
        """Calculate edge versus market price."""
        return self.probability - market_price
    
    @property
    def is_high_confidence(self) -> bool:
        """Is this a high confidence estimate?"""
        return self.confidence_level in (ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH)
    
    @property
    def uncertainty_range(self) -> float:
        """Width of uncertainty interval."""
        return self.upper_bound - self.lower_bound


def confidence_to_level(confidence: float) -> ConfidenceLevel:
    """Convert numeric confidence to level."""
    if confidence < 0.5:
        return ConfidenceLevel.VERY_LOW
    elif confidence < 0.65:
        return ConfidenceLevel.LOW
    elif confidence < 0.8:
        return ConfidenceLevel.MEDIUM
    elif confidence < 0.9:
        return ConfidenceLevel.HIGH
    else:
        return ConfidenceLevel.VERY_HIGH


class CategoryProbabilityEstimator(ABC):
    """
    Abstract base class for category-specific probability estimators.
    
    Each category (Politics, Sports, Economics, Pop Culture) has its own
    specialized estimator that uses domain-specific features and data sources.
    
    Usage:
        class MyEstimator(CategoryProbabilityEstimator):
            @property
            def category(self) -> MarketCategory:
                return MarketCategory.POLITICS
                
            async def estimate(self, market: EventMarket, **context) -> ProbabilityResult:
                # Implementation
                pass
    """
    
    def __init__(self, name: str):
        self.name = name
        self._estimates_cache: dict[str, ProbabilityResult] = {}
        self._cache_time: dict[str, datetime] = {}
    
    @property
    @abstractmethod
    def category(self) -> MarketCategory:
        """The market category this estimator handles."""
        pass
    
    @abstractmethod
    async def estimate(
        self,
        market: EventMarket,
        **context
    ) -> ProbabilityResult:
        """
        Estimate probability for a market.
        
        Args:
            market: The event market to estimate
            **context: Additional context (external data, etc.)
            
        Returns:
            ProbabilityResult with estimate and confidence
        """
        pass
    
    def _get_cache_key(self, market: EventMarket) -> str:
        """Generate cache key for market."""
        return f"{market.condition_id}:{market.question[:50]}"
    
    def get_cached_estimate(
        self,
        market: EventMarket,
        max_age_seconds: float = 300.0
    ) -> ProbabilityResult | None:
        """
        Get cached estimate if fresh.
        
        Args:
            market: Market to look up
            max_age_seconds: Maximum age of cached result
            
        Returns:
            Cached ProbabilityResult or None
        """
        key = self._get_cache_key(market)
        
        if key not in self._estimates_cache:
            return None
        
        cache_time = self._cache_time.get(key)
        if cache_time is None:
            return None
        
        age = (datetime.now(timezone.utc) - cache_time).total_seconds()
        if age > max_age_seconds:
            return None
        
        return self._estimates_cache[key]
    
    def cache_estimate(
        self,
        market: EventMarket,
        result: ProbabilityResult
    ) -> None:
        """Cache an estimate."""
        key = self._get_cache_key(market)
        self._estimates_cache[key] = result
        self._cache_time[key] = datetime.now(timezone.utc)
    
    def clear_cache(self) -> None:
        """Clear estimate cache."""
        self._estimates_cache.clear()
        self._cache_time.clear()
    
    def calculate_base_rate_adjustment(
        self,
        base_rate: float,
        time_to_event_hours: float,
        event_type: str = "general"
    ) -> float:
        """
        Adjust base rate based on time to event.
        
        As event approaches, base rates become less relevant
        and specific information becomes more important.
        
        Args:
            base_rate: Historical base rate (0-1)
            time_to_event_hours: Hours until event
            event_type: Type of event for context
            
        Returns:
            Adjustment factor (0-1, where 1 = full base rate weight)
        """
        # Base rates matter less as event approaches
        # At 1 week out, full weight
        # At 1 day out, half weight
        # At 1 hour out, minimal weight
        
        if time_to_event_hours > 168:  # > 1 week
            return 1.0
        elif time_to_event_hours > 24:  # 1 day to 1 week
            return 0.7 + 0.3 * (time_to_event_hours - 24) / 144
        elif time_to_event_hours > 1:  # 1 hour to 1 day
            return 0.3 + 0.4 * (time_to_event_hours - 1) / 23
        else:  # < 1 hour
            return 0.3
    
    def combine_estimates(
        self,
        estimates: list[tuple[float, float]],
        model_name: str = "combined"
    ) -> ProbabilityResult:
        """
        Combine multiple estimates with confidence weighting.
        
        Args:
            estimates: List of (probability, confidence) tuples
            model_name: Name for the combined model
            
        Returns:
            Combined ProbabilityResult
        """
        if not estimates:
            return ProbabilityResult(
                probability=0.5,
                confidence=0.0,
                confidence_level=ConfidenceLevel.VERY_LOW,
                lower_bound=0.25,
                upper_bound=0.75,
                model_name=model_name,
                category=self.category,
            )
        
        # Normalize confidences
        total_confidence = sum(conf for _, conf in estimates)
        if total_confidence == 0:
            weights = [1.0 / len(estimates)] * len(estimates)
        else:
            weights = [conf / total_confidence for _, conf in estimates]
        
        # Weighted average
        combined_prob = sum(
            prob * weight for (prob, _), weight in zip(estimates, weights)
        )
        
        # Combined confidence (not just weighted, but also considers agreement)
        probs = [prob for prob, _ in estimates]
        if len(probs) > 1:
            # Calculate variance
            mean_prob = sum(probs) / len(probs)
            variance = sum((p - mean_prob) ** 2 for p in probs) / (len(probs) - 1)
            agreement = max(0, 1 - variance * 10)  # Higher variance = lower agreement
        else:
            agreement = 1.0
        
        combined_confidence = (total_confidence / len(estimates)) * agreement
        
        # Uncertainty bounds based on variance
        std_dev = (variance ** 0.5) if len(probs) > 1 else 0.15
        lower = combined_prob - 1.96 * std_dev
        upper = combined_prob + 1.96 * std_dev
        
        return ProbabilityResult(
            probability=combined_prob,
            confidence=combined_confidence,
            confidence_level=confidence_to_level(combined_confidence),
            lower_bound=lower,
            upper_bound=upper,
            model_name=model_name,
            category=self.category,
            source_breakdown={f"source_{i}": prob for i, (prob, _) in enumerate(estimates)},
        )
