"""Event ensemble model - combines category models with dynamic weighting."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .base_estimator import (
    CategoryProbabilityEstimator,
    ProbabilityResult,
    ConfidenceLevel,
    confidence_to_level
)
from .politics_model import PoliticsProbabilityModel
from .sports_model import SportsProbabilityModel
from .economics_model import EconomicsProbabilityModel
from .pop_culture_model import PopCultureProbabilityModel

from src.ingestion.event_market_discovery import EventMarket, MarketCategory
from src.strategy.features.order_imbalance import OBIResult
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EnsembleWeights:
    """Weights for ensemble combination."""
    
    category_model: float = 0.60    # Domain-specific estimate
    orderbook_signal: float = 0.20  # OBI signal (65-75% accuracy!)
    base_rate: float = 0.10         # Historical frequencies
    agreement_bonus: float = 0.10   # Bonus when sources agree
    
    def normalize(self) -> "EnsembleWeights":
        """Normalize weights to sum to 1.0."""
        total = self.category_model + self.orderbook_signal + self.base_rate + self.agreement_bonus
        if total == 0:
            return EnsembleWeights(0.25, 0.25, 0.25, 0.25)
        
        return EnsembleWeights(
            category_model=self.category_model / total,
            orderbook_signal=self.orderbook_signal / total,
            base_rate=self.base_rate / total,
            agreement_bonus=self.agreement_bonus / total,
        )


@dataclass
class EnsembleResult:
    """Result from ensemble model."""
    
    # Final probability
    probability: float
    confidence: float
    confidence_level: ConfidenceLevel
    
    # Component results
    category_result: ProbabilityResult | None = None
    obi_adjustment: float = 0.0
    base_rate_prob: float = 0.5
    
    # Metadata
    weights_used: EnsembleWeights = field(default_factory=lambda: EnsembleWeights())
    source_agreement: float = 0.0  # 0-1, how much do sources agree
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # For debugging
    component_probs: dict[str, float] = field(default_factory=dict)


class EventEnsembleModel:
    """
    Ensemble model combining multiple probability sources.
    
    Ensemble composition:
    - Category model: 60% (domain-specific probability)
    - Order book signal: 20% (65-75% accuracy!)
    - Base rate: 10% (historical frequencies)
    - Agreement bonus: 10% (boost when sources agree)
    
    Key insight from research:
    Order Book Imbalance achieves 65-75% accuracy vs 55-65% for traditional models.
    The edge comes from microstructure features, not just probability estimation.
    
    Usage:
        ensemble = EventEnsembleModel()
        
        result = await ensemble.estimate(
            market=event_market,
            category=MarketCategory.POLITICS,
            obi_result=obi_result,
        )
        
        probability = result.probability
        confidence = result.confidence
    """
    
    # Default weights
    DEFAULT_WEIGHTS = EnsembleWeights(
        category_model=0.60,
        orderbook_signal=0.20,
        base_rate=0.10,
        agreement_bonus=0.10,
    )
    
    def __init__(
        self,
        politics_model: PoliticsProbabilityModel | None = None,
        sports_model: SportsProbabilityModel | None = None,
        economics_model: EconomicsProbabilityModel | None = None,
        pop_culture_model: PopCultureProbabilityModel | None = None,
        weights: EnsembleWeights | None = None,
    ):
        self.politics_model = politics_model or PoliticsProbabilityModel()
        self.sports_model = sports_model or SportsProbabilityModel()
        self.economics_model = economics_model or EconomicsProbabilityModel()
        self.pop_culture_model = pop_culture_model or PopCultureProbabilityModel()
        
        self.weights = weights or self.DEFAULT_WEIGHTS
        
        # Base rates by category (simplified)
        self.base_rates = {
            MarketCategory.POLITICS: 0.50,      # Elections are roughly 50/50
            MarketCategory.SPORTS: 0.55,        # Slight home field advantage
            MarketCategory.ECONOMICS: 0.50,     # Neutral for most econ events
            MarketCategory.POP_CULTURE: 0.30,   # Favorites often win
        }
    
    def get_model_for_category(
        self,
        category: MarketCategory
    ) -> CategoryProbabilityEstimator:
        """Get the appropriate model for a category."""
        models = {
            MarketCategory.POLITICS: self.politics_model,
            MarketCategory.SPORTS: self.sports_model,
            MarketCategory.ECONOMICS: self.economics_model,
            MarketCategory.POP_CULTURE: self.pop_culture_model,
        }
        
        return models.get(category, self.politics_model)  # Default to politics
    
    async def estimate(
        self,
        market: EventMarket,
        category: MarketCategory | None = None,
        obi_result: OBIResult | None = None,
        context: Any | None = None,
        **kwargs
    ) -> EnsembleResult:
        """
        Estimate probability using ensemble of models.
        
        Args:
            market: Event market to estimate
            category: Market category (auto-detected if None)
            obi_result: Order book imbalance result
            context: Category-specific context
            **kwargs: Additional parameters
            
        Returns:
            EnsembleResult with combined probability
        """
        # Determine category
        if category is None:
            category = market.category
        
        # Get category model
        category_model = self.get_model_for_category(category)
        
        # 1. Get category model estimate
        category_prob = await category_model.estimate(
            market,
            context=context,
            obi_signal=obi_result.obi if obi_result else None,
        )
        
        # 2. Extract OBI signal (already factored into category model, but track separately)
        obi_prob = 0.5
        if obi_result and obi_result.is_reliable:
            # Convert OBI to probability
            # OBI > 0 = more bids = buying pressure = higher YES price expected
            obi_prob = 0.5 + (obi_result.obi * 0.3)  # Scale OBI
            obi_prob = max(0.1, min(0.9, obi_prob))
        
        # 3. Get base rate
        base_rate = self.base_rates.get(category, 0.5)
        
        # 4. Calculate agreement
        probs = [category_prob.probability, obi_prob, base_rate]
        agreement = self._calculate_agreement(probs)
        
        # Adjust weights based on confidence
        weights = self._adjust_weights(category_prob.confidence, obi_result)
        
        # Combine estimates
        # Category model contribution
        category_contrib = category_prob.probability * weights.category_model
        
        # OBI contribution
        obi_contrib = obi_prob * weights.orderbook_signal
        
        # Base rate contribution
        base_contrib = base_rate * weights.base_rate
        
        # Agreement bonus (push toward category model when sources agree)
        agreement_contrib = 0.0
        if agreement > 0.7:
            # Sources agree - boost category model
            agreement_contrib = category_prob.probability * weights.agreement_bonus
        else:
            # Sources disagree - distribute bonus
            agreement_contrib = sum(probs) / len(probs) * weights.agreement_bonus
        
        # Final probability
        combined_prob = category_contrib + obi_contrib + base_contrib + agreement_contrib
        
        # Normalize (weights should sum to 1, but just in case)
        total_weight = (
            weights.category_model + weights.orderbook_signal + 
            weights.base_rate + weights.agreement_bonus
        )
        combined_prob /= total_weight
        
        # Calculate confidence
        # Start with category model confidence
        confidence = category_prob.confidence
        
        # Boost confidence if OBI agrees
        if obi_result and obi_result.is_reliable:
            obi_direction = "up" if obi_result.obi > 0 else "down"
            model_direction = "up" if category_prob.probability > 0.5 else "down"
            if obi_direction == model_direction:
                confidence = min(0.95, confidence * 1.1)
        
        # Reduce confidence if low agreement
        if agreement < 0.5:
            confidence *= 0.8
        
        # Calculate uncertainty bounds
        std_dev = category_prob.uncertainty_range / 2
        lower = max(0.01, combined_prob - 1.96 * std_dev)
        upper = min(0.99, combined_prob + 1.96 * std_dev)
        
        return EnsembleResult(
            probability=combined_prob,
            confidence=confidence,
            confidence_level=confidence_to_level(confidence),
            category_result=category_prob,
            obi_adjustment=obi_prob - 0.5,
            base_rate_prob=base_rate,
            weights_used=weights,
            source_agreement=agreement,
            component_probs={
                "category_model": category_prob.probability,
                "obi_signal": obi_prob,
                "base_rate": base_rate,
            },
        )
    
    def _calculate_agreement(self, probs: list[float]) -> float:
        """
        Calculate agreement between probability estimates.
        
        Returns:
            Agreement score (0-1, higher = more agreement)
        """
        if len(probs) < 2:
            return 1.0
        
        mean = sum(probs) / len(probs)
        variance = sum((p - mean) ** 2 for p in probs) / len(probs)
        
        # Max variance at 0.5 spread is 0.25
        # Normalize to 0-1
        agreement = max(0, 1 - variance * 4)
        
        return agreement
    
    def _adjust_weights(
        self,
        category_confidence: float,
        obi_result: OBIResult | None
    ) -> EnsembleWeights:
        """
        Adjust weights based on signal quality.
        
        Args:
            category_confidence: Confidence in category model
            obi_result: OBI result
            
        Returns:
            Adjusted weights
        """
        weights = EnsembleWeights(
            category_model=self.DEFAULT_WEIGHTS.category_model,
            orderbook_signal=self.DEFAULT_WEIGHTS.orderbook_signal,
            base_rate=self.DEFAULT_WEIGHTS.base_rate,
            agreement_bonus=self.DEFAULT_WEIGHTS.agreement_bonus,
        )
        
        # If category model is low confidence, reduce its weight
        if category_confidence < 0.5:
            reduction = (0.5 - category_confidence) * 0.5
            weights.category_model -= reduction
            weights.orderbook_signal += reduction * 0.5
            weights.base_rate += reduction * 0.5
        
        # If OBI is not reliable, reduce its weight
        if obi_result is None or not obi_result.is_reliable:
            obi_reduction = weights.orderbook_signal * 0.5
            weights.orderbook_signal -= obi_reduction
            weights.category_model += obi_reduction
        
        # Normalize
        return weights.normalize()
    
    def update_base_rate(
        self,
        category: MarketCategory,
        outcome: bool,
        learning_rate: float = 0.05
    ) -> None:
        """
        Update base rate based on observed outcomes.
        
        Args:
            category: Market category
            outcome: True if YES outcome occurred
            learning_rate: How much to adjust base rate
        """
        current = self.base_rates.get(category, 0.5)
        target = 1.0 if outcome else 0.0
        
        new_rate = current * (1 - learning_rate) + target * learning_rate
        self.base_rates[category] = new_rate
        
        logger.debug(
            "Updated base rate",
            category=category.value,
            old_rate=current,
            new_rate=new_rate,
        )


# Pre-instantiated model
event_ensemble = EventEnsembleModel()
