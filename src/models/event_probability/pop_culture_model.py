"""Pop Culture probability model - market aggregation + sentiment."""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .base_estimator import (
    CategoryProbabilityEstimator, 
    ProbabilityResult, 
    ConfidenceLevel
)
from src.ingestion.event_market_discovery import EventMarket, MarketCategory
from src.ingestion.external_data.sentiment_analyzer import (
    SentimentAnalyzer,
    SentimentData,
    SentimentSource
)
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PopCultureContext:
    """Additional context for pop culture markets."""
    
    sentiment_data: SentimentData | None = None
    
    # Market data from other platforms
    metaculus_probability: float | None = None
    manifold_probability: float | None = None
    predictit_probability: float | None = None
    
    # Expert predictions (critics, etc.)
    expert_predictions: dict[str, float] = None  # Expert -> probability
    
    # Historical data
    historical_base_rate: float | None = None  # How often does favorite win?


class PopCultureProbabilityModel(CategoryProbabilityEstimator):
    """
    Market aggregation + sentiment analysis for pop culture markets.
    
    Pop culture markets (Oscars, Grammys, etc.) have unique characteristics:
    - Hard to model mathematically
    - Social sentiment can be predictive
    - Expert opinions matter (critics)
    - Cross-platform aggregation provides signal
    
    Model components:
    1. Cross-platform consensus (35% weight)
       - Metaculus community
       - Manifold Markets
       - PredictIt
    
    2. Social sentiment (30% weight)
       - Twitter/X sentiment analysis
       - Reddit mentions
       - News sentiment
    
    3. Expert predictions (25% weight)
       - Aggregated critic predictions
       - Industry expert forecasts
    
    4. Historical base rates (10% weight)
       - How often do favorites win?
    
    Usage:
        model = PopCultureProbabilityModel()
        result = await model.estimate(market, context=pop_context)
    """
    
    def __init__(self, sentiment_analyzer: SentimentAnalyzer | None = None):
        super().__init__(name="PopCultureProbabilityModel")
        self.sentiment_analyzer = sentiment_analyzer or SentimentAnalyzer()
        
        # Model weights
        self.PLATFORM_WEIGHT = 0.35
        self.SENTIMENT_WEIGHT = 0.30
        self.EXPERT_WEIGHT = 0.25
        self.BASE_RATE_WEIGHT = 0.10
    
    @property
    def category(self) -> MarketCategory:
        return MarketCategory.POP_CULTURE
    
    async def estimate(
        self,
        market: EventMarket,
        **context
    ) -> ProbabilityResult:
        """
        Estimate probability for a pop culture market.
        
        Args:
            market: Pop culture event market
            **context: Optional PopCultureContext
            
        Returns:
            ProbabilityResult
        """
        pop_context = context.get("context", PopCultureContext())
        
        # Component estimates
        estimates: list[tuple[float, float]] = []
        source_breakdown: dict[str, float] = {}
        
        # 1. Cross-platform consensus
        platform_prob = self._estimate_from_platforms(pop_context)
        if platform_prob is not None:
            estimates.append((platform_prob, self.PLATFORM_WEIGHT))
            source_breakdown["platform_consensus"] = platform_prob
        
        # 2. Social sentiment
        sentiment_prob = self._estimate_from_sentiment(pop_context)
        if sentiment_prob is not None:
            estimates.append((sentiment_prob, self.SENTIMENT_WEIGHT))
            source_breakdown["sentiment"] = sentiment_prob
        
        # 3. Expert predictions
        expert_prob = self._estimate_from_experts(pop_context)
        if expert_prob is not None:
            estimates.append((expert_prob, self.EXPERT_WEIGHT))
            source_breakdown["experts"] = expert_prob
        
        # 4. Historical base rate
        base_rate_prob = self._estimate_from_base_rate(pop_context)
        if base_rate_prob is not None:
            estimates.append((base_rate_prob, self.BASE_RATE_WEIGHT))
            source_breakdown["base_rate"] = base_rate_prob
        
        # 5. Market signal (OBI)
        obi_signal = context.get("obi_signal")
        if obi_signal is not None:
            market_prob = 0.5 + (obi_signal * 0.15)
            estimates.append((market_prob, 0.15))
            source_breakdown["obi"] = market_prob
        
        # Combine estimates
        if not estimates:
            return ProbabilityResult(
                probability=market.yes_price,
                confidence=0.25,
                confidence_level=ConfidenceLevel.LOW,
                lower_bound=market.yes_price - 0.2,
                upper_bound=market.yes_price + 0.2,
                model_name=self.name,
                category=self.category,
                features={"fallback_to_market": True},
            )
        
        # Weighted combination
        total_weight = sum(weight for _, weight in estimates)
        combined_prob = sum(
            prob * weight for prob, weight in estimates
        ) / total_weight
        
        # Confidence based on data diversity
        num_sources = len(estimates)
        if num_sources >= 4:
            confidence = 0.70
        elif num_sources >= 3:
            confidence = 0.60
        elif num_sources >= 2:
            confidence = 0.50
        else:
            confidence = 0.40
        
        # Pop culture has inherently higher uncertainty
        std_dev = 0.15
        
        result = ProbabilityResult(
            probability=combined_prob,
            confidence=confidence,
            confidence_level=ConfidenceLevel.MEDIUM if confidence >= 0.5 else ConfidenceLevel.LOW,
            lower_bound=max(0.01, combined_prob - 1.96 * std_dev),
            upper_bound=min(0.99, combined_prob + 1.96 * std_dev),
            model_name=self.name,
            category=self.category,
            source_breakdown=source_breakdown,
            features={
                "num_sources": num_sources,
                "has_sentiment": sentiment_prob is not None,
                "has_expert": expert_prob is not None,
                "platform_divergence": self._calculate_divergence(pop_context),
            },
        )
        
        return result
    
    def _estimate_from_platforms(self, context: PopCultureContext) -> float | None:
        """
        Estimate from cross-platform consensus.
        
        Returns:
            Probability or None
        """
        probs: list[tuple[float, float]] = []  # (prob, weight)
        
        if context.metaculus_probability is not None:
            probs.append((context.metaculus_probability, 1.0))
        
        if context.manifold_probability is not None:
            probs.append((context.manifold_probability, 0.8))
        
        if context.predictit_probability is not None:
            probs.append((context.predictit_probability, 0.9))
        
        if not probs:
            return None
        
        # Weighted average
        total_weight = sum(weight for _, weight in probs)
        combined = sum(prob * weight for prob, weight in probs) / total_weight
        
        return combined
    
    def _estimate_from_sentiment(self, context: PopCultureContext) -> float | None:
        """
        Estimate from social sentiment.
        
        Returns:
            Probability or None
        """
        if context.sentiment_data is None:
            return None
        
        # Convert sentiment to probability
        return context.sentiment_data.to_probability(baseline=0.5)
    
    def _estimate_from_experts(self, context: PopCultureContext) -> float | None:
        """
        Estimate from expert predictions.
        
        Returns:
            Probability or None
        """
        if context.expert_predictions is None:
            return None
        
        predictions = list(context.expert_predictions.values())
        if not predictions:
            return None
        
        # Simple average of expert predictions
        return sum(predictions) / len(predictions)
    
    def _estimate_from_base_rate(self, context: PopCultureContext) -> float | None:
        """
        Estimate from historical base rate.
        
        Returns:
            Probability or None
        """
        if context.historical_base_rate is None:
            return None
        
        return context.historical_base_rate
    
    def _calculate_divergence(self, context: PopCultureContext) -> float:
        """
        Calculate divergence between platforms.
        
        High divergence = uncertainty = lower confidence.
        
        Returns:
            Standard deviation of platform probabilities
        """
        probs: list[float] = []
        
        if context.metaculus_probability is not None:
            probs.append(context.metaculus_probability)
        if context.manifold_probability is not None:
            probs.append(context.manifold_probability)
        if context.predictit_probability is not None:
            probs.append(context.predictit_probability)
        
        if len(probs) < 2:
            return 0.0
        
        mean = sum(probs) / len(probs)
        variance = sum((p - mean) ** 2 for p in probs) / len(probs)
        
        return variance ** 0.5


# Pre-instantiated model
pop_culture_model = PopCultureProbabilityModel()
