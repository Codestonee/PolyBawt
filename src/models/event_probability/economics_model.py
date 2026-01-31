"""Economics probability model - Fed futures + consensus forecasts."""

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any

from .base_estimator import (
    CategoryProbabilityEstimator, 
    ProbabilityResult, 
    ConfidenceLevel
)
from src.ingestion.event_market_discovery import EventMarket, MarketCategory
from src.ingestion.external_data.economic_forecasts import (
    EconomicForecasts,
    EconomicData,
    EconomicIndicator,
    FedFutures
)
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EconomicsContext:
    """Additional context for economics markets."""
    
    economic_data: EconomicData | None = None
    
    # Leading indicators
    leading_indicators: dict[str, float] = None
    
    # Historical accuracy
    historical_forecast_errors: list[float] = None


class EconomicsProbabilityModel(CategoryProbabilityEstimator):
    """
    Fed futures + consensus forecasts for economic markets.
    
    Model components:
    1. Fed funds futures (60% weight when available)
       - 90%+ accuracy 30 days out
       - Direct market pricing of Fed policy
    
    2. Consensus forecasts (25% weight)
       - Economist surveys
       - Leading indicators
    
    3. Base rate / historical frequency (15% weight)
       - How often does this outcome occur historically?
    
    Time weighting:
    - Far from event: Weight consensus more
    - Close to event: Weight futures more
    - Fed futures have 90%+ accuracy 30 days out
    
    Usage:
        model = EconomicsProbabilityModel()
        result = await model.estimate(market, context=econ_context)
    """
    
    def __init__(self, forecasts_source: EconomicForecasts | None = None):
        super().__init__(name="EconomicsProbabilityModel")
        self.forecasts_source = forecasts_source or EconomicForecasts()
    
    @property
    def category(self) -> MarketCategory:
        return MarketCategory.ECONOMICS
    
    async def estimate(
        self,
        market: EventMarket,
        **context
    ) -> ProbabilityResult:
        """
        Estimate probability for an economics market.
        
        Args:
            market: Economics event market
            **context: Optional EconomicsContext
            
        Returns:
            ProbabilityResult
        """
        econ_context = context.get("context", EconomicsContext())
        econ_data = econ_context.economic_data
        
        # Time to event
        time_to_event = market.time_to_resolution
        time_to_event_days = time_to_event / 24 if time_to_event else 30
        
        # Component estimates
        estimates: list[tuple[float, float]] = []
        source_breakdown: dict[str, float] = {}
        
        # 1. Fed futures (if applicable)
        futures_prob = self._estimate_from_futures(econ_data)
        if futures_prob is not None:
            # Weight increases as event approaches
            futures_weight = self._futures_weight(time_to_event_days)
            estimates.append((futures_prob, futures_weight))
            source_breakdown["fed_futures"] = futures_prob
        
        # 2. Consensus forecast
        consensus_prob = self._estimate_from_consensus(econ_data)
        if consensus_prob is not None:
            consensus_weight = 1.0 - (estimates[-1][1] if estimates else 0.0)
            if consensus_weight > 0:
                estimates.append((consensus_prob, consensus_weight * 0.8))
                source_breakdown["consensus"] = consensus_prob
        
        # 3. Leading indicators
        leading_prob = self._estimate_from_leading(econ_context)
        if leading_prob is not None:
            estimates.append((leading_prob, 0.3))
            source_breakdown["leading"] = leading_prob
        
        # 4. Market signal (OBI)
        obi_signal = context.get("obi_signal")
        if obi_signal is not None:
            market_prob = 0.5 + (obi_signal * 0.15)
            estimates.append((market_prob, 0.15))
            source_breakdown["obi"] = market_prob
        
        # Combine estimates
        if not estimates:
            return ProbabilityResult(
                probability=market.yes_price,
                confidence=0.3,
                confidence_level=ConfidenceLevel.LOW,
                lower_bound=market.yes_price - 0.15,
                upper_bound=market.yes_price + 0.15,
                model_name=self.name,
                category=self.category,
                features={"fallback_to_market": True},
            )
        
        # Weighted combination
        total_weight = sum(weight for _, weight in estimates)
        combined_prob = sum(
            prob * weight for prob, weight in estimates
        ) / total_weight
        
        # Confidence is high for Fed futures close to event
        if futures_prob is not None and time_to_event_days < 30:
            confidence = 0.85
        elif futures_prob is not None:
            confidence = 0.75
        elif consensus_prob is not None:
            confidence = 0.65
        else:
            confidence = 0.50
        
        # Uncertainty based on time to event
        # Closer = less uncertainty
        if time_to_event_days < 7:
            std_dev = 0.05
        elif time_to_event_days < 30:
            std_dev = 0.08
        else:
            std_dev = 0.12
        
        result = ProbabilityResult(
            probability=combined_prob,
            confidence=confidence,
            confidence_level=ConfidenceLevel.HIGH if confidence > 0.75 else ConfidenceLevel.MEDIUM,
            lower_bound=max(0.01, combined_prob - 1.96 * std_dev),
            upper_bound=min(0.99, combined_prob + 1.96 * std_dev),
            model_name=self.name,
            category=self.category,
            source_breakdown=source_breakdown,
            features={
                "time_to_event_days": time_to_event_days,
                "has_futures": futures_prob is not None,
                "has_consensus": consensus_prob is not None,
                "futures_weight": source_breakdown.get("fed_futures", 0),
            },
        )
        
        return result
    
    def _estimate_from_futures(self, econ_data: EconomicData | None) -> float | None:
        """
        Estimate from Fed funds futures.
        
        Returns:
            Probability or None
        """
        if econ_data is None or econ_data.fed_futures is None:
            return None
        
        futures = econ_data.fed_futures
        
        # Determine what the market is asking
        # For binary markets, we need to map futures to the question
        
        # Example: "Will Fed hike rates in March?"
        # prob = prob_25bp_hike + prob_50bp_hike
        hike_prob = futures.prob_25bp_hike + futures.prob_50bp_hike
        cut_prob = futures.prob_25bp_cut + futures.prob_50bp_cut
        
        # For "Will Fed hike?" market
        return hike_prob / (hike_prob + cut_prob + futures.prob_no_change)
    
    def _estimate_from_consensus(self, econ_data: EconomicData | None) -> float | None:
        """
        Estimate from consensus forecasts.
        
        Returns:
            Probability or None
        """
        if econ_data is None or econ_data.consensus is None:
            return None
        
        consensus = econ_data.consensus
        
        # For binary markets, convert continuous forecast to probability
        # Using the probability of being above/below a threshold
        
        # Simplified: Assume normal distribution around consensus
        # P(X > threshold) based on std dev
        
        # For now, return a neutral-ish estimate based on consensus vs previous
        if consensus.previous_value == 0:
            return 0.5
        
        # If consensus is higher than previous, probability of "increase" is > 0.5
        ratio = consensus.consensus / consensus.previous_value
        prob = 1 / (1 + 2.71828 ** (-10 * (ratio - 1)))  # Logistic transform
        
        return prob
    
    def _estimate_from_leading(self, context: EconomicsContext) -> float | None:
        """
        Estimate from leading indicators.
        
        Returns:
            Probability or None
        """
        if context.leading_indicators is None:
            return None
        
        # Aggregate leading indicator signals
        # Positive values = indicator suggests YES outcome
        
        signals = list(context.leading_indicators.values())
        if not signals:
            return None
        
        avg_signal = sum(signals) / len(signals)
        
        # Convert to probability
        prob = 0.5 + (avg_signal * 0.3)  # Scale to 0.2-0.8 range
        return max(0.1, min(0.9, prob))
    
    def _futures_weight(self, time_to_event_days: float) -> float:
        """
        Calculate weight for Fed futures based on time to event.
        
        Fed futures are very accurate close to event.
        
        Args:
            time_to_event_days: Days until event
            
        Returns:
            Weight (0-1)
        """
        if time_to_event_days < 7:
            return 0.70  # Very high weight when close
        elif time_to_event_days < 30:
            return 0.60  # High weight
        elif time_to_event_days < 90:
            return 0.50  # Moderate weight
        else:
            return 0.40  # Lower weight when far out
    
    def check_divergence(
        self,
        futures: FedFutures | None,
        consensus: float | None
    ) -> dict[str, Any]:
        """
        Check for divergence between futures and consensus.
        
        Divergence = trading opportunity.
        
        Returns:
            Divergence analysis
        """
        if futures is None or consensus is None:
            return {"divergence": False}
        
        futures_expected = futures.expected_rate
        
        # Divergence magnitude
        divergence = abs(futures_expected - consensus)
        
        return {
            "divergence": divergence > 0.25,  # 25bp difference
            "divergence_bp": divergence * 100,  # In basis points
            "futures_expected": futures_expected,
            "consensus": consensus,
            "signal": "futures_higher" if futures_expected > consensus else "consensus_higher",
        }


# Pre-instantiated model
economics_model = EconomicsProbabilityModel()
