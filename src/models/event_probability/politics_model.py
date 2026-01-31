"""Politics probability model - poll aggregation with base rates."""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .base_estimator import (
    CategoryProbabilityEstimator, 
    ProbabilityResult, 
    ConfidenceLevel,
    confidence_to_level
)
from src.ingestion.event_market_discovery import EventMarket, MarketCategory
from src.ingestion.external_data.poll_aggregator import PollAggregator, AggregatedPolls
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PoliticalContext:
    """Additional context for political markets."""
    
    # Incumbency
    is_incumbent_running: bool = False
    incumbent_approval_rating: float | None = None
    
    # Economic context
    gdp_growth_qoq: float | None = None  # GDP growth quarter-over-quarter
    unemployment_rate: float | None = None
    inflation_rate: float | None = None
    
    # Historical
    party_last_won: bool | None = None  # Did this party win last time?
    
    # Poll data
    aggregated_polls: AggregatedPolls | None = None


class PoliticsProbabilityModel(CategoryProbabilityEstimator):
    """
    Poll aggregation with recency weighting and base rates.
    
    Model components:
    1. Poll aggregation (60% weight)
       - Recency weighting (7-day half-life)
       - Sample size weighting
       - Pollster rating weighting
    
    2. Base rate adjustments (20% weight)
       - Incumbency advantage (+3% historically)
       - Economic correlation (approval -> vote)
    
    3. Market signal (20% weight)
       - Order book imbalance
       - Recent price action
    
    Usage:
        model = PoliticsProbabilityModel()
        result = await model.estimate(market, context=political_context)
    """
    
    def __init__(self, poll_aggregator: PollAggregator | None = None):
        super().__init__(name="PoliticsProbabilityModel")
        self.poll_aggregator = poll_aggregator or PollAggregator()
        
        # Historical base rates
        self.INCUMBENCY_ADVANTAGE = 0.03  # 3% boost for incumbents
        self.ECONOMIC_CORRELATION = 0.6   # 60% of approval translates to vote
    
    @property
    def category(self) -> MarketCategory:
        return MarketCategory.POLITICS
    
    async def estimate(
        self,
        market: EventMarket,
        **context
    ) -> ProbabilityResult:
        """
        Estimate probability for a political market.
        
        Args:
            market: Political event market
            **context: Optional PoliticalContext
            
        Returns:
            ProbabilityResult
        """
        political_context = context.get("context", PoliticalContext())
        
        # Component estimates
        estimates: list[tuple[float, float]] = []  # (probability, confidence)
        
        # 1. Poll-based estimate
        poll_estimate = self._estimate_from_polls(market, political_context)
        if poll_estimate:
            estimates.append(poll_estimate)
        
        # 2. Base rate estimate (incumbency + economy)
        base_rate_estimate = self._estimate_from_base_rates(political_context)
        if base_rate_estimate:
            estimates.append(base_rate_estimate)
        
        # 3. Market-implied (order book signal)
        obi_signal = context.get("obi_signal")
        if obi_signal is not None:
            # Convert OBI (-1 to +1) to probability adjustment
            market_prob = 0.5 + (obi_signal * 0.2)  # Scale OBI
            estimates.append((market_prob, 0.6))  # 60% confidence in OBI
        
        # Combine estimates
        if not estimates:
            # Fallback to market price with low confidence
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
        
        result = self.combine_estimates(estimates, model_name=self.name)
        result.category = self.category
        
        # Add political-specific features
        result.features = {
            "poll_count": political_context.aggregated_polls.poll_count if political_context.aggregated_polls else 0,
            "incumbent_running": political_context.is_incumbent_running,
            "has_approval_data": political_context.incumbent_approval_rating is not None,
        }
        
        return result
    
    def _estimate_from_polls(
        self,
        market: EventMarket,
        context: PoliticalContext
    ) -> tuple[float, float] | None:
        """
        Estimate from aggregated polls.
        
        Returns:
            (probability, confidence) or None
        """
        polls = context.aggregated_polls
        if polls is None:
            return None
        
        # Convert poll average to probability
        # Assuming candidate_a_avg is the "YES" outcome
        probability = polls.candidate_a_avg / 100.0
        
        # Confidence based on:
        # - Number of polls
        # - Agreement between polls
        # - Sample size
        poll_confidence = min(1.0, polls.poll_count / 10)  # Max at 10+ polls
        agreement_confidence = polls.agreement_score
        sample_confidence = min(1.0, polls.weighted_sample_size / 5000)
        
        confidence = (poll_confidence + agreement_confidence + sample_confidence) / 3
        
        return (probability, confidence)
    
    def _estimate_from_base_rates(
        self,
        context: PoliticalContext
    ) -> tuple[float, float] | None:
        """
        Estimate from historical base rates.
        
        Returns:
            (probability, confidence) or None
        """
        if not context.is_incumbent_running:
            return None
        
        # Start with neutral
        base_probability = 0.5
        
        # Incumbency advantage
        base_probability += self.INCUMBENCY_ADVANTAGE
        
        # Economic adjustment (if approval data available)
        if context.incumbent_approval_rating is not None:
            # Normalize approval (50% is neutral)
            approval_effect = (context.incumbent_approval_rating - 50) / 100
            base_probability += approval_effect * self.ECONOMIC_CORRELATION
        
        # Base rate confidence is lower than polls
        confidence = 0.5  # Base rates are less certain
        
        return (base_probability, confidence)
    
    def apply_time_adjustment(
        self,
        probability: float,
        time_to_election_days: float
    ) -> float:
        """
        Adjust probability based on time to election.
        
        Early in cycle: More uncertainty, revert to 0.5
        Late in cycle: Polls more reliable
        
        Args:
            probability: Current probability estimate
            time_to_election_days: Days until election
            
        Returns:
            Adjusted probability
        """
        if time_to_election_days > 180:
            # > 6 months: high uncertainty
            reversion_factor = 0.5  # Move halfway to 0.5
        elif time_to_election_days > 30:
            # 1-6 months: moderate uncertainty
            reversion_factor = 0.3
        elif time_to_election_days > 7:
            # 1 week to 1 month: low uncertainty
            reversion_factor = 0.1
        else:
            # < 1 week: minimal adjustment
            reversion_factor = 0.0
        
        # Revert probability toward 0.5
        adjusted = probability * (1 - reversion_factor) + 0.5 * reversion_factor
        return adjusted


# Pre-instantiated model
politics_model = PoliticsProbabilityModel()
