"""Sports probability model - ELO + betting line comparison."""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .base_estimator import (
    CategoryProbabilityEstimator, 
    ProbabilityResult, 
    ConfidenceLevel
)
from src.ingestion.event_market_discovery import EventMarket, MarketCategory
from src.ingestion.external_data.sports_stats import (
    SportsStatsSource, 
    SportsData, 
    EloRatings,
    BettingLine
)
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SportsContext:
    """Additional context for sports markets."""
    
    # Team data
    sports_data: SportsData | None = None
    
    # Injury info
    team_a_key_injuries: list[str] = None
    team_b_key_injuries: list[str] = None
    
    # Context
    is_playoffs: bool = False
    is_home_game_for_a: bool = True
    
    # Weather (for outdoor sports)
    weather_impact: float = 0.0  # -1 to +1, impact on team A


class SportsProbabilityModel(CategoryProbabilityEstimator):
    """
    ELO + statistics + betting line edge detection.
    
    Model components:
    1. ELO-based probability (40% weight)
       - Base ELO ratings
       - Home advantage adjustment
       - Rest days adjustment
       - Injury adjustments
    
    2. Vegas implied probability (30% weight)
       - Moneyline consensus
       - Vig removal
       - Line movement analysis
    
    3. Recent form (20% weight)
       - Last 10 games
       - Head-to-head history
    
    4. Market signal (10% weight)
       - Order book imbalance
    
    Edge exists when ELO model diverges from market.
    
    Usage:
        model = SportsProbabilityModel()
        result = await model.estimate(market, context=sports_context)
    """
    
    def __init__(self, stats_source: SportsStatsSource | None = None):
        super().__init__(name="SportsProbabilityModel")
        self.stats_source = stats_source or SportsStatsSource()
        
        # Model weights
        self.ELO_WEIGHT = 0.40
        self.VEGAS_WEIGHT = 0.30
        self.FORM_WEIGHT = 0.20
        self.MARKET_WEIGHT = 0.10
    
    @property
    def category(self) -> MarketCategory:
        return MarketCategory.SPORTS
    
    async def estimate(
        self,
        market: EventMarket,
        **context
    ) -> ProbabilityResult:
        """
        Estimate probability for a sports market.
        
        Args:
            market: Sports event market
            **context: Optional SportsContext
            
        Returns:
            ProbabilityResult
        """
        sports_context = context.get("context", SportsContext())
        sports_data = sports_context.sports_data
        
        # Component estimates
        estimates: list[tuple[float, float]] = []
        source_breakdown: dict[str, float] = {}
        
        # 1. ELO-based estimate
        elo_prob = self._estimate_from_elo(sports_context)
        if elo_prob is not None:
            estimates.append((elo_prob, self.ELO_WEIGHT))
            source_breakdown["elo"] = elo_prob
        
        # 2. Vegas implied probability
        vegas_prob = self._estimate_from_vegas(sports_data)
        if vegas_prob is not None:
            estimates.append((vegas_prob, self.VEGAS_WEIGHT))
            source_breakdown["vegas"] = vegas_prob
        
        # 3. Recent form
        form_prob = self._estimate_from_form(sports_context)
        if form_prob is not None:
            estimates.append((form_prob, self.FORM_WEIGHT))
            source_breakdown["form"] = form_prob
        
        # 4. Market signal (OBI)
        obi_signal = context.get("obi_signal")
        if obi_signal is not None:
            market_prob = 0.5 + (obi_signal * 0.15)
            estimates.append((market_prob, self.MARKET_WEIGHT))
            source_breakdown["obi"] = market_prob
        
        # Combine estimates
        if not estimates:
            return ProbabilityResult(
                probability=market.yes_price,
                confidence=0.3,
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
        
        # Confidence based on data availability
        confidence = min(0.9, 0.5 + len(estimates) * 0.15)
        
        # Uncertainty based on model disagreement
        probs = [prob for prob, _ in estimates]
        if len(probs) > 1:
            mean_prob = sum(probs) / len(probs)
            variance = sum((p - mean_prob) ** 2 for p in probs) / len(probs)
            std_dev = variance ** 0.5
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
                "has_elo": elo_prob is not None,
                "has_vegas": vegas_prob is not None,
                "has_form": form_prob is not None,
                "elo_vegas_edge": (elo_prob - vegas_prob) if elo_prob and vegas_prob else 0,
            },
        )
        
        return result
    
    def _estimate_from_elo(self, context: SportsContext) -> float | None:
        """
        Estimate from ELO ratings.
        
        Returns:
            Probability or None
        """
        if context.sports_data is None or context.sports_data.elo is None:
            return None
        
        elo = context.sports_data.elo
        
        # Apply adjustments
        is_home = context.is_home_game_for_a
        
        # Apply injury adjustments
        injury_penalty = 50  # ELO points per key injury
        team_a_adjusted = elo.team_a_elo - (len(context.team_a_key_injuries or []) * injury_penalty)
        team_b_adjusted = elo.team_b_elo - (len(context.team_b_key_injuries or []) * injury_penalty)
        
        # Calculate win probability
        # Create temporary ELO object with adjusted ratings
        adjusted_elo = EloRatings(
            team_a_elo=team_a_adjusted,
            team_b_elo=team_b_adjusted,
            home_advantage=elo.home_advantage,
            team_a_rest_days=elo.team_a_rest_days,
            team_b_rest_days=elo.team_b_rest_days,
        )
        
        return adjusted_elo.win_probability(is_home)
    
    def _estimate_from_vegas(self, sports_data: SportsData | None) -> float | None:
        """
        Estimate from Vegas betting lines.
        
        Returns:
            Probability or None
        """
        if sports_data is None:
            return None
        
        return sports_data.vegas_consensus_prob
    
    def _estimate_from_form(self, context: SportsContext) -> float | None:
        """
        Estimate from recent form.
        
        Returns:
            Probability or None
        """
        if context.sports_data is None:
            return None
        
        data = context.sports_data
        
        # Calculate win rate from last 10
        total_games = 10  # Assuming we track last 10
        team_a_wins = data.team_a_last_10_wins
        team_b_wins = data.team_b_last_10_wins
        
        if team_a_wins + team_b_wins == 0:
            return None
        
        # Simple win rate model
        team_a_rate = team_a_wins / 10
        team_b_rate = data.team_b_last_10_wins / 10
        
        # Compare win rates
        if team_a_rate + team_b_rate == 0:
            return 0.5
        
        # Probability proportional to win rate
        prob = team_a_rate / (team_a_rate + team_b_rate)
        
        # Regress toward 0.5 (small sample)
        prob = prob * 0.7 + 0.5 * 0.3
        
        return prob
    
    def detect_edge(
        self,
        model_prob: float,
        market_price: float,
        vegas_prob: float | None = None
    ) -> dict[str, Any]:
        """
        Detect edge between model, market, and Vegas.
        
        Returns:
            Edge analysis dict
        """
        edge_vs_market = model_prob - market_price
        edge_vs_vegas = model_prob - (vegas_prob or market_price)
        
        return {
            "model_prob": model_prob,
            "market_price": market_price,
            "vegas_prob": vegas_prob,
            "edge_vs_market": edge_vs_market,
            "edge_vs_vegas": edge_vs_vegas,
            "edge_direction": "team_a" if edge_vs_market > 0 else "team_b",
            "edge_magnitude": abs(edge_vs_market),
        }


# Pre-instantiated model
sports_model = SportsProbabilityModel()
