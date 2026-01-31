"""Social sentiment analyzer for pop culture markets."""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any
from enum import Enum
import re

import aiohttp

from .base_source import ExternalDataSource, DataSourceResult, DataSourceStatus
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class SentimentSource(Enum):
    """Sources of sentiment data."""
    TWITTER = "twitter"
    REDDIT = "reddit"
    METACULUS = "metaculus"
    MANIFOLD = "manifold"
    NEWS = "news"


@dataclass
class SentimentScore:
    """Sentiment score from a source."""
    source: SentimentSource
    score: float  # -1.0 (very negative) to +1.0 (very positive)
    volume: int  # Number of mentions/posts
    timestamp: datetime
    confidence: float = 1.0
    
    @property
    def weighted_score(self) -> float:
        """Score weighted by confidence and log volume."""
        import math
        volume_weight = 1 + math.log10(max(1, self.volume))
        return self.score * self.confidence * volume_weight


@dataclass
class MetaculusForecast:
    """Metaculus prediction market data."""
    question_id: str
    question_title: str
    community_prediction: float  # 0-1
    num_forecasters: int
    timestamp: datetime
    
    @property
    def confidence(self) -> float:
        """Confidence based on number of forecasters."""
        return min(1.0, self.num_forecasters / 100)


@dataclass
class SentimentData:
    """Aggregated sentiment data."""
    topic: str
    timestamp: datetime
    
    # Source-specific scores
    source_scores: list[SentimentScore] = field(default_factory=list)
    metaculus: MetaculusForecast | None = None
    
    # Aggregated
    aggregate_score: float = 0.0
    aggregate_confidence: float = 0.0
    
    # Trend (change from previous period)
    score_change_24h: float = 0.0
    
    def __post_init__(self):
        if not self.aggregate_score and self.source_scores:
            self._calculate_aggregate()
    
    def _calculate_aggregate(self) -> None:
        """Calculate aggregate sentiment score."""
        if not self.source_scores:
            return
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for score in self.source_scores:
            weight = score.confidence * (1 + score.volume / 100)
            weighted_sum += score.score * weight
            total_weight += weight
        
        if total_weight > 0:
            self.aggregate_score = weighted_sum / total_weight
            self.aggregate_confidence = min(1.0, total_weight / len(self.source_scores))
    
    @property
    def sentiment_category(self) -> str:
        """Categorize sentiment."""
        if self.aggregate_score > 0.3:
            return "very_positive"
        elif self.aggregate_score > 0.1:
            return "positive"
        elif self.aggregate_score < -0.3:
            return "very_negative"
        elif self.aggregate_score < -0.1:
            return "negative"
        else:
            return "neutral"
    
    def to_probability(self, baseline: float = 0.5) -> float:
        """
        Convert sentiment to probability estimate.
        
        Maps sentiment (-1 to +1) to probability (0 to 1).
        """
        # Sentiment of 0 -> baseline probability
        # Sentiment of +1 -> 1.0
        # Sentiment of -1 -> 0.0
        
        # Scale and shift
        prob = baseline + (self.aggregate_score * 0.5)
        return max(0.05, min(0.95, prob))  # Clamp to reasonable range


class SentimentAnalyzer(ExternalDataSource):
    """
    Analyzes social sentiment for pop culture markets.
    
    Data sources:
    - Twitter/X API (if available)
    - Reddit mentions
    - Metaculus forecasts
    - Manifold Markets
    - News sentiment
    
    Usage:
        analyzer = SentimentAnalyzer()
        result = await analyzer.fetch_cached(topic="Oscar Best Picture 2024")
        
        if result.is_valid:
            sentiment = result.data.get("sentiment")
            probability = sentiment.to_probability(baseline=0.25)  # 4 nominees
    """
    
    def __init__(self, api_key: str | None = None):
        super().__init__(name="SentimentAnalyzer", api_key=api_key)
        self._session: aiohttp.ClientSession | None = None
        
        # Metaculus API
        self.metaculus_base_url = "https://www.metaculus.com/api2"
        
        # Manifold API
        self.manifold_base_url = "https://api.manifold.markets/v0"
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    @property
    def update_frequency_seconds(self) -> int:
        return 1800  # 30 minutes
    
    async def fetch(
        self,
        topic: str,
        metaculus_question_id: str | None = None,
        **kwargs
    ) -> DataSourceResult:
        """
        Fetch sentiment data for a topic.
        
        Args:
            topic: Topic/entity to analyze
            metaculus_question_id: Optional Metaculus question ID
            
        Returns:
            DataSourceResult with SentimentData
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            source_scores: list[SentimentScore] = []
            metaculus_forecast: MetaculusForecast | None = None
            
            # Fetch Metaculus forecast
            if metaculus_question_id:
                metaculus_forecast = await self._fetch_metaculus(metaculus_question_id)
                if metaculus_forecast:
                    # Convert to sentiment score
                    score = (metaculus_forecast.community_prediction - 0.5) * 2
                    source_scores.append(SentimentScore(
                        source=SentimentSource.METACULUS,
                        score=score,
                        volume=metaculus_forecast.num_forecasters,
                        timestamp=start_time,
                        confidence=metaculus_forecast.confidence,
                    ))
            
            # Fetch Manifold markets
            manifold_score = await self._fetch_manifold(topic)
            if manifold_score:
                source_scores.append(manifold_score)
            
            # Fetch Twitter/X sentiment (if API available)
            twitter_score = await self._fetch_twitter(topic)
            if twitter_score:
                source_scores.append(twitter_score)
            
            # Build sentiment data
            sentiment_data = SentimentData(
                topic=topic,
                timestamp=start_time,
                source_scores=source_scores,
                metaculus=metaculus_forecast,
            )
            
            latency_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            return DataSourceResult(
                data={"sentiment": sentiment_data},
                source_name=self.name,
                status=DataSourceStatus.HEALTHY if source_scores else DataSourceStatus.DEGRADED,
                timestamp=start_time,
                latency_ms=latency_ms,
                confidence=sentiment_data.aggregate_confidence,
                sample_size=len(source_scores),
            )
            
        except Exception as e:
            logger.error("Sentiment analysis failed", error=str(e))
            return DataSourceResult(
                source_name=self.name,
                status=DataSourceStatus.DOWN,
                error_message=str(e),
            )
    
    async def _fetch_metaculus(self, question_id: str) -> MetaculusForecast | None:
        """Fetch Metaculus community prediction."""
        try:
            session = await self._get_session()
            url = f"{self.metaculus_base_url}/questions/{question_id}/"
            
            async with session.get(url, timeout=15) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    prediction = data.get("community_prediction", {}).get("full", {}).get("q2", 0.5)
                    num_forecasters = data.get("number_of_forecasters", 0)
                    
                    return MetaculusForecast(
                        question_id=question_id,
                        question_title=data.get("title", ""),
                        community_prediction=prediction,
                        num_forecasters=num_forecasters,
                        timestamp=datetime.now(timezone.utc),
                    )
                    
        except Exception as e:
            logger.warning("Metaculus fetch failed", error=str(e))
        
        return None
    
    async def _fetch_manifold(self, topic: str) -> SentimentScore | None:
        """Fetch Manifold market data as sentiment signal."""
        try:
            session = await self._get_session()
            
            # Search for markets matching topic
            url = f"{self.manifold_base_url}/search-markets"
            params = {"term": topic, "limit": 5}
            
            async with session.get(url, params=params, timeout=15) as resp:
                if resp.status == 200:
                    markets = await resp.json()
                    
                    if markets:
                        # Use highest volume market
                        best_market = max(markets, key=lambda m: m.get("volume24h", 0))
                        
                        probability = best_market.get("probability", 0.5)
                        volume = best_market.get("volume24h", 0)
                        
                        # Convert to sentiment (-1 to +1)
                        score = (probability - 0.5) * 2
                        
                        return SentimentScore(
                            source=SentimentSource.MANIFOLD,
                            score=score,
                            volume=int(volume),
                            timestamp=datetime.now(timezone.utc),
                            confidence=min(1.0, volume / 10000),
                        )
                        
        except Exception as e:
            logger.warning("Manifold fetch failed", error=str(e))
        
        return None
    
    async def _fetch_twitter(self, topic: str) -> SentimentScore | None:
        """Fetch Twitter/X sentiment."""
        # NOTE: Requires Twitter API v2 access
        # For now, return None
        
        # In production:
        # 1. Search for tweets about topic
        # 2. Run sentiment analysis on text
        # 3. Aggregate scores weighted by engagement
        
        return None
    
    async def close(self) -> None:
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


# Pre-instantiated instance
sentiment_analyzer = SentimentAnalyzer()
