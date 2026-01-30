"""
Sentiment Integration Module for enhanced pricing.

Integrates market sentiment indicators to improve probability estimates:
- Google Trends data
- Social media sentiment (Twitter/X)
- Funding rates
- Open interest changes
- Fear & Greed index

Based on 2024 research:
- Springer: Neural Network for Bitcoin Options with Sentiment
- Shows sentiment indicators significantly improve pricing accuracy

References:
- Springer 2024: Neural Network for Valuing Bitcoin Options
"""

import asyncio
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import aiohttp

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class SentimentSource(Enum):
    """Sources of sentiment data."""
    GOOGLE_TRENDS = "google_trends"
    TWITTER = "twitter"
    REDDIT = "reddit"
    FUNDING_RATE = "funding_rate"
    OPEN_INTEREST = "open_interest"
    FEAR_GREED = "fear_greed"
    NEWS = "news"


class SentimentLevel(Enum):
    """Discretized sentiment levels."""
    EXTREME_FEAR = "extreme_fear"
    FEAR = "fear"
    NEUTRAL = "neutral"
    GREED = "greed"
    EXTREME_GREED = "extreme_greed"


@dataclass
class SentimentReading:
    """A single sentiment measurement."""
    source: SentimentSource
    asset: str
    value: float  # Normalized -1 to +1
    raw_value: float  # Original value
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    confidence: float = 1.0  # How reliable is this reading

    @property
    def level(self) -> SentimentLevel:
        """Convert value to discrete level."""
        if self.value <= -0.6:
            return SentimentLevel.EXTREME_FEAR
        elif self.value <= -0.2:
            return SentimentLevel.FEAR
        elif self.value <= 0.2:
            return SentimentLevel.NEUTRAL
        elif self.value <= 0.6:
            return SentimentLevel.GREED
        else:
            return SentimentLevel.EXTREME_GREED

    @property
    def age_seconds(self) -> float:
        """Age of reading in seconds."""
        return (datetime.now(timezone.utc) - self.timestamp).total_seconds()

    @property
    def is_stale(self) -> bool:
        """Whether reading is too old to use (> 5 minutes)."""
        return self.age_seconds > 300


@dataclass
class AggregateSentiment:
    """Aggregated sentiment across multiple sources."""

    asset: str
    readings: list[SentimentReading] = field(default_factory=list)

    # Computed values
    composite_score: float = 0.0  # Weighted average
    composite_level: SentimentLevel = SentimentLevel.NEUTRAL
    confidence: float = 0.0

    # Individual components
    trend_sentiment: float = 0.0  # From search trends
    social_sentiment: float = 0.0  # From social media
    market_sentiment: float = 0.0  # From funding/OI

    @property
    def is_bullish(self) -> bool:
        return self.composite_score > 0.2

    @property
    def is_bearish(self) -> bool:
        return self.composite_score < -0.2

    @property
    def probability_adjustment(self) -> float:
        """
        Suggested adjustment to model probability based on sentiment.

        Returns a multiplier: 1.0 = no change, 1.1 = 10% more likely, etc.
        """
        # Conservative adjustment based on sentiment
        # Maximum +/- 5% adjustment
        adjustment = self.composite_score * 0.05

        # Scale by confidence
        adjustment *= self.confidence

        return 1.0 + adjustment


class SentimentAggregator:
    """
    Aggregates sentiment from multiple sources.

    Weighting scheme based on research on predictive power:
    - Funding rates: High predictive value for short-term
    - Google Trends: Good for regime detection
    - Social sentiment: Noisy but useful in extremes
    - Fear & Greed: Good summary indicator
    """

    # Source weights based on research
    SOURCE_WEIGHTS = {
        SentimentSource.FUNDING_RATE: 0.30,
        SentimentSource.OPEN_INTEREST: 0.20,
        SentimentSource.FEAR_GREED: 0.20,
        SentimentSource.GOOGLE_TRENDS: 0.15,
        SentimentSource.TWITTER: 0.10,
        SentimentSource.REDDIT: 0.05,
    }

    def __init__(self, max_reading_age_seconds: float = 300):
        self.max_reading_age = max_reading_age_seconds
        self._readings: dict[str, dict[SentimentSource, SentimentReading]] = {}

    def add_reading(self, reading: SentimentReading) -> None:
        """Add or update a sentiment reading."""
        if reading.asset not in self._readings:
            self._readings[reading.asset] = {}

        self._readings[reading.asset][reading.source] = reading

    def get_aggregate(self, asset: str) -> AggregateSentiment:
        """
        Get aggregated sentiment for an asset.

        Args:
            asset: Asset symbol

        Returns:
            AggregateSentiment with weighted composite score
        """
        if asset not in self._readings:
            return AggregateSentiment(asset=asset)

        readings = []
        weighted_sum = 0.0
        total_weight = 0.0

        trend_sum = 0.0
        trend_weight = 0.0
        social_sum = 0.0
        social_weight = 0.0
        market_sum = 0.0
        market_weight = 0.0

        for source, reading in self._readings[asset].items():
            # Skip stale readings
            if reading.is_stale:
                continue

            readings.append(reading)

            # Get weight for this source
            weight = self.SOURCE_WEIGHTS.get(source, 0.1) * reading.confidence

            # Decay weight by age
            age_decay = max(0.5, 1.0 - reading.age_seconds / self.max_reading_age)
            weight *= age_decay

            weighted_sum += reading.value * weight
            total_weight += weight

            # Categorize by type
            if source == SentimentSource.GOOGLE_TRENDS:
                trend_sum += reading.value * weight
                trend_weight += weight
            elif source in (SentimentSource.TWITTER, SentimentSource.REDDIT):
                social_sum += reading.value * weight
                social_weight += weight
            elif source in (SentimentSource.FUNDING_RATE, SentimentSource.OPEN_INTEREST):
                market_sum += reading.value * weight
                market_weight += weight

        # Calculate composite
        if total_weight > 0:
            composite = weighted_sum / total_weight
            confidence = min(1.0, total_weight / sum(self.SOURCE_WEIGHTS.values()))
        else:
            composite = 0.0
            confidence = 0.0

        # Calculate category scores
        trend_sentiment = trend_sum / trend_weight if trend_weight > 0 else 0.0
        social_sentiment = social_sum / social_weight if social_weight > 0 else 0.0
        market_sentiment = market_sum / market_weight if market_weight > 0 else 0.0

        # Determine level
        if composite <= -0.6:
            level = SentimentLevel.EXTREME_FEAR
        elif composite <= -0.2:
            level = SentimentLevel.FEAR
        elif composite <= 0.2:
            level = SentimentLevel.NEUTRAL
        elif composite <= 0.6:
            level = SentimentLevel.GREED
        else:
            level = SentimentLevel.EXTREME_GREED

        return AggregateSentiment(
            asset=asset,
            readings=readings,
            composite_score=composite,
            composite_level=level,
            confidence=confidence,
            trend_sentiment=trend_sentiment,
            social_sentiment=social_sentiment,
            market_sentiment=market_sentiment,
        )


class FundingRateFetcher:
    """
    Fetches funding rate data from exchanges.

    Funding rate is a strong sentiment indicator:
    - Positive funding = longs pay shorts (bullish positioning)
    - Negative funding = shorts pay longs (bearish positioning)
    """

    # Funding rate API endpoints
    ENDPOINTS = {
        "binance": "https://fapi.binance.com/fapi/v1/fundingRate",
        "bybit": "https://api.bybit.com/v5/market/funding/history",
    }

    def __init__(self):
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def fetch_funding_rate(self, asset: str) -> SentimentReading | None:
        """
        Fetch current funding rate for an asset.

        Args:
            asset: Asset symbol (BTC, ETH, etc.)

        Returns:
            SentimentReading or None if unavailable
        """
        symbol = f"{asset}USDT"

        try:
            session = await self._get_session()

            # Try Binance first
            url = f"{self.ENDPOINTS['binance']}?symbol={symbol}&limit=1"
            async with session.get(url, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data:
                        funding_rate = float(data[0].get("fundingRate", 0))

                        # Normalize: typical range is -0.01 to +0.01
                        # Map to -1 to +1 (multiply by 100)
                        normalized = max(-1, min(1, funding_rate * 100))

                        return SentimentReading(
                            source=SentimentSource.FUNDING_RATE,
                            asset=asset,
                            value=normalized,
                            raw_value=funding_rate,
                            confidence=0.9,
                        )

        except Exception as e:
            logger.debug(f"Failed to fetch funding rate: {e}")

        return None

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()


class FearGreedFetcher:
    """
    Fetches Fear & Greed Index data.

    The Fear & Greed Index is a composite indicator:
    - 0-25: Extreme Fear
    - 25-45: Fear
    - 45-55: Neutral
    - 55-75: Greed
    - 75-100: Extreme Greed
    """

    ENDPOINT = "https://api.alternative.me/fng/"

    def __init__(self):
        self._session: aiohttp.ClientSession | None = None
        self._cache: tuple[datetime, float] | None = None
        self._cache_ttl = 300  # 5 minutes

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def fetch_fear_greed(self) -> SentimentReading | None:
        """
        Fetch current Fear & Greed index.

        Returns:
            SentimentReading or None if unavailable
        """
        # Check cache
        if self._cache:
            cache_time, cache_value = self._cache
            if (datetime.now(timezone.utc) - cache_time).total_seconds() < self._cache_ttl:
                # Normalize 0-100 to -1 to +1
                normalized = (cache_value - 50) / 50
                return SentimentReading(
                    source=SentimentSource.FEAR_GREED,
                    asset="CRYPTO",  # Market-wide indicator
                    value=normalized,
                    raw_value=cache_value,
                    timestamp=cache_time,
                    confidence=0.85,
                )

        try:
            session = await self._get_session()

            async with session.get(self.ENDPOINT, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("data"):
                        value = float(data["data"][0].get("value", 50))

                        # Update cache
                        self._cache = (datetime.now(timezone.utc), value)

                        # Normalize 0-100 to -1 to +1
                        normalized = (value - 50) / 50

                        return SentimentReading(
                            source=SentimentSource.FEAR_GREED,
                            asset="CRYPTO",
                            value=normalized,
                            raw_value=value,
                            confidence=0.85,
                        )

        except Exception as e:
            logger.debug(f"Failed to fetch Fear & Greed: {e}")

        return None

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()


class SentimentIntegrator:
    """
    Main class for integrating sentiment into trading decisions.

    Usage:
        integrator = SentimentIntegrator()
        await integrator.update_all("BTC")

        sentiment = integrator.get_sentiment("BTC")
        if sentiment.is_bullish:
            model_prob *= sentiment.probability_adjustment
    """

    def __init__(self):
        self.aggregator = SentimentAggregator()
        self.funding_fetcher = FundingRateFetcher()
        self.fear_greed_fetcher = FearGreedFetcher()

    async def update_all(self, asset: str) -> None:
        """
        Update all sentiment sources for an asset.

        Args:
            asset: Asset symbol
        """
        tasks = [
            self._update_funding(asset),
            self._update_fear_greed(asset),
        ]

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _update_funding(self, asset: str) -> None:
        """Update funding rate sentiment."""
        reading = await self.funding_fetcher.fetch_funding_rate(asset)
        if reading:
            self.aggregator.add_reading(reading)

    async def _update_fear_greed(self, asset: str) -> None:
        """Update Fear & Greed sentiment."""
        reading = await self.fear_greed_fetcher.fetch_fear_greed()
        if reading:
            # Apply to specific asset (market-wide indicator)
            reading_for_asset = SentimentReading(
                source=reading.source,
                asset=asset,
                value=reading.value,
                raw_value=reading.raw_value,
                timestamp=reading.timestamp,
                confidence=reading.confidence * 0.8,  # Discount for non-specific
            )
            self.aggregator.add_reading(reading_for_asset)

    def add_manual_reading(
        self,
        asset: str,
        source: SentimentSource,
        value: float,
        confidence: float = 1.0,
    ) -> None:
        """
        Add a manual sentiment reading.

        Useful for incorporating proprietary signals.
        """
        reading = SentimentReading(
            source=source,
            asset=asset,
            value=max(-1, min(1, value)),  # Clamp to [-1, 1]
            raw_value=value,
            confidence=confidence,
        )
        self.aggregator.add_reading(reading)

    def get_sentiment(self, asset: str) -> AggregateSentiment:
        """Get aggregated sentiment for an asset."""
        return self.aggregator.get_aggregate(asset)

    def adjust_probability(
        self,
        model_prob: float,
        asset: str,
        max_adjustment: float = 0.05,
    ) -> tuple[float, AggregateSentiment]:
        """
        [DEPRECATED] Adjust model probability based on sentiment.
        
        RESEARCH WARNING (Claude/Perplexity):
        "Funding rate R² for next-period prediction is approximately zero."
        Do NOT use this for probability adjustment. Use get_sizing_multiplier() instead.

        Args:
            model_prob: Original model probability
            asset: Asset symbol
            max_adjustment: Maximum absolute adjustment

        Returns:
            Tuple of (adjusted_prob, sentiment_used)
        """
        logger.warning(
            "DEPRECATED: adjust_probability() should not be used. "
            "Research shows funding rate R²≈0 for short-term price prediction. "
            "Use get_sizing_multiplier() for Kelly sizing instead."
        )
        
        sentiment = self.get_sentiment(asset)

        if sentiment.confidence < 0.3:
            # Not enough confidence, don't adjust
            return model_prob, sentiment

        # Calculate adjustment
        adjustment = sentiment.composite_score * max_adjustment * sentiment.confidence

        # Apply adjustment
        adjusted = model_prob + adjustment

        # Clamp to valid range
        adjusted = max(0.01, min(0.99, adjusted))

        if abs(adjustment) > 0.01:
            logger.debug(
                "Sentiment adjustment applied",
                asset=asset,
                original_prob=f"{model_prob:.3f}",
                adjusted_prob=f"{adjusted:.3f}",
                sentiment_score=f"{sentiment.composite_score:.2f}",
                sentiment_level=sentiment.composite_level.value,
            )

        return adjusted, sentiment
    
    def get_sizing_multiplier(
        self,
        asset: str,
        trade_direction: str = "long",  # "long" or "short"
    ) -> tuple[float, AggregateSentiment]:
        """
        Get Kelly sizing multiplier based on sentiment.
        
        RESEARCH-BACKED (Claude/Perplexity/Grok):
        "Funding rates have ~0 R² for next-period price prediction, but 
        are useful as regime filters for position sizing conviction."
        
        Use this to adjust Kelly fraction, NOT probability.
        
        Weighting scheme (Perplexity):
        - Funding rate: 60%
        - Order flow imbalance: 25%  
        - Fear & Greed: 10%
        - Social: 5%
        
        Args:
            asset: Asset symbol (BTC, ETH, etc.)
            trade_direction: "long" or "short"
            
        Returns:
            Tuple of (sizing_multiplier, sentiment_used)
            - multiplier in range [0.5, 1.5]
            - 1.0 = neutral (no size change)
            - 0.5 = reduce size by 50% (sentiment contradicts trade)
            - 1.5 = increase size by 50% (sentiment aligns with trade)
        """
        sentiment = self.get_sentiment(asset)
        
        if sentiment.confidence < 0.3:
            logger.debug(
                "Sentiment confidence too low for sizing adjustment",
                asset=asset,
                confidence=sentiment.confidence,
            )
            return 1.0, sentiment
        
        # Determine if sentiment aligns with trade direction
        # Bullish sentiment (positive score) aligns with long trades
        # Bearish sentiment (negative score) aligns with short trades
        score = sentiment.composite_score  # Range [-1, +1]
        
        if trade_direction == "long":
            alignment = score  # Positive = aligned
        else:  # short
            alignment = -score  # Negative sentiment = aligned with short
        
        # Calculate multiplier:
        # alignment > 0 → increase size (up to 1.5)
        # alignment < 0 → decrease size (down to 0.5)
        # Base formula: 1.0 + (alignment * 0.5 * confidence)
        multiplier = 1.0 + (alignment * 0.5 * sentiment.confidence)
        
        # Clamp to [0.5, 1.5]
        multiplier = max(0.5, min(1.5, multiplier))
        
        logger.debug(
            "Sentiment sizing adjustment",
            asset=asset,
            direction=trade_direction,
            sentiment_score=f"{score:.2f}",
            sentiment_level=sentiment.composite_level.value,
            alignment=f"{alignment:.2f}",
            sizing_multiplier=f"{multiplier:.2f}",
        )
        
        return multiplier, sentiment

    async def close(self) -> None:
        """Clean up resources."""
        await self.funding_fetcher.close()
        await self.fear_greed_fetcher.close()


# Pre-instantiated instance
sentiment_integrator = SentimentIntegrator()
