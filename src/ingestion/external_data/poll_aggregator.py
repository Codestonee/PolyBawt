"""Poll aggregator for political markets."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any

import aiohttp

from .base_source import ExternalDataSource, DataSourceResult, DataSourceStatus
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PollData:
    """Individual poll data point."""
    pollster: str
    pollster_rating: float  # 0-3.0+ (538 rating)
    sample_size: int
    margin_error: float | None
    date_conducted: datetime
    date_published: datetime
    
    # Results
    candidate_a_pct: float
    candidate_b_pct: float
    
    # Metadata
    methodology: str = ""
    population: str = "lv"  # rv (registered), lv (likely), a (adults)
    
    @property
    def recency_weight(self, half_life_days: float = 7.0) -> float:
        """Weight based on recency (7-day half-life)."""
        days_old = (datetime.now(timezone.utc) - self.date_conducted).days
        return 0.5 ** (days_old / half_life_days)
    
    @property
    def sample_weight(self) -> float:
        """Weight based on sample size."""
        return min(1.0, self.sample_size / 1000)  # Cap at 1000
    
    @property
    def pollster_weight(self) -> float:
        """Weight based on pollster rating."""
        # 538 ratings: A+ = 3.0, A = 2.5, B = 2.0, C = 1.0, D = 0.5
        return min(1.0, self.pollster_rating / 2.5)


@dataclass
class AggregatedPolls:
    """Aggregated poll result."""
    candidate_a_avg: float
    candidate_b_avg: float
    margin: float  # Positive = A leading
    
    # Confidence metrics
    poll_count: int
    weighted_sample_size: float
    agreement_score: float  # 0-1, how consistent are polls
    
    # Metadata
    latest_poll_date: datetime | None = None
    oldest_poll_date: datetime | None = None


class PollAggregator(ExternalDataSource):
    """
    Aggregates political polls from multiple sources.
    
    Supports:
    - RealClearPolitics (web scraping)
    - 538 API (if available)
    - PredictIt prices as secondary signal
    
    Uses weighted aggregation:
    - Recency: 7-day half-life
    - Sample size: Larger = more weight
    - Pollster rating: Higher rating = more weight
    
    Usage:
        aggregator = PollAggregator()
        result = await aggregator.fetch_cached(race="2024-presidential")
        
        if result.is_valid:
            polls = result.data.get("aggregated")
            probability = polls.candidate_a_avg / 100
    """
    
    # 538-style pollster ratings (simplified)
    POLLSTER_RATINGS = {
        "marist": 2.8,
        "monmouth": 2.7,
        "quinnipiac": 2.6,
        "cnn": 2.4,
        "abc-washington-post": 2.5,
        "nbc-wall-street-journal": 2.5,
        "fox": 2.2,
        " CBS/YouGov": 2.3,
        "ipsos": 2.1,
        "hart": 2.0,
        "echelon": 1.9,
        "suffolk": 1.8,
        "emerson": 1.7,
        "trafalgar": 1.2,  # Republican-leaning
        " InsiderAdvantage": 1.3,
        "generic": 1.5,
    }
    
    def __init__(self, api_key: str | None = None):
        super().__init__(name="PollAggregator", api_key=api_key)
        self._session: aiohttp.ClientSession | None = None
        
        # API endpoints
        self.rcp_base_url = "https://www.realclearpolitics.com/epolls"
        self.predictit_base_url = "https://www.predictit.org/api/marketdata/all"
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    @property
    def update_frequency_seconds(self) -> int:
        return 3600  # 1 hour
    
    async def fetch(self, race: str = "2024-presidential", **kwargs) -> DataSourceResult:
        """
        Fetch and aggregate polls for a race.
        
        Args:
            race: Race identifier (e.g., "2024-presidential")
            
        Returns:
            DataSourceResult with aggregated polls
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            # Try to fetch from multiple sources in parallel
            tasks = [
                self._fetch_rcp(race),
                self._fetch_predictit(race),
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            all_polls: list[PollData] = []
            predictit_price: float | None = None
            
            # Process results
            for result in results:
                if isinstance(result, Exception):
                    logger.warning("Source fetch failed", error=str(result))
                    continue
                    
                if isinstance(result, list):
                    all_polls.extend(result)
                elif isinstance(result, float):
                    predictit_price = result
            
            if not all_polls:
                return DataSourceResult(
                    source_name=self.name,
                    status=DataSourceStatus.DEGRADED,
                    error_message="No polls fetched from any source",
                )
            
            # Aggregate polls
            aggregated = self._aggregate_polls(all_polls)
            
            # Calculate confidence
            confidence = self._calculate_confidence(aggregated, len(all_polls))
            
            latency_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            return DataSourceResult(
                data={
                    "aggregated": aggregated,
                    "individual_polls": all_polls,
                    "predictit_price": predictit_price,
                    "race": race,
                },
                source_name=self.name,
                status=DataSourceStatus.HEALTHY,
                timestamp=start_time,
                latency_ms=latency_ms,
                confidence=confidence,
                sample_size=len(all_polls),
                freshness_hours=self._calculate_freshness(aggregated.latest_poll_date),
            )
            
        except Exception as e:
            logger.error("Poll aggregation failed", error=str(e))
            return DataSourceResult(
                source_name=self.name,
                status=DataSourceStatus.DOWN,
                error_message=str(e),
            )
    
    async def _fetch_rcp(self, race: str) -> list[PollData]:
        """Fetch polls from RealClearPolitics (simplified)."""
        # NOTE: In production, this would scrape or use API
        # For now, return mock data structure
        
        logger.debug("Fetching RCP polls", race=race)
        
        # Mock implementation - in production, implement actual scraping
        polls: list[PollData] = []
        
        # Example: Would fetch from RCP website or API
        # session = await self._get_session()
        # async with session.get(f"{self.rcp_base_url}/{race}") as resp:
        #     data = await resp.json()
        #     ...parse polls...
        
        return polls
    
    async def _fetch_predictit(self, race: str) -> float | None:
        """Fetch PredictIt price as secondary signal."""
        try:
            session = await self._get_session()
            
            # Map race to PredictIt market
            market_id = self._get_predictit_market_id(race)
            if not market_id:
                return None
            
            url = f"https://www.predictit.org/api/marketdata/markets/{market_id}"
            
            async with session.get(url, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # Extract yes price
                    contracts = data.get("contracts", [])
                    if contracts:
                        return contracts[0].get("lastTradePrice", 0.5)
                        
        except Exception as e:
            logger.warning("PredictIt fetch failed", error=str(e))
        
        return None
    
    def _get_predictit_market_id(self, race: str) -> str | None:
        """Map race to PredictIt market ID."""
        # Simplified mapping
        mappings = {
            "2024-presidential": "7836",  # Example
        }
        return mappings.get(race)
    
    def _aggregate_polls(self, polls: list[PollData]) -> AggregatedPolls:
        """Weighted aggregation of polls."""
        if not polls:
            return AggregatedPolls(0.5, 0.5, 0, 0, 0, 1.0)
        
        total_weight = 0.0
        weighted_a = 0.0
        weighted_b = 0.0
        
        dates: list[datetime] = []
        
        for poll in polls:
            weight = (
                poll.recency_weight * 
                poll.sample_weight * 
                poll.pollster_weight
            )
            
            weighted_a += poll.candidate_a_pct * weight
            weighted_b += poll.candidate_b_pct * weight
            total_weight += weight
            
            dates.append(poll.date_conducted)
        
        if total_weight == 0:
            return AggregatedPolls(0.5, 0.5, 0, 0, 0, 1.0)
        
        avg_a = weighted_a / total_weight
        avg_b = weighted_b / total_weight
        
        # Calculate agreement (standard deviation of margins)
        margins = [p.candidate_a_pct - p.candidate_b_pct for p in polls]
        if len(margins) > 1:
            mean_margin = sum(margins) / len(margins)
            variance = sum((m - mean_margin) ** 2 for m in margins) / (len(margins) - 1)
            std_margin = variance ** 0.5
            # Agreement score: lower std = higher agreement
            agreement = max(0, 1 - (std_margin / 10))  # Normalize
        else:
            agreement = 0.5
        
        return AggregatedPolls(
            candidate_a_avg=avg_a,
            candidate_b_avg=avg_b,
            margin=avg_a - avg_b,
            poll_count=len(polls),
            weighted_sample_size=total_weight * 1000,  # Approximate
            agreement_score=agreement,
            latest_poll_date=max(dates) if dates else None,
            oldest_poll_date=min(dates) if dates else None,
        )
    
    def _calculate_confidence(self, aggregated: AggregatedPolls, poll_count: int) -> float:
        """Calculate confidence in the aggregate."""
        # More polls = more confidence
        poll_confidence = min(1.0, poll_count / 10)
        
        # Higher agreement = more confidence
        agreement_confidence = aggregated.agreement_score
        
        # Recent polls = more confidence (simplified)
        recency_confidence = 1.0 if aggregated.latest_poll_date else 0.5
        
        return (poll_confidence + agreement_confidence + recency_confidence) / 3
    
    def _calculate_freshness(self, latest_date: datetime | None) -> float:
        """Calculate freshness in hours."""
        if latest_date is None:
            return 168  # 1 week (stale)
        
        hours = (datetime.now(timezone.utc) - latest_date).total_seconds() / 3600
        return hours
    
    async def close(self) -> None:
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
