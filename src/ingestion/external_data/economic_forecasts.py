"""Economic forecasts for economics markets (Fed rates, etc.)."""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any
from enum import Enum

import aiohttp

from .base_source import ExternalDataSource, DataSourceResult, DataSourceStatus
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class EconomicIndicator(Enum):
    """Types of economic indicators."""
    FED_FUNDS_RATE = "fed_funds_rate"
    CPI = "cpi"
    UNEMPLOYMENT = "unemployment"
    GDP = "gdp"
    NONFARM_PAYROLL = "nonfarm_payroll"


@dataclass
class FedFutures:
    """Fed funds futures implied probabilities."""
    meeting_date: datetime
    current_rate: float
    
    # Implied probabilities for different outcomes
    prob_no_change: float
    prob_25bp_hike: float
    prob_50bp_hike: float
    prob_25bp_cut: float
    prob_50bp_cut: float
    
    # Derived expected rate
    expected_rate: float = 0.0
    
    def __post_init__(self):
        if self.expected_rate == 0.0:
            self.expected_rate = (
                self.current_rate +
                0.25 * (self.prob_25bp_hike - self.prob_25bp_cut) +
                0.50 * (self.prob_50bp_hike - self.prob_50bp_cut)
            )
    
    @property
    def most_likely_action(self) -> str:
        """Determine most likely Fed action."""
        probs = {
            "no_change": self.prob_no_change,
            "hike_25": self.prob_25bp_hike,
            "hike_50": self.prob_50bp_hike,
            "cut_25": self.prob_25bp_cut,
            "cut_50": self.prob_50bp_cut,
        }
        return max(probs, key=probs.get)


@dataclass
class ConsensusForecast:
    """Economist consensus forecast."""
    indicator: EconomicIndicator
    release_date: datetime
    
    # Forecasts
    consensus: float
    low_estimate: float
    high_estimate: float
    
    # Previous and year-ago
    previous_value: float
    year_ago_value: float
    
    # Metadata
    num_estimates: int = 0
    std_dev: float = 0.0
    
    @property
    def surprise_range(self) -> tuple[float, float]:
        """Range that would be considered a surprise."""
        return (
            self.consensus - 1.5 * self.std_dev,
            self.consensus + 1.5 * self.std_dev,
        )


@dataclass
class EconomicData:
    """Complete economic data for a market."""
    indicator: EconomicIndicator
    target_date: datetime
    
    # Data sources
    fed_futures: FedFutures | None = None
    consensus: ConsensusForecast | None = None
    
    # Leading indicators
    leading_signals: dict[str, float] = field(default_factory=dict)
    
    # Market-implied (from prediction markets)
    market_implied_prob: float | None = None
    
    def probability_above_consensus(self) -> float:
        """
        Estimate probability of outcome above consensus.
        
        Combines Fed futures (if applicable) with consensus distribution.
        """
        if self.indicator == EconomicIndicator.FED_FUNDS_RATE and self.fed_futures:
            # Use Fed futures as primary signal
            if self.fed_futures.expected_rate > (self.consensus.consensus if self.consensus else 0):
                return 0.5 + min(0.4, abs(self.fed_futures.expected_rate - 
                    (self.consensus.consensus if self.consensus else self.fed_futures.expected_rate)) * 2)
            else:
                return 0.5 - min(0.4, abs(self.fed_futures.expected_rate - 
                    (self.consensus.consensus if self.consensus else self.fed_futures.expected_rate)) * 2)
        
        elif self.consensus:
            # Default to consensus-based estimate
            return 0.5  # Neutral without more data
        
        return 0.5


class EconomicForecasts(ExternalDataSource):
    """
    Fetches economic forecasts and Fed futures data.
    
    Data sources:
    - CME FedWatch (Fed funds futures)
    - FRED (Federal Reserve Economic Data)
    - Bloomberg/WSJ economist surveys
    
    Usage:
        forecasts = EconomicForecasts()
        result = await forecasts.fetch_cached(
            indicator=EconomicIndicator.FED_FUNDS_RATE,
            meeting_date="2024-03-20"
        )
        
        if result.is_valid:
            data = result.data.get("economic_data")
            prob = data.probability_above_consensus()
    """
    
    # CME FedWatch API (unofficial endpoints)
    CME_FEDWATCH_URL = "https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/{}/G"
    
    # FRED API
    FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
    
    def __init__(self, api_key: str | None = None):
        super().__init__(name="EconomicForecasts", api_key=api_key)
        self._session: aiohttp.ClientSession | None = None
        
        # Cache for Fed futures
        self._fed_futures_cache: dict[str, FedFutures] = {}
        self._cache_time: datetime | None = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    @property
    def update_frequency_seconds(self) -> int:
        return 900  # 15 minutes
    
    async def fetch(
        self,
        indicator: EconomicIndicator,
        target_date: str | None = None,
        **kwargs
    ) -> DataSourceResult:
        """
        Fetch economic forecast data.
        
        Args:
            indicator: Type of economic indicator
            target_date: Target date for forecast (YYYY-MM-DD)
            
        Returns:
            DataSourceResult with EconomicData
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            parsed_date = datetime.strptime(target_date, "%Y-%m-%d").replace(tzinfo=timezone.utc) if target_date else \
                          datetime.now(timezone.utc) + timedelta(days=30)
            
            # Fetch based on indicator type
            fed_futures = None
            consensus = None
            
            if indicator == EconomicIndicator.FED_FUNDS_RATE:
                fed_futures = await self._fetch_fed_futures(parsed_date)
            
            consensus = await self._fetch_consensus(indicator, parsed_date)
            
            # Build economic data
            econ_data = EconomicData(
                indicator=indicator,
                target_date=parsed_date,
                fed_futures=fed_futures,
                consensus=consensus,
            )
            
            latency_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            return DataSourceResult(
                data={"economic_data": econ_data},
                source_name=self.name,
                status=DataSourceStatus.HEALTHY,
                timestamp=start_time,
                latency_ms=latency_ms,
                confidence=0.9 if fed_futures else 0.7,
                sample_size=1,
            )
            
        except Exception as e:
            logger.error("Economic forecast fetch failed", error=str(e))
            return DataSourceResult(
                source_name=self.name,
                status=DataSourceStatus.DOWN,
                error_message=str(e),
            )
    
    async def _fetch_fed_futures(self, meeting_date: datetime) -> FedFutures | None:
        """Fetch Fed funds futures implied probabilities."""
        try:
            # Check cache
            cache_key = meeting_date.strftime("%Y-%m-%d")
            if cache_key in self._fed_futures_cache and not self._should_refresh_cache():
                return self._fed_futures_cache[cache_key]
            
            # Fetch from CME
            # NOTE: In production, use official CME API or scrape FedWatch
            # This is a simplified structure
            
            session = await self._get_session()
            
            # CME Fed funds futures product code: ZQ
            url = self.CME_FEDWATCH_URL.format("305")
            
            async with session.get(url, timeout=15) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # Parse futures data
                    # In production, extract actual probabilities
                    
                    # Mock structure for illustration
                    futures = FedFutures(
                        meeting_date=meeting_date,
                        current_rate=5.25,
                        prob_no_change=0.70,
                        prob_25bp_hike=0.10,
                        prob_50bp_hike=0.05,
                        prob_25bp_cut=0.10,
                        prob_50bp_cut=0.05,
                    )
                    
                    self._fed_futures_cache[cache_key] = futures
                    self._cache_time = datetime.now(timezone.utc)
                    
                    return futures
                    
        except Exception as e:
            logger.warning("Fed futures fetch failed", error=str(e))
        
        return None
    
    async def _fetch_consensus(
        self,
        indicator: EconomicIndicator,
        target_date: datetime
    ) -> ConsensusForecast | None:
        """Fetch economist consensus forecast."""
        # NOTE: In production, integrate with:
        # - Bloomberg API
        # - FactSet
        # - WSJ economist survey
        
        # Return None for now - would need real data source
        return None
    
    def _should_refresh_cache(self) -> bool:
        """Check if cache needs refresh."""
        if self._cache_time is None:
            return True
        
        age = datetime.now(timezone.utc) - self._cache_time
        return age > timedelta(minutes=15)
    
    async def close(self) -> None:
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


# Pre-instantiated instance
economic_forecasts = EconomicForecasts()
