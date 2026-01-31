"""Abstract base class for external data sources."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class DataSourceStatus(Enum):
    """Status of data source."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"
    RATE_LIMITED = "rate_limited"


@dataclass
class DataSourceResult:
    """Result from fetching external data."""
    
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source_name: str = ""
    status: DataSourceStatus = DataSourceStatus.HEALTHY
    error_message: str = ""
    latency_ms: float = 0.0
    
    # Metadata
    confidence: float = 1.0  # Confidence in data quality (0-1)
    sample_size: int = 0  # Number of data points
    freshness_hours: float = 0.0  # How stale is the data
    
    @property
    def is_valid(self) -> bool:
        """Is this data usable?"""
        return self.status == DataSourceStatus.HEALTHY and len(self.data) > 0
    
    @property
    def is_stale(self, max_age_hours: float = 24.0) -> bool:
        """Is data older than threshold?"""
        return self.freshness_hours > max_age_hours


class ExternalDataSource(ABC):
    """
    Abstract base class for external data sources.
    
    All external data sources (polls, sports stats, economic forecasts)
    must implement this interface for consistent error handling,
    caching, and status reporting.
    
    Usage:
        class MyDataSource(ExternalDataSource):
            async def fetch(self, **params) -> DataSourceResult:
                # Implementation
                pass
                
            @property
            def update_frequency_seconds(self) -> int:
                return 3600  # 1 hour
    """
    
    def __init__(self, name: str, api_key: str | None = None):
        self.name = name
        self.api_key = api_key
        self._last_result: DataSourceResult | None = None
        self._last_fetch_time: datetime | None = None
        self._error_count: int = 0
        self._success_count: int = 0
    
    @abstractmethod
    async def fetch(self, **params) -> DataSourceResult:
        """
        Fetch data from the external source.
        
        Args:
            **params: Source-specific parameters
            
        Returns:
            DataSourceResult with data and metadata
        """
        pass
    
    @property
    @abstractmethod
    def update_frequency_seconds(self) -> int:
        """How often should this source be refreshed?"""
        pass
    
    @property
    def status(self) -> DataSourceStatus:
        """Current health status based on recent fetches."""
        if self._error_count > 5 and self._success_count == 0:
            return DataSourceStatus.DOWN
        elif self._error_count > 2:
            return DataSourceStatus.DEGRADED
        return DataSourceStatus.HEALTHY
    
    async def fetch_cached(self, max_age_seconds: int | None = None, **params) -> DataSourceResult:
        """
        Fetch with caching - return cached result if fresh enough.
        
        Args:
            max_age_seconds: Maximum age of cached data (default: update_frequency)
            **params: Fetch parameters
            
        Returns:
            DataSourceResult (cached or fresh)
        """
        max_age = max_age_seconds or self.update_frequency_seconds
        now = datetime.now(timezone.utc)
        
        # Check if we have valid cached data
        if self._last_result and self._last_fetch_time:
            age_seconds = (now - self._last_fetch_time).total_seconds()
            if age_seconds < max_age and self._last_result.is_valid:
                logger.debug(
                    "Using cached data",
                    source=self.name,
                    age_seconds=age_seconds,
                )
                return self._last_result
        
        # Fetch fresh data
        start_time = datetime.now(timezone.utc)
        try:
            result = await self.fetch(**params)
            latency_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            result.latency_ms = latency_ms
            
            # Update stats
            if result.is_valid:
                self._success_count += 1
                self._error_count = max(0, self._error_count - 1)
            else:
                self._error_count += 1
            
            # Cache result
            self._last_result = result
            self._last_fetch_time = now
            
            return result
            
        except Exception as e:
            self._error_count += 1
            latency_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error(
                "Fetch failed",
                source=self.name,
                error=str(e),
                latency_ms=latency_ms,
            )
            
            # Return last result if available, else empty result
            if self._last_result:
                return self._last_result
            
            return DataSourceResult(
                source_name=self.name,
                status=DataSourceStatus.DOWN,
                error_message=str(e),
                latency_ms=latency_ms,
            )
    
    def get_stats(self) -> dict[str, Any]:
        """Get source statistics."""
        return {
            "name": self.name,
            "status": self.status.value,
            "success_count": self._success_count,
            "error_count": self._error_count,
            "last_fetch": self._last_fetch_time.isoformat() if self._last_fetch_time else None,
            "update_frequency_seconds": self.update_frequency_seconds,
        }
    
    def reset_stats(self) -> None:
        """Reset statistics."""
        self._error_count = 0
        self._success_count = 0
