"""External data ingestion for event markets."""

from .base_source import ExternalDataSource, DataSourceResult
from .poll_aggregator import PollAggregator, PollData
from .sports_stats import SportsStatsSource, SportsData
from .economic_forecasts import EconomicForecasts, EconomicData
from .sentiment_analyzer import SentimentAnalyzer, SentimentData

__all__ = [
    "ExternalDataSource",
    "DataSourceResult",
    "PollAggregator",
    "PollData",
    "SportsStatsSource",
    "SportsData",
    "EconomicForecasts",
    "EconomicData",
    "SentimentAnalyzer",
    "SentimentData",
]
