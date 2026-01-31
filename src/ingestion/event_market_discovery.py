"""
Non-Crypto Event Market Discovery.

Discovers and categorizes non-crypto prediction markets:
- Politics: Elections, policy decisions
- Sports: Game outcomes, championships
- Economics: Fed rates, employment data
- Pop Culture: Awards, celebrity events

Unlike crypto markets (mathematical pricing via Black-Scholes),
these require external data sources and domain expertise.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
import aiohttp

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class MarketCategory(Enum):
    """Categories of prediction markets."""
    POLITICS = "politics"
    SPORTS = "sports"
    ECONOMICS = "economics"
    POP_CULTURE = "pop_culture"
    CRYPTO = "crypto"
    OTHER = "other"


# Polymarket tag IDs by category
CATEGORY_TAGS = {
    MarketCategory.POLITICS: [3],      # Elections, government
    MarketCategory.SPORTS: [7],        # Sports events
    MarketCategory.ECONOMICS: [5],     # Fed, economic data
    MarketCategory.POP_CULTURE: [11],  # Entertainment, celebrities
    MarketCategory.CRYPTO: [1, 2],     # Crypto prices
}


@dataclass
class EventMarket:
    """A non-crypto event market."""
    condition_id: str
    question: str
    category: MarketCategory
    yes_price: float
    no_price: float
    volume_24h: float
    liquidity: float
    end_date: datetime | None = None
    tags: list[str] = field(default_factory=list)
    description: str = ""

    # For research/analysis
    external_probability: float | None = None  # Probability from external source
    edge: float | None = None  # Calculated edge
    notes: str = ""  # Manual research notes

    @property
    def has_external_estimate(self) -> bool:
        return self.external_probability is not None

    @property
    def is_tradeable(self) -> bool:
        """Is this market worth investigating?"""
        return self.volume_24h > 1000 and self.liquidity > 500

    @property
    def time_to_resolution(self) -> float | None:
        """Hours until market resolves."""
        if self.end_date is None:
            return None
        delta = self.end_date - datetime.now(timezone.utc)
        return delta.total_seconds() / 3600


class EventMarketDiscovery:
    """
    Discover non-crypto event markets.

    Unlike crypto (mathematical pricing), these require:
    - External data sources (polls, news, stats)
    - Domain-specific probability estimation
    - Manual research integration

    Usage:
        discovery = EventMarketDiscovery()

        # Get political markets
        politics = await discovery.get_political_markets()

        # Get all event markets
        events = await discovery.get_all_event_markets()

        # Filter to high-liquidity tradeable markets
        tradeable = [m for m in events if m.is_tradeable]
    """

    def __init__(
        self,
        base_url: str = "https://gamma-api.polymarket.com",
        min_volume_24h: float = 1000,
        min_liquidity: float = 500,
    ):
        self.base_url = base_url
        self.min_volume_24h = min_volume_24h
        self.min_liquidity = min_liquidity
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def _fetch_markets(
        self,
        tag_id: int | None = None,
        active: bool = True,
        limit: int = 100,
    ) -> list[dict]:
        """Fetch markets from Gamma API."""
        session = await self._get_session()

        params = {
            "active": str(active).lower(),
            "limit": limit,
            "closed": "false",
        }
        if tag_id is not None:
            params["tag_id"] = tag_id

        url = f"{self.base_url}/markets"

        try:
            async with session.get(url, params=params, timeout=10) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    logger.warning(
                        "Failed to fetch markets",
                        status=resp.status,
                        tag_id=tag_id,
                    )
                    return []
        except Exception as e:
            logger.error("Error fetching markets", error=str(e))
            return []

    def _parse_market(
        self,
        data: dict,
        category: MarketCategory,
    ) -> EventMarket | None:
        """Parse API response into EventMarket."""
        try:
            end_date = None
            if data.get("endDate"):
                end_date = datetime.fromisoformat(
                    data["endDate"].replace("Z", "+00:00")
                )

            return EventMarket(
                condition_id=data.get("conditionId", ""),
                question=data.get("question", ""),
                category=category,
                yes_price=float(data.get("outcomePrices", [0.5])[0]),
                no_price=1 - float(data.get("outcomePrices", [0.5])[0]),
                volume_24h=float(data.get("volume24hr", 0)),
                liquidity=float(data.get("liquidity", 0)),
                end_date=end_date,
                tags=data.get("tags", []),
                description=data.get("description", ""),
            )
        except Exception as e:
            logger.debug("Failed to parse market", error=str(e))
            return None

    async def get_markets_by_category(
        self,
        category: MarketCategory,
        limit: int = 50,
    ) -> list[EventMarket]:
        """Get markets for a specific category."""
        tag_ids = CATEGORY_TAGS.get(category, [])

        markets = []
        for tag_id in tag_ids:
            raw_markets = await self._fetch_markets(tag_id=tag_id, limit=limit)

            for data in raw_markets:
                market = self._parse_market(data, category)
                if market and market.volume_24h >= self.min_volume_24h:
                    markets.append(market)

        # Sort by volume
        markets.sort(key=lambda m: m.volume_24h, reverse=True)
        return markets

    async def get_political_markets(self, limit: int = 50) -> list[EventMarket]:
        """Fetch active political/election markets."""
        return await self.get_markets_by_category(MarketCategory.POLITICS, limit)

    async def get_sports_markets(self, limit: int = 50) -> list[EventMarket]:
        """Fetch active sports betting markets."""
        return await self.get_markets_by_category(MarketCategory.SPORTS, limit)

    async def get_economics_markets(self, limit: int = 50) -> list[EventMarket]:
        """Fetch active economics/Fed markets."""
        return await self.get_markets_by_category(MarketCategory.ECONOMICS, limit)

    async def get_pop_culture_markets(self, limit: int = 50) -> list[EventMarket]:
        """Fetch active pop culture markets."""
        return await self.get_markets_by_category(MarketCategory.POP_CULTURE, limit)

    async def get_all_event_markets(
        self,
        exclude_crypto: bool = True,
    ) -> list[EventMarket]:
        """
        Get all non-crypto event markets.

        Args:
            exclude_crypto: If True, skip crypto markets

        Returns:
            List of EventMarket sorted by volume
        """
        all_markets = []

        for category in MarketCategory:
            if exclude_crypto and category == MarketCategory.CRYPTO:
                continue

            try:
                markets = await self.get_markets_by_category(category)
                all_markets.extend(markets)
            except Exception as e:
                logger.error("Failed to fetch category", category=category.value, error=str(e))

        # Remove duplicates by condition_id
        seen = set()
        unique_markets = []
        for market in all_markets:
            if market.condition_id not in seen:
                seen.add(market.condition_id)
                unique_markets.append(market)

        # Sort by volume
        unique_markets.sort(key=lambda m: m.volume_24h, reverse=True)
        return unique_markets

    async def get_research_candidates(
        self,
        min_edge_potential: float = 0.05,
    ) -> list[EventMarket]:
        """
        Get markets that are good candidates for manual research.

        Criteria:
        - High liquidity (worth trading)
        - Price not at extremes (room for edge)
        - Resolves soon-ish (actionable)

        Args:
            min_edge_potential: Minimum distance from 0.5 to consider

        Returns:
            List of research-worthy markets
        """
        all_markets = await self.get_all_event_markets()

        candidates = []
        for market in all_markets:
            # Skip if not tradeable
            if not market.is_tradeable:
                continue

            # Skip if at extreme prices (already priced in)
            if market.yes_price < 0.10 or market.yes_price > 0.90:
                continue

            # Skip if resolving too soon (no time to research)
            if market.time_to_resolution is not None:
                if market.time_to_resolution < 24:  # Less than 1 day
                    continue

            candidates.append(market)

        return candidates

    def calculate_edge(
        self,
        market: EventMarket,
        estimated_probability: float,
    ) -> float:
        """
        Calculate edge given an estimated probability.

        Args:
            market: The market to calculate edge for
            estimated_probability: Your probability estimate (0-1)

        Returns:
            Edge as a decimal (e.g., 0.05 = 5% edge)
        """
        # Edge for buying YES
        edge_yes = estimated_probability - market.yes_price

        # Edge for buying NO
        edge_no = (1 - estimated_probability) - market.no_price

        # Return the larger edge
        if edge_yes > edge_no:
            return edge_yes
        else:
            return edge_no

    def annotate_market(
        self,
        market: EventMarket,
        estimated_probability: float,
        notes: str = "",
    ) -> EventMarket:
        """
        Add research annotations to a market.

        Args:
            market: Market to annotate
            estimated_probability: Your probability estimate
            notes: Research notes

        Returns:
            Annotated market
        """
        market.external_probability = estimated_probability
        market.edge = self.calculate_edge(market, estimated_probability)
        market.notes = notes
        return market
