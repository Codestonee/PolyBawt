"""
Market discovery via Polymarket Gamma API.

Provides:
- Discovery of active 15-minute crypto markets
- Parsing of market metadata (token IDs, prices, expiry)
- Filtering by asset type
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import aiohttp

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Market:
    """Parsed market data."""
    
    # Identifiers
    condition_id: str
    question: str
    slug: str
    
    # Token IDs
    yes_token_id: str
    no_token_id: str
    
    # Prices
    yes_price: float
    no_price: float
    
    # Timing
    end_date: datetime
    created_at: datetime | None = None
    
    # Metadata
    asset: str = ""  # BTC, ETH, SOL, XRP
    is_15m: bool = False
    active: bool = True
    
    # Raw data
    raw: dict[str, Any] = field(default_factory=dict, repr=False)
    
    @property
    def seconds_to_expiry(self) -> float:
        """Seconds until market expires."""
        now = datetime.now(timezone.utc)
        delta = self.end_date - now
        return max(0, delta.total_seconds())
    
    @property
    def minutes_to_expiry(self) -> float:
        """Minutes until market expires."""
        return self.seconds_to_expiry / 60
    
    @property
    def is_expired(self) -> bool:
        """Whether market has expired."""
        return self.seconds_to_expiry <= 0
    
    @property
    def mid_price(self) -> float:
        """Midpoint price."""
        return (self.yes_price + self.no_price) / 2


class MarketDiscovery:
    """
    Client for discovering markets via the Gamma API.
    
    Usage:
        discovery = MarketDiscovery()
        markets = await discovery.get_crypto_15m_markets()
        btc_markets = [m for m in markets if m.asset == "BTC"]
    """
    
    def __init__(
        self,
        base_url: str = "https://gamma-api.polymarket.com",
        timeout_seconds: int = 15,
    ):
        self.base_url = base_url
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._session: aiohttp.ClientSession | None = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session
    
    async def close(self) -> None:
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def get_markets(
        self,
        active: bool = True,
        closed: bool = False,
        limit: int = 100,
        tag_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Fetch markets from Gamma API.
        
        Args:
            active: Filter for active markets
            closed: Filter for closed markets
            limit: Maximum number of markets to return
            tag_id: Filter by tag (21 = Crypto)
        
        Returns:
            List of raw market data dictionaries
        """
        session = await self._get_session()
        
        params: dict[str, Any] = {
            "limit": limit,
            "active": str(active).lower(),
            "closed": str(closed).lower(),
        }
        if tag_id is not None:
            params["tag_id"] = tag_id
        
        try:
            url = f"{self.base_url}/markets"
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.error("Gamma API request failed", status=resp.status)
                    return []
                
                data = await resp.json()
                logger.debug("Fetched markets from Gamma API", count=len(data))
                return data
                
        except Exception as e:
            logger.error("Gamma API error", error=str(e))
            return []
    
    async def get_crypto_15m_markets(
        self,
        assets: list[str] | None = None,
    ) -> list[Market]:
        """
        Get active 15-minute crypto prediction markets.
        
        Args:
            assets: Filter by asset symbols (BTC, ETH, SOL, XRP)
        
        Returns:
            List of parsed Market objects
        """
        if assets is None:
            assets = ["BTC", "ETH", "SOL", "XRP"]
        
        # Fetch from Gamma API (tag_id=21 is Crypto)
        raw_markets = await self.get_markets(active=True, closed=False, tag_id=21)
        
        markets: list[Market] = []
        for raw in raw_markets:
            market = self._parse_market(raw)
            if market is None:
                continue
            
            # Filter for 15m markets and requested assets
            if market.is_15m and market.asset in assets:
                markets.append(market)
        
        logger.info(
            "Found 15m crypto markets",
            count=len(markets),
            assets=[m.asset for m in markets]
        )
        
        return markets
    
    def _parse_market(self, raw: dict[str, Any]) -> Market | None:
        """Parse raw market data into Market object."""
        try:
            question = raw.get("question", "")
            
            # Extract token IDs
            clob_token_ids = raw.get("clobTokenIds")
            if isinstance(clob_token_ids, str):
                clob_token_ids = json.loads(clob_token_ids)
            
            if not clob_token_ids or len(clob_token_ids) < 2:
                return None
            
            # Extract prices
            outcome_prices = raw.get("outcomePrices")
            if isinstance(outcome_prices, str):
                outcome_prices = json.loads(outcome_prices)
            
            if not outcome_prices or len(outcome_prices) < 2:
                outcome_prices = ["0.5", "0.5"]
            
            # Parse end date
            end_date_str = raw.get("endDate") or raw.get("end_date_iso")
            if not end_date_str:
                return None
            
            if end_date_str.endswith("Z"):
                end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
            else:
                end_date = datetime.fromisoformat(end_date_str)
            
            # Determine asset
            asset = self._detect_asset(question)
            
            # Determine if 15m market
            is_15m = self._is_15m_market(question, raw)
            
            return Market(
                condition_id=raw.get("conditionId", ""),
                question=question,
                slug=raw.get("slug", ""),
                yes_token_id=clob_token_ids[0],
                no_token_id=clob_token_ids[1],
                yes_price=float(outcome_prices[0]),
                no_price=float(outcome_prices[1]),
                end_date=end_date,
                asset=asset,
                is_15m=is_15m,
                active=raw.get("active", True),
                raw=raw,
            )
            
        except Exception as e:
            logger.debug("Failed to parse market", error=str(e))
            return None
    
    def _detect_asset(self, question: str) -> str:
        """Detect which crypto asset the market is for."""
        question_upper = question.upper()
        
        if "BTC" in question_upper or "BITCOIN" in question_upper:
            return "BTC"
        elif "ETH" in question_upper or "ETHEREUM" in question_upper:
            return "ETH"
        elif "SOL" in question_upper or "SOLANA" in question_upper:
            return "SOL"
        elif "XRP" in question_upper or "RIPPLE" in question_upper:
            return "XRP"
        
        return ""
    
    def _is_15m_market(self, question: str, raw: dict[str, Any]) -> bool:
        """Determine if market is a 15-minute market."""
        # Check question text
        question_lower = question.lower()
        if "15 min" in question_lower or "15min" in question_lower:
            return True
        
        # Check if end_date is within 30 minutes
        end_date_str = raw.get("endDate") or raw.get("end_date_iso")
        if end_date_str:
            try:
                if end_date_str.endswith("Z"):
                    end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                else:
                    end_date = datetime.fromisoformat(end_date_str)
                
                now = datetime.now(timezone.utc)
                delta = end_date - now
                minutes = delta.total_seconds() / 60
                
                # If expires in 0-30 minutes and is crypto, likely a 15m market
                if 0 < minutes <= 30:
                    return True
                    
            except Exception:
                pass
        
        return False


async def create_market_discovery(
    base_url: str = "https://gamma-api.polymarket.com",
) -> MarketDiscovery:
    """Factory function to create market discovery client."""
    return MarketDiscovery(base_url=base_url)
