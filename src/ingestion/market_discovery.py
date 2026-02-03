"""
Market discovery via Polymarket Gamma API.

Provides:
- Discovery of active 15-minute crypto markets
- Parsing of market metadata (token IDs, prices, expiry)
- Filtering by asset type
"""

import asyncio
import json
import re
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

    # Timing (required - must come before optional fields)
    end_date: datetime

    # Optional fields with defaults
    open_price: float | None = None  # Asset price at market creation (for UP/DOWN markets)
    created_at: datetime | None = None

    # Metadata
    asset: str = ""  # BTC, ETH, SOL, XRP
    is_15m: bool = False
    interval_minutes: int | None = None  # Explicit interval duration (e.g., 15 for 15-min markets)
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
    def seconds_since_open(self) -> float:
        """
        Seconds since market was created.

        Returns 0 if created_at is not available (fail-safe).
        """
        if self.created_at is None:
            return 0.0
        now = datetime.now(timezone.utc)
        delta = now - self.created_at
        return max(0.0, delta.total_seconds())

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
    
    async def get_active_markets(self, assets: list[str] | None = None) -> list[Market]:
        """Alias for get_crypto_15m_markets."""
        return await self.get_crypto_15m_markets(assets=assets)

    async def get_crypto_15m_markets(
        self,
        assets: list[str] | None = None,
    ) -> list[Market]:
        """
        Get active 15-minute crypto prediction markets.
        
        These markets use a special slug-based endpoint:
        /events/slug/{asset}-updown-15m-{epoch}
        
        Args:
            assets: Filter by asset symbols (BTC, ETH, SOL, XRP)
        
        Returns:
            List of parsed Market objects
        """
        if assets is None:
            assets = ["BTC", "ETH", "SOL", "XRP"]
        
        markets: list[Market] = []
        session = await self._get_session()
        
        # Calculate epochs for current and upcoming 15m windows
        import time
        now = int(time.time())
        current_15m = (now // 900) * 900  # Current 15-min aligned epoch
        
        # Try current and next few 15m windows
        epochs_to_try = [
            current_15m,          # Current window
            current_15m + 900,    # Next window
            current_15m + 1800,   # Window after next
        ]
        
        for asset in assets:
            asset_lower = asset.lower()
            
            for epoch in epochs_to_try:
                slug = f"{asset_lower}-updown-15m-{epoch}"
                url = f"{self.base_url}/events/slug/{slug}"
                
                # FIX: Add retry with exponential backoff
                max_retries = 3
                base_delay = 0.5
                
                for attempt in range(max_retries):
                    try:
                        async with session.get(url) as resp:
                            if resp.status == 404:
                                break  # This epoch doesn't exist, no retry needed
                            elif resp.status != 200:
                                if attempt < max_retries - 1:
                                    delay = base_delay * (2 ** attempt)
                                    logger.debug(
                                        "Retrying after error",
                                        slug=slug,
                                        status=resp.status,
                                        attempt=attempt + 1,
                                        delay=delay,
                                    )
                                    await asyncio.sleep(delay)
                                    continue
                                else:
                                    break  # Give up after max retries
                            
                            event_data = await resp.json()
                            event_markets = event_data.get("markets", [])
                            
                            for raw in event_markets:
                                # Merge event-level data into market
                                raw["slug"] = event_data.get("slug", slug)
                                raw["question"] = event_data.get("title", raw.get("question", ""))
                                
                                market = self._parse_market(raw)
                                if market is not None and not market.is_expired:
                                    # Force 15m flag and asset
                                    market.is_15m = True
                                    market.asset = asset.upper()
                                    market.interval_minutes = 15
                                    markets.append(market)
                                    logger.debug(
                                        "Found 15m market via slug",
                                        asset=asset,
                                        epoch=epoch,
                                        slug=slug,
                                        expires_in=f"{market.minutes_to_expiry:.1f}m"
                                    )
                            break  # Success, no more retries needed
                                    
                    except aiohttp.ClientError as e:
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt)
                            await asyncio.sleep(delay)
                        else:
                            logger.debug("Failed to fetch 15m market after retries", slug=slug, error=str(e))
                        continue
        
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

            # Parse created_at (market creation time)
            created_at = None
            created_at_str = raw.get("createdAt") or raw.get("created_at")
            if created_at_str:
                try:
                    if isinstance(created_at_str, str):
                        if created_at_str.endswith("Z"):
                            created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                        else:
                            created_at = datetime.fromisoformat(created_at_str)
                except Exception:
                    pass

            # Determine asset
            asset = self._detect_asset(question)

            # Determine if 15m market (now with strict validation)
            is_15m, interval_minutes = self._is_15m_market(question, raw, created_at, end_date)

            # Parse open_price (the asset price at market creation)
            # Check multiple possible field names
            open_price = None
            for field_name in ["openPrice", "open_price", "referencePrice", "reference_price", "initialPrice", "initial_price"]:
                if field_name in raw:
                    try:
                        open_price = float(raw[field_name])
                        logger.debug(
                            "Parsed open_price from API",
                            field_name=field_name,
                            open_price=open_price,
                        )
                        break
                    except (ValueError, TypeError):
                        pass

            # FALLBACK: Try to parse reference price from question text
            if open_price is None:
                open_price = self._parse_reference_price_from_question(question)
                if open_price is not None:
                    logger.info(
                        "Extracted open_price from question text",
                        asset=asset,
                        open_price=open_price,
                    )

            # If STILL no open_price, log warning (strategy will need to handle this)
            if open_price is None and is_15m:
                logger.warning(
                    "No open_price found - model accuracy will be degraded",
                    question=question[:50],
                )

            return Market(
                condition_id=raw.get("conditionId", ""),
                question=question,
                slug=raw.get("slug", ""),
                yes_token_id=clob_token_ids[0],
                no_token_id=clob_token_ids[1],
                yes_price=float(outcome_prices[0]),
                no_price=float(outcome_prices[1]),
                open_price=open_price,
                end_date=end_date,
                created_at=created_at,
                asset=asset,
                is_15m=is_15m,
                interval_minutes=interval_minutes,
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
    
    def _parse_reference_price_from_question(self, question: str) -> float | None:
        """
        Extract reference price from market question text.
        
        Examples:
            "Will the price of BTC be 103,500 USDT or higher..." -> 103500.0
            "Will ETH be $3,250 or higher at 2:30 PM?" -> 3250.0
            "Will SOL price be 175.50 or above..." -> 175.50
        
        Returns:
            Reference price as float, or None if not found
        """
        # Pattern matches numbers like: 103,500 or 3250.50 or $3,175
        # Captures the numeric part, handles commas and decimals
        patterns = [
            # "be $103,500 USDT or higher" or "be 103,500 or higher"
            r"be\s+\$?([\d]+(?:,\d{3})*(?:\.\d+)?)\s*(?:USDT|USD|or higher|or above)",
            # "price of $103500" or "price be 103,500"
            r"price\s+(?:of\s+)?(?:be\s+)?\$?([\d]+(?:,\d{3})*(?:\.\d+)?)",
            # "above $103,500" or "at 103,500"
            r"(?:above|at|over)\s+\$?([\d]+(?:,\d{3})*(?:\.\d+)?)",
            # Fallback: any "$" followed by a number (e.g., "$3,250")
            r"\$([\d]+(?:,\d{3})*(?:\.\d+)?)",
            # Fallback: "be <number>" more permissive
            r"be\s+([\d,]+(?:\.\d+)?)\s",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                try:
                    # Remove commas and convert to float
                    price_str = match.group(1).replace(",", "")
                    price = float(price_str)
                    if price > 0:
                        logger.debug(
                            "Parsed reference price from question",
                            price=price,
                            question=question[:60],
                        )
                        return price
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def _is_15m_market(
        self,
        question: str,
        raw: dict[str, Any],
        created_at: datetime | None,
        end_date: datetime,
    ) -> tuple[bool, int | None]:
        """
        Determine if market is a 15-minute market with strict validation.

        Returns:
            Tuple of (is_15m: bool, interval_minutes: int | None)
        """
        import re

        # Strategy 1: Check question pattern
        # Valid patterns for 15-minute markets:
        # - "Will BTC be up/down at HH:MM UTC?"
        # - "Will ETH be above/below $X at HH:MM?"
        question_lower = question.lower()
        has_time_pattern = bool(re.search(r'\d{1,2}:\d{2}', question))
        has_direction_word = any(
            word in question_lower
            for word in ["up", "down", "above", "below", "higher", "lower"]
        )

        # Strategy 2: Calculate interval from created_at to end_date
        interval_minutes = None
        if created_at:
            delta = end_date - created_at
            interval_minutes = int(delta.total_seconds() / 60)

            # 15-minute markets should have interval of 14-16 minutes (allow tolerance)
            if 14 <= interval_minutes <= 16:
                if has_time_pattern and has_direction_word:
                    logger.debug(
                        "Identified 15m market (strict validation)",
                        question=question[:50],
                        interval_minutes=interval_minutes,
                    )
                    return (True, interval_minutes)

        # Strategy 3: Fallback to text matching with warnings
        if "15 min" in question_lower or "15min" in question_lower:
            # Text match found, but no created_at to verify interval
            if created_at is None:
                logger.warning(
                    "15m market detected by text only (no created_at)",
                    question=question[:50],
                )
                return (True, 15)  # Assume 15 minutes
            elif interval_minutes and not (14 <= interval_minutes <= 16):
                logger.warning(
                    "15m text in question but interval mismatch",
                    question=question[:50],
                    interval_minutes=interval_minutes,
                )
                return (False, interval_minutes)

        # Strategy 4: Legacy heuristic (expires soon) - DEPRECATED but kept for backward compatibility
        now = datetime.now(timezone.utc)
        delta_to_expiry = end_date - now
        minutes_to_expiry = delta_to_expiry.total_seconds() / 60

        if 0 < minutes_to_expiry <= 30 and created_at is None:
            # Only use this heuristic if we don't have created_at
            logger.warning(
                "15m market detected by expiry heuristic (unreliable)",
                question=question[:50],
                minutes_to_expiry=int(minutes_to_expiry),
            )
            return (True, None)

        return (False, interval_minutes)


async def create_market_discovery(
    base_url: str = "https://gamma-api.polymarket.com",
) -> MarketDiscovery:
    """Factory function to create market discovery client."""
    return MarketDiscovery(base_url=base_url)
