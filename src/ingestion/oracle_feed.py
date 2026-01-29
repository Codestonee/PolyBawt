"""
Oracle price feed adapters.

Provides real-time crypto price data from:
- Binance (primary)
- Coinbase (fallback)
- Future: Chainlink Data Streams (for settlement alignment)
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import aiohttp

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PriceTick:
    """Single price observation."""
    
    asset: str
    price: float
    timestamp: float = field(default_factory=time.time)
    source: str = "binance"
    
    @property
    def age_seconds(self) -> float:
        """Age of this tick in seconds."""
        return time.time() - self.timestamp


@dataclass
class OracleHealth:
    """Health status of oracle feeds."""
    
    binance_healthy: bool = False
    coinbase_healthy: bool = False
    last_price_age_seconds: float = float("inf")
    
    @property
    def any_healthy(self) -> bool:
        return self.binance_healthy or self.coinbase_healthy


class OracleFeed:
    """
    Multi-source price oracle with fallback.
    
    Priority:
    1. Binance (lowest latency, highest volume)
    2. Coinbase (reliable fallback)
    
    Future: Add Chainlink Data Streams for settlement alignment.
    
    Usage:
        oracle = OracleFeed()
        btc_price = await oracle.get_price("BTC")
        all_prices = await oracle.get_all_prices()
    """
    
    # Binance trading pairs
    BINANCE_PAIRS = {
        "BTC": "BTCUSDT",
        "ETH": "ETHUSDT",
        "SOL": "SOLUSDT",
        "XRP": "XRPUSDT",
    }
    
    # Coinbase trading pairs
    COINBASE_PAIRS = {
        "BTC": "BTC-USD",
        "ETH": "ETH-USD",
        "SOL": "SOL-USD",
        "XRP": "XRP-USD",
    }
    
    def __init__(
        self,
        timeout_seconds: int = 5,
        stale_threshold_seconds: float = 10.0,
    ):
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self.stale_threshold = stale_threshold_seconds
        
        self._session: aiohttp.ClientSession | None = None
        self._cache: dict[str, PriceTick] = {}
        self._health = OracleHealth()
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session
    
    async def close(self) -> None:
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    @property
    def health(self) -> OracleHealth:
        """Current oracle health status."""
        # Update last price age
        if self._cache:
            min_age = min(tick.age_seconds for tick in self._cache.values())
            self._health.last_price_age_seconds = min_age
        return self._health
    
    async def get_price(self, asset: str) -> float | None:
        """
        Get current price for asset.
        
        Args:
            asset: Asset symbol (BTC, ETH, SOL, XRP)
        
        Returns:
            Current price in USD, or None if unavailable
        """
        # Check cache first (if fresh)
        cached = self._cache.get(asset)
        if cached and cached.age_seconds < self.stale_threshold:
            return cached.price
        
        # Try Binance first
        price = await self._get_binance_price(asset)
        if price is not None:
            self._cache[asset] = PriceTick(asset=asset, price=price, source="binance")
            return price
        
        # Fallback to Coinbase
        price = await self._get_coinbase_price(asset)
        if price is not None:
            self._cache[asset] = PriceTick(asset=asset, price=price, source="coinbase")
            return price
        
        # Use stale cache as last resort
        if cached:
            logger.warning(
                "Using stale price",
                asset=asset,
                age_seconds=cached.age_seconds
            )
            return cached.price
        
        logger.error("No price available", asset=asset)
        return None
    
    async def get_all_prices(
        self,
        assets: list[str] | None = None,
    ) -> dict[str, float]:
        """
        Get current prices for all assets.
        
        Args:
            assets: List of assets (default: BTC, ETH, SOL, XRP)
        
        Returns:
            Dict of asset -> price
        """
        if assets is None:
            assets = ["BTC", "ETH", "SOL", "XRP"]
        
        # Fetch all in parallel
        tasks = [self.get_price(asset) for asset in assets]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        prices = {}
        for asset, result in zip(assets, results):
            if isinstance(result, Exception):
                logger.warning("Failed to get price", asset=asset, error=str(result))
            elif result is not None:
                prices[asset] = result
        
        return prices
    
    async def _get_binance_price(self, asset: str) -> float | None:
        """Fetch price from Binance."""
        pair = self.BINANCE_PAIRS.get(asset)
        if not pair:
            return None
        
        try:
            session = await self._get_session()
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={pair}"
            
            async with session.get(url) as resp:
                if resp.status != 200:
                    self._health.binance_healthy = False
                    return None
                
                data = await resp.json()
                price = float(data["price"])
                self._health.binance_healthy = True
                
                logger.debug("Binance price", asset=asset, price=price)
                return price
                
        except Exception as e:
            self._health.binance_healthy = False
            logger.warning("Binance price fetch failed", asset=asset, error=str(e))
            return None
    
    async def _get_coinbase_price(self, asset: str) -> float | None:
        """Fetch price from Coinbase."""
        pair = self.COINBASE_PAIRS.get(asset)
        if not pair:
            return None
        
        try:
            session = await self._get_session()
            url = f"https://api.coinbase.com/v2/prices/{pair}/spot"
            
            async with session.get(url) as resp:
                if resp.status != 200:
                    self._health.coinbase_healthy = False
                    return None
                
                data = await resp.json()
                price = float(data["data"]["amount"])
                self._health.coinbase_healthy = True
                
                logger.debug("Coinbase price", asset=asset, price=price)
                return price
                
        except Exception as e:
            self._health.coinbase_healthy = False
            logger.warning("Coinbase price fetch failed", asset=asset, error=str(e))
            return None
    
    def get_cached_price(self, asset: str) -> PriceTick | None:
        """Get cached price tick (may be stale)."""
        return self._cache.get(asset)
    
    def is_price_stale(self, asset: str) -> bool:
        """Check if cached price is stale."""
        cached = self._cache.get(asset)
        if cached is None:
            return True
        return cached.age_seconds > self.stale_threshold


async def create_oracle_feed(
    timeout_seconds: int = 5,
    stale_threshold_seconds: float = 10.0,
) -> OracleFeed:
    """Factory function to create oracle feed."""
    return OracleFeed(
        timeout_seconds=timeout_seconds,
        stale_threshold_seconds=stale_threshold_seconds,
    )
