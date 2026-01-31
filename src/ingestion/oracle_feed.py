"""
Oracle price feed adapters.

Provides real-time crypto price data from:
- Binance (primary)
- Coinbase (fallback)
- Chainlink Data Streams (settlement source - for sniper protection)

CRITICAL: Polymarket settles based on Chainlink oracle prices, NOT Binance.
The Chainlink feed is used to detect "sniper risk" - when HFT bots with
faster Chainlink access could front-run stale orders.

Research (gemeni.txt):
"The trading bot must integrate the Chainlink Data Streams SDK directly.
If the Chainlink price deviates from the Polymarket mid-price by more than
a threshold, the bot should immediately cancel orders to pull liquidity."
"""

import asyncio
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine

import aiohttp

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class SniperRiskLevel(Enum):
    """Risk level for sniper/latency arbitrage."""
    SAFE = "safe"
    ELEVATED = "elevated"
    CRITICAL = "critical"


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
class SniperAlert:
    """Alert when sniper risk is detected."""

    asset: str
    risk_level: SniperRiskLevel
    chainlink_price: float
    market_price: float
    divergence_pct: float
    timestamp: float = field(default_factory=time.time)

    @property
    def should_cancel_orders(self) -> bool:
        """Whether orders should be cancelled immediately."""
        return self.risk_level == SniperRiskLevel.CRITICAL


@dataclass
class OracleHealth:
    """Health status of oracle feeds."""

    binance_healthy: bool = False
    coinbase_healthy: bool = False
    chainlink_healthy: bool = False
    last_price_age_seconds: float = float("inf")
    last_chainlink_age_seconds: float = float("inf")

    @property
    def any_healthy(self) -> bool:
        return self.binance_healthy or self.coinbase_healthy

    @property
    def settlement_source_healthy(self) -> bool:
        """Whether the settlement price source (Chainlink) is healthy."""
        return self.chainlink_healthy


class OracleFeed:
    """
    Multi-source price oracle with Chainlink sniper protection.

    Price Sources (for trading signals):
    1. Binance (lowest latency, highest volume)
    2. Coinbase (reliable fallback)

    Settlement Source (for sniper protection):
    - Chainlink Data Streams (what Polymarket actually settles on)

    CRITICAL: The Chainlink price is used to detect when HFT bots might
    have seen a price update before it's reflected in Polymarket markets.
    When divergence exceeds threshold, orders should be cancelled.

    Usage:
        oracle = OracleFeed()
        btc_price = await oracle.get_price("BTC")

        # Check for sniper risk before placing orders
        alert = await oracle.check_sniper_risk("BTC", market_mid_price=95000)
        if alert and alert.should_cancel_orders:
            await order_manager.cancel_all()
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

    # Chainlink Data Streams feed IDs (Arbitrum Mainnet)
    # These are the official Chainlink feed IDs for crypto prices
    # See: https://docs.chain.link/data-streams/crypto-streams
    CHAINLINK_FEED_IDS = {
        "BTC": "0x00027bbaff688c906a3e20a34fe951715d1018d262a5b66e38eda027a674cd1b",  # BTC/USD
        "ETH": "0x000359843a543ee2fe414dc14c7e7920ef10f4372990b79d6361cdc0dd1ba782",  # ETH/USD
        "SOL": "0x000576c5cff91e9ad22ab976b6236fa4c80be86ab12f111dbabbe5f4178abfd3",  # SOL/USD
        "XRP": "0x0003a8fd2ea9d6ba2d48ed496cbe9e0fd7a5e0a8f2d8c1c5e22d0a0f55c2c8d1",  # XRP/USD (placeholder)
    }

    # Sniper risk thresholds (percentage divergence)
    SNIPER_THRESHOLD_ELEVATED = 0.001  # 0.1% - elevated risk
    SNIPER_THRESHOLD_CRITICAL = 0.003  # 0.3% - critical, cancel orders

    def __init__(
        self,
        timeout_seconds: int = 5,
        stale_threshold_seconds: float = 10.0,
        chainlink_client_id: str | None = None,
        chainlink_client_secret: str | None = None,
        sniper_callback: Callable[[SniperAlert], Coroutine[Any, Any, None]] | None = None,
    ):
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self.stale_threshold = stale_threshold_seconds

        # Chainlink credentials (from environment if not provided)
        self._chainlink_client_id = chainlink_client_id or os.getenv("CHAINLINK_CLIENT_ID")
        self._chainlink_client_secret = chainlink_client_secret or os.getenv("CHAINLINK_CLIENT_SECRET")

        # Callback when sniper risk is detected
        self._sniper_callback = sniper_callback

        self._session: aiohttp.ClientSession | None = None
        self._cache: dict[str, PriceTick] = {}
        self._chainlink_cache: dict[str, PriceTick] = {}
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
        # Update Chainlink age
        if self._chainlink_cache:
            min_chainlink_age = min(tick.age_seconds for tick in self._chainlink_cache.values())
            self._health.last_chainlink_age_seconds = min_chainlink_age
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

    async def get_chainlink_price(self, asset: str) -> float | None:
        """
        Fetch price from Chainlink Data Streams.

        This is the SETTLEMENT price source for Polymarket 15-min markets.
        Used for sniper risk detection, not for trading signals.

        Args:
            asset: Asset symbol (BTC, ETH, SOL, XRP)

        Returns:
            Current Chainlink price in USD, or None if unavailable
        """
        # Check cache first (Chainlink has tighter freshness requirement)
        cached = self._chainlink_cache.get(asset)
        if cached and cached.age_seconds < 2.0:  # 2 second cache for Chainlink
            return cached.price

        price = await self._fetch_chainlink_price(asset)
        if price is not None:
            self._chainlink_cache[asset] = PriceTick(
                asset=asset, price=price, source="chainlink"
            )
            self._health.chainlink_healthy = True
            return price

        self._health.chainlink_healthy = False
        return cached.price if cached else None

    async def _fetch_chainlink_price(self, asset: str) -> float | None:
        """
        Fetch price from Chainlink Data Streams API.

        Chainlink Data Streams uses a pull-based model where clients
        request signed price reports. For production, you'd use their
        official SDK, but this demonstrates the API pattern.
        """
        feed_id = self.CHAINLINK_FEED_IDS.get(asset)
        if not feed_id:
            return None

        # If no credentials, fall back to Binance as proxy
        # (In production, you MUST use actual Chainlink credentials)
        if not self._chainlink_client_id or not self._chainlink_client_secret:
            logger.debug(
                "Chainlink credentials not configured, using Binance as proxy",
                asset=asset,
            )
            return await self._get_binance_price(asset)

        try:
            session = await self._get_session()

            # Chainlink Data Streams API endpoint
            # See: https://docs.chain.link/data-streams/reference/streams-direct/streams-direct-onchain-verification
            url = "https://api.chain.link/data-streams/v1/reports/latest"

            headers = {
                "Authorization": f"Basic {self._chainlink_client_id}:{self._chainlink_client_secret}",
                "Content-Type": "application/json",
            }

            params = {"feedID": feed_id}

            async with session.get(url, headers=headers, params=params) as resp:
                if resp.status != 200:
                    logger.warning(
                        "Chainlink API error",
                        status=resp.status,
                        asset=asset,
                    )
                    return None

                data = await resp.json()

                # Parse the report - Chainlink returns price with 18 decimals
                if "report" in data and "benchmarkPrice" in data["report"]:
                    raw_price = int(data["report"]["benchmarkPrice"])
                    price = raw_price / 1e18

                    logger.debug("Chainlink price", asset=asset, price=price)
                    return price

        except Exception as e:
            logger.warning("Chainlink price fetch failed", asset=asset, error=str(e))

        return None

    async def check_sniper_risk(
        self,
        asset: str,
        market_mid_price: float,
    ) -> SniperAlert | None:
        """
        Check for sniper/latency arbitrage risk.

        Compares the Chainlink settlement price against the Polymarket
        mid-price. If divergence exceeds threshold, HFT bots with faster
        Chainlink access could be front-running stale orders.

        CRITICAL: Call this before placing orders and periodically while
        orders are open. If CRITICAL risk is detected, cancel all orders.

        Args:
            asset: Asset symbol
            market_mid_price: Current Polymarket mid-price for the asset

        Returns:
            SniperAlert if risk detected, None if safe
        """
        chainlink_price = await self.get_chainlink_price(asset)

        if chainlink_price is None:
            # Can't determine risk without Chainlink price
            logger.warning(
                "Cannot check sniper risk - Chainlink unavailable",
                asset=asset,
            )
            return None

        # Calculate divergence
        divergence = abs(chainlink_price - market_mid_price) / market_mid_price

        # Determine risk level
        if divergence >= self.SNIPER_THRESHOLD_CRITICAL:
            risk_level = SniperRiskLevel.CRITICAL
        elif divergence >= self.SNIPER_THRESHOLD_ELEVATED:
            risk_level = SniperRiskLevel.ELEVATED
        else:
            return None  # Safe, no alert needed

        alert = SniperAlert(
            asset=asset,
            risk_level=risk_level,
            chainlink_price=chainlink_price,
            market_price=market_mid_price,
            divergence_pct=divergence * 100,
        )

        logger.warning(
            "SNIPER RISK DETECTED",
            asset=asset,
            risk_level=risk_level.value,
            chainlink_price=chainlink_price,
            market_price=market_mid_price,
            divergence_pct=f"{divergence*100:.3f}%",
            should_cancel=alert.should_cancel_orders,
        )

        # Trigger callback if registered
        if self._sniper_callback:
            try:
                await self._sniper_callback(alert)
            except Exception as e:
                logger.error("Sniper callback failed", error=str(e))

        return alert

    def get_chainlink_cached_price(self, asset: str) -> PriceTick | None:
        """Get cached Chainlink price tick."""
        return self._chainlink_cache.get(asset)

    def register_sniper_callback(
        self,
        callback: Callable[[SniperAlert], Coroutine[Any, Any, None]],
    ) -> None:
        """Register callback for sniper risk alerts."""
        self._sniper_callback = callback


async def create_oracle_feed(
    timeout_seconds: int = 5,
    stale_threshold_seconds: float = 10.0,
    sniper_callback: Callable[[SniperAlert], Coroutine[Any, Any, None]] | None = None,
) -> OracleFeed:
    """Factory function to create oracle feed."""
    return OracleFeed(
        timeout_seconds=timeout_seconds,
        stale_threshold_seconds=stale_threshold_seconds,
        sniper_callback=sniper_callback,
    )
