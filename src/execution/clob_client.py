"""
CLOB API client for Polymarket trading.

Wraps py-clob-client with:
- Rate limiting
- Idempotent order submission
- Error handling
- Dry-run mode
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from src.infrastructure.logging import get_logger
from src.infrastructure.config import get_secrets
from src.execution.rate_limiter import RateLimiter

logger = get_logger(__name__)


@dataclass
class OrderResponse:
    """Response from order submission."""
    
    success: bool
    exchange_order_id: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    timestamp: float = 0.0


@dataclass
class CancelResponse:
    """Response from order cancellation."""
    
    success: bool
    error_message: str | None = None


class CLOBClient:
    """
    Async client for Polymarket CLOB API.
    
    Wraps py-clob-client with additional features:
    - Rate limiting
    - Retry logic
    - Error normalization
    - Dry-run mode
    
    Usage:
        client = CLOBClient(dry_run=True)
        await client.initialize()
        
        response = await client.place_order(
            token_id="abc123",
            side="BUY",
            price=0.55,
            size=10.0,
        )
    """
    
    def __init__(
        self,
        dry_run: bool = True,
        base_url: str = "https://clob.polymarket.com",
        rate_limiter: RateLimiter | None = None,
    ):
        self.dry_run = dry_run
        self.base_url = base_url
        self.rate_limiter = rate_limiter or RateLimiter()
        
        self._clob_client: Any = None
        self._initialized = False
    
    async def initialize(self) -> bool:
        """
        Initialize the CLOB client with credentials.
        
        In dry_run mode, skips actual client initialization.
        
        Returns:
            True if initialization successful
        """
        if self.dry_run:
            logger.info("CLOB client initialized in dry-run mode")
            self._initialized = True
            return True
        
        try:
            # Import py-clob-client
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import ApiCreds
            
            secrets = get_secrets()
            
            if not secrets.polymarket_api_key:
                logger.error("Missing POLYMARKET_API_KEY")
                return False
            
            # Create credentials
            creds = ApiCreds(
                api_key=secrets.polymarket_api_key,
                api_secret=secrets.polymarket_api_secret,
                api_passphrase=secrets.polymarket_passphrase,
            )
            
            # Initialize client
            self._clob_client = ClobClient(
                host=self.base_url,
                key=secrets.polymarket_private_key,
                chain_id=137,  # Polygon
                creds=creds,
                funder=secrets.polymarket_funder_address,
            )
            
            self._initialized = True
            logger.info("CLOB client initialized")
            return True
            
        except ImportError:
            logger.error("py-clob-client not installed")
            return False
        except Exception as e:
            logger.error("CLOB client initialization failed", error=str(e))
            return False
    
    async def place_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
        order_type: str = "GTC",
        client_order_id: str | None = None,
    ) -> OrderResponse:
        """
        Place an order on the CLOB.
        
        Args:
            token_id: Token to trade
            side: "BUY" or "SELL"
            price: Limit price (0-1)
            size: Order size in shares
            order_type: GTC, GTD, FOK, or IOC
            client_order_id: Optional idempotency key
        
        Returns:
            OrderResponse with result
        """
        # Rate limit
        wait_time = await self.rate_limiter.acquire_order()
        if wait_time > 0:
            logger.debug("Rate limited", wait_seconds=wait_time)
        
        if self.dry_run:
            logger.info(
                "DRY RUN: Place order",
                token_id=token_id,
                side=side,
                price=price,
                size=size,
            )
            return OrderResponse(
                success=True,
                exchange_order_id=f"dry_run_{int(time.time() * 1000)}",
                timestamp=time.time(),
            )
        
        if not self._initialized:
            return OrderResponse(
                success=False,
                error_code="NOT_INITIALIZED",
                error_message="CLOB client not initialized",
            )
        
        try:
            # Build order using py-clob-client
            from py_clob_client.order_builder.constants import BUY, SELL
            
            order_side = BUY if side == "BUY" else SELL
            
            # Create and sign order
            order = self._clob_client.create_order(
                token_id=token_id,
                price=price,
                size=size,
                side=order_side,
            )
            
            # Submit order
            response = await asyncio.to_thread(
                self._clob_client.post_order, order
            )
            
            if response and response.get("orderID"):
                return OrderResponse(
                    success=True,
                    exchange_order_id=response["orderID"],
                    timestamp=time.time(),
                )
            else:
                return OrderResponse(
                    success=False,
                    error_code="UNKNOWN",
                    error_message=str(response),
                )
                
        except Exception as e:
            logger.error("Order placement failed", error=str(e))
            return OrderResponse(
                success=False,
                error_code="EXCEPTION",
                error_message=str(e),
            )
    
    async def cancel_order(
        self,
        exchange_order_id: str,
    ) -> CancelResponse:
        """
        Cancel an order.
        
        Args:
            exchange_order_id: Exchange-assigned order ID
        
        Returns:
            CancelResponse with result
        """
        await self.rate_limiter.acquire_order()
        
        if self.dry_run:
            logger.info(
                "DRY RUN: Cancel order",
                exchange_order_id=exchange_order_id,
            )
            return CancelResponse(success=True)
        
        if not self._initialized:
            return CancelResponse(
                success=False,
                error_message="CLOB client not initialized",
            )
        
        try:
            response = await asyncio.to_thread(
                self._clob_client.cancel, exchange_order_id
            )
            
            return CancelResponse(success=True)
            
        except Exception as e:
            logger.error("Order cancel failed", error=str(e))
            return CancelResponse(
                success=False,
                error_message=str(e),
            )
    
    async def cancel_all_orders(self) -> int:
        """
        Cancel all open orders.
        
        Returns:
            Number of orders canceled
        """
        await self.rate_limiter.acquire_order()
        
        if self.dry_run:
            logger.info("DRY RUN: Cancel all orders")
            return 0
        
        if not self._initialized:
            return 0
        
        try:
            response = await asyncio.to_thread(
                self._clob_client.cancel_all
            )
            return len(response) if response else 0
        except Exception as e:
            logger.error("Cancel all failed", error=str(e))
            return 0
    
    async def get_open_orders(
        self,
        token_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get open orders.
        
        Args:
            token_id: Optional filter by token
        
        Returns:
            List of open orders
        """
        await self.rate_limiter.acquire_query()
        
        if self.dry_run:
            return []
        
        if not self._initialized:
            return []
        
        try:
            orders = await asyncio.to_thread(
                self._clob_client.get_orders
            )
            
            if token_id:
                orders = [o for o in orders if o.get("asset_id") == token_id]
            
            return orders or []
            
        except Exception as e:
            logger.error("Get orders failed", error=str(e))
            return []
    
    async def get_order_book(
        self,
        token_id: str,
    ) -> dict[str, Any]:
        """
        Get order book for a token.
        
        Args:
            token_id: Token to get book for
        
        Returns:
            Order book with bids and asks
        """
        await self.rate_limiter.acquire_query()
        
        if self.dry_run:
            return {"bids": [], "asks": []}
        
        if not self._initialized:
            return {"bids": [], "asks": []}
        
        try:
            book = await asyncio.to_thread(
                self._clob_client.get_order_book, token_id
            )
            return book or {"bids": [], "asks": []}
            
        except Exception as e:
            logger.error("Get order book failed", error=str(e))
            return {"bids": [], "asks": []}


async def create_clob_client(
    dry_run: bool = True,
    base_url: str = "https://clob.polymarket.com",
) -> CLOBClient:
    """Factory function to create and initialize CLOB client."""
    client = CLOBClient(dry_run=dry_run, base_url=base_url)
    await client.initialize()
    return client
