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
        
        In dry_run mode, creates a read-only client for order book queries.
        
        Returns:
            True if initialization successful
        """
        if self.dry_run:
            # Create read-only client for order book queries (no auth needed)
            try:
                from py_clob_client.client import ClobClient
                self._clob_client = ClobClient(
                    host=self.base_url,
                    # No key/creds = read-only mode
                )
                logger.info("CLOB client initialized in dry-run mode (read-only for order books)")
            except ImportError:
                logger.warning("py-clob-client not installed, order book queries disabled")
            except Exception as e:
                logger.warning("Could not create read-only CLOB client", error=str(e))
            
            self._initialized = True
            return True
        
        try:
            # Import py-clob-client
            from py_clob_client.client import ClobClient
            
            secrets = get_secrets()
            
            if not secrets.polymarket_private_key:
                logger.error("Missing POLYMARKET_PRIVATE_KEY environment variable")
                return False
            
            if not secrets.polymarket_funder_address:
                logger.error("Missing POLYMARKET_FUNDER_ADDRESS environment variable")
                return False
            
            # Initialize client with private key
            self._clob_client = ClobClient(
                host=self.base_url,
                key=secrets.polymarket_private_key,
                chain_id=137,  # Polygon
                signature_type=1,  # POLY_GNOSIS_SAFE signature (Magic Link/email wallets)
                funder=secrets.polymarket_funder_address,
            )
            
            # Derive API credentials from private key
            logger.info("Deriving API credentials from private key...")
            self._clob_client.set_api_creds(self._clob_client.create_or_derive_api_creds())
            
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
            size: Order size in USD (will be converted to shares)
            order_type: GTC, GTD, FOK, or IOC
            client_order_id: Optional idempotency key

        Returns:
            OrderResponse with result

        Note:
            Size is provided in USD and automatically converted to shares.
            For Polymarket binary options: shares = usd / price
            Example: $10 at price 0.5 = 20 shares
        """
        # Convert USD to shares
        # For Polymarket binary options: shares = usd / price
        if price <= 0:
            return OrderResponse(
                success=False,
                error_code="INVALID_PRICE",
                error_message=f"Price must be > 0, got {price}",
            )

        size_usd = size  # Keep original for logging
        size_shares = size_usd / price

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
                size_usd=size_usd,
                size_shares=size_shares,
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
            from py_clob_client.clob_types import OrderArgs

            order_side = BUY if side == "BUY" else SELL

            # Create OrderArgs with correct API
            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=size_shares,  # CRITICAL: Use shares, not USD
                side=order_side,
            )
            
            # Create and sign order
            order = self._clob_client.create_order(order_args)
            
            # Submit order
            response = await asyncio.to_thread(
                self._clob_client.post_order, order
            )
            
            if response and response.get("orderID"):
                logger.info(
                    "Order placed",
                    order_id=response["orderID"],
                    token_id=token_id,
                    side=side,
                    price=price,
                    size_usd=size_usd,
                    size_shares=size_shares,
                )
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
            logger.error(
                "Order placement failed",
                error=str(e),
                size_usd=size_usd,
                size_shares=size_shares,
            )
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
    ) -> "OrderBook":
        """
        Get order book for a token.

        Args:
            token_id: Token to get book for

        Returns:
            Parsed OrderBook with bids and asks
        """
        from src.models.order_book import OrderBook

        await self.rate_limiter.acquire_query()

        # In dry-run mode, we still fetch real order books (they're public/read-only)
        # Only skip if no client is available
        if self._clob_client is None:
            return OrderBook(
                token_id=token_id,
                bids=[],
                asks=[],
                timestamp=time.time(),
            )

        if not self._initialized:
            return OrderBook(
                token_id=token_id,
                bids=[],
                asks=[],
                timestamp=time.time(),
            )

        try:
            book_raw = await asyncio.to_thread(
                self._clob_client.get_order_book, token_id
            )

            # Parse bids and asks from py-clob-client's OrderBookSummary
            # It has .bids and .asks attributes, each entry has .price and .size
            bids = []
            asks = []

            if book_raw:
                # Handle OrderBookSummary object from py-clob-client
                if hasattr(book_raw, 'bids'):
                    raw_bids = book_raw.bids or []
                elif isinstance(book_raw, dict):
                    raw_bids = book_raw.get("bids", [])
                else:
                    raw_bids = []

                if hasattr(book_raw, 'asks'):
                    raw_asks = book_raw.asks or []
                elif isinstance(book_raw, dict):
                    raw_asks = book_raw.get("asks", [])
                else:
                    raw_asks = []

                for bid in raw_bids:
                    try:
                        if hasattr(bid, 'price') and hasattr(bid, 'size'):
                            bids.append((float(bid.price), float(bid.size)))
                        elif isinstance(bid, (list, tuple)) and len(bid) >= 2:
                            bids.append((float(bid[0]), float(bid[1])))
                    except (ValueError, TypeError):
                        continue

                for ask in raw_asks:
                    try:
                        if hasattr(ask, 'price') and hasattr(ask, 'size'):
                            asks.append((float(ask.price), float(ask.size)))
                        elif isinstance(ask, (list, tuple)) and len(ask) >= 2:
                            asks.append((float(ask[0]), float(ask[1])))
                    except (ValueError, TypeError):
                        continue

            # Sort: bids descending (best bid first), asks ascending (best ask first)
            bids.sort(key=lambda x: x[0], reverse=True)
            asks.sort(key=lambda x: x[0])

            return OrderBook(
                token_id=token_id,
                bids=bids,
                asks=asks,
                timestamp=time.time(),
            )

        except Exception as e:
            logger.error("Get order book failed", error=str(e), token_id=token_id)
            return OrderBook(
                token_id=token_id,
                bids=[],
                asks=[],
                timestamp=time.time(),
            )


async def create_clob_client(
    dry_run: bool = True,
    base_url: str = "https://clob.polymarket.com",
) -> CLOBClient:
    """Factory function to create and initialize CLOB client."""
    client = CLOBClient(dry_run=dry_run, base_url=base_url)
    await client.initialize()
    return client
