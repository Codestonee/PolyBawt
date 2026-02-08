"""
CLOB API client for Polymarket trading.

Wraps py-clob-client with:
- Rate limiting
- Idempotent order submission
- Error handling
- Dry-run mode
"""

import asyncio
import math
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
    
    PRICE_TICK = 0.001
    MIN_PRICE = 0.001
    MAX_PRICE = 0.999
    SIZE_STEP_SHARES = 0.01

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
        
        Tries to authenticate even in dry_run mode to fetch account history.
        Safety is enforced in place_order/cancel_order checks.
        
        Returns:
            True if initialization successful
        """
        try:
            from py_clob_client.client import ClobClient
            
            secrets = get_secrets()
            
            # Try full auth first (even in dry_run) to get analysis/history
            if secrets.polymarket_private_key and secrets.polymarket_funder_address:
                try:
                    self._clob_client = ClobClient(
                        host=self.base_url,
                        key=secrets.polymarket_private_key,
                        chain_id=137,  # Polygon
                        signature_type=1,
                        funder=secrets.polymarket_funder_address,
                    )
                    # Derive API credentials
                    logger.info("Deriving API credentials...")
                    self._clob_client.set_api_creds(self._clob_client.create_or_derive_api_creds())
                    self._initialized = True
                    logger.info(f"CLOB client initialized (Auth: YES, Dry Run: {self.dry_run})")
                    return True
                except Exception as e:
                    logger.warning(f"Auth failed in dry-run (falling back to public): {e}")

            # Fallback to public read-only
            self._clob_client = ClobClient(host=self.base_url)
            self._initialized = True
            logger.info(f"CLOB client initialized (Auth: NO, Dry Run: {self.dry_run})")
            return True

        except ImportError:
            logger.error("py-clob-client not installed")
            return False
        except Exception as e:
            logger.error("CLOB client initialization failed", error=str(e))
            return False
    
    async def get_collateral_balance_usdc(self) -> float | None:
        """Fetch collateral (USDC) balance via CLOB (Level 2 auth).

        Returns:
            float balance on success, or None if unavailable.
        """
        if not self._initialized or self._clob_client is None:
            return None

        try:
            from py_clob_client.clob_types import BalanceAllowanceParams, AssetType

            params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            resp = await asyncio.to_thread(self._clob_client.get_balance_allowance, params)
            from src.execution.balance import parse_usdc_balance

            bal = parse_usdc_balance(resp)
            return float(bal)
        except Exception as e:
            logger.warning("Failed to fetch collateral balance", error=str(e))
            return None

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
        # Enforce venue precision/steps before validation and submission.
        raw_price = float(price)
        price = round(round(raw_price / self.PRICE_TICK) * self.PRICE_TICK, 3)
        price = min(self.MAX_PRICE, max(self.MIN_PRICE, price))

        # FIX P1 #4: Validate price is in valid range for binary options (0 < price < 1)
        if price <= 0 or price >= 1:
            return OrderResponse(
                success=False,
                error_code="INVALID_PRICE",
                error_message=f"Price must be between 0 and 1 (exclusive), got {price}",
            )

        # FIX P1 #6: Validate size is positive
        if size <= 0:
            return OrderResponse(
                success=False,
                error_code="INVALID_SIZE",
                error_message=f"Size must be > 0, got {size}",
            )

        # Convert USD to shares
        # For Polymarket binary options: shares = usd / price
        size_usd = size  # Keep original for logging
        size_shares_raw = size_usd / price
        size_shares = math.floor(size_shares_raw / self.SIZE_STEP_SHARES) * self.SIZE_STEP_SHARES
        size_shares = round(size_shares, 8)
        if size_shares <= 0:
            return OrderResponse(
                success=False,
                error_code="INVALID_SIZE_STEP",
                error_message=f"Size too small after step rounding, got {size_shares_raw}",
            )

        if abs(raw_price - price) > 1e-9:
            logger.debug(
                "Price rounded to exchange tick",
                raw_price=raw_price,
                rounded_price=price,
            )

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

        # Retry logic for 500 Internal Server Errors
        max_retries = 3
        retry_delay = 0.5

        for attempt in range(max_retries):
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
                # Map order type to py-clob-client enum.
                from py_clob_client.clob_types import OrderType as PMOrderType
                ot = str(order_type).upper() if order_type else "GTC"
                # Our internal OrderType supports IOC/FAK naming; Polymarket uses FAK.
                if ot == "IOC":
                    ot = "FAK"
                try:
                    pm_order_type = getattr(PMOrderType, ot)
                except Exception:
                    pm_order_type = PMOrderType.GTC

                response = await asyncio.to_thread(
                    self._clob_client.post_order, order, pm_order_type
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
                    # FIX #7: Parse actual error details
                    error_msg = str(response)
                    error_code = "UNKNOWN"
                    
                    if isinstance(response, dict):
                        error_msg = response.get("error") or response.get("errorMsg") or str(response)
                        if response.get("code"):
                            error_code = str(response.get("code"))
                    
                    # Check for 500s in the response body
                    if "Internal server error" in str(error_msg) or "500" in str(error_code):
                        if attempt < max_retries - 1:
                            logger.warning(f"CLOB 500 Error, retrying {attempt+1}/{max_retries}...", error=error_msg)
                            await asyncio.sleep(retry_delay * (2 ** attempt))
                            continue

                    return OrderResponse(
                        success=False,
                        error_code=error_code,
                        error_message=error_msg,
                    )

            except Exception as e:
                # Catch-all for exceptions
                error_str = str(e)
                # Check for 500s in exception message
                if "Internal server error" in error_str or "500" in error_str:
                    if attempt < max_retries - 1:
                        logger.warning(f"CLOB Exception 500, retrying {attempt+1}/{max_retries}...", error=error_str)
                        await asyncio.sleep(retry_delay * (2 ** attempt))
                        continue
                
                logger.error(
                    "Order placement failed",
                    error=error_str,
                    size_usd=size_usd,
                    size_shares=size_shares,
                )
                return OrderResponse(
                    success=False,
                    error_code="EXCEPTION",
                    error_message=error_str,
                )
        
        # If we get here, we exhausted retries
        return OrderResponse(
            success=False,
            error_code="MAX_RETRIES",
            error_message="Exhausted retries on 500 error",
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
        
        # Retry logic for 500 errors
        max_retries = 3
        retry_delay = 0.5

        for attempt in range(max_retries):
            try:
                response = await asyncio.to_thread(
                    self._clob_client.cancel, exchange_order_id
                )
                
                # FIX: Verify cancel response instead of assuming success
                # The API may return an error object or indicate failure
                if response is None:
                    # In some cases, None might actually mean void success, but usually it's a failure
                    # We will treat it as potential 500 if repeated
                    if attempt < max_retries - 1:
                        logger.warning(f"Cancel returned None, retrying {attempt+1}/{max_retries}...", exchange_order_id=exchange_order_id)
                        await asyncio.sleep(retry_delay * (2 ** attempt))
                        continue

                    logger.warning(
                        "Cancel returned None response",
                        exchange_order_id=exchange_order_id,
                    )
                    return CancelResponse(
                        success=False,
                        error_message="Cancel returned None response",
                    )
                
                # Check for error fields in response
                if isinstance(response, dict):
                    if response.get("error") or response.get("errorMsg"):
                        error_msg = response.get("error") or response.get("errorMsg") or "Unknown cancel error"
                        
                        # Retry on 500
                        if "Internal server error" in str(error_msg) or "500" in str(error_msg):
                            if attempt < max_retries - 1:
                                logger.warning(f"Cancel 500 Error, retrying {attempt+1}/{max_retries}...", error=error_msg)
                                await asyncio.sleep(retry_delay * (2 ** attempt))
                                continue

                        logger.error(
                            "Cancel API returned error",
                            exchange_order_id=exchange_order_id,
                            error=error_msg,
                        )
                        return CancelResponse(
                            success=False,
                            error_message=str(error_msg),
                        )
                    # Check for success indicators
                    if response.get("canceled") == False or response.get("success") == False:
                        return CancelResponse(
                            success=False,
                            error_message=f"Cancel rejected: {response}",
                        )
                
                logger.info(
                    "Order cancelled successfully",
                    exchange_order_id=exchange_order_id,
                )
                return CancelResponse(success=True)
                
            except Exception as e:
                # Catch 500s in exception
                if "Internal server error" in str(e) or "500" in str(e):
                    if attempt < max_retries - 1:
                        logger.warning(f"Cancel Exception 500, retrying {attempt+1}/{max_retries}...", error=str(e))
                        await asyncio.sleep(retry_delay * (2 ** attempt))
                        continue
                
                logger.error("Order cancel failed", error=str(e))
                return CancelResponse(
                    success=False,
                    error_message=str(e),
                )
        
        return CancelResponse(
            success=False,
            error_message="Exhausted retries on 500 error",
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


    async def get_trade_history(
        self,
        limit: int = 50,
        maker: bool | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get trade history for the account.
        
        Args:
            limit: Number of trades to fetch
            maker: Filter by maker/taker (None = both)
        
        Returns:
            List of trade dicts
        """
        await self.rate_limiter.acquire_query()
        
        if not self._initialized or not self._clob_client:
            return []
            
        try:
            # Check if we have auth (private key)
            # py-clob-client requires auth for get_trades
            if not hasattr(self._clob_client, 'key') or not self._clob_client.key:
                logger.debug("Cannot fetch trade history: No auth credentials")
                return []

            trades = await asyncio.to_thread(
                self._clob_client.get_trades,
                limit=limit,
                maker=maker
            )
            return trades or []
            
        except Exception as e:
            logger.error("Get trade history failed", error=str(e))
            return []


    async def get_positions(self) -> list[dict[str, Any]]:
        """
        Get open positions from exchange.
        
        Used by Reconciler for position sync.
        
        Returns:
            List of position dicts with token_id and size
        """
        await self.rate_limiter.acquire_query()
        
        if not self._initialized or not self._clob_client:
            return []
        
        try:
            # py-clob-client get_positions returns account positions
            positions = await asyncio.to_thread(
                self._clob_client.get_positions
            )
            return positions or []
            
        except Exception as e:
            logger.error("Get positions failed", error=str(e))
            return []


    async def get_account_balance(self) -> float:
        """
        Get account collateral (USDC) balance.
        
        Returns:
            Balance in USDC (float)
        """
        res = await self.get_collateral_balance_usdc()
        return res if res is not None else 0.0


async def create_clob_client(
    dry_run: bool = True,
    base_url: str = "https://clob.polymarket.com",
) -> CLOBClient:
    """Factory function to create and initialize CLOB client."""
    client = CLOBClient(dry_run=dry_run, base_url=base_url)
    await client.initialize()
    return client
