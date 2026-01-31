"""
Idempotent Order Manager for Polymarket CLOB.

Features:
- Client-order-ID based idempotency
- Order lifecycle state machine
- Partial fill tracking
- Cancel/replace support
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine

from src.infrastructure.logging import get_logger
from src.execution.rate_limiter import RateLimiter

logger = get_logger(__name__)


class OrderState(Enum):
    """Order lifecycle states."""
    
    CREATING = "creating"        # Order being prepared
    PENDING_NEW = "pending_new"  # Sent to exchange, awaiting ack
    NEW = "new"                  # Resting on book
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"           # Completely executed
    PENDING_CANCEL = "pending_cancel"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    FAILED = "failed"           # Client-side failure
    
    @property
    def is_terminal(self) -> bool:
        """Whether this is a terminal state."""
        return self in (
            OrderState.FILLED,
            OrderState.CANCELED,
            OrderState.REJECTED,
            OrderState.EXPIRED,
            OrderState.FAILED,
        )
    
    @property
    def is_active(self) -> bool:
        """Whether order is active on the book."""
        return self in (OrderState.NEW, OrderState.PARTIALLY_FILLED)


class OrderType(Enum):
    """Order types supported by Polymarket CLOB."""
    
    GTC = "GTC"  # Good-till-cancelled
    GTD = "GTD"  # Good-till-date
    FOK = "FOK"  # Fill-or-kill
    IOC = "IOC"  # Immediate-or-cancel (FAK)


class OrderSide(Enum):
    """Order side."""
    
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Order:
    """
    Order with full lifecycle tracking.
    
    Uses client_order_id for idempotency.
    """
    
    # Identifiers
    client_order_id: str
    token_id: str
    
    # Order details
    side: OrderSide
    price: float
    size: float
    order_type: OrderType = OrderType.GTC
    
    # State
    state: OrderState = OrderState.CREATING
    
    # Fills
    filled_size: float = 0.0
    average_fill_price: float = 0.0
    
    # Exchange data
    exchange_order_id: str | None = None
    
    # Timestamps
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    submitted_at: float | None = None
    filled_at: float | None = None
    
    # Metadata
    strategy_id: str = "default"
    error_message: str = ""
    
    @property
    def remaining_size(self) -> float:
        """Size not yet filled."""
        return max(0, self.size - self.filled_size)
    
    @property
    def fill_pct(self) -> float:
        """Percentage filled."""
        if self.size == 0:
            return 0
        return self.filled_size / self.size
    
    @property
    def age_seconds(self) -> float:
        """Time since order creation."""
        return time.time() - self.created_at
    
    def update_state(self, new_state: OrderState) -> None:
        """Update state with timestamp."""
        self.state = new_state
        self.updated_at = time.time()
        # Note: Persistence happens in OrderManager, not here
    
    def add_fill(self, fill_size: float, fill_price: float) -> None:
        """Record a fill."""
        if fill_size <= 0:
            return
        
        # FIX: Clamp fill_size to remaining_size to prevent overfills
        remaining = self.remaining_size
        if fill_size > remaining:
            from src.infrastructure.logging import get_logger
            logger = get_logger(__name__)
            logger.warning(
                "Fill exceeds remaining size, clamping",
                client_order_id=self.client_order_id,
                fill_size=fill_size,
                remaining_size=remaining,
                clamped_to=remaining,
            )
            fill_size = remaining
        
        if fill_size <= 0:
            return  # Nothing left to fill
        
        # Update weighted average price
        old_value = self.filled_size * self.average_fill_price
        new_value = fill_size * fill_price
        self.filled_size += fill_size
        
        if self.filled_size > 0:
            self.average_fill_price = (old_value + new_value) / self.filled_size
        
        self.updated_at = time.time()
        
        # Update state
        if self.filled_size >= self.size:
            self.state = OrderState.FILLED
            self.filled_at = time.time()
        else:
            self.state = OrderState.PARTIALLY_FILLED


def generate_client_order_id(
    strategy_id: str,
    token_id: str,
    nonce: str | None = None,
    deterministic: bool = False,
    price: float | None = None,
    side: str | None = None,
) -> str:
    """
    Generate idempotent client order ID.

    Format (random): {strategy_id}_{token_id_prefix}_{timestamp_ms}_{nonce}
    Format (deterministic): {strategy_id}_{token_id_prefix}_{price_int}_{side}

    Args:
        strategy_id: Strategy identifier
        token_id: Token being traded
        nonce: Optional nonce (random if None)
        deterministic: If True, use price+side instead of timestamp+nonce
        price: Price for deterministic mode
        side: Side for deterministic mode (BUY/SELL)

    Returns:
        Client order ID string
    """
    token_prefix = token_id[:8] if len(token_id) >= 8 else token_id

    if deterministic:
        # FIX #8: Deterministic IDs for crash recovery
        # Use price and side as the nonce component
        if price is None or side is None:
            raise ValueError("price and side required for deterministic IDs")
        price_int = int(price * 10000)  # Convert to basis points
        return f"{strategy_id}_{token_prefix}_{price_int}_{side}"
    else:
        # Random mode (backward compatible)
        timestamp_ms = int(time.time() * 1000)
        nonce = nonce or uuid.uuid4().hex[:6]
        return f"{strategy_id}_{token_prefix}_{timestamp_ms}_{nonce}"


OrderCallback = Callable[[Order], Coroutine[Any, Any, None]]


class OrderManager:
    """
    Manages order lifecycle with idempotency guarantees.
    
    Features:
    - Client-order-ID based idempotency
    - Order state tracking
    - Partial fill handling
    - Callbacks for state changes
    
    Usage:
        manager = OrderManager()
        
        order = manager.create_order(
            token_id="abc123",
            side=OrderSide.BUY,
            price=0.55,
            size=10.0,
        )
        
        await manager.submit_order(order)
        
        # Later, check status
        status = manager.get_order(order.client_order_id)
    """
    
    def __init__(
        self,
        dry_run: bool = True,
        submit_callback: OrderCallback | None = None,
        cancel_callback: OrderCallback | None = None,
        persistence_file: str | None = None,  # FIX #8: Optional persistence
        rate_limiter: RateLimiter | None = None,  # FIX #6: Rate limiter enforcement
    ):
        self.dry_run = dry_run
        self._submit_callback = submit_callback
        self._cancel_callback = cancel_callback
        self._rate_limiter = rate_limiter  # FIX #6: Store for use in submit/cancel

        # FIX #8: Order state persistence for crash recovery
        self._persistence: "OrderStateStore | None" = None
        if persistence_file:
            from src.execution.order_persistence import OrderStateStore
            self._persistence = OrderStateStore(persistence_file)
            logger.info("Order persistence enabled", file=persistence_file)
        
        # Order tracking
        self._orders: dict[str, Order] = {}
        self._orders_by_token: dict[str, list[str]] = {}
        self._pending_orders: set[str] = set()
        
        # Metrics
        self._total_submitted = 0
        self._total_filled = 0
        self._total_canceled = 0
        self._total_rejected = 0
    
    def create_order(
        self,
        token_id: str,
        side: OrderSide,
        price: float,
        size: float,
        order_type: OrderType = OrderType.GTC,
        strategy_id: str = "default",
        client_order_id: str | None = None,
    ) -> Order:
        """
        Create a new order (but don't submit it).
        
        Args:
            token_id: Token to trade
            side: BUY or SELL
            price: Limit price
            size: Order size in USD
            order_type: GTC, GTD, FOK, or IOC
            strategy_id: Strategy identifier for idempotency
            client_order_id: Optional custom ID (auto-generated if None)
        
        Returns:
            Order object in CREATING state
        """
        # Generate or validate client order ID
        if client_order_id is None:
            client_order_id = generate_client_order_id(strategy_id, token_id)
        
        # Check for duplicate
        if client_order_id in self._orders:
            existing = self._orders[client_order_id]
            logger.warning(
                "Duplicate client_order_id",
                client_order_id=client_order_id,
                existing_state=existing.state.value,
            )
            return existing
        
        # Create order
        order = Order(
            client_order_id=client_order_id,
            token_id=token_id,
            side=side,
            price=price,
            size=size,
            order_type=order_type,
            strategy_id=strategy_id,
        )
        
        # Track order
        self._orders[client_order_id] = order

        if token_id not in self._orders_by_token:
            self._orders_by_token[token_id] = []
        self._orders_by_token[token_id].append(client_order_id)

        # FIX #8: Persist order state
        if self._persistence:
            self._persistence.save_order(order)

        logger.debug(
            "Order created",
            client_order_id=client_order_id,
            token_id=token_id,
            side=side.value,
            price=price,
            size=size,
        )

        return order
    
    async def submit_order(self, order: Order) -> bool:
        """
        Submit order to exchange.
        
        In dry_run mode, simulates submission without actual placement.
        
        Args:
            order: Order to submit
        
        Returns:
            True if submission successful, False otherwise
        """
        if order.state not in (OrderState.CREATING,):
            logger.warning(
                "Cannot submit order in current state",
                client_order_id=order.client_order_id,
                state=order.state.value,
            )
            return False
        
        # FIX #6: Enforce rate limit before submission
        if self._rate_limiter:
            wait_time = await self._rate_limiter.acquire_order()
            if wait_time > 0:
                logger.debug(
                    "Rate limited before submit",
                    client_order_id=order.client_order_id,
                    wait_seconds=wait_time,
                )
        
        # Update state
        order.update_state(OrderState.PENDING_NEW)
        order.submitted_at = time.time()
        self._pending_orders.add(order.client_order_id)

        # FIX #8: Persist updated state
        if self._persistence:
            self._persistence.save_order(order)

        if self.dry_run:
            # Simulate immediate acknowledgment
            logger.info(
                "DRY RUN: Order submitted",
                client_order_id=order.client_order_id,
                token_id=order.token_id,
                side=order.side.value,
                price=order.price,
                size=order.size,
            )
            order.update_state(OrderState.NEW)
            order.exchange_order_id = f"dry_run_{order.client_order_id}"
            self._pending_orders.discard(order.client_order_id)
            self._total_submitted += 1

            # FIX #8: Persist NEW state
            if self._persistence:
                self._persistence.save_order(order)

            return True
        
        # Real submission via callback
        if self._submit_callback:
            try:
                # The submit callback may either:
                # 1) Return a response-like object with .success/.exchange_order_id/.error_message
                # 2) Mutate the order state directly (legacy behavior)
                result = await self._submit_callback(order)

                self._pending_orders.discard(order.client_order_id)

                # Prefer interpreting a response object if returned
                if result is not None and hasattr(result, "success"):
                    if bool(getattr(result, "success")):
                        order.exchange_order_id = getattr(result, "exchange_order_id", None) or getattr(result, "orderID", None)
                        order.update_state(OrderState.NEW)
                    else:
                        order.update_state(OrderState.REJECTED)
                        order.error_message = str(getattr(result, "error_message", ""))

                # If callback didn't set a state (or returned None), default to NEW.
                if order.state == OrderState.PENDING_NEW:
                    order.update_state(OrderState.NEW)

                # Persist post-submit state (NEW/REJECTED/FAILED/etc.)
                if self._persistence:
                    self._persistence.save_order(order)

                if order.state.is_terminal and order.state not in (OrderState.FILLED, OrderState.CANCELED):
                    if order.state == OrderState.REJECTED:
                        self._total_rejected += 1
                    return False

                self._total_submitted += 1
                return True

            except Exception as e:
                order.update_state(OrderState.FAILED)
                order.error_message = str(e)
                self._pending_orders.discard(order.client_order_id)

                if self._persistence:
                    self._persistence.save_order(order)

                logger.error(
                    "Order submission failed",
                    client_order_id=order.client_order_id,
                    error=str(e),
                )
                return False
        
        logger.error("No submit callback configured for live trading")
        order.update_state(OrderState.FAILED)
        self._pending_orders.discard(order.client_order_id)
        return False
    
    async def cancel_order(self, order: Order) -> bool:
        """
        Request order cancellation.
        
        Args:
            order: Order to cancel
        
        Returns:
            True if cancel request successful
        """
        if not order.state.is_active:
            logger.warning(
                "Cannot cancel order in current state",
                client_order_id=order.client_order_id,
                state=order.state.value,
            )
            return False
        
        # FIX #6: Enforce rate limit before cancellation
        if self._rate_limiter:
            wait_time = await self._rate_limiter.acquire_order()
            if wait_time > 0:
                logger.debug(
                    "Rate limited before cancel",
                    client_order_id=order.client_order_id,
                    wait_seconds=wait_time,
                )
        
        order.update_state(OrderState.PENDING_CANCEL)
        
        if self.dry_run:
            logger.info(
                "DRY RUN: Order canceled",
                client_order_id=order.client_order_id,
            )
            order.update_state(OrderState.CANCELED)
            self._total_canceled += 1
            return True
        
        if self._cancel_callback:
            try:
                await self._cancel_callback(order)
                order.update_state(OrderState.CANCELED)
                self._total_canceled += 1
                
                # FIX #8: Persist updated state after cancel
                if self._persistence:
                    self._persistence.save_order(order)
                    
                return True
            except Exception as e:
                # Revert to previous active state
                order.update_state(OrderState.NEW)
                logger.error(
                    "Order cancel failed",
                    client_order_id=order.client_order_id,
                    error=str(e),
                )
                return False
        
        return False
    
    async def cancel_all_orders(self, token_id: str | None = None) -> int:
        """
        Cancel all active orders.
        
        Args:
            token_id: Optional filter by token
        
        Returns:
            Number of orders canceled
        """
        canceled = 0
        
        for order in self.get_active_orders(token_id):
            if await self.cancel_order(order):
                canceled += 1
        
        return canceled
    
    def get_order(self, client_order_id: str) -> Order | None:
        """Get order by client order ID."""
        return self._orders.get(client_order_id)
    
    def get_active_orders(self, token_id: str | None = None) -> list[Order]:
        """Get all active orders, optionally filtered by token."""
        active = []
        
        if token_id:
            order_ids = self._orders_by_token.get(token_id, [])
        else:
            order_ids = list(self._orders.keys())
        
        for order_id in order_ids:
            order = self._orders.get(order_id)
            if order and order.state.is_active:
                active.append(order)
        
        return active
    
    def get_pending_orders(self) -> list[Order]:
        """Get orders waiting for exchange acknowledgment."""
        return [
            self._orders[oid] 
            for oid in self._pending_orders 
            if oid in self._orders
        ]
    
    def handle_fill(
        self,
        client_order_id: str,
        fill_size: float,
        fill_price: float,
    ) -> bool:
        """
        Handle fill event from exchange.
        
        Args:
            client_order_id: Order that was filled
            fill_size: Size filled in this event
            fill_price: Price of this fill
        
        Returns:
            True if fill was applied
        """
        order = self._orders.get(client_order_id)
        if order is None:
            logger.warning("Fill for unknown order", client_order_id=client_order_id)
            return False
        
        order.add_fill(fill_size, fill_price)
        
        logger.info(
            "Order filled",
            client_order_id=client_order_id,
            fill_size=fill_size,
            fill_price=fill_price,
            filled_pct=f"{order.fill_pct:.0%}",
            state=order.state.value,
        )
        
        if order.state == OrderState.FILLED:
            self._total_filled += 1
        
        # FIX #8: Persist updated state after fill
        if self._persistence:
            self._persistence.save_order(order)
            
        return True
    
    def handle_rejection(
        self,
        client_order_id: str,
        reason: str,
    ) -> None:
        """Handle order rejection from exchange."""
        order = self._orders.get(client_order_id)
        if order is None:
            return
        
        order.update_state(OrderState.REJECTED)
        order.error_message = reason
        self._pending_orders.discard(client_order_id)
        self._total_rejected += 1
        
        # FIX #8: Persist updated state after rejection
        if self._persistence:
            self._persistence.save_order(order)
        
        logger.warning(
            "Order rejected",
            client_order_id=client_order_id,
            reason=reason,
        )
    
    async def timeout_stale_orders(
        self,
        max_pending_age_seconds: float = 30.0,
        max_active_age_seconds: float = 300.0,
    ) -> list[Order]:
        """
        Cancel orders that have been pending or active too long.

        Args:
            max_pending_age_seconds: Max time for pending acknowledgment
            max_active_age_seconds: Max time for unfilled active orders

        Returns:
            List of orders that were timed out
        """
        timed_out = []
        now = time.time()

        for order in list(self._orders.values()):
            should_timeout = False
            reason = ""

            # Check pending orders
            if order.state == OrderState.PENDING_NEW:
                if order.submitted_at and (now - order.submitted_at) > max_pending_age_seconds:
                    should_timeout = True
                    reason = f"Pending for {now - order.submitted_at:.1f}s"

            # Check active unfilled orders
            elif order.state.is_active and order.filled_size == 0:
                if (now - order.created_at) > max_active_age_seconds:
                    should_timeout = True
                    reason = f"Active unfilled for {now - order.created_at:.1f}s"

            if should_timeout:
                logger.warning(
                    "Timing out stale order",
                    client_order_id=order.client_order_id,
                    state=order.state.value,
                    reason=reason,
                )

                if order.state.is_active:
                    await self.cancel_order(order)
                else:
                    order.update_state(OrderState.EXPIRED)
                    order.error_message = f"Timed out: {reason}"
                    self._pending_orders.discard(order.client_order_id)

                timed_out.append(order)

        return timed_out

    async def amend_order(
        self,
        order: Order,
        new_price: float | None = None,
        new_size: float | None = None,
        amend_callback: OrderCallback | None = None,
    ) -> bool:
        """
        Amend an existing order (cancel-replace).

        This performs an atomic cancel-replace operation. If the exchange
        supports native amend, use that. Otherwise, cancel and recreate.

        Args:
            order: Order to amend
            new_price: New price (None = keep current)
            new_size: New size (None = keep current)
            amend_callback: Optional callback for native amend

        Returns:
            True if amend was successful
        """
        if not order.state.is_active:
            logger.warning(
                "Cannot amend order in current state",
                client_order_id=order.client_order_id,
                state=order.state.value,
            )
            return False

        # Use provided values or keep existing
        final_price = new_price if new_price is not None else order.price
        final_size = new_size if new_size is not None else order.size

        # Check if amend is actually needed
        if final_price == order.price and final_size == order.size:
            logger.debug("Amend requested but no changes", client_order_id=order.client_order_id)
            return True

        logger.info(
            "Amending order",
            client_order_id=order.client_order_id,
            old_price=order.price,
            new_price=final_price,
            old_size=order.size,
            new_size=final_size,
        )

        if self.dry_run:
            # Simulate amend
            order.price = final_price
            order.size = final_size
            order.updated_at = time.time()
            logger.info(
                "DRY RUN: Order amended",
                client_order_id=order.client_order_id,
            )
            return True

        # Try native amend if available
        if amend_callback:
            try:
                # Create amended order info
                amended = Order(
                    client_order_id=order.client_order_id,
                    token_id=order.token_id,
                    side=order.side,
                    price=final_price,
                    size=final_size,
                    order_type=order.order_type,
                    exchange_order_id=order.exchange_order_id,
                )
                await amend_callback(amended)

                # Update local order
                order.price = final_price
                order.size = final_size
                order.updated_at = time.time()
                return True

            except Exception as e:
                logger.warning(
                    "Native amend failed, falling back to cancel-replace",
                    error=str(e),
                )

        # Fallback: Cancel and replace
        if not await self.cancel_order(order):
            logger.error("Amend failed: could not cancel original order")
            return False

        # Create replacement order
        new_order = self.create_order(
            token_id=order.token_id,
            side=order.side,
            price=final_price,
            size=final_size,
            order_type=order.order_type,
            strategy_id=order.strategy_id,
        )

        return await self.submit_order(new_order)

    def get_orders_by_state(self, state: OrderState) -> list[Order]:
        """Get all orders in a specific state."""
        return [o for o in self._orders.values() if o.state == state]

    def get_fill_rate(self) -> float:
        """Get the fill rate (filled / submitted)."""
        if self._total_submitted == 0:
            return 0.0
        return self._total_filled / self._total_submitted

    @property
    def stats(self) -> dict[str, int | float]:
        """Get order statistics."""
        return {
            "total_orders": len(self._orders),
            "active_orders": len(self.get_active_orders()),
            "pending_orders": len(self._pending_orders),
            "submitted": self._total_submitted,
            "filled": self._total_filled,
            "canceled": self._total_canceled,
            "rejected": self._total_rejected,
            "fill_rate": self.get_fill_rate(),
        }
