"""
Reconciliation service for position and order synchronization.

Periodically syncs local state with exchange state to detect and fix:
- Orphaned orders (local thinks filled, exchange still open)
- Missing fills (exchange filled, local didn't record)
- Position drift (local position != exchange position)

Critical for production reliability.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any
import time

from src.infrastructure.logging import get_logger
from src.execution.order_manager import OrderManager, OrderState
from src.execution.clob_client import CLOBClient
from src.portfolio.tracker import Portfolio

logger = get_logger(__name__)


@dataclass
class ReconciliationResult:
    """Result of a reconciliation run."""
    
    timestamp: float = field(default_factory=time.time)
    
    # Order reconciliation
    orders_checked: int = 0
    orphaned_orders_found: int = 0
    orphaned_orders_cancelled: int = 0
    missing_fills_detected: int = 0
    missing_fills_applied: int = 0
    
    # Position reconciliation
    positions_checked: int = 0
    position_mismatches: int = 0
    position_corrections: int = 0
    
    # Errors
    errors: list[str] = field(default_factory=list)
    
    @property
    def has_discrepancies(self) -> bool:
        """Check if any discrepancies were found."""
        return (
            self.orphaned_orders_found > 0 or
            self.missing_fills_detected > 0 or
            self.position_mismatches > 0
        )
    
    @property
    def success(self) -> bool:
        """Check if reconciliation completed without errors."""
        return len(self.errors) == 0


class Reconciler:
    """
    Reconciles local state with exchange state.
    
    Runs periodically to ensure local order/position state matches
    what the exchange reports. This catches edge cases like:
    - Network failures during order placement/fill
    - Bot restarts mid-trade
    - Exchange-side order modifications
    
    Usage:
        reconciler = Reconciler(
            order_manager=order_manager,
            clob_client=clob_client,
            portfolio=portfolio,
        )
        
        # Run once
        result = await reconciler.run()
        
        # Or start periodic background task
        await reconciler.start_periodic(interval_seconds=60)
    """
    
    def __init__(
        self,
        order_manager: OrderManager,
        clob_client: CLOBClient,
        portfolio: Portfolio,
        auto_fix: bool = True,  # Automatically fix discrepancies
    ):
        self.order_manager = order_manager
        self.clob_client = clob_client
        self.portfolio = portfolio
        self.auto_fix = auto_fix
        
        self._running = False
        self._task: asyncio.Task | None = None
        self._last_result: ReconciliationResult | None = None
    
    async def run(self) -> ReconciliationResult:
        """
        Run a single reconciliation pass.
        
        Returns:
            ReconciliationResult with details of what was found/fixed.
        """
        result = ReconciliationResult()
        
        try:
            # 1. Reconcile orders
            await self._reconcile_orders(result)
            
            # 2. Reconcile positions
            await self._reconcile_positions(result)
            
        except Exception as e:
            result.errors.append(f"Reconciliation failed: {e}")
            logger.exception("Reconciliation error", error=str(e))
        
        self._last_result = result
        
        if result.has_discrepancies:
            logger.warning(
                "Reconciliation found discrepancies",
                orphaned_orders=result.orphaned_orders_found,
                missing_fills=result.missing_fills_detected,
                position_mismatches=result.position_mismatches,
            )
        else:
            logger.debug("Reconciliation complete - no discrepancies")
        
        return result
    
    async def _reconcile_orders(self, result: ReconciliationResult) -> None:
        """Reconcile local order state with exchange."""
        # Get active orders from local state
        local_orders = self.order_manager.get_active_orders()
        result.orders_checked = len(local_orders)
        
        if not local_orders:
            return
        
        try:
            # Get open orders from exchange
            exchange_orders = await self.clob_client.get_open_orders()
            exchange_order_ids = {o.get("id") for o in exchange_orders}
        except Exception as e:
            result.errors.append(f"Failed to fetch exchange orders: {e}")
            return
        
        for order in local_orders:
            if not order.exchange_order_id:
                continue
            
            # Check if order still exists on exchange
            if order.exchange_order_id not in exchange_order_ids:
                result.orphaned_orders_found += 1
                logger.warning(
                    "Orphaned order detected - exists locally but not on exchange",
                    client_order_id=order.client_order_id,
                    exchange_order_id=order.exchange_order_id,
                )
                
                if self.auto_fix:
                    # Mark as filled or cancelled based on context
                    # Check if there were partial fills
                    if order.filled_size > 0:
                        # Likely filled completely
                        order.update_state(OrderState.FILLED)
                        result.missing_fills_applied += 1
                        logger.info(
                            "Auto-fixed: marked orphaned order as filled",
                            client_order_id=order.client_order_id,
                        )
                    else:
                        # Likely cancelled externally
                        order.update_state(OrderState.CANCELED)
                        result.orphaned_orders_cancelled += 1
                        logger.info(
                            "Auto-fixed: marked orphaned order as cancelled",
                            client_order_id=order.client_order_id,
                        )
    
    async def _reconcile_positions(self, result: ReconciliationResult) -> None:
        """Reconcile local positions with exchange."""
        try:
            # Get positions from exchange
            exchange_positions = await self.clob_client.get_positions()
        except Exception as e:
            result.errors.append(f"Failed to fetch exchange positions: {e}")
            return
        
        result.positions_checked = len(exchange_positions)
        
        for position in exchange_positions:
            token_id = position.get("token_id", "")
            exchange_size = float(position.get("size", 0))
            
            # Get local position for this token
            local_position = self.portfolio.get_position(token_id)
            local_size = local_position.size if local_position else 0.0
            
            # Check for mismatch (with small tolerance for rounding)
            if abs(exchange_size - local_size) > 0.01:
                result.position_mismatches += 1
                logger.warning(
                    "Position mismatch detected",
                    token_id=token_id[:16] + "...",
                    local_size=local_size,
                    exchange_size=exchange_size,
                    delta=exchange_size - local_size,
                )
                
                if self.auto_fix:
                    # Update local position to match exchange
                    self.portfolio.sync_position(token_id, exchange_size)
                    result.position_corrections += 1
                    logger.info(
                        "Auto-fixed: synced local position to exchange",
                        token_id=token_id[:16] + "...",
                        new_size=exchange_size,
                    )
    
    async def start_periodic(self, interval_seconds: float = 60.0) -> None:
        """
        Start periodic reconciliation in background.
        
        Args:
            interval_seconds: Time between reconciliation runs.
        """
        if self._running:
            logger.warning("Reconciler already running")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._run_loop(interval_seconds))
        logger.info(
            "Started periodic reconciliation",
            interval_seconds=interval_seconds,
        )
    
    async def stop(self) -> None:
        """Stop periodic reconciliation."""
        self._running = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        
        logger.info("Stopped periodic reconciliation")
    
    async def _run_loop(self, interval: float) -> None:
        """Background loop for periodic reconciliation."""
        while self._running:
            try:
                await self.run()
            except Exception as e:
                logger.error("Periodic reconciliation error", error=str(e))
            
            await asyncio.sleep(interval)
    
    @property
    def last_result(self) -> ReconciliationResult | None:
        """Get result of last reconciliation run."""
        return self._last_result
    
    @property
    def is_running(self) -> bool:
        """Check if periodic reconciliation is active."""
        return self._running
