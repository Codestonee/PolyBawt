"""Order state persistence for crash recovery."""

import json
import time
from pathlib import Path
from typing import Any
from dataclasses import asdict

from src.infrastructure.logging import get_logger
from src.execution.order_manager import Order, OrderState

logger = get_logger(__name__)


class OrderStateStore:
    """
    Persists order state to JSON for crash recovery.

    Ensures orders are not duplicated across restarts.
    """

    def __init__(self, file_path: Path | str):
        """
        Initialize order state store.

        Args:
            file_path: Path to JSON persistence file
        """
        self.file_path = Path(file_path)
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        """Create parent directory if it doesn't exist."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def save_order(self, order: Order) -> None:
        """
        Save or update an order in the persistence file.

        Uses atomic write (write to temp, then rename) to prevent corruption.

        Args:
            order: Order to persist
        """
        try:
            # Load existing orders
            orders = self.load_orders()

            # Update or add this order
            orders[order.client_order_id] = {
                "order": self._order_to_dict(order),
                "state": order.state.value,
                "last_updated": time.time(),
            }

            # Atomic write
            temp_path = self.file_path.with_suffix(".tmp")
            with open(temp_path, "w") as f:
                json.dump(orders, f, indent=2)

            # Atomic rename (overwrites existing file)
            temp_path.replace(self.file_path)

            logger.debug(
                "Order persisted",
                client_order_id=order.client_order_id,
                state=order.state.value,
            )

        except Exception as e:
            logger.error(
                "Failed to persist order",
                client_order_id=order.client_order_id,
                error=str(e),
            )

    def load_orders(self) -> dict[str, Any]:
        """
        Load all persisted orders.

        Returns:
            Dict mapping client_order_id -> order data
        """
        if not self.file_path.exists():
            return {}

        try:
            with open(self.file_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error("Corrupted persistence file", error=str(e))
            # Backup corrupted file
            backup_path = self.file_path.with_suffix(f".backup.{int(time.time())}")
            self.file_path.rename(backup_path)
            logger.warning("Backed up corrupted file", backup_path=str(backup_path))
            return {}
        except Exception as e:
            logger.error("Failed to load orders", error=str(e))
            return {}

    def get_pending_orders(self) -> list[dict[str, Any]]:
        """
        Get all orders that are not in terminal states.

        Returns:
            List of order data dicts for non-terminal orders
        """
        terminal_states = {
            OrderState.FILLED.value,
            OrderState.CANCELED.value,
            OrderState.REJECTED.value,
            OrderState.EXPIRED.value,
            OrderState.FAILED.value,
        }

        orders = self.load_orders()
        return [
            data
            for data in orders.values()
            if data.get("state") not in terminal_states
        ]

    def remove_order(self, client_order_id: str) -> None:
        """
        Remove an order from persistence (e.g., after confirmation).

        Args:
            client_order_id: ID of order to remove
        """
        try:
            orders = self.load_orders()
            if client_order_id in orders:
                del orders[client_order_id]

                # Atomic write
                temp_path = self.file_path.with_suffix(".tmp")
                with open(temp_path, "w") as f:
                    json.dump(orders, f, indent=2)
                temp_path.replace(self.file_path)

                logger.debug("Order removed from persistence", client_order_id=client_order_id)
        except Exception as e:
            logger.error("Failed to remove order", client_order_id=client_order_id, error=str(e))

    def _order_to_dict(self, order: Order) -> dict[str, Any]:
        """
        Convert Order to JSON-serializable dict.

        Args:
            order: Order to serialize

        Returns:
            Dict representation
        """
        data = asdict(order)
        # Convert enums to strings
        data["state"] = order.state.value
        data["side"] = order.side.value
        data["order_type"] = order.order_type.value
        return data

    def clear_all(self) -> None:
        """Clear all persisted orders (for testing)."""
        if self.file_path.exists():
            self.file_path.unlink()
            logger.info("Cleared all persisted orders")
