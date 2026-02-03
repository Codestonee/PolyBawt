from abc import ABC, abstractmethod
from typing import Any, List, Optional
from dataclasses import dataclass
import logging

from src.ingestion.market_discovery import Market
from src.execution.order_manager import OrderManager
from src.execution.clob_client import CLOBClient
from src.infrastructure.config import AppConfig

@dataclass
class TradeContext:
    """Context passed to strategies during scan."""
    market: Market
    spot_price: float
    order_book: Any  # OrderBook object for YES token
    order_book_no: Any = None  # OrderBook object for NO token (for arbitrage)
    open_exposure: float = 0.0
    daily_pnl: float = 0.0

class BaseStrategy(ABC):
    """
    Abstract base class for all modular strategies.
    
    Each strategy is responsible for:
    1. Scanning a market for specific entry conditions.
    2. Generating orders if conditions are met.
    3. Managing its own local state (if any).
    """

    def __init__(self, name: str, config: AppConfig):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"strategy.{name}")

    @abstractmethod
    async def scan(self, context: TradeContext) -> List[dict]:
        """
        Scan market for opportunities.
        
        Args:
            context: Current market context (prices, book, risk state)
            
        Returns:
            List of order parameters dicts to be executed by the orchestrator.
            Empty list if no opportunity.
            
            Example output:
            [
                {
                    "side": "BUY",
                    "token_id": "...",
                    "price": 0.55,
                    "size": 5.0,
                    "order_type": "GTC"
                }
            ]
        """
        pass

    @abstractmethod
    async def on_order_update(self, order_id: str, new_state: str):
        """Callback for order updates (filled, cancelled, etc)."""
        pass
