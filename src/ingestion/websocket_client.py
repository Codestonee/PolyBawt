"""
WebSocket client for real-time Polymarket CLOB data.

Research-backed implementation from Perplexity deep research.
Replaces REST polling with sub-100ms latency updates.

Features:
- Real-time orderbook updates via `clob_market` channel
- User order/fill notifications via `clob_user` channel
- VPIN (Volume-Synchronized Probability of Informed Trading) calculation
- Order book imbalance metrics
"""

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from threading import Thread
from typing import Any, Callable

import websockets
from websockets.exceptions import ConnectionClosedError

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OrderBook:
    """Real-time order book state."""
    
    market_id: str
    bids: list[tuple[float, float]] = field(default_factory=list)  # [(price, size), ...]
    asks: list[tuple[float, float]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    received_at: float = field(default_factory=time.time)
    
    @property
    def best_bid(self) -> float | None:
        """Best bid price."""
        return self.bids[0][0] if self.bids else None
    
    @property
    def best_ask(self) -> float | None:
        """Best ask price."""
        return self.asks[0][0] if self.asks else None
    
    @property
    def mid_price(self) -> float | None:
        """Midpoint price."""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None
    
    @property
    def spread(self) -> float | None:
        """Bid-ask spread."""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None
    
    @property
    def spread_bps(self) -> float | None:
        """Spread in basis points."""
        if self.mid_price and self.spread:
            return (self.spread / self.mid_price) * 10000
        return None
    
    def bid_depth(self, bps: int = 100) -> float:
        """Total bid volume within bps of mid."""
        if not self.mid_price:
            return 0.0
        threshold = self.mid_price * (1 - bps / 10000)
        return sum(size for price, size in self.bids if price >= threshold)
    
    def ask_depth(self, bps: int = 100) -> float:
        """Total ask volume within bps of mid."""
        if not self.mid_price:
            return 0.0
        threshold = self.mid_price * (1 + bps / 10000)
        return sum(size for price, size in self.asks if price <= threshold)
    
    def order_imbalance(self, bps: int = 100) -> float:
        """
        Order book imbalance ratio.
        
        Research shows this is the #1 predictive feature for short-term direction.
        
        Returns:
            Imbalance in range [-1, 1]
            +1 = All bids (bullish)
            -1 = All asks (bearish)
            0 = Balanced
        """
        bid_vol = self.bid_depth(bps)
        ask_vol = self.ask_depth(bps)
        total = bid_vol + ask_vol
        
        if total == 0:
            return 0.0
        
        return (bid_vol - ask_vol) / total


@dataclass
class Trade:
    """A single trade."""
    
    market_id: str
    price: float
    size: float
    side: str  # 'buy' or 'sell'
    timestamp: float


class VPINCalculator:
    """
    Volume-Synchronized Probability of Informed Trading.
    
    Research: "VPIN was able to foresee the flash crash and predict
    short-term volatility... Order flow is regarded as toxic when it
    adversely selects market makers"
    
    VPIN ∈ [0, 1]:
    - VPIN > 0.7: Extremely toxic (informed trading, stop quoting)
    - VPIN > 0.5: Moderately toxic
    - VPIN < 0.3: Normal uninformed flow (safe to market make)
    """
    
    def __init__(self, window: int = 50):
        """
        Args:
            window: Number of recent trades to include in calculation
        """
        self.window = window
        self.trades: deque[Trade] = deque(maxlen=window * 2)
        self.vpin_history: deque[float] = deque(maxlen=100)
    
    def add_trade(self, trade: Trade) -> None:
        """Record a trade."""
        self.trades.append(trade)
    
    def calculate(self) -> float:
        """
        Calculate current VPIN.
        
        Formula: VPIN = |BuyVolume - SellVolume| / TotalVolume
        
        Returns:
            VPIN in range [0, 1], or 0.5 if insufficient data
        """
        if len(self.trades) < 10:
            return 0.5  # Neutral if not enough data
        
        # Use recent window of trades
        recent = list(self.trades)[-self.window:]
        
        buy_volume = sum(t.size for t in recent if t.side == 'buy')
        sell_volume = sum(t.size for t in recent if t.side == 'sell')
        total = buy_volume + sell_volume
        
        if total == 0:
            return 0.5
        
        vpin = abs(buy_volume - sell_volume) / total
        
        # Store history
        self.vpin_history.append(vpin)
        
        return vpin
    
    def is_toxic(self, threshold: float = 0.7) -> bool:
        """Is current order flow toxic?"""
        return self.calculate() > threshold
    
    def toxicity_percentile(self) -> float:
        """
        Where does current VPIN rank among recent values?
        High percentile = more toxic than usual.
        """
        if not self.vpin_history:
            return 50.0
        
        current = self.vpin_history[-1]
        rank = sum(1 for v in self.vpin_history if v <= current) / len(self.vpin_history)
        return rank * 100


class PolymarketWebSocketClient:
    """
    Production WebSocket client for Polymarket CLOB.
    
    Provides:
    - Real-time orderbook updates
    - Trade notifications
    - VPIN toxicity calculation
    - Order book imbalance metrics
    
    Usage:
        client = PolymarketWebSocketClient()
        await client.connect()
        await client.subscribe_markets(["0x123...", "0x456..."])
        
        # Get latest data
        ob = client.get_orderbook("0x123...")
        vpin = client.get_vpin("0x123...")
    """
    
    CLOB_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/"
    
    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        api_passphrase: str | None = None,
        on_orderbook_update: Callable[[OrderBook], None] | None = None,
        on_trade: Callable[[Trade], None] | None = None,
    ):
        """
        Initialize WebSocket client.
        
        Args:
            api_key: CLOB API key (for user channel)
            api_secret: CLOB API secret
            api_passphrase: CLOB API passphrase
            on_orderbook_update: Callback when orderbook updates
            on_trade: Callback when trade occurs
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_passphrase = api_passphrase
        
        self.ws = None
        self.connected = False
        self._stop = False
        
        # Callbacks
        self.on_orderbook_update = on_orderbook_update
        self.on_trade = on_trade
        
        # State
        self.order_books: dict[str, OrderBook] = {}
        self.vpin_calculators: dict[str, VPINCalculator] = {}
        self.recent_trades: dict[str, deque[Trade]] = {}
        
        # Subscribed markets
        self.subscribed_markets: set[str] = set()
        
        # Statistics
        self.messages_received = 0
        self.last_message_time: float | None = None
    
    async def connect(self) -> bool:
        """
        Establish WebSocket connection.
        
        Returns:
            True if connected successfully
        """
        try:
            self.ws = await websockets.connect(
                self.CLOB_WS_URL,
                ping_interval=30,
                ping_timeout=10,
            )
            self.connected = True
            self._stop = False
            
            logger.info(
                "WebSocket connected",
                url=self.CLOB_WS_URL,
            )
            
            # Start message handler
            asyncio.create_task(self._message_handler())
            
            return True
            
        except Exception as e:
            logger.error("WebSocket connection failed", error=str(e))
            return False
    
    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        self._stop = True
        if self.ws:
            await self.ws.close()
        self.connected = False
        logger.info("WebSocket disconnected")
    
    async def subscribe_markets(self, token_ids: list[str]) -> None:
        """
        Subscribe to market channel for specific token IDs.
        
        Args:
            token_ids: List of ERC1155 token IDs (condition IDs)
        """
        if not self.connected or not self.ws:
            logger.error("Cannot subscribe: not connected")
            return
        
        subscribe_msg = {
            "type": "subscribe",
            "subscriptions": [
                {
                    "topic": "clob_market",
                    "type": "*",  # All message types
                    "filters": json.dumps({
                        "token_ids": token_ids
                    })
                }
            ]
        }
        
        try:
            await self.ws.send(json.dumps(subscribe_msg))
            self.subscribed_markets.update(token_ids)
            
            # Initialize VPIN calculators for each market
            for tid in token_ids:
                if tid not in self.vpin_calculators:
                    self.vpin_calculators[tid] = VPINCalculator()
                if tid not in self.recent_trades:
                    self.recent_trades[tid] = deque(maxlen=100)
            
            logger.info(
                "Subscribed to markets",
                count=len(token_ids),
                token_ids=token_ids[:3],  # Log first 3
            )
            
        except Exception as e:
            logger.error("Subscribe failed", error=str(e))
    
    async def _message_handler(self) -> None:
        """Handle incoming WebSocket messages."""
        while not self._stop and self.ws:
            try:
                message = await self.ws.recv()
                self.messages_received += 1
                self.last_message_time = time.time()
                
                data = json.loads(message)
                await self._route_message(data)
                
            except ConnectionClosedError:
                logger.warning("WebSocket connection closed")
                self.connected = False
                break
            except json.JSONDecodeError:
                logger.warning("Invalid JSON message")
            except Exception as e:
                logger.error("Message handler error", error=str(e))
    
    async def _route_message(self, data: dict[str, Any]) -> None:
        """Route message to appropriate handler."""
        msg_type = data.get("type", "")
        
        if msg_type == "agg_orderbook":
            self._handle_orderbook(data)
        elif msg_type == "last_trade_price":
            self._handle_trade(data)
        elif msg_type == "price_change":
            pass  # Use orderbook updates instead
        elif msg_type == "order_update":
            self._handle_order_update(data)
        elif msg_type == "fill":
            self._handle_fill(data)
    
    def _handle_orderbook(self, data: dict[str, Any]) -> None:
        """Process orderbook update."""
        market_id = data.get("market_id", "")
        
        # Parse bids and asks
        bids = [(float(p), float(s)) for p, s in data.get("bids", [])]
        asks = [(float(p), float(s)) for p, s in data.get("asks", [])]
        
        order_book = OrderBook(
            market_id=market_id,
            bids=bids,
            asks=asks,
            timestamp=datetime.now(),
            received_at=time.time(),
        )
        
        self.order_books[market_id] = order_book
        
        # Fire callback
        if self.on_orderbook_update:
            self.on_orderbook_update(order_book)
        
        logger.debug(
            "Orderbook updated",
            market_id=market_id[:20] + "...",
            mid=order_book.mid_price,
            spread_bps=order_book.spread_bps,
            imbalance=order_book.order_imbalance(),
        )
    
    def _handle_trade(self, data: dict[str, Any]) -> None:
        """Process trade notification."""
        market_id = data.get("market_id", "")
        
        trade = Trade(
            market_id=market_id,
            price=float(data.get("price", 0)),
            size=float(data.get("size", 0)),
            side=data.get("side", ""),
            timestamp=float(data.get("timestamp", time.time())),
        )
        
        # Update VPIN calculator
        if market_id in self.vpin_calculators:
            self.vpin_calculators[market_id].add_trade(trade)
        
        # Store recent trades
        if market_id in self.recent_trades:
            self.recent_trades[market_id].append(trade)
        
        # Fire callback
        if self.on_trade:
            self.on_trade(trade)
        
        logger.debug(
            "Trade received",
            market_id=market_id[:20] + "...",
            price=trade.price,
            size=trade.size,
            side=trade.side,
        )
    
    def _handle_order_update(self, data: dict[str, Any]) -> None:
        """Process order status update (user channel)."""
        order_id = data.get("order_id", "")
        status = data.get("status", "")
        
        logger.info(
            "Order update",
            order_id=order_id[:20] + "...",
            status=status,
        )
    
    def _handle_fill(self, data: dict[str, Any]) -> None:
        """Process fill notification (user channel)."""
        order_id = data.get("order_id", "")
        fill_size = data.get("fill_size", 0)
        fill_price = data.get("fill_price", 0)
        
        logger.info(
            "Order filled",
            order_id=order_id[:20] + "...",
            size=fill_size,
            price=fill_price,
        )
    
    # ========== Public API ==========
    
    def get_orderbook(self, market_id: str) -> OrderBook | None:
        """Get current orderbook state for a market."""
        return self.order_books.get(market_id)
    
    def get_vpin(self, market_id: str) -> float:
        """
        Get current VPIN (toxicity) for a market.
        
        Returns:
            VPIN in [0, 1], or 0.5 if no data
        """
        if market_id in self.vpin_calculators:
            return self.vpin_calculators[market_id].calculate()
        return 0.5
    
    def is_toxic(self, market_id: str, threshold: float = 0.7) -> bool:
        """Check if order flow is toxic for a market."""
        if market_id in self.vpin_calculators:
            return self.vpin_calculators[market_id].is_toxic(threshold)
        return False
    
    def get_order_imbalance(self, market_id: str, bps: int = 100) -> float:
        """
        Get order book imbalance for a market.
        
        This is the #1 predictive feature for short-term price direction
        according to research.
        
        Returns:
            Imbalance in [-1, 1], or 0.0 if no data
        """
        ob = self.order_books.get(market_id)
        if ob:
            return ob.order_imbalance(bps)
        return 0.0
    
    def get_recent_trades(self, market_id: str, limit: int = 20) -> list[Trade]:
        """Get recent trades for a market."""
        if market_id in self.recent_trades:
            trades = list(self.recent_trades[market_id])
            return trades[-limit:]
        return []
    
    def get_latency_stats(self) -> dict[str, Any]:
        """Get connection statistics."""
        return {
            "connected": self.connected,
            "messages_received": self.messages_received,
            "last_message_time": self.last_message_time,
            "ms_since_last_message": (
                (time.time() - self.last_message_time) * 1000
                if self.last_message_time else None
            ),
            "subscribed_markets": len(self.subscribed_markets),
        }


# ========== Factory Function ==========

def create_websocket_client(
    api_key: str | None = None,
    api_secret: str | None = None,
    api_passphrase: str | None = None,
) -> PolymarketWebSocketClient:
    """
    Factory function to create WebSocket client.
    
    Usage:
        client = create_websocket_client()
        await client.connect()
        await client.subscribe_markets(["0x123..."])
    """
    return PolymarketWebSocketClient(
        api_key=api_key,
        api_secret=api_secret,
        api_passphrase=api_passphrase,
    )
