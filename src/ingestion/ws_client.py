"""
WebSocket client for Polymarket CLOB with automatic reconnection.

Features:
- Exponential backoff with jitter
- Automatic reconnection
- Message handlers for order book updates
- Health monitoring
"""

import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class ConnectionState(Enum):
    """WebSocket connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    CLOSED = "closed"


@dataclass
class ReconnectPolicy:
    """Exponential backoff configuration."""
    initial_delay_ms: int = 100
    max_delay_ms: int = 5000
    multiplier: float = 2.0
    jitter_pct: float = 0.20
    max_attempts: int = 10
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay in seconds for given attempt number."""
        base_delay = self.initial_delay_ms * (self.multiplier ** attempt)
        capped_delay = min(base_delay, self.max_delay_ms)
        jitter = capped_delay * self.jitter_pct * (random.random() * 2 - 1)
        return (capped_delay + jitter) / 1000.0  # Convert to seconds


@dataclass
class WSMessage:
    """Parsed WebSocket message."""
    event_type: str
    asset_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


MessageHandler = Callable[[WSMessage], Coroutine[Any, Any, None]]


class WebSocketClient:
    """
    Polymarket CLOB WebSocket client with automatic reconnection.
    
    Usage:
        client = WebSocketClient(url="wss://ws-subscriptions-clob.polymarket.com/ws/market")
        client.on_book_update(handle_book_update)
        await client.connect()
        await client.subscribe(["token_id_1", "token_id_2"])
    """
    
    def __init__(
        self,
        url: str,
        reconnect_policy: ReconnectPolicy | None = None,
    ):
        self.url = url
        self.reconnect_policy = reconnect_policy or ReconnectPolicy()
        
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._state = ConnectionState.DISCONNECTED
        self._subscribed_tokens: set[str] = set()
        self._handlers: dict[str, list[MessageHandler]] = {}
        self._reconnect_attempt = 0
        self._running = False
        self._last_message_time: float = 0
        self._receive_task: asyncio.Task | None = None
    
    @property
    def state(self) -> ConnectionState:
        return self._state
    
    @property
    def is_connected(self) -> bool:
        return self._state == ConnectionState.CONNECTED
    
    @property
    def last_message_age_seconds(self) -> float:
        """Time since last message received."""
        if self._last_message_time == 0:
            return float("inf")
        return time.time() - self._last_message_time
    
    def on(self, event_type: str, handler: MessageHandler) -> None:
        """Register handler for event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
    
    def on_book_update(self, handler: MessageHandler) -> None:
        """Register handler for order book updates."""
        self.on("book", handler)
    
    def on_price_change(self, handler: MessageHandler) -> None:
        """Register handler for price changes."""
        self.on("price_change", handler)
    
    def on_trade(self, handler: MessageHandler) -> None:
        """Register handler for trade events."""
        self.on("last_trade_price", handler)
    
    async def connect(self) -> None:
        """
        Establish WebSocket connection.
        
        Raises:
            WebSocketException: If connection fails after max retries
        """
        if self._running:
            logger.warning("WebSocket already running")
            return
        
        self._running = True
        await self._connect_with_retry()
    
    async def _connect_with_retry(self) -> None:
        """Connect with exponential backoff retry."""
        while self._running and self._reconnect_attempt < self.reconnect_policy.max_attempts:
            try:
                self._state = ConnectionState.CONNECTING
                logger.info(
                    "Connecting to WebSocket",
                    url=self.url,
                    attempt=self._reconnect_attempt + 1
                )
                
                self._ws = await websockets.connect(
                    self.url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                )
                
                self._state = ConnectionState.CONNECTED
                self._reconnect_attempt = 0
                self._last_message_time = time.time()
                
                logger.info("WebSocket connected", url=self.url)
                
                # Resubscribe to tokens
                if self._subscribed_tokens:
                    await self._send_subscribe(list(self._subscribed_tokens))
                
                # Start receive loop
                self._receive_task = asyncio.create_task(self._receive_loop())
                return
                
            except Exception as e:
                self._reconnect_attempt += 1
                delay = self.reconnect_policy.get_delay(self._reconnect_attempt)
                
                logger.warning(
                    "WebSocket connection failed",
                    error=str(e),
                    attempt=self._reconnect_attempt,
                    retry_delay_s=delay
                )
                
                if self._reconnect_attempt >= self.reconnect_policy.max_attempts:
                    self._state = ConnectionState.CLOSED
                    raise WebSocketException(
                        f"Failed to connect after {self._reconnect_attempt} attempts"
                    )
                
                await asyncio.sleep(delay)
    
    async def _receive_loop(self) -> None:
        """Main loop for receiving and processing messages."""
        try:
            async for message in self._ws:
                self._last_message_time = time.time()
                await self._handle_message(message)
                
        except ConnectionClosed as e:
            logger.warning("WebSocket connection closed", code=e.code, reason=e.reason)
            await self._handle_disconnect()
            
        except Exception as e:
            logger.error("WebSocket receive error", error=str(e))
            await self._handle_disconnect()
    
    async def _handle_message(self, raw_message: str) -> None:
        """Parse and dispatch message to handlers."""
        try:
            data = json.loads(raw_message)
            
            # Parse message format
            event_type = data.get("event_type") or data.get("type", "unknown")
            asset_id = data.get("asset_id") or data.get("token_id")
            
            msg = WSMessage(
                event_type=event_type,
                asset_id=asset_id,
                data=data,
                timestamp=time.time()
            )
            
            # Dispatch to handlers
            handlers = self._handlers.get(event_type, [])
            for handler in handlers:
                try:
                    await handler(msg)
                except Exception as e:
                    logger.error(
                        "Message handler error",
                        event_type=event_type,
                        error=str(e)
                    )
            
            # Also dispatch to wildcard handlers
            for handler in self._handlers.get("*", []):
                try:
                    await handler(msg)
                except Exception as e:
                    logger.error("Wildcard handler error", error=str(e))
                    
        except json.JSONDecodeError as e:
            logger.warning("Invalid JSON message", error=str(e))
    
    async def _handle_disconnect(self) -> None:
        """Handle disconnection and attempt reconnection."""
        if not self._running:
            return
        
        self._state = ConnectionState.RECONNECTING
        self._ws = None
        
        logger.info("Attempting reconnection")
        await self._connect_with_retry()
    
    async def subscribe(self, token_ids: list[str]) -> None:
        """
        Subscribe to market updates for given token IDs.
        
        Args:
            token_ids: List of token IDs to subscribe to
        """
        self._subscribed_tokens.update(token_ids)
        
        if self.is_connected:
            await self._send_subscribe(token_ids)
    
    async def unsubscribe(self, token_ids: list[str]) -> None:
        """Unsubscribe from token updates."""
        for token_id in token_ids:
            self._subscribed_tokens.discard(token_id)
        
        if self.is_connected:
            await self._send_unsubscribe(token_ids)
    
    async def _send_subscribe(self, token_ids: list[str]) -> None:
        """Send subscription message."""
        if not self._ws:
            return
        
        message = {
            "type": "market",
            "assets_ids": token_ids
        }
        await self._ws.send(json.dumps(message))
        logger.debug("Subscribed to tokens", count=len(token_ids))
    
    async def _send_unsubscribe(self, token_ids: list[str]) -> None:
        """Send unsubscription message."""
        if not self._ws:
            return
        
        message = {
            "type": "unsubscribe",
            "assets_ids": token_ids
        }
        await self._ws.send(json.dumps(message))
        logger.debug("Unsubscribed from tokens", count=len(token_ids))
    
    async def close(self) -> None:
        """Close the WebSocket connection."""
        self._running = False
        self._state = ConnectionState.CLOSED
        
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        
        if self._ws:
            await self._ws.close()
            self._ws = None
        
        logger.info("WebSocket closed")
    
    async def health_check(self) -> bool:
        """
        Check if connection is healthy.
        
        Returns:
            True if connected and receiving messages
        """
        if not self.is_connected:
            return False
        
        # Consider unhealthy if no message in 60 seconds
        return self.last_message_age_seconds < 60


async def create_market_ws_client(
    url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market",
    reconnect_policy: ReconnectPolicy | None = None,
) -> WebSocketClient:
    """
    Factory function to create a configured WebSocket client.
    
    Args:
        url: WebSocket URL
        reconnect_policy: Custom reconnection policy
    
    Returns:
        Configured WebSocket client (not yet connected)
    """
    return WebSocketClient(url=url, reconnect_policy=reconnect_policy)
