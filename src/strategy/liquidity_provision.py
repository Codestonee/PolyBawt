"""
Liquidity Provision (Market Making) Module for Polymarket.

Implements market making strategies for prediction markets:
- Two-sided quoting
- Inventory management
- Spread optimization
- Risk limits

New Polymarket markets have low liquidity and pay 80-200% APY equivalent.

Based on research:
- ChainCatcher 2025: LP strategy returns declining but still viable
- Classical market making literature (Avellaneda-Stoikov)

WARNING: Market making carries significant risks including:
- Adverse selection (trading against informed traders)
- Inventory risk (getting stuck with large positions)
- Execution risk (orders may not fill symmetrically)
"""

import asyncio
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine

from src.infrastructure.logging import get_logger
from src.models.order_book import OrderBook
from src.execution.order_manager import OrderManager, Order, OrderSide, OrderType

logger = get_logger(__name__)


class LPState(Enum):
    """Liquidity provider state."""
    IDLE = "idle"
    QUOTING = "quoting"
    REBALANCING = "rebalancing"
    PAUSED = "paused"
    STOPPED = "stopped"


@dataclass
class InventoryPosition:
    """Current inventory position for a market."""
    token_id: str
    yes_shares: float = 0.0
    no_shares: float = 0.0
    avg_yes_cost: float = 0.5
    avg_no_cost: float = 0.5
    realized_pnl: float = 0.0

    @property
    def net_position(self) -> float:
        """Net position (positive = long YES, negative = long NO)."""
        return self.yes_shares - self.no_shares

    @property
    def total_exposure(self) -> float:
        """Total absolute exposure."""
        return self.yes_shares + self.no_shares

    @property
    def inventory_skew(self) -> float:
        """Inventory skew from -1 (all NO) to +1 (all YES)."""
        total = self.total_exposure
        if total == 0:
            return 0.0
        return (self.yes_shares - self.no_shares) / total


@dataclass
class QuoteParams:
    """Parameters for a two-sided quote."""

    # Prices
    bid_price: float  # Price to buy YES
    ask_price: float  # Price to sell YES (or buy NO)
    spread: float     # ask - bid

    # Sizes
    bid_size: float
    ask_size: float

    # Derived
    mid_price: float = 0.5

    def __post_init__(self):
        self.mid_price = (self.bid_price + self.ask_price) / 2


@dataclass
class LPConfig:
    """Configuration for liquidity provision."""

    # Spread parameters
    base_spread: float = 0.04          # 4% base spread
    min_spread: float = 0.02           # 2% minimum spread
    max_spread: float = 0.15           # 15% maximum spread
    spread_volatility_mult: float = 2.0  # Spread multiplier per unit vol

    # Inventory limits
    max_inventory: float = 100.0       # Maximum position in USD
    target_inventory: float = 0.0      # Target net position
    inventory_skew_mult: float = 0.02  # Price skew per unit inventory

    # Size parameters
    quote_size_usd: float = 10.0       # Size per side
    min_quote_size_usd: float = 1.0    # Minimum size to quote

    # Risk limits
    max_loss_per_market: float = 20.0  # Stop LP if loss exceeds this
    max_spread_to_quote: float = 0.20  # Don't quote if market spread > 20%

    # Timing
    requote_interval_seconds: float = 5.0
    min_time_to_expiry_seconds: float = 120.0


@dataclass
class LPMetrics:
    """Metrics for liquidity provision performance."""

    # Volume
    total_volume_usd: float = 0.0
    buy_volume_usd: float = 0.0
    sell_volume_usd: float = 0.0

    # PnL
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    fees_earned: float = 0.0

    # Efficiency
    quotes_placed: int = 0
    quotes_filled: int = 0
    average_spread_captured: float = 0.0

    @property
    def fill_rate(self) -> float:
        if self.quotes_placed == 0:
            return 0.0
        return self.quotes_filled / self.quotes_placed

    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl


class SpreadCalculator:
    """
    Calculates optimal spread for market making.

    Uses Avellaneda-Stoikov style approach:
    - Base spread covers adverse selection
    - Volatility adjustment for risk
    - Inventory skew to manage position
    """

    def __init__(self, config: LPConfig):
        self.config = config

    def calculate_spread(
        self,
        fair_value: float,
        volatility: float,
        inventory_skew: float,
        time_to_expiry_seconds: float,
    ) -> QuoteParams:
        """
        Calculate optimal bid/ask prices.

        Args:
            fair_value: Estimated fair probability
            volatility: Current annualized volatility
            inventory_skew: Current inventory skew (-1 to 1)
            time_to_expiry_seconds: Time remaining

        Returns:
            QuoteParams with bid/ask prices and sizes
        """
        cfg = self.config

        # Base spread
        spread = cfg.base_spread

        # Volatility adjustment (higher vol = wider spread)
        vol_adj = volatility * cfg.spread_volatility_mult * 0.1
        spread += vol_adj

        # Time decay adjustment (tighter near expiry for more fills)
        if time_to_expiry_seconds < 300:  # Last 5 minutes
            time_mult = max(0.5, time_to_expiry_seconds / 300)
            spread *= time_mult

        # Clamp spread
        spread = max(cfg.min_spread, min(cfg.max_spread, spread))

        # Calculate mid with inventory skew
        # If we're long YES, shade bid down (less willing to buy more)
        # If we're long NO, shade ask up (less willing to sell YES)
        skew_adjustment = inventory_skew * cfg.inventory_skew_mult

        # Bid and ask
        half_spread = spread / 2
        bid = fair_value - half_spread - skew_adjustment
        ask = fair_value + half_spread - skew_adjustment

        # Clamp to valid range
        bid = max(0.01, min(0.98, bid))
        ask = max(0.02, min(0.99, ask))

        # Ensure bid < ask
        if bid >= ask:
            mid = (bid + ask) / 2
            bid = mid - 0.01
            ask = mid + 0.01

        # Calculate sizes with inventory adjustment
        bid_size = cfg.quote_size_usd
        ask_size = cfg.quote_size_usd

        # Reduce size on the side where we're already exposed
        if inventory_skew > 0.3:
            # Long YES, reduce bid size
            bid_size *= (1 - inventory_skew)
        elif inventory_skew < -0.3:
            # Long NO, reduce ask size
            ask_size *= (1 + inventory_skew)

        bid_size = max(cfg.min_quote_size_usd, bid_size)
        ask_size = max(cfg.min_quote_size_usd, ask_size)

        return QuoteParams(
            bid_price=round(bid, 4),
            ask_price=round(ask, 4),
            spread=round(ask - bid, 4),
            bid_size=round(bid_size, 2),
            ask_size=round(ask_size, 2),
        )


class LiquidityProvider:
    """
    Market maker for Polymarket prediction markets.

    Provides two-sided liquidity to earn the bid-ask spread.

    Strategy:
    1. Calculate fair value using pricing model
    2. Determine optimal spread based on volatility and inventory
    3. Place bid and ask orders
    4. Manage fills and rebalance inventory
    5. Monitor risk limits

    Usage:
        lp = LiquidityProvider(
            config=LPConfig(),
            order_manager=order_manager,
            fair_value_fn=lambda: model.prob_up(...),
        )

        await lp.start_quoting(token_id="abc123")
    """

    def __init__(
        self,
        config: LPConfig,
        order_manager: OrderManager,
        fair_value_fn: Callable[[], float] | None = None,
        volatility_fn: Callable[[], float] | None = None,
    ):
        self.config = config
        self.order_manager = order_manager
        self.fair_value_fn = fair_value_fn or (lambda: 0.5)
        self.volatility_fn = volatility_fn or (lambda: 0.6)

        self.spread_calculator = SpreadCalculator(config)

        # State
        self.state = LPState.IDLE
        self._positions: dict[str, InventoryPosition] = {}
        self._active_quotes: dict[str, tuple[Order | None, Order | None]] = {}  # bid, ask
        self._metrics = LPMetrics()

        # Control
        self._running = False
        self._markets: set[str] = set()

    def get_position(self, token_id: str) -> InventoryPosition:
        """Get or create inventory position for a market."""
        if token_id not in self._positions:
            self._positions[token_id] = InventoryPosition(token_id=token_id)
        return self._positions[token_id]

    def update_position(
        self,
        token_id: str,
        side: str,  # "YES" or "NO"
        quantity: float,
        price: float,
        is_buy: bool,
    ) -> None:
        """
        Update inventory position after a fill.

        Args:
            token_id: Market token
            side: "YES" or "NO"
            quantity: Number of shares
            price: Fill price
            is_buy: True if we bought, False if we sold
        """
        pos = self.get_position(token_id)

        if side == "YES":
            if is_buy:
                # Bought YES
                old_value = pos.yes_shares * pos.avg_yes_cost
                new_value = quantity * price
                pos.yes_shares += quantity
                if pos.yes_shares > 0:
                    pos.avg_yes_cost = (old_value + new_value) / pos.yes_shares
            else:
                # Sold YES
                pnl = quantity * (price - pos.avg_yes_cost)
                pos.realized_pnl += pnl
                pos.yes_shares -= quantity
        else:
            if is_buy:
                # Bought NO
                old_value = pos.no_shares * pos.avg_no_cost
                new_value = quantity * price
                pos.no_shares += quantity
                if pos.no_shares > 0:
                    pos.avg_no_cost = (old_value + new_value) / pos.no_shares
            else:
                # Sold NO
                pnl = quantity * (price - pos.avg_no_cost)
                pos.realized_pnl += pnl
                pos.no_shares -= quantity

        # Update metrics
        volume = quantity * price
        if is_buy:
            self._metrics.buy_volume_usd += volume
        else:
            self._metrics.sell_volume_usd += volume
        self._metrics.total_volume_usd += volume
        self._metrics.realized_pnl = sum(p.realized_pnl for p in self._positions.values())

    async def calculate_quotes(
        self,
        token_id: str,
        time_to_expiry_seconds: float,
    ) -> QuoteParams | None:
        """
        Calculate optimal quotes for a market.

        Args:
            token_id: Market token
            time_to_expiry_seconds: Time until market resolves

        Returns:
            QuoteParams or None if shouldn't quote
        """
        # Check time to expiry
        if time_to_expiry_seconds < self.config.min_time_to_expiry_seconds:
            logger.debug("Too close to expiry for LP", token_id=token_id)
            return None

        # Get current position
        pos = self.get_position(token_id)

        # Check inventory limits
        if pos.total_exposure >= self.config.max_inventory:
            logger.info("Inventory limit reached", token_id=token_id, exposure=pos.total_exposure)
            return None

        # Check loss limit
        if pos.realized_pnl < -self.config.max_loss_per_market:
            logger.warning("Loss limit reached, stopping LP", token_id=token_id, pnl=pos.realized_pnl)
            return None

        # Get fair value and volatility
        fair_value = self.fair_value_fn()
        volatility = self.volatility_fn()

        # Calculate quotes
        quotes = self.spread_calculator.calculate_spread(
            fair_value=fair_value,
            volatility=volatility,
            inventory_skew=pos.inventory_skew,
            time_to_expiry_seconds=time_to_expiry_seconds,
        )

        return quotes

    async def place_quotes(
        self,
        token_id: str,
        quotes: QuoteParams,
    ) -> tuple[Order | None, Order | None]:
        """
        Place bid and ask orders.

        Args:
            token_id: Market token
            quotes: Quote parameters

        Returns:
            Tuple of (bid_order, ask_order)
        """
        # Cancel existing quotes first
        await self.cancel_quotes(token_id)

        bid_order = None
        ask_order = None

        # Place bid (buy YES)
        if quotes.bid_size >= self.config.min_quote_size_usd:
            bid_order = self.order_manager.create_order(
                token_id=token_id,
                side=OrderSide.BUY,
                price=quotes.bid_price,
                size=quotes.bid_size,
                order_type=OrderType.GTC,
                strategy_id="liquidity_provision",
            )
            await self.order_manager.submit_order(bid_order)
            self._metrics.quotes_placed += 1

        # Place ask (sell YES / buy NO)
        if quotes.ask_size >= self.config.min_quote_size_usd:
            ask_order = self.order_manager.create_order(
                token_id=token_id,
                side=OrderSide.SELL,
                price=quotes.ask_price,
                size=quotes.ask_size,
                order_type=OrderType.GTC,
                strategy_id="liquidity_provision",
            )
            await self.order_manager.submit_order(ask_order)
            self._metrics.quotes_placed += 1

        self._active_quotes[token_id] = (bid_order, ask_order)

        logger.debug(
            "Quotes placed",
            token_id=token_id,
            bid=f"{quotes.bid_price:.3f}x{quotes.bid_size:.1f}",
            ask=f"{quotes.ask_price:.3f}x{quotes.ask_size:.1f}",
            spread=f"{quotes.spread:.3f}",
        )

        return bid_order, ask_order

    async def cancel_quotes(self, token_id: str) -> None:
        """Cancel existing quotes for a market."""
        if token_id not in self._active_quotes:
            return

        bid_order, ask_order = self._active_quotes[token_id]

        if bid_order and bid_order.state.is_active:
            await self.order_manager.cancel_order(bid_order)

        if ask_order and ask_order.state.is_active:
            await self.order_manager.cancel_order(ask_order)

        del self._active_quotes[token_id]

    async def cancel_all_quotes(self) -> None:
        """Cancel all active quotes."""
        for token_id in list(self._active_quotes.keys()):
            await self.cancel_quotes(token_id)

    async def run_iteration(
        self,
        token_id: str,
        time_to_expiry_seconds: float,
    ) -> None:
        """
        Run one iteration of LP for a market.

        Args:
            token_id: Market token
            time_to_expiry_seconds: Time until market resolves
        """
        if self.state != LPState.QUOTING:
            return

        # Calculate new quotes
        quotes = await self.calculate_quotes(token_id, time_to_expiry_seconds)

        if quotes is None:
            await self.cancel_quotes(token_id)
            return

        # Check if quotes need updating
        current = self._active_quotes.get(token_id, (None, None))
        bid_order, ask_order = current

        needs_update = False

        if bid_order is None or ask_order is None:
            needs_update = True
        elif not bid_order.state.is_active or not ask_order.state.is_active:
            needs_update = True
        elif abs(bid_order.price - quotes.bid_price) > 0.005:
            needs_update = True
        elif abs(ask_order.price - quotes.ask_price) > 0.005:
            needs_update = True

        if needs_update:
            await self.place_quotes(token_id, quotes)

    async def start(self, markets: list[str]) -> None:
        """
        Start liquidity provision.

        Args:
            markets: List of token IDs to provide liquidity for
        """
        self._markets = set(markets)
        self._running = True
        self.state = LPState.QUOTING

        logger.info("Liquidity provision started", markets=len(markets))

    async def stop(self) -> None:
        """Stop liquidity provision and cancel all quotes."""
        self._running = False
        self.state = LPState.STOPPED

        await self.cancel_all_quotes()

        logger.info(
            "Liquidity provision stopped",
            total_volume=self._metrics.total_volume_usd,
            realized_pnl=self._metrics.realized_pnl,
            fill_rate=self._metrics.fill_rate,
        )

    def pause(self) -> None:
        """Pause quoting (keeps existing quotes)."""
        self.state = LPState.PAUSED

    def resume(self) -> None:
        """Resume quoting."""
        if self._running:
            self.state = LPState.QUOTING

    @property
    def metrics(self) -> LPMetrics:
        """Get current LP metrics."""
        return self._metrics

    @property
    def is_running(self) -> bool:
        return self._running and self.state == LPState.QUOTING


# Pre-instantiated with defaults
default_lp_config = LPConfig()
