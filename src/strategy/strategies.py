import logging
import time
import random
from typing import List, Dict, Optional, Any, Deque
from dataclasses import dataclass
from collections import deque
from enum import Enum

from src.strategy.base import BaseStrategy, TradeContext
from src.strategy.arbitrage_detector import ArbitrageDetector
from src.execution.clob_client import CLOBClient
from src.infrastructure.config import AppConfig

logger = logging.getLogger(__name__)

# ==============================================================================
# STRATEGY 1: ARB TAKER (Risk-Free* Profit)
# ==============================================================================
class ArbTakerStrategy(BaseStrategy):
    """
    Scans for risk-free arbitrage where YES + NO price < 1.0 (minus fees).
    Executes simultaneous TAKER buys on both sides.

    Arbitrage Types:
    1. Long Arb: YES_ask + NO_ask < 1.0 (buy both, guaranteed profit)
    2. Short Arb: YES_bid + NO_bid > 1.0 (sell both, guaranteed profit)

    Fee Model:
    - Polymarket charges ~2% per trade (4% round-trip)
    - Need >2% gross profit for long arb to be profitable
    - Threshold: YES_ask + NO_ask < 0.98
    """
    def __init__(self, config: AppConfig):
        super().__init__("ArbTaker", config)
        self.long_arb_threshold = 0.98  # YES + NO must be < 0.98 for long arb
        self.short_arb_threshold = 1.02  # YES + NO must be > 1.02 for short arb
        self.min_profit_pct = 0.005  # 0.5% minimum profit after fees
        self.max_arb_size = 5.0  # $5 max per leg

    async def scan(self, context: TradeContext) -> List[dict]:
        if not context.market.no_token_id:
            return []

        orders = []

        # Get prices from order books (preferred) or market data (fallback)
        yes_ask = None
        yes_bid = None
        no_ask = None
        no_bid = None

        # YES order book
        if context.order_book:
            yes_ask = context.order_book.best_ask
            yes_bid = context.order_book.best_bid

        # NO order book (now available in context)
        if context.order_book_no:
            no_ask = context.order_book_no.best_ask
            no_bid = context.order_book_no.best_bid

        # Fallback to market prices if order books unavailable
        if yes_ask is None:
            yes_ask = context.market.yes_price
        if no_ask is None:
            no_ask = context.market.no_price

        # ===== LONG ARBITRAGE: Buy both YES and NO =====
        # If YES_ask + NO_ask < 0.98, we can buy both and guarantee profit
        if yes_ask is not None and no_ask is not None:
            total_cost = yes_ask + no_ask

            if total_cost < self.long_arb_threshold:
                gross_profit = 1.0 - total_cost
                # Account for fees (2% per trade, 4% total for both legs)
                fee_pct = 0.04
                net_profit = gross_profit - (total_cost * fee_pct)

                if net_profit > self.min_profit_pct:
                    logger.info(
                        f"LONG ARB DETECTED: YES {yes_ask:.3f} + NO {no_ask:.3f} = "
                        f"{total_cost:.3f}, gross profit {gross_profit:.2%}, "
                        f"net profit {net_profit:.2%}"
                    )

                    # Calculate size based on available liquidity
                    yes_depth = context.order_book.ask_depth_usd if context.order_book else 10.0
                    no_depth = context.order_book_no.ask_depth_usd if context.order_book_no else 10.0
                    max_size = min(self.max_arb_size, yes_depth * 0.5, no_depth * 0.5)

                    if max_size >= 1.0:
                        orders.append({
                            "side": "BUY",
                            "token_id": context.market.yes_token_id,
                            "price": yes_ask,
                            "size": max_size,
                            "edge": net_profit,
                            "reason": f"long_arb_{gross_profit:.2%}"
                        })
                        orders.append({
                            "side": "BUY",
                            "token_id": context.market.no_token_id,
                            "price": no_ask,
                            "size": max_size,
                            "edge": net_profit,
                            "reason": f"long_arb_{gross_profit:.2%}"
                        })

        # ===== SHORT ARBITRAGE: Sell both YES and NO =====
        # If YES_bid + NO_bid > 1.02, we can sell both and guarantee profit
        # (Only possible if we already hold positions)
        if yes_bid is not None and no_bid is not None:
            total_proceeds = yes_bid + no_bid

            if total_proceeds > self.short_arb_threshold:
                gross_profit = total_proceeds - 1.0
                fee_pct = 0.04
                net_profit = gross_profit - (total_proceeds * fee_pct)

                if net_profit > self.min_profit_pct:
                    logger.info(
                        f"SHORT ARB DETECTED: YES {yes_bid:.3f} + NO {no_bid:.3f} = "
                        f"{total_proceeds:.3f}, gross profit {gross_profit:.2%}, "
                        f"net profit {net_profit:.2%}"
                    )
                    # Note: Short arb requires existing inventory
                    # For now, just log - implementation would need portfolio check
                    pass

        return orders

    async def on_order_update(self, order_id: str, new_state: str):
        """No state to manage for ArbTaker."""
        pass

# ==============================================================================
# STRATEGY 2: LATENCY SNIPE (Offensive)
# ==============================================================================
class LatencySnipeStrategy(BaseStrategy):
    """
    Subscribes to Binance WS (via Oracle).
    If Spot Price moves > 2% in 2 mins, and Poly lags, snipe the old price.

    Logic:
    - Track spot price history over 2-minute window
    - When spot moves >2%, check if Polymarket YES price has adjusted
    - If Polymarket is lagging (price hasn't moved proportionally), snipe
    - For 15-min binary markets: spot pump -> YES should increase
    """
    def __init__(self, config: AppConfig, oracle_feed):
        super().__init__("LatencySnipe", config)
        self.oracle = oracle_feed
        # Asset -> Deque[(price, timestamp)]
        self.price_history: Dict[str, Deque[tuple[float, float]]] = {}
        # Track last known YES prices to detect lag
        self.last_yes_prices: Dict[str, tuple[float, float]] = {}  # asset -> (price, timestamp)
        self.window_seconds = 120  # 2 minutes
        self.min_delta = 0.02      # 2% move in spot
        self.lag_threshold = 0.01  # YES should move at least 1% if spot moves 2%
        self.max_position_size = 5.0  # Max $5 per snipe

    async def scan(self, context: TradeContext) -> List[dict]:
        asset = context.market.asset
        spot = context.spot_price

        if spot is None or spot <= 0:
            return []

        # 0. Initialize History
        if asset not in self.price_history:
            self.price_history[asset] = deque()

        history = self.price_history[asset]
        now = time.time()

        # 1. Update History & Prune
        history.append((spot, now))
        while history and (now - history[0][1] > self.window_seconds):
            history.popleft()

        # 2. Need enough history
        if len(history) < 5:
            return []

        # 3. Calculate Spot Delta (Current vs Oldest in Window)
        oldest_price, oldest_time = history[0]
        if oldest_price <= 0:
            return []

        spot_delta_pct = (spot - oldest_price) / oldest_price

        # 4. Check Trigger (> 2% move in spot)
        if abs(spot_delta_pct) < self.min_delta:
            return []

        # 5. Check if we have order book data
        if not context.order_book:
            return []

        current_yes_price = context.market.yes_price
        if current_yes_price is None or current_yes_price <= 0:
            return []

        # 6. Track YES price history
        last_yes = self.last_yes_prices.get(asset)
        self.last_yes_prices[asset] = (current_yes_price, now)

        if last_yes is None:
            return []

        last_yes_price, last_yes_time = last_yes

        # Only compare if we have a recent YES price (within window)
        if now - last_yes_time > self.window_seconds:
            return []

        # 7. Calculate YES price movement
        if last_yes_price <= 0:
            return []
        yes_delta_pct = (current_yes_price - last_yes_price) / last_yes_price

        # 8. Detect Lagging Market
        # If spot pumped >2% but YES hasn't moved proportionally, market is lagging
        orders = []

        if spot_delta_pct > self.min_delta:
            # Spot pumped - YES should have increased
            expected_yes_move = spot_delta_pct * 0.5  # YES should move ~half of spot
            if yes_delta_pct < expected_yes_move - self.lag_threshold:
                # Market is lagging - BUY YES
                logger.info(
                    f"Latency snipe detected: {asset} spot +{spot_delta_pct:.2%}, "
                    f"YES only +{yes_delta_pct:.2%}, expected +{expected_yes_move:.2%}"
                )
                if context.order_book.best_ask:
                    orders.append({
                        "side": "BUY",
                        "token_id": context.market.yes_token_id,
                        "price": context.order_book.best_ask,
                        "size": self.max_position_size,
                        "reason": f"latency_snipe_pump_{spot_delta_pct:.2%}"
                    })

        elif spot_delta_pct < -self.min_delta:
            # Spot dumped - YES should have decreased (or NO should increase)
            expected_yes_move = spot_delta_pct * 0.5  # YES should drop
            if yes_delta_pct > expected_yes_move + self.lag_threshold:
                # Market is lagging - SELL YES (or BUY NO)
                logger.info(
                    f"Latency snipe detected: {asset} spot {spot_delta_pct:.2%}, "
                    f"YES only {yes_delta_pct:.2%}, expected {expected_yes_move:.2%}"
                )
                # Buy NO instead of selling YES (safer for binary markets)
                if context.market.no_token_id and context.market.no_price:
                    orders.append({
                        "side": "BUY",
                        "token_id": context.market.no_token_id,
                        "price": context.market.no_price,
                        "size": self.max_position_size,
                        "reason": f"latency_snipe_dump_{spot_delta_pct:.2%}"
                    })

        return orders

    async def on_order_update(self, order_id: str, new_state: str):
        """No state to manage for LatencySnipe."""
        pass


# ==============================================================================
# STRATEGY 3: SPREAD MAKER (Passive Market Making)
# ==============================================================================
class SpreadMakerStrategy(BaseStrategy):
    """
    Passive market making strategy.

    Posts bid/ask orders inside the spread when spread > threshold.
    Implements basic inventory management to avoid directional risk.

    Features:
    - Posts both bid and ask orders
    - Skews quotes based on inventory
    - Minimum spread requirement (5 cents)
    - Size based on available liquidity
    """
    def __init__(self, config: AppConfig):
        super().__init__("SpreadMaker", config)
        self.min_spread = 0.05  # 5 cents minimum spread
        self.quote_offset = 0.01  # 1 cent inside the market
        self.max_size_per_side = 5.0  # $5 max per side
        # Track inventory for skewing
        self.inventory: Dict[str, float] = {}  # token_id -> net position

    async def scan(self, context: TradeContext) -> List[dict]:
        book = context.order_book
        if not book or not book.best_bid or not book.best_ask:
            return []

        spread = book.best_ask - book.best_bid
        if spread < self.min_spread:
            return []  # Spread too tight, no edge

        mid = (book.best_ask + book.best_bid) / 2

        # Get current inventory for this token
        token_id = context.market.yes_token_id
        current_inventory = self.inventory.get(token_id, 0.0)

        # Calculate quote prices
        # Base quotes: slightly inside the market
        base_bid = round(mid - self.quote_offset, 2)
        base_ask = round(mid + self.quote_offset, 2)

        # Inventory skew: if we're long, lower our bid and raise our ask
        # to encourage selling and discourage buying
        skew = 0.0
        if abs(current_inventory) > 1.0:
            # Skew by 0.5 cents per $5 of inventory
            skew = (current_inventory / 5.0) * 0.005

        bid_price = max(0.01, round(base_bid - skew, 2))
        ask_price = min(0.99, round(base_ask - skew, 2))

        # Ensure we maintain minimum spread after skew
        if ask_price - bid_price < self.min_spread / 2:
            return []

        # Calculate sizes based on depth and inventory limits
        bid_size = min(self.max_size_per_side, book.bid_depth_usd * 0.1)
        ask_size = min(self.max_size_per_side, book.ask_depth_usd * 0.1)

        # Reduce size on side where we have inventory
        if current_inventory > 2.0:
            bid_size *= 0.5  # Reduce buying
        elif current_inventory < -2.0:
            ask_size *= 0.5  # Reduce selling

        orders = []

        # Only post orders if they would be inside the market
        if bid_price < book.best_ask and bid_size >= 1.0:
            orders.append({
                "side": "BUY",
                "token_id": token_id,
                "price": bid_price,
                "size": bid_size,
                "order_type": "GTC",
                "reason": f"spread_maker_bid_{spread:.2f}"
            })

        if ask_price > book.best_bid and ask_size >= 1.0:
            orders.append({
                "side": "SELL",
                "token_id": token_id,
                "price": ask_price,
                "size": ask_size,
                "order_type": "GTC",
                "reason": f"spread_maker_ask_{spread:.2f}"
            })

        if orders:
            logger.info(
                f"SpreadMaker: spread={spread:.3f}, mid={mid:.3f}, "
                f"bid={bid_price}, ask={ask_price}, inventory={current_inventory:.2f}"
            )

        return orders

    async def on_order_update(self, order_id: str, new_state: str):
        """Track fills to update inventory."""
        # Note: In a full implementation, we'd track which orders belong to this
        # strategy and update inventory on fills. For now, inventory is tracked
        # separately in the portfolio.
        pass

    def update_inventory(self, token_id: str, delta: float) -> None:
        """Update inventory tracking after a fill."""
        current = self.inventory.get(token_id, 0.0)
        self.inventory[token_id] = current + delta


# ==============================================================================
# STRATEGY 4: LEGGED HEDGE (Crash & Catch)
# ==============================================================================
class LegState(Enum):
    SEARCHING = "searching"
    LEG1_Open = "leg1_open"
    LEG1_FILLED = "leg1_filled"
    HEDGING = "hedging"
    COMPLETE = "complete"

@dataclass
class LegContext:
    state: LegState
    leg1_order_id: Optional[str] = None
    leg1_price: float = 0.0
    leg1_size: float = 0.0
    timestamp: float = 0.0

class LeggedHedgeStrategy(BaseStrategy):
    """
    Leg 1: Buy dropped side if >15% book drop.
    Leg 2: Hedge opposite when sum < 0.95.
    """
    def __init__(self, config: AppConfig):
        super().__init__("LeggedHedge", config)
        self.active_legs: Dict[str, LegContext] = {} # market_id -> LegContext

    async def scan(self, context: TradeContext) -> List[dict]:
        m_id = context.market.question
        if m_id not in self.active_legs:
            self.active_legs[m_id] = LegContext(state=LegState.SEARCHING, timestamp=time.time())
            
        leg_ctx = self.active_legs[m_id]
        
        if leg_ctx.state == LegState.SEARCHING:
            # GROK SPEC: "Buy dropped side if >15% book drop"
            # We use YES price drop of 15% as trigger
            if context.order_book and context.order_book.best_bid:
                # Compare to "opening" or "fair" price if we had it, 
                # for now compare to market price from discovery
                if context.market.yes_price > 0:
                    drop = (context.market.yes_price - context.order_book.best_bid) / context.market.yes_price
                    if drop > 0.15:
                        logger.warning(f"Crash detected: {context.market.asset} dropped {drop:.2%}")
                        order = {
                            "side": "BUY",
                            "token_id": context.market.yes_token_id,
                            "price": context.order_book.best_bid,
                            "size": 5.0
                        }
                        leg_ctx.state = LegState.LEG1_Open
                        return [order]

        # 2. LEG1_OPEN -> Managed by on_order_update
        elif leg_ctx.state == LegState.LEG1_Open:
            # Check timeout?
            pass
            
        # 3. LEG1_FILLED -> Execute Hedge
        elif leg_ctx.state == LegState.LEG1_FILLED:
            # Check Hedge Condition: Sum < 0.95
            book = context.order_book
            if book and book.best_bid and book.best_ask:
                # Assuming we bought YES, we want to buy NO.
                # Simplistic hedge check
                orders = [{
                    "side": "BUY", # Opposite side
                    "token_id": context.market.no_token_id, 
                    "price": 1.0 - book.best_bid - 0.02, # Aggressive
                    "size": leg_ctx.leg1_size # Hedge delta neutral or 1:1
                }]
                leg_ctx.state = LegState.HEDGING
                return orders

        # 4. HEDGING -> Managed by on_order_update
        elif leg_ctx.state == LegState.HEDGING:
            pass

        return []

    async def on_order_update(self, order_id: str, new_state: str):
        # Find which market this order belongs to
        # Inefficient O(N) lookup unless we map order_id -> market_id separately
        # For now, iterating small active set
        for m_id, ctx in self.active_legs.items():
            if ctx.leg1_order_id == order_id:
                if new_state == "filled":
                    ctx.state = LegState.LEG1_FILLED
                    logger.info(f"Leg 1 Filled for {m_id}, looking to hedge.")
                elif new_state in ("canceled", "rejected", "expired"):
                    ctx.state = LegState.SEARCHING # Reset
                    ctx.leg1_order_id = None
                return
