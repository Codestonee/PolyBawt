"""
Market Making Strategy.

Two-sided quoting to earn spread and rebates.

Key principles:
1. Quote both sides (bid + ask) simultaneously
2. Monitor VPIN - halt if toxicity > 0.6
3. Manage inventory - don't get too long/short
4. Target probability extremes (lowest fees)

Research basis:
- Makers pay zero fees + earn USDC rebates
- Best when market is "quiet" (low toxicity)
"""

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any

from src.infrastructure.logging import get_logger
from src.risk.vpin import VPINCalculator, VPINResult, ToxicityLevel
from src.strategy.features.order_imbalance import OrderBookImbalance, OBIResult

logger = get_logger(__name__)


class MarketMakingState(Enum):
    """Market making operational state."""
    ACTIVE = "active"          # Quoting both sides
    SKEWED = "skewed"          # Inventory adjustment needed
    REDUCED = "reduced"        # High toxicity, reduced quoting
    HALTED = "halted"          # VPIN too high, no quotes


@dataclass
class QuoteConfig:
    """Configuration for market making quotes."""
    spread_bps: float = 50.0        # Bid-ask spread in basis points
    quote_size_usd: float = 10.0    # Size per side
    max_inventory: float = 100.0    # Max position before skewing
    skew_bps_per_inventory: float = 1.0  # BPS to skew per $10 inventory
    vpin_halt_threshold: float = 0.6    # VPIN level to halt
    vpin_reduce_threshold: float = 0.4  # VPIN level to reduce size
    min_profit_margin: float = 0.005    # 0.5% minimum expected profit


@dataclass
class Quote:
    """A market making quote."""
    side: str  # "bid" or "ask"
    price: float
    size: float
    token_id: str

    @property
    def is_bid(self) -> bool:
        return self.side == "bid"


@dataclass
class MarketMakingSignal:
    """Signal from market making strategy."""
    state: MarketMakingState
    bid_quote: Quote | None = None
    ask_quote: Quote | None = None
    vpin: VPINResult | None = None
    obi: OBIResult | None = None
    inventory: float = 0.0
    skew_applied: float = 0.0  # BPS of skew applied

    @property
    def is_quoting(self) -> bool:
        return self.bid_quote is not None or self.ask_quote is not None

    @property
    def should_cancel_all(self) -> bool:
        return self.state == MarketMakingState.HALTED


class MarketMakingStrategy:
    """
    Two-sided quoting to earn spread and rebates.

    Usage:
        mm = MarketMakingStrategy(config, vpin_calc, obi_calc)

        signal = mm.generate_signal(
            market_id="xyz",
            mid_price=0.50,
            token_id="abc123",
            current_inventory=25.0,
        )

        if signal.is_quoting:
            # Submit bid_quote and ask_quote
            ...
    """

    def __init__(
        self,
        config: QuoteConfig | None = None,
        vpin_calculator: VPINCalculator | None = None,
        obi_calculator: OrderBookImbalance | None = None,
    ):
        self.config = config or QuoteConfig()
        self.vpin_calc = vpin_calculator or VPINCalculator()
        self.obi_calc = obi_calculator or OrderBookImbalance()

        # Stats
        self._quotes_generated: int = 0
        self._quotes_halted: int = 0
        self._total_inventory_skew: float = 0.0

    def generate_signal(
        self,
        market_id: str,
        mid_price: float,
        token_id: str,
        current_inventory: float = 0.0,
        order_book: Any | None = None,
    ) -> MarketMakingSignal:
        """
        Generate market making quotes.

        Args:
            market_id: Market identifier for VPIN lookup
            mid_price: Current mid price
            token_id: Token ID for quotes
            current_inventory: Current position in USD (positive = long)
            order_book: Optional order book for OBI calculation

        Returns:
            MarketMakingSignal with quotes and state
        """
        # Get VPIN toxicity
        vpin_result = self.vpin_calc.get_vpin(market_id)

        # Get OBI if order book provided
        obi_result = None
        if order_book:
            obi_result = self.obi_calc.calculate_from_book(order_book)

        # Determine state based on VPIN
        if vpin_result.vpin >= self.config.vpin_halt_threshold:
            self._quotes_halted += 1
            logger.warning(
                "MM halted due to high VPIN",
                market_id=market_id,
                vpin=f"{vpin_result.vpin:.2f}",
            )
            return MarketMakingSignal(
                state=MarketMakingState.HALTED,
                vpin=vpin_result,
                obi=obi_result,
                inventory=current_inventory,
            )

        # Determine if we're skewed due to inventory
        state = MarketMakingState.ACTIVE
        if abs(current_inventory) > self.config.max_inventory * 0.5:
            state = MarketMakingState.SKEWED

        if vpin_result.vpin >= self.config.vpin_reduce_threshold:
            state = MarketMakingState.REDUCED

        # Calculate spread
        half_spread_bps = self.config.spread_bps / 2
        half_spread = (half_spread_bps / 10000) * mid_price

        # Calculate inventory skew
        # If long, want to sell more aggressively (lower ask, higher bid)
        skew_bps = (current_inventory / 10) * self.config.skew_bps_per_inventory
        skew = (skew_bps / 10000) * mid_price
        self._total_inventory_skew += abs(skew_bps)

        # Apply OBI skew if available (fade the imbalance)
        obi_skew = 0.0
        if obi_result and obi_result.is_reliable:
            # If strong buy imbalance (positive OBI), skew quotes up
            # This provides liquidity to eager buyers at better prices for us
            obi_skew = obi_result.obi * 0.002 * mid_price  # 20 bps at OBI=1

        # Calculate quote prices
        bid_price = mid_price - half_spread - skew + obi_skew
        ask_price = mid_price + half_spread - skew + obi_skew

        # Adjust size based on toxicity
        size_multiplier = vpin_result.size_multiplier
        quote_size = self.config.quote_size_usd * size_multiplier

        # Don't quote if size would be too small
        if quote_size < 1.0:
            return MarketMakingSignal(
                state=MarketMakingState.REDUCED,
                vpin=vpin_result,
                obi=obi_result,
                inventory=current_inventory,
                skew_applied=skew_bps,
            )

        # Create quotes
        bid_quote = Quote(
            side="bid",
            price=round(bid_price, 4),
            size=quote_size,
            token_id=token_id,
        )
        ask_quote = Quote(
            side="ask",
            price=round(ask_price, 4),
            size=quote_size,
            token_id=token_id,
        )

        self._quotes_generated += 1

        logger.debug(
            "MM quotes generated",
            market_id=market_id,
            mid=f"{mid_price:.4f}",
            bid=f"{bid_quote.price:.4f}",
            ask=f"{ask_quote.price:.4f}",
            size=f"${quote_size:.2f}",
            vpin=f"{vpin_result.vpin:.2f}",
            inventory=f"${current_inventory:.2f}",
        )

        return MarketMakingSignal(
            state=state,
            bid_quote=bid_quote,
            ask_quote=ask_quote,
            vpin=vpin_result,
            obi=obi_result,
            inventory=current_inventory,
            skew_applied=skew_bps,
        )

    def update_vpin(
        self,
        market_id: str,
        volume: float,
        is_buy: bool,
    ) -> None:
        """Update VPIN with a trade."""
        self.vpin_calc.update(market_id, volume, is_buy)

    def should_quote_market(
        self,
        market_price: float,
        expected_volatility: float = 0.05,
    ) -> bool:
        """
        Determine if a market is suitable for market making.

        Market making works best when:
        - Price is at extremes (low fees)
        - Volatility is moderate (not too jumpy)
        - Spread covers expected movement
        """
        # Prefer extreme prices (lower fees)
        is_extreme = market_price < 0.1 or market_price > 0.9

        # Check if spread covers expected volatility
        spread_covers_vol = self.config.spread_bps / 10000 > expected_volatility * 0.5

        return is_extreme or spread_covers_vol

    def get_stats(self) -> dict:
        """Get strategy statistics."""
        return {
            "quotes_generated": self._quotes_generated,
            "quotes_halted": self._quotes_halted,
            "halt_rate": (
                self._quotes_halted / max(1, self._quotes_generated + self._quotes_halted)
            ),
            "avg_inventory_skew_bps": (
                self._total_inventory_skew / max(1, self._quotes_generated)
            ),
        }
