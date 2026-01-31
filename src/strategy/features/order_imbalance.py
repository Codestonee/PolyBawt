"""
Order Book Imbalance (OBI) Signal.

OBI measures the imbalance between bid and ask depth in the order book.
Strong imbalance predicts short-term price movement direction.

Research basis:
- "Most predictive feature for <15m horizons" (Perplexity/Grok research)
- OBI > 0.2 → Strong buy pressure → Signal BUY
- OBI < -0.2 → Strong sell pressure → Signal SELL
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class OBISignal(Enum):
    """Order Book Imbalance trading signal."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class OBIResult:
    """Result of OBI calculation."""
    obi: float  # -1 to +1, positive = more bids
    signal: OBISignal
    bid_volume: float
    ask_volume: float
    levels_analyzed: int
    is_reliable: bool  # True if enough liquidity for reliable estimate

    @property
    def signal_strength(self) -> float:
        """Signal strength from 0 (neutral) to 1 (extreme)."""
        return abs(self.obi)

    @property
    def is_actionable(self) -> bool:
        """Whether the signal is strong enough to trade on."""
        return self.signal in (
            OBISignal.STRONG_BUY,
            OBISignal.BUY,
            OBISignal.SELL,
            OBISignal.STRONG_SELL,
        )


class OrderBookImbalance:
    """
    Order Book Imbalance calculator.

    OBI = (BidVol - AskVol) / (BidVol + AskVol)

    - OBI > 0 → More bids than asks (buying pressure)
    - OBI < 0 → More asks than bids (selling pressure)
    - OBI near 0 → Balanced book

    Usage:
        obi = OrderBookImbalance()

        result = obi.calculate(
            bids=[(0.55, 100), (0.54, 200), ...],  # (price, size) tuples
            asks=[(0.56, 150), (0.57, 100), ...],
        )

        if result.signal == OBISignal.STRONG_BUY:
            # Consider buying
    """

    def __init__(
        self,
        levels: int = 5,                    # Number of price levels to analyze
        strong_threshold: float = 0.4,      # Threshold for STRONG_BUY/STRONG_SELL
        weak_threshold: float = 0.2,        # Threshold for BUY/SELL
        min_volume_usd: float = 100.0,      # Minimum total volume for reliable signal
        weighted: bool = True,              # Weight volume by distance from mid
    ):
        self.levels = levels
        self.strong_threshold = strong_threshold
        self.weak_threshold = weak_threshold
        self.min_volume_usd = min_volume_usd
        self.weighted = weighted

    def calculate(
        self,
        bids: list[tuple[float, float]],  # (price, size_usd)
        asks: list[tuple[float, float]],
        mid_price: float | None = None,
    ) -> OBIResult:
        """
        Calculate Order Book Imbalance.

        Args:
            bids: List of (price, size_usd) tuples, sorted descending by price
            asks: List of (price, size_usd) tuples, sorted ascending by price
            mid_price: Optional mid price for weighted calculation

        Returns:
            OBIResult with imbalance and signal
        """
        # Take top N levels
        top_bids = bids[:self.levels]
        top_asks = asks[:self.levels]

        if not top_bids and not top_asks:
            return OBIResult(
                obi=0.0,
                signal=OBISignal.NEUTRAL,
                bid_volume=0.0,
                ask_volume=0.0,
                levels_analyzed=0,
                is_reliable=False,
            )

        # Calculate mid price if not provided
        if mid_price is None and top_bids and top_asks:
            mid_price = (top_bids[0][0] + top_asks[0][0]) / 2
        elif mid_price is None:
            mid_price = top_bids[0][0] if top_bids else top_asks[0][0]

        # Sum volumes (optionally weighted by distance from mid)
        if self.weighted and mid_price > 0:
            bid_volume = self._weighted_volume(top_bids, mid_price, is_bid=True)
            ask_volume = self._weighted_volume(top_asks, mid_price, is_bid=False)
        else:
            bid_volume = sum(size for _, size in top_bids)
            ask_volume = sum(size for _, size in top_asks)

        total_volume = bid_volume + ask_volume
        levels_analyzed = max(len(top_bids), len(top_asks))
        is_reliable = total_volume >= self.min_volume_usd

        # Calculate OBI
        if total_volume == 0:
            obi = 0.0
        else:
            obi = (bid_volume - ask_volume) / total_volume

        # Determine signal
        signal = self._classify_signal(obi)

        return OBIResult(
            obi=obi,
            signal=signal,
            bid_volume=bid_volume,
            ask_volume=ask_volume,
            levels_analyzed=levels_analyzed,
            is_reliable=is_reliable,
        )

    def _weighted_volume(
        self,
        orders: list[tuple[float, float]],
        mid_price: float,
        is_bid: bool,
    ) -> float:
        """
        Calculate volume weighted by distance from mid price.

        Orders closer to mid price are weighted more heavily.
        """
        if not orders:
            return 0.0

        weighted_sum = 0.0
        for price, size in orders:
            # Distance from mid as percentage
            distance = abs(price - mid_price) / mid_price

            # Weight inversely proportional to distance
            # Orders at mid get weight 1.0, orders 1% away get ~0.9, etc.
            weight = 1.0 / (1.0 + distance * 10)
            weighted_sum += size * weight

        return weighted_sum

    def _classify_signal(self, obi: float) -> OBISignal:
        """Classify OBI into trading signal."""
        if obi >= self.strong_threshold:
            return OBISignal.STRONG_BUY
        elif obi >= self.weak_threshold:
            return OBISignal.BUY
        elif obi <= -self.strong_threshold:
            return OBISignal.STRONG_SELL
        elif obi <= -self.weak_threshold:
            return OBISignal.SELL
        else:
            return OBISignal.NEUTRAL

    def calculate_from_book(self, order_book: Any) -> OBIResult:
        """
        Calculate OBI from an OrderBook object.

        Args:
            order_book: Object with bids/asks as list of (price, size) tuples

        Returns:
            OBIResult
        """
        bids = getattr(order_book, "bids", [])
        asks = getattr(order_book, "asks", [])
        mid = getattr(order_book, "mid_price", None)

        return self.calculate(bids=bids, asks=asks, mid_price=mid)


# Default instance
order_book_imbalance = OrderBookImbalance()


def calculate_obi(
    bids: list[tuple[float, float]],
    asks: list[tuple[float, float]],
    levels: int = 5,
) -> float:
    """
    Simple OBI calculation function.

    Args:
        bids: List of (price, size) tuples
        asks: List of (price, size) tuples
        levels: Number of levels to analyze

    Returns:
        OBI value from -1 to +1
    """
    bid_vol = sum(size for _, size in bids[:levels])
    ask_vol = sum(size for _, size in asks[:levels])

    if bid_vol + ask_vol == 0:
        return 0.0

    return (bid_vol - ask_vol) / (bid_vol + ask_vol)
