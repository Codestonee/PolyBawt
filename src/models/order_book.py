"""Order book data structures with NumPy vectorization for performance."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class OrderBook:
    """
    Parsed order book with bids and asks.

    For Polymarket binary options, prices are 0-1 and sizes are in shares.

    Performance optimized with NumPy arrays for:
    - Fast depth calculations
    - Efficient slippage estimation
    - Vectorized VWAP computation
    """

    token_id: str
    bids: list[tuple[float, float]]  # [(price, size), ...] sorted descending
    asks: list[tuple[float, float]]  # [(price, size), ...] sorted ascending
    timestamp: float

    # Cached NumPy arrays (computed lazily)
    _bids_arr: Optional[np.ndarray] = field(default=None, repr=False)
    _asks_arr: Optional[np.ndarray] = field(default=None, repr=False)

    def _ensure_arrays(self) -> None:
        """Convert lists to NumPy arrays if not already done."""
        if self._bids_arr is None and self.bids:
            self._bids_arr = np.array(self.bids, dtype=np.float64)
        if self._asks_arr is None and self.asks:
            self._asks_arr = np.array(self.asks, dtype=np.float64)

    @property
    def bids_array(self) -> np.ndarray:
        """Get bids as NumPy array [price, size]."""
        self._ensure_arrays()
        return self._bids_arr if self._bids_arr is not None else np.empty((0, 2))

    @property
    def asks_array(self) -> np.ndarray:
        """Get asks as NumPy array [price, size]."""
        self._ensure_arrays()
        return self._asks_arr if self._asks_arr is not None else np.empty((0, 2))

    @property
    def best_bid(self) -> float | None:
        """Best bid price (highest price someone will pay)."""
        if not self.bids:
            return None
        return self.bids[0][0]

    @property
    def best_ask(self) -> float | None:
        """Best ask price (lowest price someone will sell at)."""
        if not self.asks:
            return None
        return self.asks[0][0]

    @property
    def spread(self) -> float:
        """
        Bid-ask spread in dollars.

        For Polymarket: spread = best_ask - best_bid
        Returns 0 if book is empty.
        """
        if self.best_bid is None or self.best_ask is None:
            return 0.0
        return self.best_ask - self.best_bid

    @property
    def spread_bps(self) -> float:
        """Spread in basis points relative to mid price."""
        mid = self.mid_price
        if mid is None or mid == 0:
            return 0.0
        return (self.spread / mid) * 10000

    @property
    def mid_price(self) -> float | None:
        """Midpoint between best bid and ask."""
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2

    def weighted_mid_price(self, depth_usd: float = 1000.0) -> float | None:
        """
        Volume-weighted mid price up to specified USD depth.

        More accurate than simple mid for illiquid markets.
        """
        bid_vwap = self.vwap_bid(depth_usd)
        ask_vwap = self.vwap_ask(depth_usd)

        if bid_vwap is None or ask_vwap is None:
            return self.mid_price

        return (bid_vwap + ask_vwap) / 2

    def vwap_bid(self, depth_usd: float) -> float | None:
        """Volume-weighted average price for bid side up to depth."""
        arr = self.bids_array
        if arr.size == 0:
            return None

        prices = arr[:, 0]
        sizes = arr[:, 1]
        values = prices * sizes

        cumulative = np.cumsum(values)
        mask = cumulative <= depth_usd

        if not mask.any():
            # Not enough depth, use all available
            if values.sum() == 0:
                return None
            return np.average(prices, weights=sizes)

        # Include partial fill at the boundary
        valid_idx = mask.sum()
        if valid_idx < len(prices):
            # Partial fill calculation
            remaining = depth_usd - (cumulative[valid_idx - 1] if valid_idx > 0 else 0)
            partial_size = remaining / prices[valid_idx] if prices[valid_idx] > 0 else 0

            weights = np.concatenate([sizes[:valid_idx], [partial_size]])
            price_subset = np.concatenate([prices[:valid_idx], [prices[valid_idx]]])
        else:
            weights = sizes[mask]
            price_subset = prices[mask]

        if weights.sum() == 0:
            return None
        return np.average(price_subset, weights=weights)

    def vwap_ask(self, depth_usd: float) -> float | None:
        """Volume-weighted average price for ask side up to depth."""
        arr = self.asks_array
        if arr.size == 0:
            return None

        prices = arr[:, 0]
        sizes = arr[:, 1]
        values = prices * sizes

        cumulative = np.cumsum(values)
        mask = cumulative <= depth_usd

        if not mask.any():
            if values.sum() == 0:
                return None
            return np.average(prices, weights=sizes)

        valid_idx = mask.sum()
        if valid_idx < len(prices):
            remaining = depth_usd - (cumulative[valid_idx - 1] if valid_idx > 0 else 0)
            partial_size = remaining / prices[valid_idx] if prices[valid_idx] > 0 else 0

            weights = np.concatenate([sizes[:valid_idx], [partial_size]])
            price_subset = np.concatenate([prices[:valid_idx], [prices[valid_idx]]])
        else:
            weights = sizes[mask]
            price_subset = prices[mask]

        if weights.sum() == 0:
            return None
        return np.average(price_subset, weights=weights)

    def estimate_slippage(self, order_size_usd: float, side: str = "BUY") -> float:
        """
        Estimate slippage for a given order size.

        Args:
            order_size_usd: Order size in USD
            side: "BUY" or "SELL"

        Returns:
            Slippage as a fraction (0.01 = 1% slippage)
        """
        mid = self.mid_price
        if mid is None or mid == 0:
            return float('inf')

        if side == "BUY":
            vwap = self.vwap_ask(order_size_usd)
        else:
            vwap = self.vwap_bid(order_size_usd)

        if vwap is None:
            return float('inf')

        return abs(vwap - mid) / mid

    def depth_at_price(self, side: str, price: float, levels: int = 5) -> float:
        """
        Calculate total size available at or better than the given price.

        Vectorized implementation for performance.

        Args:
            side: "BUY" or "SELL"
            price: Price level to check
            levels: Number of price levels to sum (default 5)

        Returns:
            Total size in shares
        """
        if side == "BUY":
            arr = self.asks_array
            if arr.size == 0:
                return 0.0
            arr = arr[:levels]
            mask = arr[:, 0] <= price
            return float(np.sum(arr[mask, 1]))
        else:  # SELL
            arr = self.bids_array
            if arr.size == 0:
                return 0.0
            arr = arr[:levels]
            mask = arr[:, 0] >= price
            return float(np.sum(arr[mask, 1]))

    @property
    def bid_depth_usd(self) -> float:
        """
        Total USD value on bid side (top 5 levels).

        Vectorized: value = sum(price * size) for each level
        """
        arr = self.bids_array
        if arr.size == 0:
            return 0.0
        arr = arr[:5]
        return float(np.sum(arr[:, 0] * arr[:, 1]))

    @property
    def ask_depth_usd(self) -> float:
        """Total USD value on ask side (top 5 levels)."""
        arr = self.asks_array
        if arr.size == 0:
            return 0.0
        arr = arr[:5]
        return float(np.sum(arr[:, 0] * arr[:, 1]))

    def depth_usd(self, levels: int = 5) -> float:
        """Total USD liquidity (bid + ask depth) for given number of levels."""
        bid_arr = self.bids_array
        ask_arr = self.asks_array

        bid_depth = 0.0
        ask_depth = 0.0

        if bid_arr.size > 0:
            bid_subset = bid_arr[:levels]
            bid_depth = float(np.sum(bid_subset[:, 0] * bid_subset[:, 1]))

        if ask_arr.size > 0:
            ask_subset = ask_arr[:levels]
            ask_depth = float(np.sum(ask_subset[:, 0] * ask_subset[:, 1]))

        return bid_depth + ask_depth

    def order_book_imbalance(self, levels: int = 5) -> float:
        """
        Calculate order book imbalance.

        Returns value between -1 and 1:
        - Positive: More bids (bullish)
        - Negative: More asks (bearish)
        - 0: Balanced
        """
        bid_depth = self.bid_depth_usd
        ask_depth = self.ask_depth_usd

        total = bid_depth + ask_depth
        if total == 0:
            return 0.0

        return (bid_depth - ask_depth) / total

    def price_impact(self, order_size_usd: float, side: str = "BUY") -> float:
        """
        Estimate price impact using Kyle's Lambda model.

        impact = λ * √(order_size)

        Args:
            order_size_usd: Order size in USD
            side: "BUY" or "SELL"

        Returns:
            Expected price impact as fraction of mid price
        """
        # Estimate lambda from book depth
        depth = self.depth_usd(10)
        if depth == 0:
            return float('inf')

        # Kyle's lambda approximation
        # λ ≈ spread / (2 * sqrt(depth))
        lambda_est = self.spread / (2 * np.sqrt(depth)) if depth > 0 else 1.0

        # Impact = λ * √(size)
        impact = lambda_est * np.sqrt(order_size_usd)

        mid = self.mid_price
        if mid is None or mid == 0:
            return impact

        return impact / mid

    def get(self, key: str, default: any = None) -> any:
        """Dict-like access for compatibility."""
        return getattr(self, key, default)
