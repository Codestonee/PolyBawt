"""Order book data structures."""

from dataclasses import dataclass


@dataclass
class OrderBook:
    """
    Parsed order book with bids and asks.

    For Polymarket binary options, prices are 0-1 and sizes are in shares.
    """

    token_id: str
    bids: list[tuple[float, float]]  # [(price, size), ...] sorted descending
    asks: list[tuple[float, float]]  # [(price, size), ...] sorted ascending
    timestamp: float

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
    def mid_price(self) -> float | None:
        """Midpoint between best bid and ask."""
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2

    def depth_at_price(self, side: str, price: float, levels: int = 5) -> float:
        """
        Calculate total size available at or better than the given price.

        Args:
            side: "BUY" or "SELL"
            price: Price level to check
            levels: Number of price levels to sum (default 5)

        Returns:
            Total size in shares
        """
        total_size = 0.0

        if side == "BUY":
            # For buying, we care about asks at or below our price
            for ask_price, ask_size in self.asks[:levels]:
                if ask_price <= price:
                    total_size += ask_size
                else:
                    break
        else:  # SELL
            # For selling, we care about bids at or above our price
            for bid_price, bid_size in self.bids[:levels]:
                if bid_price >= price:
                    total_size += bid_size
                else:
                    break

        return total_size

    @property
    def bid_depth_usd(self) -> float:
        """
        Total USD value on bid side (top 5 levels).

        For Polymarket: value = sum(price * size) for each level
        """
        return sum(price * size for price, size in self.bids[:5])

    @property
    def ask_depth_usd(self) -> float:
        """Total USD value on ask side (top 5 levels)."""
        return sum(price * size for price, size in self.asks[:5])

    def depth_usd(self, levels: int = 5) -> float:
        """Total USD liquidity (bid + ask depth) for given number of levels."""
        bid_depth = sum(price * size for price, size in self.bids[:levels])
        ask_depth = sum(price * size for price, size in self.asks[:levels])
        return bid_depth + ask_depth
