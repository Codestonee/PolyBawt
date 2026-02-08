"""
Local Order Book for PolyBawt.

Sequence-validated local order book with:
- Sequence gap detection
- Snapshot recovery
- Staleness detection
- Microprice calculation

Usage:
    book = LocalBook(token_id)
    book.apply_delta(delta_event)
    microprice = book.microprice
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BookLevel:
    """Single price level in the book."""
    price: float
    size: float
    order_count: int = 1
    
    def __lt__(self, other):
        return self.price < other.price


@dataclass
class BookSnapshot:
    """Full order book snapshot."""
    token_id: str
    bids: List[BookLevel]
    asks: List[BookLevel]
    sequence: int
    timestamp_ns: int


class LocalBook:
    """
    Sequence-validated local order book.
    
    Features:
    - Sequence gap detection with recovery trigger
    - Staleness detection
    - Microprice calculation (volume-weighted mid)
    - Delta application with validation
    """
    
    def __init__(
        self,
        token_id: str,
        stale_threshold_ms: float = 1500.0,
        max_sequence_gap: int = 10,
    ):
        self.token_id = token_id
        self.stale_threshold_ns = int(stale_threshold_ms * 1_000_000)
        self.max_sequence_gap = max_sequence_gap
        
        # Book state
        self._bids: Dict[float, BookLevel] = {}
        self._asks: Dict[float, BookLevel] = {}
        
        # Sequence tracking
        self._last_sequence: int = 0
        self._last_update_ns: int = 0
        self._needs_snapshot: bool = True
        self._gap_count: int = 0
        
        # Sorted price caches (invalidated on update)
        self._sorted_bids: Optional[List[float]] = None
        self._sorted_asks: Optional[List[float]] = None
    
    @property
    def best_bid(self) -> Optional[float]:
        """Get best bid price."""
        if not self._bids:
            return None
        if self._sorted_bids is None:
            self._sorted_bids = sorted(self._bids.keys(), reverse=True)
        return self._sorted_bids[0] if self._sorted_bids else None
    
    @property
    def best_ask(self) -> Optional[float]:
        """Get best ask price."""
        if not self._asks:
            return None
        if self._sorted_asks is None:
            self._sorted_asks = sorted(self._asks.keys())
        return self._sorted_asks[0] if self._sorted_asks else None
    
    @property
    def spread(self) -> Optional[float]:
        """Get current spread."""
        bid, ask = self.best_bid, self.best_ask
        if bid is not None and ask is not None:
            return ask - bid
        return None
    
    @property
    def midpoint(self) -> Optional[float]:
        """Get simple midpoint."""
        bid, ask = self.best_bid, self.best_ask
        if bid is not None and ask is not None:
            return (bid + ask) / 2
        return None
    
    @property
    def microprice(self) -> Optional[float]:
        """
        Calculate volume-weighted microprice.
        
        microprice = (bid_size * ask_price + ask_size * bid_price) / (bid_size + ask_size)
        
        This is a better fair value estimate than simple midpoint.
        """
        best_bid = self.best_bid
        best_ask = self.best_ask
        
        if best_bid is None or best_ask is None:
            return None
        
        bid_level = self._bids.get(best_bid)
        ask_level = self._asks.get(best_ask)
        
        if not bid_level or not ask_level:
            return self.midpoint
        
        bid_size = bid_level.size
        ask_size = ask_level.size
        
        if bid_size + ask_size == 0:
            return self.midpoint
        
        return (bid_size * best_ask + ask_size * best_bid) / (bid_size + ask_size)
    
    @property
    def is_stale(self) -> bool:
        """Check if book is stale."""
        if self._needs_snapshot:
            return True
        age_ns = time.time_ns() - self._last_update_ns
        return age_ns > self.stale_threshold_ns
    
    @property
    def age_ms(self) -> float:
        """Age of last update in milliseconds."""
        return (time.time_ns() - self._last_update_ns) / 1_000_000
    
    @property
    def needs_snapshot(self) -> bool:
        """Check if a snapshot is needed."""
        return self._needs_snapshot
    
    def apply_snapshot(self, snapshot: BookSnapshot) -> bool:
        """
        Apply a full book snapshot.
        
        Returns:
            True if applied successfully
        """
        if snapshot.token_id != self.token_id:
            logger.warning(f"[LocalBook] Token mismatch: {snapshot.token_id} != {self.token_id}")
            return False
        
        # Clear existing state
        self._bids.clear()
        self._asks.clear()
        
        # Apply snapshot
        for level in snapshot.bids:
            self._bids[level.price] = level
        for level in snapshot.asks:
            self._asks[level.price] = level
        
        self._last_sequence = snapshot.sequence
        self._last_update_ns = snapshot.timestamp_ns or time.time_ns()
        self._needs_snapshot = False
        self._gap_count = 0
        
        # Invalidate caches
        self._sorted_bids = None
        self._sorted_asks = None
        
        logger.info(
            f"[LocalBook] Applied snapshot for {self.token_id}: "
            f"{len(self._bids)} bids, {len(self._asks)} asks, seq={self._last_sequence}"
        )
        return True
    
    def apply_delta(
        self,
        side: str,
        price: float,
        new_size: float,
        sequence: int,
        timestamp_ns: Optional[int] = None,
    ) -> bool:
        """
        Apply a book delta (update to a single price level).
        
        Args:
            side: "BUY" or "SELL"
            price: Price level
            new_size: New size (0 = level removed)
            sequence: Sequence number for gap detection
            timestamp_ns: Optional timestamp
            
        Returns:
            True if applied successfully, False if gap detected
        """
        now_ns = timestamp_ns or time.time_ns()
        
        # Sequence validation
        if self._last_sequence > 0:
            expected = self._last_sequence + 1
            if sequence != expected:
                gap = sequence - expected
                if gap > 0 and gap <= self.max_sequence_gap:
                    # Small gap - apply but log
                    self._gap_count += gap
                    logger.warning(
                        f"[LocalBook] Sequence gap: expected {expected}, got {sequence} "
                        f"(total gaps: {self._gap_count})"
                    )
                elif gap > self.max_sequence_gap:
                    # Large gap - need snapshot
                    logger.error(
                        f"[LocalBook] Large sequence gap ({gap}): triggering snapshot recovery"
                    )
                    self._needs_snapshot = True
                    return False
                # If gap < 0, it's a duplicate/old message - ignore
                if gap < 0:
                    return True
        
        # Apply update
        book = self._bids if side == "BUY" else self._asks
        
        if new_size <= 0:
            # Remove level
            book.pop(price, None)
        else:
            book[price] = BookLevel(price=price, size=new_size)
        
        self._last_sequence = sequence
        self._last_update_ns = now_ns
        
        # Invalidate caches
        if side == "BUY":
            self._sorted_bids = None
        else:
            self._sorted_asks = None
        
        return True
    
    def get_depth(self, levels: int = 5) -> Dict[str, List[Tuple[float, float]]]:
        """Get order book depth for top N levels."""
        if self._sorted_bids is None:
            self._sorted_bids = sorted(self._bids.keys(), reverse=True)
        if self._sorted_asks is None:
            self._sorted_asks = sorted(self._asks.keys())
        
        bids = [
            (p, self._bids[p].size)
            for p in self._sorted_bids[:levels]
        ]
        asks = [
            (p, self._asks[p].size)
            for p in self._sorted_asks[:levels]
        ]
        
        return {"bids": bids, "asks": asks}
    
    def get_total_depth(self, side: str, price_range: float = 0.05) -> float:
        """Get total size within price range of best price."""
        if side == "BUY":
            best = self.best_bid
            if best is None:
                return 0.0
            return sum(
                l.size for p, l in self._bids.items()
                if p >= best - price_range
            )
        else:
            best = self.best_ask
            if best is None:
                return 0.0
            return sum(
                l.size for p, l in self._asks.items()
                if p <= best + price_range
            )
    
    def clear(self):
        """Clear the book state."""
        self._bids.clear()
        self._asks.clear()
        self._last_sequence = 0
        self._last_update_ns = 0
        self._needs_snapshot = True
        self._sorted_bids = None
        self._sorted_asks = None


class LocalBookManager:
    """
    Manages multiple LocalBook instances.
    
    Usage:
        manager = LocalBookManager()
        manager.get_book("token_123").apply_delta(...)
    """
    
    def __init__(self, stale_threshold_ms: float = 1500.0):
        self.stale_threshold_ms = stale_threshold_ms
        self._books: Dict[str, LocalBook] = {}
    
    def get_book(self, token_id: str) -> LocalBook:
        """Get or create a LocalBook for a token."""
        if token_id not in self._books:
            self._books[token_id] = LocalBook(
                token_id, 
                stale_threshold_ms=self.stale_threshold_ms
            )
        return self._books[token_id]
    
    def get_microprice(self, token_id: str) -> Optional[float]:
        """Get microprice for a token."""
        book = self._books.get(token_id)
        return book.microprice if book else None
    
    def all_healthy(self) -> bool:
        """Check if all books are healthy (not stale, no gaps)."""
        return all(not b.is_stale for b in self._books.values())
    
    def stale_tokens(self) -> List[str]:
        """Get list of tokens with stale books."""
        return [t for t, b in self._books.items() if b.is_stale]
    
    def needs_snapshots(self) -> List[str]:
        """Get list of tokens that need snapshots."""
        return [t for t, b in self._books.items() if b.needs_snapshot]
