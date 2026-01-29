"""
Portfolio and PnL tracking.

Tracks:
- Open positions
- Realized/unrealized PnL
- Daily performance
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class PositionSide(Enum):
    """Position side."""
    LONG_YES = "long_yes"
    LONG_NO = "long_no"


@dataclass
class Position:
    """An open position."""
    
    # Identifiers
    token_id: str
    asset: str
    market_question: str
    
    # Position details
    side: PositionSide
    entry_price: float
    size_usd: float
    shares: float
    
    # Timestamps
    opened_at: float = field(default_factory=time.time)
    expires_at: float = 0  # Market expiry
    
    # State
    closed: bool = False
    exit_price: float = 0
    realized_pnl: float = 0
    
    @property
    def current_value(self) -> float:
        """Current position value (at entry price)."""
        return self.shares * self.entry_price
    
    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL at current price."""
        if self.closed:
            return 0
        return self.shares * (current_price - self.entry_price)
    
    def close(self, exit_price: float, outcome: bool | None = None) -> float:
        """
        Close the position.
        
        Args:
            exit_price: Price at close
            outcome: True if YES won, False if NO won, None if sold
        
        Returns:
            Realized PnL
        """
        if self.closed:
            return 0
        
        if outcome is not None:
            # Settlement
            if self.side == PositionSide.LONG_YES:
                final_price = 1.0 if outcome else 0.0
            else:
                final_price = 0.0 if outcome else 1.0
            exit_price = final_price
        
        self.exit_price = exit_price
        self.realized_pnl = self.shares * (exit_price - self.entry_price)
        self.closed = True
        
        return self.realized_pnl


@dataclass
class DailyPerformance:
    """Daily PnL tracking."""
    
    date: str  # YYYY-MM-DD
    starting_capital: float
    realized_pnl: float = 0
    unrealized_pnl: float = 0
    trades_entered: int = 0
    trades_exited: int = 0
    win_count: int = 0
    loss_count: int = 0
    
    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def return_pct(self) -> float:
        if self.starting_capital == 0:
            return 0
        return self.total_pnl / self.starting_capital
    
    @property
    def win_rate(self) -> float:
        total = self.win_count + self.loss_count
        if total == 0:
            return 0
        return self.win_count / total


class Portfolio:
    """
    Portfolio manager for position and PnL tracking.
    
    Usage:
        portfolio = Portfolio(starting_capital=1000)
        
        # Open position
        position = portfolio.open_position(
            token_id="abc",
            asset="BTC",
            side=PositionSide.LONG_YES,
            price=0.55,
            size_usd=10,
        )
        
        # Check unrealized PnL
        pnl = portfolio.unrealized_pnl({"abc": 0.60})
        
        # Close on settlement
        portfolio.settle_position("abc", outcome=True)
    """
    
    def __init__(self, starting_capital: float):
        self.starting_capital = starting_capital
        self.current_capital = starting_capital
        self.peak_capital = starting_capital
        
        self._positions: dict[str, Position] = {}
        self._closed_positions: list[Position] = []
        self._daily: dict[str, DailyPerformance] = {}
        
        # Initialize today
        self._ensure_daily(self._today())
    
    def _today(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    def _ensure_daily(self, date: str) -> DailyPerformance:
        if date not in self._daily:
            self._daily[date] = DailyPerformance(
                date=date,
                starting_capital=self.current_capital,
            )
        return self._daily[date]
    
    def open_position(
        self,
        token_id: str,
        asset: str,
        market_question: str,
        side: PositionSide,
        price: float,
        size_usd: float,
        expires_at: float = 0,
    ) -> Position:
        """
        Open a new position.
        
        Args:
            token_id: Token identifier
            asset: Asset (BTC, ETH, etc.)
            market_question: Market description
            side: LONG_YES or LONG_NO
            price: Entry price
            size_usd: Position size in USD
            expires_at: Market expiry timestamp
        
        Returns:
            Position object
        """
        shares = size_usd / price if price > 0 else 0
        
        position = Position(
            token_id=token_id,
            asset=asset,
            market_question=market_question,
            side=side,
            entry_price=price,
            size_usd=size_usd,
            shares=shares,
            expires_at=expires_at,
        )
        
        self._positions[token_id] = position
        
        daily = self._ensure_daily(self._today())
        daily.trades_entered += 1
        
        logger.info(
            "Position opened",
            token_id=token_id,
            asset=asset,
            side=side.value,
            price=price,
            size_usd=size_usd,
        )
        
        return position
    
    def settle_position(
        self,
        token_id: str,
        outcome: bool,
    ) -> float:
        """
        Settle a position at market expiry.
        
        Args:
            token_id: Token identifier
            outcome: True if YES won, False if NO won
        
        Returns:
            Realized PnL
        """
        position = self._positions.get(token_id)
        if position is None:
            return 0
        
        pnl = position.close(0, outcome=outcome)
        
        # Update tracking
        self._closed_positions.append(position)
        del self._positions[token_id]
        
        self.current_capital += pnl
        self.peak_capital = max(self.peak_capital, self.current_capital)
        
        daily = self._ensure_daily(self._today())
        daily.realized_pnl += pnl
        daily.trades_exited += 1
        if pnl >= 0:
            daily.win_count += 1
        else:
            daily.loss_count += 1
        
        logger.info(
            "Position settled",
            token_id=token_id,
            outcome="YES" if outcome else "NO",
            pnl=pnl,
            capital=self.current_capital,
        )
        
        return pnl
    
    def unrealized_pnl(self, current_prices: dict[str, float]) -> float:
        """
        Calculate total unrealized PnL.
        
        Args:
            current_prices: Dict of token_id -> current price
        
        Returns:
            Total unrealized PnL
        """
        total = 0
        for token_id, position in self._positions.items():
            if token_id in current_prices:
                total += position.unrealized_pnl(current_prices[token_id])
        return total
    
    def get_position(self, token_id: str) -> Position | None:
        return self._positions.get(token_id)
    
    def get_open_positions(self) -> list[Position]:
        return list(self._positions.values())
    
    def get_exposure_by_asset(self) -> dict[str, float]:
        """Get total exposure per asset."""
        exposure: dict[str, float] = {}
        for position in self._positions.values():
            exposure[position.asset] = exposure.get(position.asset, 0) + position.size_usd
        return exposure
    
    @property
    def total_exposure(self) -> float:
        return sum(p.size_usd for p in self._positions.values())
    
    @property
    def realized_pnl(self) -> float:
        return sum(p.realized_pnl for p in self._closed_positions)
    
    @property
    def drawdown(self) -> float:
        if self.peak_capital == 0:
            return 0
        return (self.peak_capital - self.current_capital) / self.peak_capital
    
    def get_daily_performance(self, date: str | None = None) -> DailyPerformance:
        if date is None:
            date = self._today()
        return self._ensure_daily(date)
    
    def summary(self) -> dict:
        """Get portfolio summary."""
        daily = self.get_daily_performance()
        return {
            "starting_capital": self.starting_capital,
            "current_capital": round(self.current_capital, 2),
            "peak_capital": round(self.peak_capital, 2),
            "total_exposure": round(self.total_exposure, 2),
            "open_positions": len(self._positions),
            "closed_positions": len(self._closed_positions),
            "realized_pnl": round(self.realized_pnl, 2),
            "drawdown_pct": round(self.drawdown * 100, 2),
            "daily_pnl": round(daily.total_pnl, 2),
            "daily_return_pct": round(daily.return_pct * 100, 2),
            "daily_win_rate": round(daily.win_rate * 100, 1),
        }
