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

        # All-time win/loss tracking
        self.total_wins: int = 0
        self.total_losses: int = 0
        self.total_trades: int = 0
        self.biggest_win: float = 0.0
        self.biggest_loss: float = 0.0
        self._current_streak: int = 0  # positive = wins, negative = losses
        self.max_consecutive_wins: int = 0
        self.max_consecutive_losses: int = 0

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

        # Update all-time tracking
        self.total_trades += 1
        if pnl >= 0:
            daily.win_count += 1
            self.total_wins += 1
            self.biggest_win = max(self.biggest_win, pnl)
            # Update streak
            if self._current_streak >= 0:
                self._current_streak += 1
            else:
                self._current_streak = 1
            self.max_consecutive_wins = max(self.max_consecutive_wins, self._current_streak)
        else:
            daily.loss_count += 1
            self.total_losses += 1
            self.biggest_loss = min(self.biggest_loss, pnl)
            # Update streak
            if self._current_streak <= 0:
                self._current_streak -= 1
            else:
                self._current_streak = -1
            self.max_consecutive_losses = max(self.max_consecutive_losses, abs(self._current_streak))

        logger.info(
            "Position settled",
            token_id=token_id,
            outcome="YES" if outcome else "NO",
            pnl=pnl,
            capital=self.current_capital,
            win_rate=f"{self.win_rate * 100:.1f}%",
        )
        
        # PERSISTENCE: Save state after every settlement
        self.save_state()

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

    @property
    def win_rate(self) -> float:
        """All-time win rate as a decimal (0.0 to 1.0)."""
        if self.total_trades == 0:
            return 0.0
        return self.total_wins / self.total_trades

    @property
    def current_streak(self) -> int:
        """Current win/loss streak. Positive = wins, negative = losses."""
        return self._current_streak

    @property
    def average_win(self) -> float:
        """Average profit on winning trades."""
        if self.total_wins == 0:
            return 0.0
        wins = [p.realized_pnl for p in self._closed_positions if p.realized_pnl >= 0]
        return sum(wins) / len(wins) if wins else 0.0

    @property
    def average_loss(self) -> float:
        """Average loss on losing trades (returns negative value)."""
        if self.total_losses == 0:
            return 0.0
        losses = [p.realized_pnl for p in self._closed_positions if p.realized_pnl < 0]
        return sum(losses) / len(losses) if losses else 0.0

    @property
    def profit_factor(self) -> float:
        """Ratio of gross profit to gross loss. >1 is profitable."""
        gross_profit = sum(p.realized_pnl for p in self._closed_positions if p.realized_pnl >= 0)
        gross_loss = abs(sum(p.realized_pnl for p in self._closed_positions if p.realized_pnl < 0))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss
    
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
            # All-time win rate stats
            "total_trades": self.total_trades,
            "total_wins": self.total_wins,
            "total_losses": self.total_losses,
            "win_rate": round(self.win_rate * 100, 1),
            "current_streak": self.current_streak,
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
            "biggest_win": round(self.biggest_win, 2),
            "biggest_loss": round(self.biggest_loss, 2),
            "average_win": round(self.average_win, 2),
            "average_loss": round(self.average_loss, 2),
            "profit_factor": round(self.profit_factor, 2) if self.profit_factor != float("inf") else "inf",
        }

    def save_state(self, filepath: str = "data/portfolio_state.json") -> None:
        """Save portfolio state to disk atomically."""
        import json
        import os
        from pathlib import Path
        
        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "current_capital": self.current_capital,
                "peak_capital": self.peak_capital,
                "total_trades": self.total_trades,
                "total_wins": self.total_wins,
                "total_losses": self.total_losses,
                "biggest_win": self.biggest_win,
                "biggest_loss": self.biggest_loss,
                "current_streak": self._current_streak,
                "max_consecutive_wins": self.max_consecutive_wins,
                "max_consecutive_losses": self.max_consecutive_losses,
                "saved_at": datetime.now(timezone.utc).isoformat(),
                # Note: We don't save open positions as they might desync from chain
                # We start fresh with positions but keep PnL history
            }
            
            # Atomic write: write to temp file, then rename
            temp_path = path.with_suffix('.tmp')
            with open(temp_path, "w") as f:
                json.dump(data, f, indent=2)
                f.flush()  # Ensure data is written to disk
                os.fsync(f.fileno())  # Force OS to flush to disk
            
            # Atomic rename (POSIX guaranteed atomic)
            temp_path.replace(path)
            
            logger.debug("Portfolio state saved atomically", path=str(filepath))
                
        except Exception as e:
            logger.error("Failed to save portfolio state", error=str(e))

    def load_state(self, filepath: str = "data/portfolio_state.json") -> None:
        """Load portfolio state from disk."""
        import json
        import os
        
        if not os.path.exists(filepath):
            return
            
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                
            self.current_capital = data.get("current_capital", self.current_capital)
            self.peak_capital = data.get("peak_capital", self.peak_capital)
            self.total_trades = data.get("total_trades", 0)
            self.total_wins = data.get("total_wins", 0)
            self.total_losses = data.get("total_losses", 0)
            self.biggest_win = data.get("biggest_win", 0.0)
            self.biggest_loss = data.get("biggest_loss", 0.0)
            self._current_streak = data.get("current_streak", 0)
            self.max_consecutive_wins = data.get("max_consecutive_wins", 0)
            self.max_consecutive_losses = data.get("max_consecutive_losses", 0)
            
            logger.info("Portfolio state loaded", trades=self.total_trades, capital=self.current_capital)
            
        except Exception as e:
            logger.error("Failed to load portfolio state", error=str(e))
