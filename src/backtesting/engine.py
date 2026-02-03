"""
Backtesting Engine for strategy evaluation.

Simulates historical trading to evaluate strategy performance:
- Historical price replay
- Order book simulation
- Fill simulation with slippage
- Performance metrics (Sharpe, max DD, etc.)

Usage:
    engine = BacktestEngine(initial_capital=1000)
    engine.load_historical_data("btc_15m_2024.csv")

    strategy = EnsembleStrategy(...)
    results = await engine.run(strategy)

    print(results.sharpe_ratio)
    print(results.max_drawdown)
"""

import asyncio
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator
import math
import random

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


@dataclass
class HistoricalBar:
    """A single bar of historical data."""
    timestamp: datetime
    asset: str
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float = 0.0

    # Market data
    yes_price: float = 0.5
    no_price: float = 0.5
    spread: float = 0.02

    # Outcome (for resolved markets)
    outcome: bool | None = None  # True = UP, False = DOWN


@dataclass
class SimulatedFill:
    """A simulated order fill."""
    order_id: str
    timestamp: datetime
    side: str
    price: float
    size: float
    slippage: float = 0.0


@dataclass
class BacktestTrade:
    """Record of a backtested trade."""
    trade_id: str
    entry_time: datetime
    exit_time: datetime | None
    asset: str
    side: str  # "LONG_YES" or "LONG_NO"
    entry_price: float
    exit_price: float | None
    size_usd: float
    model_prob: float
    market_price: float
    outcome: bool | None
    pnl: float = 0.0
    fees: float = 0.0

    @property
    def duration_minutes(self) -> float:
        if self.exit_time is None:
            return 0.0
        return (self.exit_time - self.entry_time).total_seconds() / 60

    @property
    def net_pnl(self) -> float:
        return self.pnl - self.fees

    @property
    def is_winner(self) -> bool:
        return self.net_pnl > 0


@dataclass
class BacktestResults:
    """Complete backtest results."""

    # Summary
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    # Returns
    total_pnl: float = 0.0
    total_fees: float = 0.0
    net_pnl: float = 0.0

    # Performance metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration_hours: float = 0.0
    calmar_ratio: float = 0.0

    # Win/loss stats
    win_rate: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0

    # Risk metrics
    volatility_annualized: float = 0.0
    var_95: float = 0.0  # Value at Risk
    cvar_95: float = 0.0  # Conditional VaR

    # Timing
    start_time: datetime | None = None
    end_time: datetime | None = None

    # Detailed data
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[tuple[datetime, float]] = field(default_factory=list)
    daily_returns: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_pnl": self.total_pnl,
            "net_pnl": self.net_pnl,
            "total_fees": self.total_fees,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "average_win": self.average_win,
            "average_loss": self.average_loss,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
            "volatility_annualized": self.volatility_annualized,
        }


class SlippageModel:
    """Models execution slippage."""

    def __init__(
        self,
        base_slippage_bps: float = 10,  # 10 basis points
        size_impact_bps_per_100usd: float = 5,
        volatility_multiplier: float = 1.0,
    ):
        self.base_slippage_bps = base_slippage_bps
        self.size_impact = size_impact_bps_per_100usd
        self.vol_mult = volatility_multiplier

    def calculate_slippage(
        self,
        size_usd: float,
        volatility: float = 0.6,
        spread: float = 0.02,
    ) -> float:
        """
        Calculate expected slippage.

        Args:
            size_usd: Order size in USD
            volatility: Current volatility
            spread: Current bid-ask spread

        Returns:
            Slippage as a decimal (0.001 = 0.1%)
        """
        # Base slippage
        slippage = self.base_slippage_bps / 10000

        # Size impact
        size_impact = (size_usd / 100) * (self.size_impact / 10000)
        slippage += size_impact

        # Volatility adjustment
        vol_adjustment = (volatility - 0.6) * self.vol_mult * 0.001
        slippage += max(0, vol_adjustment)

        # Spread component
        slippage += spread * 0.5  # Half the spread

        # Add some randomness
        slippage *= random.uniform(0.8, 1.2)

        return slippage


class FeeModel:
    """Models trading fees."""

    def __init__(self, fee_coefficient: float = 0.25):
        self.fee_coefficient = fee_coefficient

    def calculate_fee(self, price: float, size_usd: float) -> float:
        """
        Calculate Polymarket dynamic fee.

        Fee = size * 0.25 * (p * (1-p))^2
        """
        p = max(0.01, min(0.99, price))
        fee_rate = self.fee_coefficient * (p * (1 - p)) ** 2
        return size_usd * fee_rate


class BacktestEngine:
    """
    Engine for running historical backtests.

    Features:
    - Event-driven simulation
    - Realistic fill simulation with slippage
    - Dynamic fee modeling
    - Comprehensive performance metrics
    """

    def __init__(
        self,
        initial_capital: float = 1000,
        slippage_model: SlippageModel | None = None,
        fee_model: FeeModel | None = None,
    ):
        self.initial_capital = initial_capital
        self.slippage_model = slippage_model or SlippageModel()
        self.fee_model = fee_model or FeeModel()

        # Historical data
        self._bars: list[HistoricalBar] = []
        self._current_idx = 0

        # State
        self._capital = initial_capital
        self._positions: dict[str, dict] = {}
        self._trades: list[BacktestTrade] = []
        self._equity_curve: list[tuple[datetime, float]] = []

        # Counters
        self._trade_counter = 0

    def load_historical_data(self, file_path: str | Path) -> int:
        """
        Load historical data from CSV file.

        Expected columns: timestamp, asset, open, high, low, close, volume,
                         yes_price, no_price, outcome (optional)

        Returns:
            Number of bars loaded
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        self._bars = []

        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                bar = HistoricalBar(
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    asset=row.get('asset', 'BTC'),
                    open_price=float(row['open']),
                    high_price=float(row['high']),
                    low_price=float(row['low']),
                    close_price=float(row['close']),
                    volume=float(row.get('volume', 0)),
                    yes_price=float(row.get('yes_price', 0.5)),
                    no_price=float(row.get('no_price', 0.5)),
                    spread=float(row.get('spread', 0.02)),
                    outcome=row.get('outcome', '').lower() == 'true' if row.get('outcome') else None,
                )
                self._bars.append(bar)

        # Sort by timestamp
        self._bars.sort(key=lambda x: x.timestamp)

        logger.info(f"Loaded {len(self._bars)} historical bars")
        return len(self._bars)

    def generate_synthetic_data(
        self,
        asset: str = "BTC",
        start_price: float = 100000,
        num_bars: int = 1000,
        interval_minutes: int = 15,
        volatility: float = 0.6,
    ) -> int:
        """
        Generate synthetic historical data for testing.

        Uses geometric Brownian motion with jumps.
        """
        self._bars = []

        current_price = start_price
        current_time = datetime.now(timezone.utc) - timedelta(minutes=num_bars * interval_minutes)

        # Annualized to per-bar volatility
        bar_vol = volatility * math.sqrt(interval_minutes / 525600)

        for i in range(num_bars):
            # Generate price movement (GBM with jumps)
            drift = 0
            random_return = random.gauss(drift, bar_vol)

            # Occasional jumps
            if random.random() < 0.02:  # 2% chance of jump
                random_return += random.gauss(0, bar_vol * 3)

            new_price = current_price * math.exp(random_return)

            # Generate OHLC
            high_price = max(current_price, new_price) * (1 + random.uniform(0, bar_vol))
            low_price = min(current_price, new_price) * (1 - random.uniform(0, bar_vol))

            # Determine if UP or DOWN outcome
            outcome = new_price >= current_price

            # YES price based on some model estimate
            # Simplified: if price went up, market leans YES
            if outcome:
                yes_price = 0.5 + random.uniform(0, 0.3)
            else:
                yes_price = 0.5 - random.uniform(0, 0.3)

            bar = HistoricalBar(
                timestamp=current_time,
                asset=asset,
                open_price=current_price,
                high_price=high_price,
                low_price=low_price,
                close_price=new_price,
                volume=random.uniform(10000, 100000),
                yes_price=yes_price,
                no_price=1 - yes_price,
                spread=random.uniform(0.01, 0.05),
                outcome=outcome,
            )
            self._bars.append(bar)

            current_price = new_price
            current_time += timedelta(minutes=interval_minutes)

        logger.info(f"Generated {len(self._bars)} synthetic bars")
        return len(self._bars)

    def bars(self) -> Iterator[HistoricalBar]:
        """Iterate through historical bars."""
        for bar in self._bars:
            yield bar

    async def run(
        self,
        strategy_fn: Callable[[HistoricalBar, float], tuple[str | None, float, float]],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> BacktestResults:
        """
        Run backtest with a strategy function.

        Args:
            strategy_fn: Function that takes (bar, capital) and returns
                        (side, size_usd, model_prob) or (None, 0, 0) for no trade
            progress_callback: Optional callback for progress updates

        Returns:
            BacktestResults with complete statistics
        """
        self._capital = self.initial_capital
        self._positions = {}
        self._trades = []
        self._equity_curve = [(self._bars[0].timestamp, self.initial_capital)]
        self._trade_counter = 0

        total_bars = len(self._bars)

        for idx, bar in enumerate(self._bars):
            # Progress callback
            if progress_callback and idx % 100 == 0:
                progress_callback(idx, total_bars)

            # Close any positions that have resolved
            await self._process_resolutions(bar)

            # Get strategy signal
            side, size_usd, model_prob = strategy_fn(bar, self._capital)

            if side and size_usd > 0:
                await self._execute_trade(bar, side, size_usd, model_prob)

            # Update equity curve
            self._equity_curve.append((bar.timestamp, self._capital))

        # Calculate results
        results = self._calculate_results()

        return results

    async def _execute_trade(
        self,
        bar: HistoricalBar,
        side: str,
        size_usd: float,
        model_prob: float,
    ) -> None:
        """Execute a trade with simulated fills."""
        self._trade_counter += 1
        trade_id = f"bt_{self._trade_counter}"

        # Calculate entry price with slippage
        if side == "LONG_YES":
            base_price = bar.yes_price
            slippage = self.slippage_model.calculate_slippage(size_usd, spread=bar.spread)
            entry_price = base_price * (1 + slippage)
        else:
            base_price = bar.no_price
            slippage = self.slippage_model.calculate_slippage(size_usd, spread=bar.spread)
            entry_price = base_price * (1 + slippage)

        # Calculate fees
        fees = self.fee_model.calculate_fee(entry_price, size_usd)

        # Check if we have enough capital
        total_cost = size_usd + fees
        if total_cost > self._capital:
            size_usd = (self._capital - fees) * 0.95  # Leave some buffer
            if size_usd <= 0:
                return

        # Create trade record
        trade = BacktestTrade(
            trade_id=trade_id,
            entry_time=bar.timestamp,
            exit_time=None,
            asset=bar.asset,
            side=side,
            entry_price=entry_price,
            exit_price=None,
            size_usd=size_usd,
            model_prob=model_prob,
            market_price=bar.yes_price,
            outcome=None,
            fees=fees,
        )

        # Deduct capital
        self._capital -= total_cost

        # Store position
        position_key = f"{bar.asset}_{trade_id}"
        self._positions[position_key] = {
            "trade": trade,
            "bar": bar,
        }

    async def _process_resolutions(self, current_bar: HistoricalBar) -> None:
        """Process any positions that have resolved."""
        resolved_keys = []

        for key, position in self._positions.items():
            trade = position["trade"]
            entry_bar = position["bar"]

            # Check if this market has resolved (outcome known)
            if current_bar.outcome is not None:
                # Calculate P&L
                if trade.side == "LONG_YES":
                    if current_bar.outcome:  # Won
                        trade.pnl = trade.size_usd * (1 / trade.entry_price - 1)
                    else:  # Lost
                        trade.pnl = -trade.size_usd
                else:  # LONG_NO
                    if not current_bar.outcome:  # Won
                        trade.pnl = trade.size_usd * (1 / trade.entry_price - 1)
                    else:  # Lost
                        trade.pnl = -trade.size_usd

                trade.exit_time = current_bar.timestamp
                trade.outcome = current_bar.outcome
                trade.exit_price = 1.0 if (
                    (trade.side == "LONG_YES" and current_bar.outcome) or
                    (trade.side == "LONG_NO" and not current_bar.outcome)
                ) else 0.0

                # Return capital
                self._capital += trade.size_usd + trade.pnl

                self._trades.append(trade)
                resolved_keys.append(key)

        # Remove resolved positions
        for key in resolved_keys:
            del self._positions[key]

    def _calculate_results(self) -> BacktestResults:
        """Calculate comprehensive backtest results."""
        results = BacktestResults(
            trades=self._trades,
            equity_curve=self._equity_curve,
        )

        if not self._trades:
            return results

        # Basic counts
        results.total_trades = len(self._trades)
        results.winning_trades = sum(1 for t in self._trades if t.is_winner)
        results.losing_trades = results.total_trades - results.winning_trades

        # P&L
        results.total_pnl = sum(t.pnl for t in self._trades)
        results.total_fees = sum(t.fees for t in self._trades)
        results.net_pnl = results.total_pnl - results.total_fees

        # Win/loss stats
        results.win_rate = results.winning_trades / results.total_trades if results.total_trades > 0 else 0

        winners = [t.net_pnl for t in self._trades if t.is_winner]
        losers = [t.net_pnl for t in self._trades if not t.is_winner]

        results.average_win = sum(winners) / len(winners) if winners else 0
        results.average_loss = sum(losers) / len(losers) if losers else 0

        total_wins = sum(winners) if winners else 0
        total_losses = abs(sum(losers)) if losers else 0
        results.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        results.expectancy = (
            results.win_rate * results.average_win +
            (1 - results.win_rate) * results.average_loss
        )

        # Calculate returns
        if len(self._equity_curve) > 1:
            returns = []
            for i in range(1, len(self._equity_curve)):
                prev_equity = self._equity_curve[i-1][1]
                curr_equity = self._equity_curve[i][1]
                if prev_equity > 0:
                    returns.append((curr_equity - prev_equity) / prev_equity)

            results.daily_returns = returns

            # Volatility
            if len(returns) > 1:
                mean_ret = sum(returns) / len(returns)
                var = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
                std = math.sqrt(var)
                results.volatility_annualized = std * math.sqrt(252 * 24 * 4)  # Assume 15min bars

                # Sharpe ratio (assuming 0% risk-free rate)
                if std > 0:
                    results.sharpe_ratio = (mean_ret * 252 * 24 * 4) / results.volatility_annualized

                # Sortino ratio
                downside_returns = [r for r in returns if r < 0]
                if downside_returns:
                    downside_var = sum(r ** 2 for r in downside_returns) / len(downside_returns)
                    downside_std = math.sqrt(downside_var)
                    if downside_std > 0:
                        results.sortino_ratio = (mean_ret * 252 * 24 * 4) / (downside_std * math.sqrt(252 * 24 * 4))

            # Max drawdown
            peak = self.initial_capital
            max_dd = 0
            for _, equity in self._equity_curve:
                peak = max(peak, equity)
                dd = (peak - equity) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)

            results.max_drawdown = max_dd

            # Calmar ratio
            if results.max_drawdown > 0:
                annual_return = results.net_pnl / self.initial_capital * (252 * 24 * 4 / len(self._equity_curve))
                results.calmar_ratio = annual_return / results.max_drawdown

            # VaR and CVaR
            if returns:
                sorted_returns = sorted(returns)
                var_idx = int(len(sorted_returns) * 0.05)
                results.var_95 = -sorted_returns[var_idx] if var_idx < len(sorted_returns) else 0
                results.cvar_95 = -sum(sorted_returns[:var_idx+1]) / (var_idx + 1) if var_idx > 0 else 0

        # Timing
        if self._equity_curve:
            results.start_time = self._equity_curve[0][0]
            results.end_time = self._equity_curve[-1][0]

        return results


def run_simple_backtest(
    strategy_name: str = "value_betting",
    initial_capital: float = 1000,
    num_bars: int = 1000,
) -> BacktestResults:
    """
    Run a simple backtest with synthetic data.

    Convenience function for quick testing.
    """
    engine = BacktestEngine(initial_capital=initial_capital)
    engine.generate_synthetic_data(num_bars=num_bars)

    def simple_strategy(bar: HistoricalBar, capital: float) -> tuple[str | None, float, float]:
        """Simple momentum strategy for testing."""
        # If price went up recently, bet YES
        if bar.yes_price > 0.55:
            return "LONG_NO", min(10, capital * 0.02), 1 - bar.yes_price + 0.05
        elif bar.yes_price < 0.45:
            return "LONG_YES", min(10, capital * 0.02), bar.yes_price + 0.05
        return None, 0, 0

    # Run synchronously
    import asyncio
    results = asyncio.get_event_loop().run_until_complete(
        engine.run(simple_strategy)
    )

    return results
