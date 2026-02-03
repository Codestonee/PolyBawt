"""
Value Betting Strategy - Grok Ensemble Edition.

Integrates:
- ArbTakerStrategy (Arb < 0.98)
- LatencySnipeStrategy (Delta > 2%)
- SpreadMakerStrategy (Spread > 5c)
- LeggedHedgeStrategy (Crash > 15%)
"""

import asyncio
import time
import random
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional

from src.infrastructure.logging import get_logger, bind_context
from src.infrastructure.config import AppConfig
from src.ingestion.market_discovery import Market, MarketDiscovery
from src.ingestion.oracle_feed import OracleFeed
from src.risk.circuit_breaker import CircuitBreaker
from src.risk.kelly_sizer import KellySizer, AdaptiveMode, correlation_matrix
from src.execution.order_manager import OrderManager, OrderSide, OrderType, Order
from src.execution.rate_limiter import RateLimiter
from src.execution.clob_client import CLOBClient
from src.models.order_book import OrderBook
from src.portfolio.tracker import Portfolio, PositionSide
from src.strategy.strategies import (
    ArbTakerStrategy,
    LatencySnipeStrategy,
    SpreadMakerStrategy,
    LeggedHedgeStrategy
)
from src.strategy.base import TradeContext

logger = get_logger(__name__)

@dataclass
class StrategyMetrics:
    """Trading metrics for performance tracking."""
    signals_generated: int = 0
    orders_placed: int = 0
    orders_filled: int = 0
    realized_pnl: float = 0.0

@dataclass
class PendingFill:
    """Metadata for an order that has been submitted but not yet filled."""
    token_id: str
    asset: str
    side: PositionSide
    price: float
    size_usd: float
    market_question: str
    expires_at: float

class EnsembleStrategy:
    """
    Orchestrator for the Grok Strategy Ensemble.
    """
    def __init__(
        self,
        config: AppConfig,
        market_discovery: MarketDiscovery,
        oracle: OracleFeed,
        order_manager: OrderManager,
        rate_limiter: RateLimiter,
        circuit_breaker: CircuitBreaker,
        clob_client: CLOBClient | None = None,
        portfolio: Portfolio | None = None,
        bankroll: float = 100.0,
    ):
        self.config = config
        self.market_discovery = market_discovery
        self.oracle = oracle
        self.order_manager = order_manager
        self.rate_limiter = rate_limiter
        self.circuit_breaker = circuit_breaker
        self.clob_client = clob_client
        self.portfolio = portfolio or Portfolio(starting_capital=bankroll)
        self.bankroll = bankroll

        logger.info("Initializing Grok sub-strategies...")
        self._running = False
        self._pending_fills: Dict[str, PendingFill] = {}
        self._order_book_cache: Dict[str, OrderBook] = {}

        # Initialize Kelly Sizer for optimal position sizing
        self.kelly_sizer = KellySizer(
            bankroll=bankroll,
            kelly_fraction=config.trading.kelly_fraction,
            max_position_pct=config.trading.max_position_pct,
            max_asset_exposure_pct=config.trading.max_asset_exposure_pct,
            max_position_usd=5.0,  # Grok spec: $5 max per trade
            adaptive_mode=AdaptiveMode.FULL,
        )
        logger.info(
            "Kelly sizer initialized",
            kelly_fraction=config.trading.kelly_fraction,
            max_position_pct=config.trading.max_position_pct,
        )

        # Initialize Grok Strategies
        self.strategies_list = [
            ArbTakerStrategy(config),
            LatencySnipeStrategy(config, oracle),
            SpreadMakerStrategy(config),
            LeggedHedgeStrategy(config)
        ]
        logger.info(f"Initialized {len(self.strategies_list)} sub-strategies")
        self.strategies = {s.name: s for s in self.strategies_list}
        self.metrics = StrategyMetrics()

        # Register for order updates
        self.order_manager.add_listener(self._handle_order_update)

    def _handle_order_update(self, order: Order) -> None:
        """Route updates to the originating strategy."""
        strategy_name = order.strategy_id
        if strategy_name in self.strategies:
            asyncio.create_task(
                self.strategies[strategy_name].on_order_update(
                    order.client_order_id, 
                    order.state.value
                )
            )

    def _get_pending_exposure(self) -> float:
        """Calculate total exposure of active orders."""
        exposure = 0.0
        for order in self.order_manager._orders.values():
            if not order.state.is_terminal:
                exposure += order.remaining_size
        return exposure

    async def run(self):
        """Main strategy loop."""
        self._running = True
        logger.info("Starting Grok Ensemble loop...")
        
        while self._running:
            try:
                # 1. Update Market Scope (15m markets)
                markets = await self.market_discovery.get_active_markets(
                    assets=self.config.trading.assets
                )
                
                # 2. Cycle Markets
                for market in markets:
                    if not self.circuit_breaker.can_trade():
                        logger.warning("Circuit breaker active")
                        break
                        
                    await self.process_market(market)
                    await asyncio.sleep(random.uniform(0.1, 0.5)) 

                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error("Error in ensemble loop", error=str(e))
                await asyncio.sleep(5.0)

    async def process_market(self, market: Market):
        """Process a single market across all strategies."""
        spot_price = await self.oracle.get_price(market.asset)
        if spot_price is None:
            return

        book_yes = None
        book_no = None

        if self.clob_client:
            # Fetch both YES and NO order books in parallel for arbitrage detection
            try:
                tasks = [self.clob_client.get_order_book(market.yes_token_id)]
                if market.no_token_id:
                    tasks.append(self.clob_client.get_order_book(market.no_token_id))

                results = await asyncio.gather(*tasks, return_exceptions=True)

                if not isinstance(results[0], Exception):
                    book_yes = results[0]

                if len(results) > 1 and not isinstance(results[1], Exception):
                    book_no = results[1]

            except Exception as e:
                logger.warning(f"Failed to fetch order books: {e}")

        # Cache order books for reference
        if book_yes:
            self._order_book_cache[market.yes_token_id] = book_yes
        if book_no:
            self._order_book_cache[market.no_token_id] = book_no

        context = TradeContext(
            market=market,
            spot_price=spot_price,
            order_book=book_yes,
            order_book_no=book_no,
            open_exposure=self.portfolio.total_exposure,
            daily_pnl=self.portfolio.get_daily_performance().total_pnl,
        )

        all_orders = []
        for strategy in self.strategies_list:
            try:
                orders = await strategy.scan(context)
                if orders:
                    for o in orders:
                        o['strategy'] = strategy.name
                    all_orders.extend(orders)
            except Exception as e:
                logger.error(f"Strategy {strategy.name} failed", error=str(e))

        for order_params in all_orders:
            await self.execute_order_params(order_params, market)

    async def execute_order_params(self, params: dict, market: Market):
        """Execute trade check and submission with Kelly-optimal sizing."""
        if not self.circuit_breaker.can_enter_new_position():
            return

        price = params.get('price')
        if not price or price <= 0 or price >= 1:
            return

        strategy_name = params.get('strategy', 'ensemble')

        # Calculate current drawdown for adaptive Kelly
        current_drawdown = 0.0
        if self.portfolio.peak_capital > 0:
            current_drawdown = (self.portfolio.peak_capital - self.portfolio.current_capital) / self.portfolio.peak_capital

        # Get current asset exposure for correlation adjustment
        current_asset_exposure = sum(
            pos.size_usd for pos in self.portfolio.positions.values()
            if pos.asset == market.asset and not pos.closed
        )

        # Calculate Kelly-optimal position size
        # For arbitrage, we assume higher edge and confidence
        is_arb = 'arb' in strategy_name.lower()
        if is_arb:
            # Arbitrage has near-certain edge
            win_prob = 0.98
            edge_confidence = 0.95
        else:
            # For other strategies, estimate edge from price
            # If we're buying at price P, our implied win prob is slightly higher
            edge_estimate = params.get('edge', 0.04)  # Default 4% edge
            win_prob = price + edge_estimate
            win_prob = max(0.01, min(0.99, win_prob))
            edge_confidence = 0.6  # Moderate confidence for non-arb

        # Get current volatility from config defaults
        current_vol = self.config.models.default_volatility.get(market.asset, 0.70)

        # Calculate optimal size using Kelly
        kelly_result = self.kelly_sizer.calculate(
            win_prob=win_prob,
            market_price=price,
            side="YES" if params.get('side', 'BUY') == 'BUY' else "NO",
            current_asset_exposure=current_asset_exposure,
            current_volatility=current_vol,
            current_drawdown_pct=current_drawdown,
            edge_confidence=edge_confidence,
        )

        # Get the recommended size (already capped by Kelly)
        kelly_size = kelly_result.recommended_size_usd

        # Apply strategy-specific sizing if provided, but cap at Kelly recommendation
        strategy_size = params.get('size', kelly_size)
        final_size = min(strategy_size, kelly_size, 5.0)  # Grok spec: $5 max

        # Minimum viable order size
        if final_size < self.config.trading.min_order_size_usd:
            logger.debug(
                "Order too small after Kelly sizing",
                kelly_size=kelly_size,
                final_size=final_size,
                min_size=self.config.trading.min_order_size_usd,
            )
            return

        # Check total exposure cap ($20 Grok spec)
        current_exposure = self.portfolio.total_exposure
        pending_exposure = self._get_pending_exposure()

        if current_exposure + pending_exposure + final_size > 20.0:
            # Reduce size to fit within cap
            available = 20.0 - current_exposure - pending_exposure
            if available < self.config.trading.min_order_size_usd:
                logger.debug("Max exposure $20 reached")
                return
            final_size = min(final_size, available)

        # Apply correlation adjustment for correlated assets
        open_positions = {
            pos.asset: pos.size_usd
            for pos in self.portfolio.positions.values()
            if not pos.closed
        }
        corr_mult = correlation_matrix.correlation_adjustment(
            market.asset, open_positions, threshold=0.7
        )
        if corr_mult < 1.0:
            final_size *= corr_mult
            logger.debug(
                "Correlation adjustment applied",
                asset=market.asset,
                multiplier=corr_mult,
            )

        # Final size check
        if final_size < self.config.trading.min_order_size_usd:
            return

        order = self.order_manager.create_order(
            token_id=params.get('token_id', market.yes_token_id),
            side=OrderSide.BUY if params.get('side', 'BUY') == 'BUY' else OrderSide.SELL,
            price=price,
            size=final_size,
            order_type=OrderType.GTC,
            strategy_id=strategy_name,
        )

        success = await self.order_manager.submit_order(order)
        if success:
            self.metrics.orders_placed += 1
            self._track_order_metadata(order, market)
            logger.info(
                f"Placed {strategy_name} order",
                asset=market.asset,
                price=price,
                size=final_size,
                kelly_fraction=kelly_result.kelly_fraction,
                kelly_recommended=kelly_result.recommended_size_usd,
            )

    def _track_order_metadata(self, order: Order, market: Market):
        """Track metadata for Portfolio fill handling."""
        # Determine side based on token_id
        side = PositionSide.LONG_YES if order.token_id == market.yes_token_id else PositionSide.LONG_NO
        
        self._pending_fills[order.client_order_id] = PendingFill(
            token_id=order.token_id,
            asset=market.asset,
            side=side,
            price=order.price,
            size_usd=order.size,
            market_question=market.question,
            expires_at=market.end_date.timestamp()
        )

    def handle_fill(self, client_order_id: str, fill_size: float, fill_price: float) -> None:
        """Process fill and update portfolio."""
        fill_usd = float(fill_size) * float(fill_price)
        self.order_manager.handle_fill(client_order_id, fill_usd, fill_price)
        
        metadata = self._pending_fills.get(client_order_id)
        if metadata:
            self.portfolio.open_position(
                token_id=metadata.token_id,
                asset=metadata.asset,
                market_question=metadata.market_question,
                side=metadata.side,
                price=fill_price,
                size_usd=fill_usd,
                expires_at=metadata.expires_at,
            )
            self.metrics.orders_filled += 1
            
            order = self.order_manager.get_order(client_order_id)
            if order and order.state.is_terminal:
                self._pending_fills.pop(client_order_id, None)

    def stop(self) -> None:
        self._running = False

    def get_metrics(self) -> dict:
        return {
            "orders_placed": self.metrics.orders_placed,
            "orders_filled": self.metrics.orders_filled,
            "portfolio": self.portfolio.summary(),
            "circuit_breaker_tripped": not self.circuit_breaker.can_trade(),
        }
