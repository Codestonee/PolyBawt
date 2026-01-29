"""
Value Betting Strategy for 15-minute crypto prediction markets.

Integrates all components:
- Market discovery
- Oracle pricing
- Jump-Diffusion model
- EV calculation
- NO-TRADE gate
- Kelly sizing
- Order execution
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.infrastructure.logging import get_logger, bind_context
from src.infrastructure.config import AppConfig
from src.ingestion.market_discovery import Market, MarketDiscovery
from src.ingestion.oracle_feed import OracleFeed
from src.models.jump_diffusion import JumpDiffusionModel, JumpDiffusionParams, minutes_to_years
from src.models.ev_calculator import EVCalculator, EVResult, TradeSide
from src.models.no_trade_gate import NoTradeGate, GateConfig, TradeContext
from src.risk.kelly_sizer import KellySizer, CorrelationMatrix
from src.risk.circuit_breaker import CircuitBreaker
from src.execution.order_manager import OrderManager, OrderSide, OrderType
from src.execution.rate_limiter import RateLimiter

logger = get_logger(__name__)


@dataclass
class TradeSignal:
    """A potential trade opportunity."""
    
    market: Market
    side: TradeSide
    model_prob: float
    market_price: float
    ev_result: EVResult
    kelly_size: float
    passes_gate: bool
    rejection_reason: str = ""
    
    @property
    def is_actionable(self) -> bool:
        return self.passes_gate and self.kelly_size > 0


@dataclass
class StrategyMetrics:
    """Trading metrics for Brier score and performance tracking."""
    
    # Brier score components
    predictions: list[tuple[float, bool]] = field(default_factory=list)  # (prob, outcome)
    
    # Trade counts
    signals_generated: int = 0
    signals_passed_gate: int = 0
    orders_placed: int = 0
    orders_filled: int = 0
    
    # PnL
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    
    @property
    def brier_score(self) -> float:
        """
        Calculate Brier score (lower is better).
        
        Brier = (1/N) Σ (p_i - o_i)²
        where p_i is prediction and o_i is outcome (0 or 1)
        """
        if not self.predictions:
            return 0.0
        
        total = sum((p - (1 if o else 0)) ** 2 for p, o in self.predictions)
        return total / len(self.predictions)
    
    def add_prediction(self, probability: float, outcome: bool) -> None:
        """Record a prediction and its outcome for Brier score."""
        self.predictions.append((probability, outcome))


class ValueBettingStrategy:
    """
    Value betting strategy for 15-minute markets.
    
    Flow:
    1. Discover active 15m markets
    2. Get oracle prices
    3. Calculate model probabilities
    4. Evaluate EV and check NO-TRADE gate
    5. Size positions with Kelly
    6. Execute trades
    
    Usage:
        strategy = ValueBettingStrategy(config, ...)
        await strategy.run()
    """
    
    def __init__(
        self,
        config: AppConfig,
        market_discovery: MarketDiscovery,
        oracle: OracleFeed,
        order_manager: OrderManager,
        rate_limiter: RateLimiter,
        circuit_breaker: CircuitBreaker,
        bankroll: float = 100.0,
    ):
        self.config = config
        self.market_discovery = market_discovery
        self.oracle = oracle
        self.order_manager = order_manager
        self.rate_limiter = rate_limiter
        self.circuit_breaker = circuit_breaker
        
        # Models
        self.pricing_model = JumpDiffusionModel()
        self.ev_calculator = EVCalculator()
        self.no_trade_gate = NoTradeGate(GateConfig(
            min_edge_threshold=config.trading.min_edge_threshold,
            min_seconds_to_expiry=config.trading.cutoff_seconds,
            min_seconds_after_open=config.trading.min_seconds_after_open,
        ))
        
        # Risk
        self.kelly_sizer = KellySizer(
            bankroll=bankroll,
            kelly_fraction=config.trading.kelly_fraction,
            max_position_pct=config.trading.max_position_pct,
            max_asset_exposure_pct=config.trading.max_asset_exposure_pct,
        )
        self.correlation_matrix = CorrelationMatrix()
        
        # State
        self.metrics = StrategyMetrics()
        self._open_positions: dict[str, float] = {}  # asset -> exposure
        self._running = False
    
    async def analyze_market(
        self,
        market: Market,
        spot_price: float,
    ) -> TradeSignal | None:
        """
        Analyze a single market for trading opportunity.
        
        Args:
            market: Market to analyze
            spot_price: Current spot price for the asset
        
        Returns:
            TradeSignal if opportunity found, None otherwise
        """
        # Get model parameters for this asset
        params = JumpDiffusionParams.for_asset(market.asset)
        
        # Calculate model probability
        time_years = minutes_to_years(market.minutes_to_expiry)
        
        # Assume initial price = spot price at interval start
        # In reality, we'd track the actual opening price
        initial_price = spot_price  # Simplified for MVP
        
        model_prob = self.pricing_model.prob_up(
            spot=spot_price,
            initial=initial_price,
            time_years=time_years,
            params=params,
        )
        
        # Calculate EV
        ev_result = self.ev_calculator.calculate(
            model_prob=model_prob,
            market_price=market.yes_price,
            size_usd=10.0,  # Placeholder, Kelly will override
        )
        
        self.metrics.signals_generated += 1
        
        # Check NO-TRADE gate
        context = TradeContext(
            ev_result=ev_result,
            asset=market.asset,
            token_id=market.yes_token_id,
            seconds_to_expiry=market.seconds_to_expiry,
            oracle_age_seconds=self.oracle.get_cached_price(market.asset).age_seconds if self.oracle.get_cached_price(market.asset) else 0,
            rate_limit_usage_pct=self.rate_limiter.order_usage_pct(),
            trading_halted=not self.circuit_breaker.can_trade(),
            correlation_with_portfolio=self.correlation_matrix.max_correlation_with(
                market.asset, self._open_positions
            ),
        )
        
        gate_result = self.no_trade_gate.evaluate(context)
        
        # Calculate Kelly size
        kelly_result = self.kelly_sizer.calculate(
            win_prob=model_prob if ev_result.side == TradeSide.BUY_YES else 1 - model_prob,
            market_price=market.yes_price if ev_result.side == TradeSide.BUY_YES else 1 - market.yes_price,
            side="YES" if ev_result.side == TradeSide.BUY_YES else "NO",
            current_asset_exposure=self._open_positions.get(market.asset, 0),
        )
        
        # Apply correlation adjustment
        corr_adj = self.correlation_matrix.correlation_adjustment(
            market.asset, self._open_positions
        )
        adjusted_size = kelly_result.recommended_size_usd * corr_adj
        
        signal = TradeSignal(
            market=market,
            side=ev_result.side,
            model_prob=model_prob,
            market_price=market.yes_price,
            ev_result=ev_result,
            kelly_size=adjusted_size,
            passes_gate=gate_result.passed,
            rejection_reason=gate_result.rejection_reason.value if gate_result.rejection_reason else "",
        )
        
        if gate_result.passed:
            self.metrics.signals_passed_gate += 1
        
        return signal
    
    async def execute_signal(self, signal: TradeSignal) -> bool:
        """
        Execute a trade signal.
        
        Args:
            signal: Signal to execute
        
        Returns:
            True if order placed successfully
        """
        if not signal.is_actionable:
            return False
        
        if not self.circuit_breaker.can_enter_new_position():
            logger.info("Circuit breaker blocking new positions")
            return False
        
        # Determine token and side
        if signal.side == TradeSide.BUY_YES:
            token_id = signal.market.yes_token_id
            order_side = OrderSide.BUY
            price = signal.market.yes_price
        else:
            token_id = signal.market.no_token_id
            order_side = OrderSide.BUY
            price = 1 - signal.market.yes_price
        
        # Create order
        order = self.order_manager.create_order(
            token_id=token_id,
            side=order_side,
            price=price,
            size=signal.kelly_size,
            strategy_id="value_betting",
        )
        
        # Submit
        success = await self.order_manager.submit_order(order)
        
        if success:
            self.metrics.orders_placed += 1
            # Track exposure
            self._open_positions[signal.market.asset] = (
                self._open_positions.get(signal.market.asset, 0) + signal.kelly_size
            )
            
            logger.info(
                "Order placed",
                asset=signal.market.asset,
                side=signal.side.value,
                price=price,
                size=signal.kelly_size,
                model_prob=signal.model_prob,
                net_ev=signal.ev_result.net_ev,
            )
        
        return success
    
    async def run_iteration(self) -> None:
        """Run single iteration of the strategy loop."""
        # Check circuit breakers
        if not self.circuit_breaker.can_trade():
            logger.warning("Trading halted by circuit breaker")
            return
        
        # Get markets
        markets = await self.market_discovery.get_crypto_15m_markets(
            assets=self.config.trading.assets
        )
        
        if not markets:
            logger.debug("No active 15m markets")
            return
        
        # Get prices
        prices = await self.oracle.get_all_prices(self.config.trading.assets)
        
        # Analyze and trade each market
        for market in markets:
            spot_price = prices.get(market.asset)
            if spot_price is None:
                continue
            
            # Analyze
            signal = await self.analyze_market(market, spot_price)
            if signal is None:
                continue
            
            # Log opportunity
            if signal.passes_gate:
                logger.info(
                    "Trade opportunity",
                    asset=market.asset,
                    side=signal.side.value,
                    model_prob=f"{signal.model_prob:.3f}",
                    market_price=f"{signal.market_price:.3f}",
                    edge=f"{signal.ev_result.gross_edge:.3f}",
                    net_ev=f"${signal.ev_result.net_ev:.4f}",
                    kelly_size=f"${signal.kelly_size:.2f}",
                )
                
                # Execute
                await self.execute_signal(signal)
            else:
                logger.debug(
                    "Signal rejected",
                    asset=market.asset,
                    reason=signal.rejection_reason,
                )
    
    async def run(self, max_iterations: int | None = None) -> None:
        """
        Run the strategy loop.
        
        Args:
            max_iterations: Stop after N iterations (None = run forever)
        """
        self._running = True
        iteration = 0
        
        logger.info(
            "Strategy starting",
            dry_run=self.config.dry_run,
            bankroll=self.kelly_sizer.bankroll,
            kelly_fraction=self.config.trading.kelly_fraction,
        )
        
        try:
            while self._running:
                iteration += 1
                bind_context(iteration=iteration)
                
                try:
                    await self.run_iteration()
                except Exception as e:
                    logger.error("Iteration error", error=str(e))
                
                if max_iterations and iteration >= max_iterations:
                    break
                
                # Sleep between iterations
                await asyncio.sleep(5)
                
        except asyncio.CancelledError:
            logger.info("Strategy cancelled")
        finally:
            self._running = False
            logger.info(
                "Strategy stopped",
                iterations=iteration,
                signals=self.metrics.signals_generated,
                passed_gate=self.metrics.signals_passed_gate,
                orders=self.metrics.orders_placed,
            )
    
    def stop(self) -> None:
        """Stop the strategy loop."""
        self._running = False
    
    def get_metrics(self) -> dict[str, Any]:
        """Get current strategy metrics."""
        return {
            "signals_generated": self.metrics.signals_generated,
            "signals_passed_gate": self.metrics.signals_passed_gate,
            "orders_placed": self.metrics.orders_placed,
            "orders_filled": self.metrics.orders_filled,
            "brier_score": self.metrics.brier_score,
            "realized_pnl": self.metrics.realized_pnl,
            "open_positions": dict(self._open_positions),
            "order_stats": self.order_manager.stats,
        }
