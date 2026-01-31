"""Event betting strategy for non-crypto prediction markets."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.infrastructure.logging import get_logger, bind_context
from src.infrastructure.config import AppConfig
from src.ingestion.event_market_discovery import (
    EventMarket,
    EventMarketDiscovery,
    MarketCategory
)
from src.models.event_probability import (
    EventEnsembleModel,
    EnsembleResult,
    PoliticsProbabilityModel,
    SportsProbabilityModel,
    EconomicsProbabilityModel,
    PopCultureProbabilityModel,
)
from src.models.ev_calculator import EVCalculator, EVResult, TradeSide
from src.models.no_trade_gate import NoTradeGate, GateConfig, TradeContext
from src.risk.kelly_sizer import KellySizer, CorrelationMatrix
from src.risk.vpin import VPINCalculator
from src.strategy.features.order_imbalance import OrderBookImbalance, OBIResult
from src.execution.order_manager import OrderManager, OrderSide, OrderType
from src.execution.rate_limiter import RateLimiter
from src.execution.clob_client import CLOBClient
from src.risk.circuit_breaker import CircuitBreaker
from src.portfolio.tracker import Portfolio, PositionSide
from src.calibration.brier_tracker import BrierTracker
from src.calibration.calibration_database import CalibrationDatabase, PredictionRecord

logger = get_logger(__name__)


@dataclass
class EventTradeSignal:
    """A potential event market trade."""
    
    market: EventMarket
    side: TradeSide
    
    # Probability estimates
    ensemble_result: EnsembleResult
    model_prob: float
    market_price: float
    
    # EV analysis
    ev_result: EVResult
    
    # Position sizing
    kelly_size: float
    
    # Gate status
    passes_gate: bool
    rejection_reason: str = ""
    
    # Metadata
    category: MarketCategory = field(default=MarketCategory.OTHER)
    prediction_id: str = ""
    
    @property
    def is_actionable(self) -> bool:
        """Can this trade be executed?"""
        return self.passes_gate and self.kelly_size is not None and self.kelly_size > 0
    
    @property
    def edge(self) -> float:
        """Edge over market price."""
        return self.model_prob - self.market_price


@dataclass
class EventStrategyMetrics:
    """Metrics for event betting strategy."""
    
    # Trade counts
    markets_analyzed: int = 0
    signals_generated: int = 0
    signals_passed_gate: int = 0
    orders_placed: int = 0
    orders_filled: int = 0
    
    # By category
    category_counts: dict[str, int] = field(default_factory=lambda: {
        cat.value: 0 for cat in MarketCategory
    })
    
    # Brier tracking
    brier_tracker: BrierTracker = field(default_factory=BrierTracker)
    
    # Timestamp
    last_reset: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def reset(self) -> None:
        """Reset metrics."""
        self.markets_analyzed = 0
        self.signals_generated = 0
        self.signals_passed_gate = 0
        self.orders_placed = 0
        self.orders_filled = 0
        self.category_counts = {cat.value: 0 for cat in MarketCategory}
        self.last_reset = datetime.now(timezone.utc)


@dataclass
class PendingEventFill:
    """Metadata for pending event market order."""
    prediction_id: str
    market_id: str
    token_id: str
    category: MarketCategory
    side: PositionSide
    price: float
    size_usd: float
    predicted_probability: float
    market_question: str
    expires_at: float


class EventBettingStrategy:
    """
    Event market betting strategy for Politics, Sports, Economics, Pop Culture.
    
    Key differences from crypto strategy:
    - External data instead of mathematical models
    - Longer time horizons (days/weeks vs 15 min)
    - Position rebalancing on new information
    - Brier score tracking for model calibration
    
    Architecture:
        Event Market Discovery → Category Router → Probability Model → 
        EV Calculator → NO-TRADE Gate → Kelly Sizer → Execution
    
    Reused components:
    - EV Calculator (same calculation)
    - NO-TRADE Gate (same filtering logic)
    - Kelly Sizer (same position sizing)
    - VPIN (toxicity detection)
    - Order Book Imbalance (65-75% accuracy!)
    - Portfolio tracker
    
    Usage:
        strategy = EventBettingStrategy(config, ...)
        await strategy.run()
    """
    
    # Category-specific minimum edge thresholds
    DEFAULT_MIN_EDGE = {
        MarketCategory.POLITICS: 0.05,      # 5% minimum edge
        MarketCategory.SPORTS: 0.06,        # 6% for sports
        MarketCategory.ECONOMICS: 0.04,     # 4% for economics (more predictable)
        MarketCategory.POP_CULTURE: 0.07,   # 7% for pop culture (higher uncertainty)
    }
    
    # Refresh intervals for external data
    REFRESH_INTERVALS = {
        "polls": 3600,        # 1 hour
        "betting_lines": 900, # 15 minutes
        "fed_futures": 900,   # 15 minutes
        "sentiment": 1800,    # 30 minutes
    }
    
    def __init__(
        self,
        config: AppConfig,
        market_discovery: EventMarketDiscovery,
        order_manager: OrderManager,
        rate_limiter: RateLimiter,
        circuit_breaker: CircuitBreaker,
        clob_client: CLOBClient | None = None,
        portfolio: Portfolio | None = None,
        bankroll: float = 1000.0,
    ):
        self.config = config
        self.market_discovery = market_discovery
        self.order_manager = order_manager
        self.rate_limiter = rate_limiter
        self.circuit_breaker = circuit_breaker
        self.clob_client = clob_client
        
        # Portfolio tracking
        self.portfolio = portfolio or Portfolio(starting_capital=bankroll)
        self._pending_fills: dict[str, PendingEventFill] = {}
        
        # Probability models
        self.ensemble_model = EventEnsembleModel()
        self.category_models = {
            MarketCategory.POLITICS: PoliticsProbabilityModel(),
            MarketCategory.SPORTS: SportsProbabilityModel(),
            MarketCategory.ECONOMICS: EconomicsProbabilityModel(),
            MarketCategory.POP_CULTURE: PopCultureProbabilityModel(),
        }
        
        # Reused components from crypto strategy
        self.ev_calculator = EVCalculator()
        self.no_trade_gate = NoTradeGate(GateConfig(
            min_edge_threshold=config.trading.min_edge_threshold,
            min_seconds_to_expiry=3600,  # 1 hour minimum
            min_seconds_after_open=300,   # 5 minutes after open
        ))
        self.kelly_sizer = KellySizer(
            bankroll=bankroll,
            kelly_fraction=config.trading.kelly_fraction,
            max_position_pct=config.trading.max_position_pct,
            max_asset_exposure_pct=config.trading.max_asset_exposure_pct,
        )
        self.vpin_calculator = VPINCalculator()
        self.orderbook_imbalance = OrderBookImbalance()
        self.correlation_matrix = CorrelationMatrix()
        
        # Calibration
        self.brier_tracker = BrierTracker()
        self.calibration_db = CalibrationDatabase()
        
        # Metrics
        self.metrics = EventStrategyMetrics()
        
        # State
        self._running = False
        self._traded_markets: dict[str, float] = {}  # market_id -> timestamp
        self._last_reconciliation = 0.0
        self._reconciliation_interval = 60.0
        
        # Order book cache
        self._order_book_cache: dict[str, Any] = {}
        
        # Rebalancing
        self._active_positions: dict[str, dict[str, Any]] = {}  # market_id -> position info
    
    def _get_min_edge(self, category: MarketCategory) -> float:
        """Get minimum edge threshold for category."""
        return self.DEFAULT_MIN_EDGE.get(category, 0.05)
    
    async def analyze_market(
        self,
        market: EventMarket,
        category_context: Any | None = None,
    ) -> EventTradeSignal | None:
        """
        Analyze an event market for trading opportunity.
        
        Args:
            market: Event market to analyze
            category_context: Category-specific context (polls, stats, etc.)
            
        Returns:
            EventTradeSignal if opportunity found, None otherwise
        """
        self.metrics.markets_analyzed += 1
        self.metrics.category_counts[market.category.value] += 1
        
        # Get order book for OBI signal
        obi_result = await self._fetch_obi(market)
        
        # Get VPIN for toxicity check
        vpin = self.vpin_calculator.get_vpin(market.condition_id)
        
        # Skip if toxic
        if vpin.should_halt:
            logger.debug(
                "Skipping toxic market",
                market=market.question[:50],
                vpin=vpin.vpin,
            )
            return None
        
        # Ensemble probability estimate
        ensemble_result = await self.ensemble_model.estimate(
            market=market,
            category=market.category,
            obi_result=obi_result,
            context=category_context,
        )
        
        model_prob = ensemble_result.probability
        
        # Record prediction for Brier tracking
        prediction_id = self.brier_tracker.record_prediction(
            market_id=market.condition_id,
            category=market.category,
            probability=model_prob,
        )
        
        # Calculate EV
        ev_result = self.ev_calculator.calculate(
            model_prob=model_prob,
            market_price=market.yes_price,
            size_usd=10.0,  # Placeholder, Kelly will size
        )
        
        self.metrics.signals_generated += 1
        
        # Get order book data for gate
        spread = 0.0
        book_depth = 1000.0
        
        if self.clob_client and market.yes_token_id:
            try:
                book = await self._get_cached_order_book(market.yes_token_id)
                spread = book.spread if hasattr(book, 'spread') else 0.0
                book_depth = book.ask_depth_usd if hasattr(book, 'ask_depth_usd') else 1000.0
            except Exception as e:
                logger.debug("Failed to fetch order book", error=str(e))
        
        # Check NO-TRADE gate
        context = TradeContext(
            ev_result=ev_result,
            asset=market.category.value,  # Use category as "asset"
            token_id=market.yes_token_id or "",
            seconds_to_expiry=market.time_to_resolution * 3600 if market.time_to_resolution else 86400,
            seconds_since_open=3600,  # Assume 1 hour for event markets
            spread=spread,
            book_depth_usd=book_depth,
            oracle_age_seconds=60,  # External data is fresh
            rate_limit_usage_pct=self.rate_limiter.order_usage_pct(),
            trading_halted=not self.circuit_breaker.can_trade(),
            correlation_with_portfolio=0.0,  # Event markets have low correlation
            override_min_edge=self._get_min_edge(market.category),
        )
        
        gate_result = self.no_trade_gate.evaluate(context)
        
        # Calculate Kelly size
        kelly_result = self.kelly_sizer.calculate(
            win_prob=model_prob if ev_result.side == TradeSide.BUY_YES else 1 - model_prob,
            market_price=market.yes_price if ev_result.side == TradeSide.BUY_YES else 1 - market.yes_price,
            side="YES" if ev_result.side == TradeSide.BUY_YES else "NO",
            current_asset_exposure=self.portfolio.get_open_positions().get(
                market.category.value, 0
            ),
        )
        
        # Apply VPIN adjustment
        size_multiplier = vpin.size_multiplier
        adjusted_size = kelly_result.recommended_size_usd * size_multiplier
        
        signal = EventTradeSignal(
            market=market,
            side=ev_result.side,
            ensemble_result=ensemble_result,
            model_prob=model_prob,
            market_price=market.yes_price,
            ev_result=ev_result,
            kelly_size=adjusted_size,
            passes_gate=gate_result.passed,
            rejection_reason=gate_result.rejection_reason.value if gate_result.rejection_reason else "",
            category=market.category,
            prediction_id=prediction_id,
        )
        
        if gate_result.passed:
            self.metrics.signals_passed_gate += 1
        
        return signal
    
    async def execute_signal(self, signal: EventTradeSignal) -> bool:
        """
        Execute an event market trade signal.
        
        Args:
            signal: Trade signal to execute
            
        Returns:
            True if order placed successfully
        """
        if not signal.is_actionable:
            return False
        
        # Check for duplicate trades
        if signal.market.condition_id in self._traded_markets:
            logger.info(
                "Skipping duplicate trade",
                market=signal.market.question[:50],
            )
            return False
        
        if not self.circuit_breaker.can_enter_new_position():
            logger.info("Circuit breaker blocking new positions")
            return False
        
        # Determine token and side
        token_id = (
            signal.market.yes_token_id
            if signal.side == TradeSide.BUY_YES
            else signal.market.no_token_id
        )
        
        if not token_id:
            logger.warning("No token ID for market", market=signal.market.question[:50])
            return False
        
        # Get execution price
        price = await self._get_execution_price(token_id, signal.side, signal.market)
        
        # Create order
        order = self.order_manager.create_order(
            token_id=token_id,
            side=OrderSide.BUY,
            price=price,
            size=signal.kelly_size,
            strategy_id="event_betting",
        )
        
        # Submit
        success = await self.order_manager.submit_order(order)
        
        if success:
            self.metrics.orders_placed += 1
            
            # Track pending fill
            self._pending_fills[order.client_order_id] = PendingEventFill(
                prediction_id=signal.prediction_id,
                market_id=signal.market.condition_id,
                token_id=token_id,
                category=signal.category,
                side=PositionSide.LONG_YES if signal.side == TradeSide.BUY_YES else PositionSide.LONG_NO,
                price=price,
                size_usd=signal.kelly_size,
                predicted_probability=signal.model_prob,
                market_question=signal.market.question,
                expires_at=signal.market.end_date.timestamp() if signal.market.end_date else 0,
            )
            
            # Record that we traded this market
            self._traded_markets[signal.market.condition_id] = datetime.now(timezone.utc).timestamp()
            
            # Record prediction in calibration DB
            await self._record_prediction_in_db(signal, order.client_order_id)
            
            logger.info(
                "Event market order placed",
                market=signal.market.question[:50],
                category=signal.category.value,
                side=signal.side.value,
                price=price,
                size=signal.kelly_size,
                model_prob=signal.model_prob,
                edge=signal.edge,
            )
        
        return success
    
    async def _record_prediction_in_db(self, signal: EventTradeSignal, order_id: str) -> None:
        """Record prediction in calibration database."""
        try:
            record = PredictionRecord(
                prediction_id=signal.prediction_id,
                market_id=signal.market.condition_id,
                market_question=signal.market.question,
                category=signal.category.value,
                predicted_probability=signal.model_prob,
                model_name="event_ensemble",
                features={
                    "ensemble_weights": signal.ensemble_result.weights_used.__dict__ if hasattr(signal.ensemble_result, 'weights_used') else {},
                    "component_probs": signal.ensemble_result.component_probs if hasattr(signal.ensemble_result, 'component_probs') else {},
                    "obi_adjustment": signal.ensemble_result.obi_adjustment if hasattr(signal.ensemble_result, 'obi_adjustment') else 0.0,
                },
                market_price=signal.market_price,
                time_to_resolution_hours=signal.market.time_to_resolution,
                predicted_at=datetime.now(timezone.utc),
            )
            
            # Use async version to avoid blocking event loop
            await self.calibration_db.arecord_prediction(record)
            
        except Exception as e:
            logger.warning("Failed to record prediction", error=str(e))
    
    async def check_rebalancing(self) -> None:
        """
        Check for position rebalancing opportunities.
        
        Re-estimate probability when new data arrives.
        Exit/reduce if edge disappears (>10% probability change).
        """
        if not self.config.event_trading.rebalance_enabled:
            return
        
        rebalanced_count = 0
        threshold = self.config.event_trading.rebalance_threshold_pct
        
        # Get current positions from portfolio
        positions = self.portfolio.get_open_positions()
        
        for position in positions:
            # Skip if position doesn't have our metadata
            if not hasattr(position, 'predicted_probability'):
                continue
            
            original_prob = position.predicted_probability
            market_id = position.market_id
            
            # Fetch current market state
            try:
                # Get updated market data
                market = await self._fetch_market_by_id(market_id)
                if not market:
                    continue
                
                # Re-estimate probability with fresh external data
                category_context = await self._fetch_category_context(market)
                ensemble_result = await self.ensemble_model.estimate(
                    market=market,
                    category=market.category,
                    context=category_context,
                )
                
                new_prob = ensemble_result.probability
                prob_change = abs(new_prob - original_prob)
                
                # Check if edge has disappeared
                if prob_change > threshold:
                    logger.info(
                        "Rebalancing position - edge disappeared",
                        market=market.question[:50],
                        original_prob=original_prob,
                        new_prob=new_prob,
                        change=prob_change,
                        threshold=threshold,
                    )
                    
                    # Close position
                    success = await self._close_position(position, market)
                    if success:
                        rebalanced_count += 1
                        
                        # Record rebalancing in calibration
                        self.calibration_db.record_outcome(
                            market_id=market_id,
                            outcome=None,  # Not resolved yet
                            metadata={
                                "action": "rebalance_close",
                                "original_prob": original_prob,
                                "new_prob": new_prob,
                                "change": prob_change,
                            }
                        )
                
            except Exception as e:
                logger.error(
                    "Error during rebalancing check",
                    market_id=market_id,
                    error=str(e),
                )
        
        if rebalanced_count > 0:
            logger.info(
                "Rebalancing complete",
                positions_closed=rebalanced_count,
            )
    
    async def _fetch_market_by_id(self, market_id: str) -> EventMarket | None:
        """Fetch market by ID for rebalancing."""
        # Search in all categories
        for category in MarketCategory:
            if category == MarketCategory.CRYPTO:
                continue
            try:
                markets = await self.market_discovery.get_markets_by_category(category)
                for market in markets:
                    if market.condition_id == market_id:
                        return market
            except Exception:
                continue
        return None
    
    async def _close_position(
        self,
        position,
        market: EventMarket,
    ) -> bool:
        """
        Close a position by selling the held outcome tokens.
        
        Args:
            position: Position to close
            market: Current market state
            
        Returns:
            True if close order submitted successfully
        """
        try:
            # Determine which token to sell
            if position.side == PositionSide.LONG_YES:
                token_id = market.yes_token_id
                current_price = market.yes_price
            else:
                token_id = market.no_token_id
                current_price = 1 - market.yes_price
            
            if not token_id:
                logger.warning("No token ID for position", market_id=market.condition_id)
                return False
            
            # Calculate size to close (full position)
            close_size = position.size_usd
            
            # Create close order (sell at market or best bid)
            order = self.order_manager.create_order(
                token_id=token_id,
                side=OrderSide.SELL,
                price=current_price * 0.995,  # Slightly below market for faster fill
                size=close_size,
                order_type=OrderType.GTC,
                strategy_id="event_rebalance",
            )
            
            # Submit
            success = await self.order_manager.submit_order(order)
            
            if success:
                logger.info(
                    "Position close order submitted",
                    market=market.question[:50],
                    side=position.side.value,
                    size=close_size,
                    price=current_price,
                )
            
            return success
            
        except Exception as e:
            logger.error(
                "Failed to close position",
                market_id=market.condition_id,
                error=str(e),
            )
            return False
    
    async def handle_market_resolution(self, market_id: str, outcome: bool) -> None:
        """
        Handle a market resolution.
        
        Args:
            market_id: Market that resolved
            outcome: True if YES, False if NO
        """
        # Record outcome for Brier tracking
        self.brier_tracker.record_outcome(market_id, outcome)
        
        # Record in calibration DB
        self.calibration_db.record_outcome(market_id, outcome)
        
        # Update portfolio
        for order_id, pending in list(self._pending_fills.items()):
            if pending.market_id == market_id:
                # Calculate PnL
                # For event markets, realized PnL is either +return or -1
                if pending.side == PositionSide.LONG_YES:
                    pnl = (1 / pending.price - 1) * pending.size_usd if outcome else -pending.size_usd
                else:
                    pnl = (1 / (1 - pending.price) - 1) * pending.size_usd if not outcome else -pending.size_usd
                
                logger.info(
                    "Event market resolved",
                    market_id=market_id,
                    outcome=outcome,
                    predicted_prob=pending.predicted_probability,
                    pnl=pnl,
                )
                
                del self._pending_fills[order_id]
    
    async def _fetch_obi(self, market: EventMarket) -> OBIResult | None:
        """Fetch order book imbalance for market."""
        if not self.clob_client or not market.yes_token_id:
            return None
        
        try:
            book = await self._get_cached_order_book(market.yes_token_id)
            
            # Extract bids and asks
            bids = getattr(book, 'bids', [])
            asks = getattr(book, 'asks', [])
            
            return self.orderbook_imbalance.calculate(
                bids=bids,
                asks=asks,
            )
            
        except Exception as e:
            logger.debug("Failed to fetch OBI", error=str(e))
            return None
    
    async def _get_cached_order_book(self, token_id: str) -> Any:
        """Get order book from cache or fetch."""
        if token_id in self._order_book_cache:
            return self._order_book_cache[token_id]
        
        if not self.clob_client:
            raise RuntimeError("CLOB client not initialized")
        
        book = await self.clob_client.get_order_book(token_id)
        self._order_book_cache[token_id] = book
        return book
    
    async def _get_execution_price(
        self,
        token_id: str,
        side: TradeSide,
        market: EventMarket
    ) -> float:
        """Determine execution price."""
        price = None
        
        if self.clob_client:
            try:
                book = await self._get_cached_order_book(token_id)
                price = getattr(book, 'best_ask', None)
            except Exception as e:
                logger.debug("Failed to get execution price", error=str(e))
        
        if price is None:
            # Fallback to market price
            price = market.yes_price if side == TradeSide.BUY_YES else (1 - market.yes_price)
        
        return price
    
    def get_brier_report(self) -> dict[str, Any]:
        """Get Brier score calibration report."""
        return self.brier_tracker.get_calibration_report()
    
    async def run_iteration(self, category: MarketCategory | None = None) -> None:
        """
        Run single iteration of event strategy.
        
        Args:
            category: If set, only analyze this category
        """
        # Clear caches
        self._order_book_cache.clear()
        
        # Cleanup old trades
        self._cleanup_traded_markets()
        
        # Check circuit breaker
        if not self.circuit_breaker.can_trade():
            logger.warning("Trading halted by circuit breaker")
            return
        
        # Get markets
        if category:
            markets = await self.market_discovery.get_markets_by_category(category)
        else:
            markets = await self.market_discovery.get_all_event_markets()
        
        if not markets:
            logger.debug("No event markets found")
            return
        
        logger.info(f"Analyzing {len(markets)} event markets")
        
        # Analyze each market
        for market in markets:
            try:
                # Skip if already traded
                if market.condition_id in self._traded_markets:
                    continue
                
                # Get category context
                category_context = await self._fetch_category_context(market)
                
                # Analyze
                signal = await self.analyze_market(market, category_context)
                
                if signal and signal.passes_gate:
                    logger.info(
                        "Event trade opportunity",
                        market=market.question[:50],
                        category=market.category.value,
                        prob=signal.model_prob,
                        edge=signal.edge,
                        size=signal.kelly_size,
                    )
                    
                    await self.execute_signal(signal)
                    
            except Exception as e:
                logger.error(
                    "Error analyzing market",
                    market=market.question[:50],
                    error=str(e),
                )
    
    async def _fetch_category_context(self, market: EventMarket) -> Any:
        """Fetch category-specific context for market."""
        # This would integrate with external data sources
        # For now, return None (models will fetch their own data)
        return None
    
    def _cleanup_traded_markets(self) -> None:
        """Remove old entries from traded markets."""
        now = datetime.now(timezone.utc).timestamp()
        # Keep for 30 days
        cutoff = now - 30 * 86400
        
        expired = [
            mid for mid, ts in self._traded_markets.items()
            if ts < cutoff
        ]
        
        for mid in expired:
            del self._traded_markets[mid]
    
    async def run(
        self,
        category: MarketCategory | None = None,
        max_iterations: int | None = None
    ) -> None:
        """
        Run the event strategy loop.
        
        Args:
            category: If set, only trade this category
            max_iterations: Stop after N iterations
        """
        self._running = True
        iteration = 0
        
        logger.info(
            "Event betting strategy starting",
            category=category.value if category else "all",
            bankroll=self.kelly_sizer.bankroll,
        )
        
        try:
            while self._running:
                iteration += 1
                bind_context(iteration=iteration)
                
                try:
                    await self.run_iteration(category)
                    
                    # Check for rebalancing opportunities
                    await self.check_rebalancing()
                    
                except Exception as e:
                    logger.error("Iteration error", error=str(e))
                
                if max_iterations and iteration >= max_iterations:
                    break
                
                # Sleep between iterations (longer than crypto - event markets change slower)
                await asyncio.sleep(60)  # 1 minute
                
        except asyncio.CancelledError:
            logger.info("Strategy cancelled")
        finally:
            self._running = False
            logger.info(
                "Event strategy stopped",
                iterations=iteration,
                markets_analyzed=self.metrics.markets_analyzed,
                orders_placed=self.metrics.orders_placed,
            )
    
    def stop(self) -> None:
        """Stop the strategy."""
        self._running = False
    
    def get_metrics(self) -> dict[str, Any]:
        """Get current metrics."""
        brier_report = self.get_brier_report()
        
        return {
            "markets_analyzed": self.metrics.markets_analyzed,
            "signals_generated": self.metrics.signals_generated,
            "signals_passed_gate": self.metrics.signals_passed_gate,
            "orders_placed": self.metrics.orders_placed,
            "orders_filled": self.metrics.orders_filled,
            "by_category": self.metrics.category_counts,
            "brier_score": brier_report.get("overall_brier"),
            "calibration_grade": brier_report.get("overall_grade"),
            "portfolio": self.portfolio.summary(),
        }
