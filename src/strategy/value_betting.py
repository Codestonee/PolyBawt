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
from src.ingestion.oracle_feed import OracleFeed, SniperAlert, SniperRiskLevel
# from src.models.jump_diffusion import (
#     JumpDiffusionModel, JumpDiffusionParams, minutes_to_years,
#     logit, inv_logit, edge_in_logit_space,  # Logit utilities from arXiv paper
# )
from src.models.pricing import MertonJDCalibrator  # NEW: Research-based model
from src.models.ev_calculator import EVCalculator, EVResult, TradeSide
from src.models.no_trade_gate import NoTradeGate, GateConfig, TradeContext
from src.risk.kelly_sizer import KellySizer, CorrelationMatrix
from src.risk.circuit_breaker import CircuitBreaker
from src.execution.order_manager import OrderManager, OrderSide, OrderType
from src.execution.rate_limiter import RateLimiter
from src.execution.clob_client import CLOBClient
from src.models.order_book import OrderBook
from src.portfolio.tracker import Portfolio, PositionSide

# Research-based enhancements (arXiv:2510.15205)
from src.strategy.arbitrage_detector import ArbitrageDetector, ArbitrageType, ArbitrageOpportunity
from src.strategy.toxicity_filter import ToxicityFilter, ToxicityResult

# Research-backed enhancements (2026 deep research synthesis)
from src.models.ensemble import ResearchEnsembleModel
from src.models.orderbook_signal import OrderBookSignalModel, OrderBookFeatures
from src.models.sentiment import SentimentIntegrator, SentimentAggregator  # NEW: Sentiment
from src.ingestion.ws_client import WebSocketClient, create_market_ws_client  # NEW: WebSocket
from src.risk.vpin import VPINCalculator  # NEW: VPIN

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
        return self.passes_gate and self.kelly_size is not None and self.kelly_size > 0


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
    strategy_name: str = "Unknown"  # NEW: Track which strategy generated this
    
    @property
    def is_actionable(self) -> bool:
        return self.passes_gate and self.kelly_size is not None and self.kelly_size > 0


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
    
    Manages:
    1. ArbTakerStrategy (Risk-free arb)
    2. LatencySnipeStrategy (Offensive lag snipe)
    3. SpreadMakerStrategy (Passive spread capture)
    4. LeggedHedgeStrategy (Crash & catch)
    
    Responsibility:
    - Discovery & Data Feed
    - Global Risk Gates (Daily Loss, Max Exposure)
    - Execution
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
        self._pending_fills: dict[str, PendingFill] = {}
        
        # Risk & Sizing
        self.kelly_sizer = KellySizer(
            bankroll=bankroll,
            kelly_fraction=config.trading.kelly_fraction,
            max_position_pct=config.trading.max_position_pct,
            max_asset_exposure_pct=config.trading.max_asset_exposure_pct,
        )
        self.correlation_matrix = CorrelationMatrix()

        self.ev_calculator = EVCalculator()
        self.no_trade_gate = NoTradeGate(GateConfig(
            min_edge_threshold=config.trading.min_edge_threshold,
            min_seconds_to_expiry=getattr(config.trading, "cutoff_seconds", 300),
            min_seconds_after_open=config.trading.min_seconds_after_open,
        ))

        # Utilities
        self.arbitrage_detector = ArbitrageDetector(fee_rate=0.02)
        
        # Modular Strategies
        from src.strategy.strategies import (
            ArbTakerStrategy, LatencySnipeStrategy, 
            SpreadMakerStrategy, LeggedHedgeStrategy
        )
        self.strategies = [
            ArbTakerStrategy(config, self.arbitrage_detector),
            LatencySnipeStrategy(config, oracle),
            SpreadMakerStrategy(config),
            LeggedHedgeStrategy(config),
        ]
        
        # State
        self.metrics = StrategyMetrics()
        self._traded_markets: dict[str, float] = {}
        self._order_book_cache: dict[str, "OrderBook"] = {}
        self.ws_client = None
        self._sniper_alert_active: dict[str, SniperAlert] = {}
        self.oracle.register_sniper_callback(self._handle_sniper_alert)
        
        # VPIN / Toxicity (Research)
        self._vpin_calculators: dict[str, VPINCalculator] = {}

    async def _init_websocket(self):
        """Initialize and connect WebSocket client."""
        if not self.ws_client:
            self.ws_client = await create_market_ws_client()
            await self.ws_client.connect()
            logger.info("WebSocket initialized for ensemble strategy")

    async def run(self):
        """Main strategy loop."""
        self._running = True
        logger.info("Starting Ensemble Strategy loop...")
        await self._init_websocket()
        
        while self._running:
            try:
                # 1. Update Market Scope (15m BTC/ETH/SOL/XRP)
                markets = await self.market_discovery.get_active_markets(
                    assets=self.config.trading.assets
                )
                
                # Filter for 15-min markets specifically if needed, 
                # or trust discovery returns what we asked for.
                # Grok: "Filter '15 min' in title"
                
                # 2. Cycle Markets
                for market in markets:
                    # Refresh Risk State
                    if not self.circuit_breaker.can_trade():
                        logger.warning("Circuit breaker active - skipping scan")
                        break
                        
                    await self.process_market(market)
                    
                    # Jitter to avoid bot-like patterns
                    # Grok: "Poll every 1-3s jittered"
                    await asyncio.sleep(random.uniform(0.1, 0.5)) 

                # Cleanup
                self._cleanup_traded_markets()
                
                # Main Loop Sleep
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error("Error in ensemble loop", error=str(e))
                await asyncio.sleep(5.0)

    async def process_market(self, market: Market):
        """Process a single market across all strategies."""
        # 1. Fetch Data (Oracle, Book)
        spot_price = self.oracle.get_price(market.asset)
        if spot_price is None:
            return

        book = None
        if self.clob_client:
            try:
                # Cache book
                token_id = market.yes_token_id
                book = await self._get_cached_order_book(token_id)
            except Exception as e:
                logger.debug("Failed to fetch book", asset=market.asset)
        
        # 2. Build Context
        from src.strategy.base import TradeContext
        context = TradeContext(
            market=market,
            spot_price=spot_price,
            order_book=book,
            open_exposure=self.portfolio.total_exposure_usd,
            daily_pnl=self.portfolio.daily_pnl,
        )
        
        # 3. Scan All Strategies
        all_orders = []
        for strategy in self.strategies:
            try:
                # Check strategy-specific caps?
                # Grok: "25% exposure each" - logic can be inside strategy or here.
                # keeping it simple: scan returns desired orders
                
                orders = await strategy.scan(context)
                if orders:
                    # Enrich with strategy name for tracking
                    for o in orders:
                        o['strategy'] = strategy.name
                    all_orders.extend(orders)
            except Exception as e:
                logger.error("Strategy scan failed", strategy=strategy.name, error=str(e))

        # 4. Execute (with Risk check)
        for order_params in all_orders:
            await self.execute_order_params(order_params, market)

    async def execute_order_params(self, params: dict, market: Market):
        """Create and submit order from simplified params."""
        # Check global limits
        if not self.circuit_breaker.can_enter_new_position():
            return

        # Sizing
        # Grok: "$5 max bet"
        # Strategy might return 'size', or we cap it here.
        recommended_size = params.get('size', 5.0)
        final_size = min(recommended_size, 5.0) # Hard cap $5
        
        # Exposure check
        if self.portfolio.total_exposure_usd + final_size > 20.0: # Hard cap $20
            return

        # Create Order
        token_id = params.get('token_id', market.yes_token_id) # Default to YES if not spec
        side_str = params.get('side', 'BUY')
        price = params.get('price')
        
        if not price: return

        side = OrderSide.BUY if side_str == 'BUY' else OrderSide.SELL
        order_type_str = params.get('order_type', 'GTC')
        order_type = OrderType.GTC
        # Map Maker Types if needed
        
        order = self.order_manager.create_order(
            token_id=token_id,
            side=side,
            price=price,
            size=final_size,
            order_type=order_type,
            strategy_id=params.get('strategy', 'ensemble'),
        )
        
        success = await self.order_manager.submit_order(order)
        if success:
            self.metrics.orders_placed += 1
            logger.info(
                "Ensemble Order Placed",
                strategy=params.get('strategy'),
                asset=market.asset,
                price=price,
                size=final_size
            )

    async def _get_cached_order_book(self, token_id: str) -> "OrderBook":
        """Get order book with short caching."""
        # Simple cache wrapper
        if not self.clob_client:
            raise ValueError("No CLOB client")
            
        now = time.time()
        # Invalidate cache if older than 1s (Grok: "Poll every 1-3s")
        if token_id in self._order_book_cache:
            # Check age? For now, fetch fresh if called in new cycle
            pass 
            
        # For simplicity, always fetch in this step since we cycle markets sequentially
        # and process_market calls this once.
        book = await self.clob_client.get_order_book(token_id)
        self._order_book_cache[token_id] = book
        return book

    def stop(self):
        self._running = False
        logger.info("Ensemble Strategy stopping...")

    # ... (Keep existing helper methods like _cleanup_traded_markets, _handle_sniper_alert)


    def _cleanup_traded_markets(self) -> None:
        """Remove expired markets from the traded set to free memory."""
        now = time.time()
        expired = [q for q, expiry in self._traded_markets.items() if expiry < now]
        for q in expired:
            del self._traded_markets[q]

        if expired:
            logger.debug("Cleaned up expired traded markets", count=len(expired))

    async def _handle_sniper_alert(self, alert: SniperAlert) -> None:
        """
        Handle sniper risk alert from Chainlink price divergence.

        CRITICAL: When Chainlink price diverges significantly from market price,
        HFT bots with faster Chainlink access could front-run our orders.

        Research (gemeni.txt):
        "If the Chainlink price deviates from the Polymarket mid-price by more
        than a threshold, the bot should immediately send a DELETE /cancel-all
        request to pull liquidity."
        """
        self._sniper_alert_active[alert.asset] = alert

        if alert.should_cancel_orders:
            logger.critical(
                "SNIPER PROTECTION: Cancelling all orders due to Chainlink divergence",
                asset=alert.asset,
                chainlink_price=alert.chainlink_price,
                market_price=alert.market_price,
                divergence_pct=f"{alert.divergence_pct:.3f}%",
            )

            # Cancel all open orders for this asset
            try:
                cancelled = await self._cancel_orders_for_asset(alert.asset)
                logger.info(
                    "Sniper protection: orders cancelled",
                    asset=alert.asset,
                    count=cancelled,
                )
            except Exception as e:
                logger.error(
                    "Failed to cancel orders during sniper alert",
                    asset=alert.asset,
                    error=str(e),
                )

    async def _cancel_orders_for_asset(self, asset: str) -> int:
        """
        Cancel all open orders for a specific asset.

        Uses the pending_fills tracking to find orders by asset,
        then cancels them via the order manager.

        Args:
            asset: Asset symbol (BTC, ETH, SOL, XRP)

        Returns:
            Number of orders cancelled
        """
        cancelled = 0

        # Find all pending orders for this asset
        order_ids_to_cancel = [
            order_id
            for order_id, fill in self._pending_fills.items()
            if fill.asset == asset
        ]

        # Cancel each order
        for order_id in order_ids_to_cancel:
            order = self.order_manager.get_order(order_id)
            if order and order.state.is_active:
                success = await self.order_manager.cancel_order(order)
                if success:
                    cancelled += 1

        return cancelled

    async def _check_sniper_risk_for_market(self, market: Market) -> bool:
        """
        Check for sniper risk before placing orders.

        Returns:
            True if safe to proceed, False if sniper risk detected
        """
        # Get the market mid-price (YES price is the mid for UP markets)
        market_mid = market.yes_price

        alert = await self.oracle.check_sniper_risk(market.asset, market_mid)

        if alert is None:
            # Clear any previous alert for this asset
            self._sniper_alert_active.pop(market.asset, None)
            return True  # Safe to proceed

        if alert.risk_level == SniperRiskLevel.CRITICAL:
            return False  # Do not place orders

        if alert.risk_level == SniperRiskLevel.ELEVATED:
            logger.warning(
                "Elevated sniper risk - proceeding with caution",
                asset=market.asset,
                divergence_pct=f"{alert.divergence_pct:.3f}%",
            )
            return True  # Proceed but with caution

        return True

    async def _get_execution_price(
        self, 
        token_id: str, 
        side: TradeSide, 
        market: Market
    ) -> float:
        """Determine the best price to execute at."""
        price = None
        
        # Try order book first
        if self.clob_client:
            try:
                # Use cached book if available (it should be from analysis step)
                book = await self._get_cached_order_book(token_id)
                price = book.best_ask
                
                if price is None:
                    logger.warning("No asks in order book, using fallback price", token_id=token_id)
            except Exception as e:
                logger.warning("Failed to fetch order book for pricing", error=str(e), token_id=token_id)

        # Fallback to market price
        if price is None:
            price = market.yes_price if side == TradeSide.BUY_YES else (1 - market.yes_price)
            logger.info("Using fallback market price", price=price, token_id=token_id)
            
        return price

    def _track_order_metadata(
        self, 
        order: "Order", 
        signal: TradeSignal, 
        price: float, 
        token_id: str
    ) -> None:
        """Store metadata for fill handling."""
        self._pending_fills[order.client_order_id] = PendingFill(
            token_id=token_id,
            asset=signal.market.asset,
            side=PositionSide.LONG_YES if signal.side == TradeSide.BUY_YES else PositionSide.LONG_NO,
            price=price,
            size_usd=signal.kelly_size,
            market_question=signal.market.question,
            expires_at=signal.market.end_date.timestamp(),
        )
    
    async def check_arbitrage(self, market: Market) -> ArbitrageOpportunity | None:
        """
        Check for arbitrage opportunities in a binary market.
        
        From arXiv:2510.15205 Section 3.2.1: Market Rebalancing Arbitrage
        If YES + NO prices deviate from 1.0, guaranteed profit exists.
        
        Args:
            market: Market to check
        
        Returns:
            ArbitrageOpportunity if profitable opportunity exists
        """
        if not self.clob_client:
            return None
        
        try:
            # Fetch both order books (cached)
            yes_book = await self._get_cached_order_book(market.yes_token_id)
            no_book = await self._get_cached_order_book(market.no_token_id) if market.no_token_id else None
            
            if no_book is None:
                # Use implied NO price from YES price
                yes_ask = yes_book.best_ask or market.yes_price
                no_ask = 1.0 - (yes_book.best_bid or market.yes_price)
                opportunity = self.arbitrage_detector.check_binary_market(yes_ask, no_ask)
            else:
                # Use actual NO order book
                opportunity = self.arbitrage_detector.check_from_order_books(
                    yes_book, no_book, size_usd=10.0
                )
            
            if opportunity.is_profitable:
                logger.info(
                    "ARBITRAGE OPPORTUNITY DETECTED",
                    asset=market.asset,
                    arb_type=opportunity.arb_type.value,
                    yes_price=opportunity.yes_price,
                    no_price=opportunity.no_price,
                    net_profit_pct=f"{opportunity.net_profit_pct:.2%}",
                    action=opportunity.recommended_action,
                )
                return opportunity
                
        except Exception as e:
            logger.debug("Arbitrage check failed", error=str(e), market=market.question[:30])
        
        return None
    
    async def check_toxicity(self, market: Market) -> ToxicityResult:
        """
        Check for toxic order flow conditions.
        
        From arXiv:2510.15205 Section 4.2: Execution hygiene (anti pick-off)
        Avoid trading when order flow is heavily one-sided or spreads are wide.
        
        Args:
            market: Market to check
        
        Returns:
            ToxicityResult with analysis
        """
        from src.strategy.toxicity_filter import ToxicityReason
        
        if not self.clob_client:
            # Can't check without order book access - assume safe
            return ToxicityResult(
                is_toxic=False,
                reason=ToxicityReason.NONE,
                severity=0,
            )
        
        try:
            # Fetch order book (cached)
            book = await self._get_cached_order_book(market.yes_token_id)
            
            # Check toxicity
            result = self.toxicity_filter.analyze(
                order_book=book,
                recent_price=self.toxicity_filter.get_recent_price(),
                current_price=market.yes_price,
            )
            
            # Track price for momentum analysis
            self.toxicity_filter.update_price_history(market.yes_price)
            
            if result.is_toxic:
                logger.warning(
                    "Toxic flow detected - reducing/skipping trade",
                    asset=market.asset,
                    reason=result.reason.value if hasattr(result.reason, 'value') else str(result.reason),
                    severity=f"{result.severity:.2f}",
                    message=result.message,
                )
            
            return result
            
        except Exception as e:
            logger.debug("Toxicity check failed", error=str(e))
            return ToxicityResult(is_toxic=False, reason=ToxicityReason.NONE, severity=0)
    
    def handle_fill(self, client_order_id: str, fill_size: float, fill_price: float) -> None:
        """
        FIX #4: Handle order fill - update Portfolio on fills, not submissions.
        
        Args:
            client_order_id: Order that was filled
            fill_size: Size filled in this event
            fill_price: Price of this fill
        """
        # Forward to OrderManager (OrderManager tracks size in USD; fills arrive in shares)
        fill_usd = float(fill_size) * float(fill_price)
        self.order_manager.handle_fill(client_order_id, fill_usd, fill_price)
        
        # Check if we have metadata for this order
        metadata = self._pending_fills.get(client_order_id)
        if metadata is None:
            logger.warning("Fill for unknown order", client_order_id=client_order_id)
            return
        
        # FIX #4: Open position in Portfolio on fill
        self.portfolio.open_position(
            token_id=metadata.token_id,
            asset=metadata.asset,
            market_question=metadata.market_question,
            side=metadata.side,
            price=fill_price,
            size_usd=fill_size * fill_price,  # Convert shares to USD
            expires_at=metadata.expires_at,
        )
        
        self.metrics.orders_filled += 1
        
        # Check if order fully filled, clean up pending
        order = self.order_manager.get_order(client_order_id)
        if order and order.state.is_terminal:
            self._pending_fills.pop(client_order_id, None)
        
        logger.info(
            "Fill processed -> position opened",
            client_order_id=client_order_id,
            fill_size=fill_size,
            fill_price=fill_price,
            asset=metadata.asset,
            total_exposure=self.portfolio.total_exposure,
        )
    
    def _update_circuit_breaker(self) -> None:
        """
        FIX #7: Update CircuitBreaker with real values from Portfolio and Oracle.
        """
        # Update daily PnL from portfolio
        daily_perf = self.portfolio.get_daily_performance()
        self.circuit_breaker.update_daily_pnl(daily_perf.realized_pnl)
        
        # Update drawdown
        self.circuit_breaker.update_drawdown(self.portfolio.current_capital)

        # Wire portfolio equity into Kelly sizer for accurate sizing
        self.kelly_sizer.update_bankroll(self.portfolio.current_capital)

        # Update oracle staleness (max age across all watched assets)
        max_oracle_age = 0.0
        for asset in self.config.trading.assets:
            cached = self.oracle.get_cached_price(asset)
            if cached:
                max_oracle_age = max(max_oracle_age, cached.age_seconds)
        self.circuit_breaker.update_oracle_age(max_oracle_age)
        
        # Update rate limit usage
        self.circuit_breaker.update_rate_limit_usage(self.rate_limiter.order_usage_pct())
    
    async def _reconcile_orders(self) -> None:
        """
        FIX #10: Reconcile local order state with exchange.
        
        Periodically fetches open orders from exchange and syncs state.
        """
        import time
        now = time.time()
        
        if now - self._last_reconciliation < self._reconciliation_interval:
            return
        
        self._last_reconciliation = now
        
        if not self.clob_client:
            return
        
        try:
            # Fetch open orders from exchange
            exchange_orders = await self.clob_client.get_open_orders()
            exchange_order_ids = {
                o.get("orderID") or o.get("order_id") 
                for o in exchange_orders 
                if o.get("orderID") or o.get("order_id")
            }
            
            # Check our active orders against exchange
            local_active = self.order_manager.get_active_orders()
            
            orphaned_count = 0
            for order in local_active:
                if order.exchange_order_id and order.exchange_order_id not in exchange_order_ids:
                    # Order not on exchange but we think it's active.
                    # Do NOT mark as filled. We don't have fill details.
                    # Mark as FAILED so risk logic doesn't assume exposure or profit.
                    logger.warning(
                        "Orphaned order detected - marking as FAILED (unknown final state)",
                        client_order_id=order.client_order_id,
                        exchange_order_id=order.exchange_order_id,
                    )
                    order.update_state(order.state.FAILED)
                    order.error_message = "Orphaned during reconciliation: not present on exchange open orders"
                    orphaned_count += 1
            
            if orphaned_count > 0 or len(exchange_orders) > 0:
                logger.info(
                    "Order reconciliation complete",
                    local_active=len(local_active),
                    exchange_orders=len(exchange_orders),
                    orphaned=orphaned_count,
                )
                
        except Exception as e:
            logger.warning("Order reconciliation failed", error=str(e))
    
    async def _get_cached_order_book(self, token_id: str) -> "OrderBook":
        """Get order book from cache or fetch from CLOB."""
        if token_id in self._order_book_cache:
            return self._order_book_cache[token_id]
        
        if not self.clob_client:
            raise RuntimeError("CLOB client not initialized")
            
        book = await self.clob_client.get_order_book(token_id)
        if book is None:
            raise ValueError(f"CLOB returned None for order book {token_id}")
        self._order_book_cache[token_id] = book
        return book

    async def _scan_for_arbitrage(self, markets: list[Market]) -> None:
        """Scan markets for risk-free arbitrage opportunities."""
        if not self.config.trading.arbitrage_enabled:
            return
            
        for market in markets:
            arb_opportunity = await self.check_arbitrage(market)
            if arb_opportunity and arb_opportunity.is_profitable:
                # Check minimum profit threshold
                if arb_opportunity.net_profit_pct >= self.config.trading.arbitrage_min_profit_pct:
                    logger.info(
                        ">>> ARBITRAGE: Risk-free profit available! <<<",
                        asset=market.asset,
                        profit_pct=f"{arb_opportunity.net_profit_pct:.2%}",
                        action=arb_opportunity.recommended_action,
                    )
                    await self._execute_arbitrage(market, arb_opportunity)

    async def _execute_arbitrage(self, market: Market, opportunity: ArbitrageOpportunity) -> None:
        """
        Execute an arbitrage trade by buying both YES and NO outcomes.
        
        From arXiv:2510.15205: When YES + NO < 1.0, buying both guarantees profit.
        """
        if not self.clob_client or self.config.dry_run:
            logger.info(
                "Arbitrage execution (dry-run)",
                asset=market.asset,
                profit_pct=f"{opportunity.net_profit_pct:.2%}",
                action="Would buy YES and NO",
            )
            return
        
        try:
            # Calculate size (capped by config)
            # FIX: Clamp to min order size ($5) for Polymarket
            # The API rejects orders < $5 (CTF minimum)
            MIN_ORDER_SIZE = 5.0
            
            raw_size = min(
                self.config.trading.arbitrage_max_size,
                self.kelly_sizer.bankroll * 0.05  # Max 5% of bankroll per arb
            )
            
            # If we have a guaranteed profit, we can afford to size up to min
            # provided it's safe (e.g. < 50% bankroll)
            arb_size = max(MIN_ORDER_SIZE, raw_size)
            
            # Safety check: don't bet the house just to meet min size
            if arb_size > self.kelly_sizer.bankroll * 0.5:
                 logger.warning(
                     "Arbitrage size exceeds safety limit",
                     size=f"${arb_size:.2f}",
                     limit=f"${self.kelly_sizer.bankroll * 0.5:.2f}"
                 )
                 return

            # CRITICAL SAFETY CHECK: Ensure we have funds for BOTH legs
            # We need 2 * arb_size total (one for YES, one for NO)
            required_capital = arb_size * 2
            if self.kelly_sizer.bankroll < required_capital:
                logger.warning(
                    "Insufficient funds for full arbitrage",
                    required=f"${required_capital:.2f}",
                    available=f"${self.kelly_sizer.bankroll:.2f}",
                    asset=market.asset,
                )
                return

            # For BUY_YES_AND_NO: buy both outcomes
            # FIX: Use correct enum value LONG_REBALANCING
            if opportunity.arb_type == ArbitrageType.LONG_REBALANCING:
                # Place YES order with IOC to reduce leg risk
                yes_order = self.order_manager.create_order(
                    token_id=market.yes_token_id,
                    side=OrderSide.BUY,
                    price=opportunity.yes_price,
                    size=arb_size,
                    order_type=OrderType.IOC,
                )
                yes_success = await self.order_manager.submit_order(yes_order)

                # Place NO order if token exists
                no_success = False
                if market.no_token_id:
                    no_order = self.order_manager.create_order(
                        token_id=market.no_token_id,
                        side=OrderSide.BUY,
                        price=opportunity.no_price,
                        size=arb_size,
                        order_type=OrderType.IOC,
                    )
                    no_success = await self.order_manager.submit_order(no_order)

                logger.info(
                    "Arbitrage executed",
                    asset=market.asset,
                    profit_pct=f"{opportunity.net_profit_pct:.2%}",
                    size=f"${arb_size:.2f}",
                    yes_submitted=yes_success,
                    no_submitted=no_success,
                )
                self.metrics.orders_placed += int(yes_success) + int(no_success)
                
        except Exception as e:
            logger.error("Arbitrage execution failed", error=str(e), asset=market.asset)

    async def _process_directional_trade(self, market: Market, spot_price: float) -> None:
        """Process a single market for directional trading opportunities."""
        # Check toxicity BEFORE analyzing (save compute if toxic)
        toxicity = await self.check_toxicity(market)
        if toxicity.is_toxic and toxicity.should_pause:
            logger.info(
                "Skipping market due to toxic flow",
                asset=market.asset,
                reason=toxicity.reason.value if hasattr(toxicity.reason, 'value') else str(toxicity.reason),
            )
            return
        
        # Analyze
        signal = await self.analyze_market(market, spot_price)
        if signal is None:
            return
        
        # Apply toxicity size adjustment
        if toxicity.is_toxic and toxicity.should_reduce_size:
            original_size = signal.kelly_size
            signal.kelly_size *= toxicity.size_multiplier
            logger.info(
                "Reduced size due to toxicity",
                asset=market.asset,
                original_size=f"{original_size:.2f}",
                adjusted_size=f"{signal.kelly_size:.2f}",
                multiplier=f"{toxicity.size_multiplier:.2f}",
            )
        
        # Log opportunity and Execute
        if signal.passes_gate:
            self.log_trade_opportunity(signal)
            await self.execute_signal(signal)
            return True  # Trade was placed
        else:
            logger.debug(
                "Signal rejected",
                asset=market.asset,
                reason=signal.rejection_reason,
            )
            return False  # No trade placed

    def log_trade_opportunity(self, signal: TradeSignal) -> None:
        """Log a valid trade opportunity."""
        logger.info(
            "Trade opportunity",
            asset=signal.market.asset,
            side=signal.side.value,
            model_prob=f"{signal.model_prob:.3f}",
            market_price=f"{signal.market_price:.3f}",
            edge=f"{signal.ev_result.gross_edge:.3f}",
            net_ev=f"{signal.ev_result.net_ev:.4f}",
            kelly_size=f"{signal.kelly_size:.2f}",
        )

    async def _check_balance_floor(self) -> None:
        """Hard-stop live trading if wallet balance drops below configured floor."""
        if self.config.dry_run:
            return
        if not self.clob_client:
            return

        min_bal = float(getattr(self.config.trading, "min_balance_usd", 0.0))
        if min_bal <= 0:
            return

        interval = float(getattr(self.config.trading, "balance_check_interval_seconds", 30))
        now = time.time()
        if (now - self._last_balance_check_at) < interval:
            return
        self._last_balance_check_at = now

        bal = await self.clob_client.get_collateral_balance_usdc()
        if bal is None:
            # Can't read balance; don't halt on missing data.
            return

        if bal < min_bal:
            logger.critical("BALANCE FLOOR HIT - halting trading", balance=bal, min_balance=min_bal)
            self.circuit_breaker.manual_halt()
            self._running = False

    async def _select_order_type(self, token_id: str) -> OrderType:
        """Pick an order type based on spread (execution hygiene)."""
        if not self.clob_client:
            return OrderType.GTC

        max_spread_abs = float(getattr(self.config.trading, "gtc_max_spread", 0.0))
        max_spread_bps = float(getattr(self.config.trading, "gtc_max_spread_bps", 0.0))
        if max_spread_abs <= 0 and max_spread_bps <= 0:
            return OrderType.GTC

        try:
            book = await self._get_cached_order_book(token_id)
            if book.best_bid is None or book.best_ask is None:
                return OrderType.GTC

            bid = float(book.best_bid)
            ask = float(book.best_ask)
            spread = ask - bid
            mid = (ask + bid) / 2.0 if (ask + bid) > 0 else 0.0

            if max_spread_bps > 0 and mid > 0:
                spread_bps = (spread / mid) * 10_000.0
                if spread_bps > max_spread_bps:
                    return OrderType.IOC

            if max_spread_abs > 0 and spread > max_spread_abs:
                return OrderType.IOC

        except Exception:
            pass

        return OrderType.GTC

    def _can_vibe_bet(self) -> bool:
        cooldown = float(getattr(self.config.trading, "vibe_bet_cooldown_seconds", 3600))
        return (time.time() - self._last_vibe_bet_at) >= cooldown

    async def _place_vibe_bet(self, market: Market) -> None:
        """Place an occasional small bet even without edge.

        Bounded by:
        - one bet per cooldown window
        - max_total_exposure_pct (global)
        - max_position_pct (per trade)
        - circuit breaker + toxicity checks inside order path
        """
        if not self._can_vibe_bet():
            return

        size = float(getattr(self.config.trading, "vibe_bet_size", 5.0))
        if size <= 0:
            return

        logger.info("VIBE BET: letting the lizard brain have one", size_usd=size)
        self._last_vibe_bet_at = time.time()
        await self._place_favorite_fallback(market, override_size=size)

    def _pending_exposure_usd(self) -> float:
        """Conservative exposure estimate for orders submitted but not yet reflected in Portfolio."""
        return float(sum(p.size_usd for p in self._pending_fills.values()))

    def _effective_exposure_usd(self) -> float:
        """Exposure including pending submitted orders (conservative)."""
        return float(self.portfolio.total_exposure) + self._pending_exposure_usd()

    async def _place_favorite_fallback(self, market: Market, override_size: float | None = None) -> None:
        """
        Place a minimum bet on the most likely outcome (favorite).

        Called when no value edge is found, to maintain participation
        and capture trend premiums.
        """
        # Determine favorite: choose the higher-priced outcome
        is_yes_favorite = market.yes_price >= market.no_price
        favorite_side = TradeSide.BUY_YES if is_yes_favorite else TradeSide.BUY_NO
        favorite_price = market.yes_price if is_yes_favorite else market.no_price
        token_id = market.yes_token_id if is_yes_favorite else market.no_token_id

        # Skip if no token ID (shouldn't happen, but be safe)
        if not token_id:
            return

        # Enforce per-market limit (prevent duplicate bets in same round)
        if market.question in self._traded_markets:
            logger.info("Skipping favorite fallback - already traded this market", asset=market.asset)
            return

        # Check circuit breaker before placing bet
        if not self.circuit_breaker.can_enter_new_position():
            logger.info("Circuit breaker blocking favorite fallback bet")
            return

        bet_size = float(override_size) if override_size is not None else float(self.config.trading.favorite_bet_size)

        # Enforce minimum order size
        min_size = float(getattr(self.config.trading, "min_order_size_usd", 0.0))
        if min_size > 0 and bet_size < min_size:
            logger.info("Skipping fallback/vibe - below minimum order size", size=bet_size, min_size=min_size)
            return

        # Enforce max total exposure (global)
        max_total = getattr(self.config.trading, "max_total_exposure_pct", 1.0) * self.kelly_sizer.bankroll
        effective = self._effective_exposure_usd()
        if effective >= max_total:
            logger.info("Skipping favorite fallback - max total exposure reached", effective_exposure=effective, max_total=max_total)
            return
        # If the bet would push us over, clip to remaining room (and bail if too small)
        room = max_total - effective
        bet_size = min(bet_size, room)
        if bet_size <= 0:
            return

        # Enforce max asset exposure
        exposure_by_asset = self.portfolio.get_exposure_by_asset()
        current_asset_exposure = exposure_by_asset.get(market.asset, 0.0)
        max_additional = (self.kelly_sizer.max_asset_exposure_pct * self.kelly_sizer.bankroll) - current_asset_exposure
        if max_additional <= 0:
            logger.info("Skipping favorite fallback - max asset exposure reached", asset=market.asset)
            return
        bet_size = min(bet_size, max_additional)
        if bet_size <= 0:
            return

        # Check if we're in dry-run mode
        if self.config.dry_run:
            logger.info(
                "Favorite fallback bet (dry-run)",
                asset=market.asset,
                side=favorite_side.value,
                price=f"{favorite_price:.3f}",
                size=f"${bet_size:.2f}",
            )
            return

        # Place actual order
        try:
            order_type = await self._select_order_type(token_id)
            order = self.order_manager.create_order(
                token_id=token_id,
                side=OrderSide.BUY,
                price=favorite_price,
                size=bet_size,
                order_type=order_type,
            )

            success = await self.order_manager.submit_order(order)
            if success:
                self.metrics.orders_placed += 1
                self._pending_fills[order.client_order_id] = PendingFill(
                    token_id=token_id,
                    asset=market.asset,
                    side=PositionSide.LONG_YES if is_yes_favorite else PositionSide.LONG_NO,
                    price=favorite_price,
                    size_usd=bet_size,
                    market_question=market.question,
                    expires_at=market.end_date.timestamp(),
                )
                self._traded_markets[market.question] = market.end_date.timestamp()
                logger.info(
                    "Favorite fallback bet placed",
                    asset=market.asset,
                    side=favorite_side.value,
                    price=f"{favorite_price:.3f}",
                    size=f"${bet_size:.2f}",
                )

        except Exception as e:
            logger.error(
                "Favorite fallback bet failed",
                error=str(e),
                asset=market.asset,
            )

    async def run_iteration(self) -> None:
        """Run single iteration of the strategy loop."""
        # Initialize WebSocket if not already
        await self._init_websocket()
        
        # Clear order book cache at start of iteration
        self._order_book_cache.clear()
        
        # Cleanup expired trade history
        self._cleanup_traded_markets()
        
        # FIX #7: Update circuit breaker with real values
        self._update_circuit_breaker()
        
        # FIX #10: Periodic order reconciliation
        await self._reconcile_orders()
        
        # Live balance floor check (literal wallet USDC)
        await self._check_balance_floor()

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
        
        # STEP 1: Check for Arbitrage
        await self._scan_for_arbitrage(markets)
        
        # STEP 2: Directional Trades (with favorite fallback)
        vibe_used = False
        for market in markets:
            spot_price = prices.get(market.asset)
            if spot_price is not None:
                trade_placed = await self._process_directional_trade(market, spot_price)

                # STEP 3: Favorite Fallback Bet (if no value trade placed)
                if not trade_placed and self.config.trading.favorite_bet_enabled:
                    await self._place_favorite_fallback(market)

                # STEP 4: Optional vibe bet (at most one per cooldown window)
                if (
                    not trade_placed
                    and not vibe_used
                    and getattr(self.config.trading, "vibe_bet_enabled", False)
                    and self._can_vibe_bet()
                ):
                    await self._place_vibe_bet(market)
                    vibe_used = True

        # Update API state snapshot
        try:
            from src.api.state import get_state
            st = get_state()
            st.portfolio = self.portfolio
            st.active_orders = self.order_manager.get_active_orders()
            st.last_update = time.time()
        except Exception:
            pass
    
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
                except Exception:
                    logger.exception("Iteration error")
                
                if max_iterations and iteration >= max_iterations:
                    break
                
                # Sleep between iterations
                await asyncio.sleep(5)
                
        except asyncio.CancelledError:
            logger.info("Strategy cancelled")
        finally:
            self._running = False
            # Cleanup WebSocket
            if self.ws_client:
                await self.ws_client.close()
            
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
            "realized_pnl": self.metrics.realized_pnl,
            "order_stats": self.order_manager.stats,
            # FIX #4/7: Include portfolio and circuit breaker status
            "portfolio": self.portfolio.summary(),
            "circuit_breaker_tripped": not self.circuit_breaker.can_trade(),
        }
