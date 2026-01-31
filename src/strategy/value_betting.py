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
        clob_client: CLOBClient | None = None,  # FIX #3: For order book pricing
        portfolio: Portfolio | None = None,  # FIX #4: For fill-based tracking
        bankroll: float = 100.0,
    ):
        self.config = config
        self.market_discovery = market_discovery
        self.oracle = oracle
        self.order_manager = order_manager
        self.rate_limiter = rate_limiter
        self.circuit_breaker = circuit_breaker
        self.clob_client = clob_client
        
        # FIX #4: Portfolio for fill-based exposure tracking
        self.portfolio = portfolio or Portfolio(starting_capital=bankroll)
        self._pending_fills: dict[str, PendingFill] = {}  # client_order_id -> PendingFill
        
        # Models
        # self.pricing_model = JumpDiffusionModel()  # OLD
        self.pricing_model = MertonJDCalibrator()  # NEW: Research model
        
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
        
        # Research-based enhancements from arXiv:2510.15205
        self.arbitrage_detector = ArbitrageDetector(
            fee_rate=0.02,  # Polymarket 2% fee
            min_profit_threshold=0.005,  # 0.5% minimum profit after fees
        )
        self.toxicity_filter = ToxicityFilter(
            imbalance_threshold=0.7,  # 70% imbalance is toxic
            spread_threshold=0.08,     # 8% spread is too wide
            min_depth_usd=20.0,        # Minimum $20 liquidity (lowered for more trades)
            vpin_threshold=0.7,        # VPIN > 0.7 = extremely toxic (research)
        )
        
        # Research-backed enhancements (2026 deep research synthesis)
        self.ensemble_model = ResearchEnsembleModel()  # Combines JD + Kou + Bates + order book
        self.orderbook_signal = OrderBookSignalModel()  # Order book imbalance (65-75% accuracy)
        # self.sentiment_integrator = SentimentIntegrator()  # OLD
        self.sentiment_aggregator = SentimentAggregator() # NEW: From research
        self._vpin_calculators: dict[str, VPINCalculator] = {}  # Per-market VPIN tracking
        
        # WebSocket Client (Research)
        self.ws_client = None
        
        # State
        self.metrics = StrategyMetrics()
        self._running = False
        self._last_reconciliation = 0.0
        self._reconciliation_interval = 60.0  # FIX #10: Reconcile every 60s
        
        # FIX: Order book cache to avoid duplicate fetches per iteration
        self._order_book_cache: dict[str, "OrderBook"] = {}  # token_id -> OrderBook
        
        # FIX: Track traded markets to enforce "1 bet per asset per cycle" (User Request)
        # Map: market.question -> expiry_timestamp
        self._traded_markets: dict[str, float] = {}

        # RESEARCH FIX: Sniper protection - register callback for Chainlink price divergence
        self.oracle.register_sniper_callback(self._handle_sniper_alert)
        self._sniper_alert_active: dict[str, SniperAlert] = {}  # asset -> active alert

        # Extreme probability settings (where fees approach zero)
        self._extreme_prob_enabled = config.trading.extreme_prob_enabled
        self._extreme_prob_threshold = config.trading.extreme_prob_threshold
        self._extreme_prob_kelly_boost = config.trading.extreme_prob_kelly_boost
        self._extreme_prob_min_edge = config.trading.extreme_prob_min_edge

    async def _init_websocket(self):
        """Initialize and connect WebSocket client."""
        if not self.ws_client:
            self.ws_client = await create_market_ws_client()
            await self.ws_client.connect()
            logger.info("WebSocket initialized for strategy")

    def _is_extreme_probability(self, market: Market) -> bool:
        """
        Check if market is at extreme probability (near 0% or 100%).

        At extremes, fees approach zero:
        - Fee at p=0.01 → $0.0025 per 100 shares
        - Fee at p=0.50 → $1.56 per 100 shares (62x more!)

        This allows larger positions and lower edge thresholds.
        """
        if not self._extreme_prob_enabled:
            return False
        threshold = self._extreme_prob_threshold
        return market.yes_price < threshold or market.yes_price > (1 - threshold)

    def _get_min_edge_for_market(self, market: Market) -> float:
        """Get minimum edge threshold, lower at extreme probabilities."""
        if self._is_extreme_probability(market):
            return self._extreme_prob_min_edge
        return self.config.trading.min_edge_threshold

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
        # Calculate model probability using Research Code (MertonJD)
        # Note: In a real implementation, we would calibrate this first
        # For now, we assume params are reasonable or will be calibrated
        
        # NOTE: self.pricing_model (MertonJDCalibrator) handles calibration
        # We need historical returns for it. For now, let's use the Ensemble model 
        # which incorporates it, or fallback to simple heuristics if calibration missing.
        
        time_years = market.minutes_to_expiry / (365 * 24 * 60) # Approx

        # FIX #1: Use market.open_price if available, else fall back to spot
        if market.open_price is not None:
            initial_price = market.open_price
        else:
            # Fallback: use current spot price (suboptimal but safe)
            initial_price = spot_price
            logger.warning(
                "Using spot price as initial (open_price not available)",
                asset=market.asset,
                question=market.question[:50],
            )

        # RESEARCH ENHANCEMENT: Use ensemble model with order book features
        # Combines JD + Kou + Bates + order book imbalance signals
        orderbook_features = None
        vpin = None
        
        # FIX #3: Fetch order book for spread, depth, and signal analysis
        spread = 0.0
        book_depth_usd = 1000.0  # Default fallback
        book = None
        
        # Use WebSocket for orderbook if available (faster!)
        if self.ws_client and self.ws_client.is_connected:
             ws_book_data = self.ws_client.get_orderbook(market.yes_token_id)
             if ws_book_data:
                 # Adapt WS format to OrderBook object expected by signals
                 # This is a simplification; ideally we'd map fields properly
                 pass 

        if self.clob_client:
            try:
                # Fetch order book for YES token (primary analysis)
                token_id = market.yes_token_id
                book = await self._get_cached_order_book(token_id)

                spread = book.spread
                book_depth_usd = book.ask_depth_usd
                
                # Extract order book features for ensemble model
                orderbook_features = self.orderbook_signal.extract_features(
                    orderbook=book,
                    market_state={
                        "seconds_to_expiry": market.seconds_to_expiry,
                        "vpin": vpin,
                    },
                )
                
                # Get VPIN if we have a calculator for this market
                if token_id not in self._vpin_calculators:
                    self._vpin_calculators[token_id] = VPINCalculator()
                
                # Feed trades to VPIN calculator (if we had trade stream)
                # self._vpin_calculators[token_id].add_trade(...) 
                
                vpin = self._vpin_calculators[token_id].calculate_vpin()
                
                logger.debug(
                    "Order book analyzed",
                    token_id=token_id[:20] + "...",
                    spread=spread,
                    depth_usd=book_depth_usd,
                    imbalance=orderbook_features.imbalance_100bp if orderbook_features else 0,
                    vpin=vpin,
                )
            except Exception as e:
                logger.warning("Failed to fetch order book", error=str(e), token_id=market.yes_token_id)

        # RESEARCH: Ensemble model probability (JD + Kou + Bates + order book)
        ensemble_result = self.ensemble_model.prob_up(
            spot=spot_price,
            initial=initial_price,
            time_years=time_years,
            asset=market.asset,
            orderbook_features=orderbook_features,
            vpin=vpin,
        )
        model_prob = ensemble_result.probability
        
        logger.debug(
            "Ensemble model used",
            asset=market.asset,
            jd=f"{ensemble_result.jd_prob:.3f}",
            kou=f"{ensemble_result.kou_prob:.3f}",
            bates=f"{ensemble_result.bates_prob:.3f}",
            orderbook_adj=f"{ensemble_result.orderbook_adjustment:.3f}",
            final=f"{model_prob:.3f}",
            regime=ensemble_result.regime.value,
            confidence=f"{ensemble_result.confidence:.2f}",
        )
        
        # Calculate EV
        ev_result = self.ev_calculator.calculate(
            model_prob=model_prob,
            market_price=market.yes_price,
            size_usd=10.0,  # Placeholder, Kelly will override
        )
        
        self.metrics.signals_generated += 1

        # Determine token_id for the side we'd trade
        token_id = market.yes_token_id if ev_result.side == TradeSide.BUY_YES else market.no_token_id
        
        # Try to fetch order book for the specific side if not already done
        if book is None and self.clob_client:
            try:
                book = await self._get_cached_order_book(token_id)
                spread = book.spread
                book_depth_usd = book.ask_depth_usd if ev_result.side in (TradeSide.BUY_YES, TradeSide.BUY_NO) else book.bid_depth_usd
            except Exception as e:
                logger.warning("Failed to fetch order book", error=str(e), token_id=token_id)

        # Check NO-TRADE gate
        # Use lower edge threshold for extreme probability markets (fees approach zero)
        context = TradeContext(
            ev_result=ev_result,
            asset=market.asset,
            token_id=market.yes_token_id,
            seconds_to_expiry=market.seconds_to_expiry,
            seconds_since_open=market.seconds_since_open,  # FIX #9: Now populated from market
            spread=spread,  # FIX #3: Real spread from order book
            book_depth_usd=book_depth_usd,  # FIX #3: Real depth from order book
            oracle_age_seconds=self.oracle.get_cached_price(market.asset).age_seconds if self.oracle.get_cached_price(market.asset) else 0,
            rate_limit_usage_pct=self.rate_limiter.order_usage_pct(),
            trading_halted=not self.circuit_breaker.can_trade(),
            correlation_with_portfolio=self.correlation_matrix.max_correlation_with(
                market.asset, self.portfolio.get_exposure_by_asset()
            ),
            override_min_edge=self._get_min_edge_for_market(market),  # Lower at extremes
        )
        
        gate_result = self.no_trade_gate.evaluate(context)
        
        # Calculate Kelly size
        kelly_result = self.kelly_sizer.calculate(
            win_prob=model_prob if ev_result.side == TradeSide.BUY_YES else 1 - model_prob,
            market_price=market.yes_price if ev_result.side == TradeSide.BUY_YES else 1 - market.yes_price,
            side="YES" if ev_result.side == TradeSide.BUY_YES else "NO",
            current_asset_exposure=self.portfolio.get_exposure_by_asset().get(market.asset, 0),
        )
        
        # Apply correlation adjustment
        corr_adj = self.correlation_matrix.correlation_adjustment(
            market.asset, self.portfolio.get_exposure_by_asset()
        )
        
        # Sentiment-based sizing multiplier (NOT probability adjustment!)
        # SAFETY: Disabled by default because this code previously used placeholders.
        sentiment_mult = 1.0
        if getattr(self.config.trading, "sentiment_sizing_enabled", False):
            # TODO: Replace placeholders with real inputs (e.g. funding rate + real OBI volumes)
            market_state_sentiment = type('obj', (object,), {
                'asset': market.asset,
                'funding_rate': 0.0001,  # placeholder
                'bid_volume_5bp': 1000,  # placeholder
                'ask_volume_5bp': 1000,  # placeholder
            })

            sentiment_score = self.sentiment_aggregator.composite_sentiment_score(market_state_sentiment)
            base_fraction = self.config.trading.kelly_fraction
            adjusted_fraction = self.sentiment_aggregator.adjust_position_sizing(base_fraction, sentiment_score)
            sentiment_mult = adjusted_fraction / base_fraction if base_fraction > 0 else 1.0

        # EXTREME PROBABILITY BOOST: Fees approach zero at p < 5% or p > 95%
        # Research: Fee at p=0.01 is $0.0025 vs $1.56 at p=0.50 (62x less!)
        extreme_prob_mult = 1.0
        if self._is_extreme_probability(market):
            extreme_prob_mult = self._extreme_prob_kelly_boost
            logger.info(
                "Extreme probability detected - boosting position",
                asset=market.asset,
                yes_price=f"{market.yes_price:.2f}",
                boost=f"{extreme_prob_mult:.1f}x",
            )

        # Final adjusted size = base × correlation × sentiment × extreme_boost
        adjusted_size = kelly_result.recommended_size_usd * corr_adj * sentiment_mult * extreme_prob_mult
        
        logger.debug(
            "Position sizing",
            kelly_base=f"${kelly_result.recommended_size_usd:.2f}",
            corr_adj=f"{corr_adj:.2f}",
            sentiment_mult=f"{sentiment_mult:.2f}",
            final_size=f"${adjusted_size:.2f}",
        )
        
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

        Orchestrates:
        1. Pre-trade checks (circuit breaker, actionability, sniper risk)
        2. Pricing (order book vs fallback)
        3. Order creation & submission
        4. State tracking

        Args:
            signal: Signal to execute

        Returns:
            True if order placed successfully
        """
        if not signal.is_actionable:
            return False

        # FIX: Enforce 1 bet per market limit
        if signal.market.question in self._traded_markets:
            logger.info("Skipping trade - already traded this market", asset=signal.market.asset)
            return False

        if not self.circuit_breaker.can_enter_new_position():
            logger.info("Circuit breaker blocking new positions")
            return False

        # RESEARCH FIX: Check sniper risk before placing orders
        if not await self._check_sniper_risk_for_market(signal.market):
            logger.warning(
                "Skipping trade due to sniper risk",
                asset=signal.market.asset,
            )
            return False
        
        # Determine token and side
        token_id = signal.market.yes_token_id if signal.side == TradeSide.BUY_YES else signal.market.no_token_id
        order_side = OrderSide.BUY

        # Get execution price
        price = await self._get_execution_price(token_id, signal.side, signal.market)
        
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
            self._track_order_metadata(order, signal, price, token_id)
            
            # Record that we traded this market
            self._traded_markets[signal.market.question] = signal.market.end_date.timestamp()
            
            logger.info(
                "Order placed (awaiting fill)",
                client_order_id=order.client_order_id,
                asset=signal.market.asset,
                side=signal.side.value,
                price=price,
                size=signal.kelly_size,
                model_prob=signal.model_prob,
                net_ev=signal.ev_result.net_ev,
            )
        
        return success

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
        # Forward to OrderManager
        self.order_manager.handle_fill(client_order_id, fill_size, fill_price)
        
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

    async def _place_favorite_fallback(self, market: Market) -> None:
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

        bet_size = self.config.trading.favorite_bet_size

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
            order = self.order_manager.create_order(
                token_id=token_id,
                side=OrderSide.BUY,
                price=favorite_price,
                size=bet_size,
                order_type=OrderType.GTC,
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
        for market in markets:
            spot_price = prices.get(market.asset)
            if spot_price is not None:
                trade_placed = await self._process_directional_trade(market, spot_price)
                
                # STEP 3: Favorite Fallback Bet (if no value trade placed)
                if not trade_placed and self.config.trading.favorite_bet_enabled:
                    await self._place_favorite_fallback(market)

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
