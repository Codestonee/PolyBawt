"""
Value Betting Strategy - Grok Ensemble Edition.

Integrates:
- ArbTakerStrategy (Arb < 0.98)
- LatencySnipeStrategy (Delta > 2%)
- SpreadMakerStrategy (Spread > 5c)
- LeggedHedgeStrategy (Crash > 15%)

Wires:
- EventBus (audit trail)
- MetricsCollector (Prometheus)
- VPIN (toxicity filter)
- RiskGate (Signal -> Intent validation)
- Sniper risk check (Chainlink divergence)
- OBI signal integration
- Per-strategy circuit breakers
"""

import asyncio
import time
import random
import re
import math
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timezone
from typing import Any, List, Dict, Optional
from threading import Lock

from src.infrastructure.logging import get_logger, bind_context
from src.infrastructure.config import AppConfig
from src.infrastructure.events import (
    event_bus, EventType, OrderEvent, TradeEvent, RiskEvent, DomainEvent,
)
from src.infrastructure.metrics import metrics
from src.infrastructure.research_recorder import ResearchRecorder
from src.ingestion.market_discovery import Market, MarketDiscovery
from src.ingestion.oracle_feed import OracleFeed, SniperRiskLevel
from src.risk.circuit_breaker import CircuitBreaker
from src.risk.kelly_sizer import KellySizer, AdaptiveMode, correlation_matrix
from src.risk.vpin import VPINCalculator
from src.risk.gatekeeper import RiskGate
from src.execution.order_manager import OrderManager, OrderSide, OrderType, Order
from src.execution.rate_limiter import RateLimiter
from src.execution.clob_client import CLOBClient
from src.models.order_book import OrderBook
from src.portfolio.tracker import Portfolio, PositionSide
from src.strategy.strategies import (
    ArbTakerStrategy,
    LatencySnipeStrategy,
    SpreadMakerStrategy,
    LeggedHedgeStrategy,
)
from src.strategy.signals import Signal, SignalType, SignalSide, signal_from_order_params
from src.strategy.base import TradeContext

logger = get_logger(__name__)


@dataclass
class StrategyMetrics:
    """Trading metrics for performance tracking."""
    signals_generated: int = 0
    orders_placed: int = 0
    orders_filled: int = 0
    realized_pnl: float = 0.0
    # Per-strategy circuit breaker tracking
    consecutive_losses: int = 0
    disabled_until: float = 0.0
    size_multiplier: float = 1.0


@dataclass
class PendingFill:
    """Metadata for an order that has been submitted but not yet filled."""
    token_id: str
    condition_id: str
    asset: str
    side: PositionSide
    price: float
    size_usd: float
    market_question: str
    expires_at: float
    strategy_name: str = ""


class EnsembleStrategy:
    """
    Orchestrator for the Grok Strategy Ensemble.

    Wires all safety systems: EventBus, Metrics, VPIN, RiskGate, Sniper,
    OBI, per-strategy circuit breakers, settlement detection, and daily reset.
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
        self._order_book_cache: Dict[str, Any] = {}  # token_id -> (OrderBook, cached_at)
        self._processed_fill_ids: set[str] = set()  # Dedup for WebSocket fills
        self._processed_fill_fifo: deque[str] = deque()
        self._max_processed_fills = 20000
        self._fill_lock = Lock()
        self._unresolved_settlements: set[str] = set()
        self._resolution_cache: dict[str, tuple[bool | None, float]] = {}
        self._order_book_ttl_seconds = float(
            getattr(config.trading, "order_book_ttl_seconds", 5.0)
        )
        self._research: ResearchRecorder | None = None
        if self.config.observability.research_capture_enabled:
            self._research = ResearchRecorder(self.config.observability.research_capture_dir)

        # Initialize Kelly Sizer for optimal position sizing
        self.kelly_sizer = KellySizer(
            bankroll=bankroll,
            kelly_fraction=config.trading.kelly_fraction,
            max_position_pct=config.trading.max_position_pct,
            max_asset_exposure_pct=config.trading.max_asset_exposure_pct,
            max_position_usd=config.trading.max_trade_usd,
            adaptive_mode=AdaptiveMode.FULL,
        )
        logger.info(
            "Kelly sizer initialized",
            kelly_fraction=config.trading.kelly_fraction,
            max_position_pct=config.trading.max_position_pct,
        )

        # Initialize VPIN toxicity filter
        vpin_cfg = config.vpin
        self.vpin = VPINCalculator(
            bucket_size=vpin_cfg.bucket_size_usd,
            n_buckets=vpin_cfg.n_buckets,
        )
        logger.info("VPIN toxicity filter initialized")

        # Initialize RiskGate
        self.risk_gate = RiskGate(
            circuit_breaker=circuit_breaker,
            rate_limiter=rate_limiter,
            portfolio=self.portfolio,
            max_trade_usd=config.trading.max_trade_usd,
            max_total_exposure_usd=config.trading.max_total_exposure_usd,
            min_order_size_usd=config.trading.min_order_size_usd,
            pending_exposure_getter=order_manager.get_pending_exposure,
        )
        logger.info("RiskGate initialized")

        # Initialize Grok Strategies
        self.strategies_list = [
            ArbTakerStrategy(config),
            LatencySnipeStrategy(config, oracle),
            SpreadMakerStrategy(config),
            LeggedHedgeStrategy(config),
        ]
        logger.info(f"Initialized {len(self.strategies_list)} sub-strategies")
        self.strategies = {s.name: s for s in self.strategies_list}

        # Per-strategy metrics and circuit breakers
        self._strategy_metrics: Dict[str, StrategyMetrics] = {
            s.name: StrategyMetrics() for s in self.strategies_list
        }
        self.metrics = StrategyMetrics()

        # Timers for periodic tasks
        self._last_stale_check: float = 0.0
        self._last_settlement_check: float = 0.0
        self._current_date: str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Pass order_manager reference to SpreadMaker for cancel-before-quote
        spread_maker = self.strategies.get("SpreadMaker")
        if spread_maker and isinstance(spread_maker, SpreadMakerStrategy):
            spread_maker._order_manager = order_manager

        # Register for order updates
        self.order_manager.add_listener(self._handle_order_update)

    def _handle_order_update(self, order: Order) -> None:
        """Route updates to the originating strategy."""
        strategy_name = order.strategy_id
        if strategy_name in self.strategies:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(
                    self.strategies[strategy_name].on_order_update(
                        order.client_order_id,
                        order.state.value,
                    )
                )
            except RuntimeError:
                pass  # No running loop, skip async dispatch

        # Track fills for LeggedHedge leg1_order_id
        if order.state.value == "new":
            leg_ctx_id = getattr(order, '_leg_context_id', None)
            if leg_ctx_id and "LeggedHedge" in self.strategies:
                hedge = self.strategies["LeggedHedge"]
                if hasattr(hedge, 'active_legs') and leg_ctx_id in hedge.active_legs:
                    hedge.active_legs[leg_ctx_id].leg1_order_id = order.client_order_id
        if order.state.value == "filled":
            leg_ctx_id = getattr(order, '_leg_context_id', None)
            if leg_ctx_id and "LeggedHedge" in self.strategies:
                hedge = self.strategies["LeggedHedge"]
                if hasattr(hedge, 'active_legs') and leg_ctx_id in hedge.active_legs:
                    hedge.active_legs[leg_ctx_id].leg1_size = order.filled_size or order.size
                    hedge.active_legs[leg_ctx_id].leg1_price = order.average_fill_price or order.price

    async def run(self):
        """Main strategy loop with all safety wiring."""
        self._running = True
        logger.info("Starting Grok Ensemble loop...")

        while self._running:
            try:
                now = time.time()

                # === Daily reset check ===
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                if today != self._current_date:
                    self._current_date = today
                    self.circuit_breaker.reset_daily()
                    self.kelly_sizer.reset_tracking()
                    logger.info("Daily reset completed", date=today)

                # === Auto-reset circuit breakers ===
                self.circuit_breaker.check_auto_reset()

                # === Update rate-limit status in circuit breaker ===
                self.circuit_breaker.update_rate_limit_usage(
                    self.rate_limiter.order_usage_pct()
                )
                oracle_health = self.oracle.health
                if not math.isinf(oracle_health.last_price_age_seconds):
                    self.circuit_breaker.update_oracle_age(
                        oracle_health.last_price_age_seconds
                    )

                # === Stale order timeout (every 30s) ===
                if now - self._last_stale_check > 30.0:
                    self._last_stale_check = now
                    timed_out = await self.order_manager.timeout_stale_orders()
                    if timed_out:
                        logger.info(f"Timed out {len(timed_out)} stale orders")

                # === Settlement detection (every 10s) ===
                if now - self._last_settlement_check > 10.0:
                    self._last_settlement_check = now
                    await self._check_settlements()

                # === Order book cache TTL eviction ===
                self._order_book_cache = {
                    k: v for k, v in self._order_book_cache.items()
                    if now - v[1] < self._order_book_ttl_seconds
                } if self._order_book_cache else {}

                # === Metrics: iteration ===
                metrics.increment_iteration()

                # === Update portfolio metrics ===
                daily = self.portfolio.get_daily_performance()
                metrics.update_pnl(
                    realized=self.portfolio.realized_pnl,
                    unrealized=0.0,  # Would need current prices for accurate unrealized
                    daily=daily.total_pnl,
                )
                metrics.update_portfolio(
                    value=self.portfolio.current_capital,
                    open_positions=len(self.portfolio.get_open_positions()),
                    exposure_by_asset=self.portfolio.get_exposure_by_asset(),
                )
                metrics.update_win_rate(
                    win_rate=self.portfolio.win_rate,
                    total_trades=self.portfolio.total_trades,
                    total_wins=self.portfolio.total_wins,
                    total_losses=self.portfolio.total_losses,
                    current_streak=self.portfolio.current_streak,
                    profit_factor=self.portfolio.profit_factor,
                    biggest_win=self.portfolio.biggest_win,
                    biggest_loss=self.portfolio.biggest_loss,
                )

                # Graceful degradation: pause trading when oracle feed is stale/unhealthy.
                oracle_health = self.oracle.health
                stale_limit = getattr(self.oracle, "stale_threshold", 10.0) * 3.0
                if (
                    not oracle_health.any_healthy
                    and oracle_health.last_price_age_seconds > stale_limit
                ):
                    logger.warning(
                        "Pausing loop: price feed unhealthy",
                        last_price_age_seconds=f"{oracle_health.last_price_age_seconds:.1f}",
                    )
                    await asyncio.sleep(2.0)
                    continue

                # 1. Update Market Scope (15m markets)
                markets = await self.market_discovery.get_active_markets(
                    assets=self.config.trading.assets
                )

                # 2. Cycle markets with optional bounded concurrency.
                if not self.circuit_breaker.can_trade():
                    logger.warning("Circuit breaker active")
                    await asyncio.sleep(1.0)
                    continue

                concurrency = int(getattr(self.config.trading, "market_concurrency", 1))
                jitter_min_ms = int(getattr(self.config.trading, "inter_market_jitter_ms_min", 100))
                jitter_max_ms = int(getattr(self.config.trading, "inter_market_jitter_ms_max", 500))
                if jitter_max_ms < jitter_min_ms:
                    jitter_max_ms = jitter_min_ms

                if concurrency <= 1:
                    for market in markets:
                        if not self.circuit_breaker.can_trade():
                            logger.warning("Circuit breaker active")
                            break
                        await self.process_market(market)
                        if jitter_max_ms > 0:
                            await asyncio.sleep(random.uniform(jitter_min_ms / 1000.0, jitter_max_ms / 1000.0))
                else:
                    semaphore = asyncio.Semaphore(concurrency)

                    async def _process_single(m: Market) -> None:
                        async with semaphore:
                            if not self.circuit_breaker.can_trade():
                                return
                            await self.process_market(m)

                    results = await asyncio.gather(*(_process_single(m) for m in markets), return_exceptions=True)
                    for result in results:
                        if isinstance(result, Exception):
                            logger.error("Concurrent market processing error", error=str(result))

                await asyncio.sleep(1.0)

            except Exception as e:
                logger.error("Error in ensemble loop", error=str(e))
                await asyncio.sleep(5.0)

    async def _check_settlements(self) -> None:
        """Check for expired positions and settle them."""
        now = time.time()
        positions = self.portfolio.get_open_positions()

        for position in positions:
            if position.expires_at > 0 and now > position.expires_at:
                # Position has expired - determine outcome
                try:
                    outcome = await self._get_authoritative_outcome(position.token_id, position.condition_id)
                    if outcome is None:
                        final_price = await self.oracle.get_chainlink_price(position.asset)
                        if final_price is None:
                            final_price = await self.oracle.get_price(position.asset)

                        if self.config.trading.allow_heuristic_settlement_fallback and final_price is not None:
                            outcome = self._resolve_binary_outcome(
                                market_question=position.market_question,
                                final_price=final_price,
                            )

                    if outcome is None:
                        if position.token_id not in self._unresolved_settlements:
                            self._unresolved_settlements.add(position.token_id)
                            logger.warning(
                                "Settlement deferred: unresolved market outcome",
                                token_id=position.token_id[:16],
                                asset=position.asset,
                                condition_id=(position.condition_id or "")[:16],
                                market_question=position.market_question[:120],
                            )
                        continue

                    pnl = self.portfolio.settle_position(position.token_id, outcome)
                    self._unresolved_settlements.discard(position.token_id)
                    self.circuit_breaker.record_trade_result(pnl >= 0)
                    self.kelly_sizer.record_trade_result(pnl >= 0)

                    # Update per-strategy circuit breaker
                    if position.strategy_id in self._strategy_metrics:
                        sm = self._strategy_metrics[position.strategy_id]
                        if pnl >= 0:
                            sm.consecutive_losses = 0
                            sm.size_multiplier = 1.0
                        else:
                            sm.consecutive_losses += 1
                            if sm.consecutive_losses >= 5:
                                sm.disabled_until = now + 600  # 10 min cooldown
                                sm.size_multiplier = 0.0
                                logger.warning(
                                    "Strategy disabled (5+ losses)",
                                    strategy=position.strategy_id,
                                    cooldown_s=600,
                                )
                            elif sm.consecutive_losses >= 3:
                                sm.size_multiplier = 0.5

                    # Emit event
                    await event_bus.publish(TradeEvent(
                        event_type=EventType.POSITION_CLOSED,
                        source="settlement",
                        asset=position.asset,
                        pnl=pnl,
                        payload={"token_id": position.token_id, "outcome": outcome},
                    ))

                    # Update metrics
                    metrics.update_pnl(
                        realized=self.portfolio.realized_pnl,
                        unrealized=0.0,
                        daily=self.portfolio.get_daily_performance().total_pnl,
                    )

                    logger.info(
                        "Position settled",
                        token_id=position.token_id[:16],
                        pnl=f"${pnl:.2f}",
                        strategy=position.strategy_id,
                    )

                    # Clean up pending fills for settled token
                    self._pending_fills = {
                        k: v for k, v in self._pending_fills.items()
                        if v.token_id != position.token_id
                    }

                except Exception as e:
                    logger.error(
                        "Settlement error",
                        token_id=position.token_id[:16],
                        error=str(e),
                    )

    async def _get_authoritative_outcome(self, token_id: str, condition_id: str) -> bool | None:
        """Resolve outcome from Gamma resolved market data with short cache."""
        now = time.time()
        cache_key = condition_id or token_id
        cached = self._resolution_cache.get(cache_key)
        if cached and now - cached[1] < 10.0:
            return cached[0]

        outcome = await self.market_discovery.get_market_resolution(
            token_id=token_id,
            condition_id=condition_id or None,
        )
        self._resolution_cache[cache_key] = (outcome, now)
        return outcome

    def _extract_reference_price(self, market_question: str) -> float | None:
        """Extract binary market strike/reference price from question text."""
        patterns = [
            r"be\s+\$?([\d]+(?:,\d{3})*(?:\.\d+)?)\s*(?:USDT|USD|or higher|or above|or lower|or below)",
            r"(?:above|below|over|under|at)\s+\$?([\d]+(?:,\d{3})*(?:\.\d+)?)",
            r"\$([\d]+(?:,\d{3})*(?:\.\d+)?)",
        ]
        for pattern in patterns:
            m = re.search(pattern, market_question, re.IGNORECASE)
            if not m:
                continue
            try:
                return float(m.group(1).replace(",", ""))
            except (ValueError, IndexError):
                continue
        return None

    def _resolve_binary_outcome(self, market_question: str, final_price: float) -> bool | None:
        """Resolve YES/NO outcome from final spot and question semantics."""
        strike = self._extract_reference_price(market_question)
        if strike is None:
            return None

        q = market_question.lower()
        # Directional semantics for common 15m market wording.
        if any(kw in q for kw in ("above", "higher", "up", "over")):
            return final_price >= strike
        if any(kw in q for kw in ("below", "lower", "down", "under")):
            return final_price < strike
        return None

    async def process_market(self, market: Market):
        """Process a single market across all strategies."""
        spot_price = await self.oracle.get_price(market.asset)
        if spot_price is None:
            return
        if not self.config.dry_run and self.oracle.is_price_stale(market.asset):
            logger.warning("Skipping stale market price", asset=market.asset)
            return

        # Live-safety: require settlement source to be available.
        if not self.config.dry_run:
            chainlink_price = await self.oracle.get_chainlink_price(market.asset)
            if chainlink_price is None:
                logger.warning(
                    "Skipping market: chainlink unavailable in live mode",
                    asset=market.asset,
                )
                return

        # === Sniper risk check ===
        sniper_size_mult = 1.0
        try:
            mid_price = spot_price  # Compare chainlink vs tradable reference feed
            if self.clob_client:
                sniper = await self.oracle.check_sniper_risk(
                    market.asset, mid_price
                )
                if sniper:
                    if sniper.risk_level == SniperRiskLevel.CRITICAL:
                        logger.warning(
                            "CRITICAL sniper risk - skipping market",
                            asset=market.asset,
                            divergence=f"{sniper.divergence_pct:.3f}%",
                        )
                        return
                    elif sniper.risk_level == SniperRiskLevel.ELEVATED:
                        sniper_size_mult = 0.5
        except Exception as e:
            logger.debug("Sniper check failed", error=str(e))

        book_yes = None
        book_no = None
        now = time.time()

        if self.clob_client:
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

        # Latency telemetry: spot tick timestamp vs current Polymarket book timestamp.
        spot_tick = self.oracle.get_cached_price(market.asset)
        if spot_tick and book_yes:
            lag_seconds = abs(float(book_yes.timestamp) - float(spot_tick.timestamp))
            metrics.record_cross_venue_lag(market.asset, lag_seconds)

        # Cache order books with timestamp for TTL
        if book_yes:
            self._order_book_cache[market.yes_token_id] = (book_yes, now)
        if book_no and market.no_token_id:
            self._order_book_cache[market.no_token_id] = (book_no, now)

        if self._research:
            self._research.record_market_snapshot({
                "asset": market.asset,
                "condition_id": market.condition_id,
                "yes_token_id": market.yes_token_id,
                "no_token_id": market.no_token_id,
                "spot_price": spot_price,
                "yes_best_bid": getattr(book_yes, "best_bid", None),
                "yes_best_ask": getattr(book_yes, "best_ask", None),
                "no_best_bid": getattr(book_no, "best_bid", None) if book_no else None,
                "no_best_ask": getattr(book_no, "best_ask", None) if book_no else None,
                "seconds_to_expiry": market.seconds_to_expiry,
            })

        # === Feed data to VPIN (optional proxy mode) ===
        if (
            self.config.vpin.enabled
            and self.config.vpin.use_order_book_proxy
            and book_yes
            and hasattr(book_yes, 'asks')
            and book_yes.asks
        ):
            # Approximation only; disabled by default because depth != trade flow.
            total_ask_vol = sum(s for _, s in book_yes.asks[:3]) if book_yes.asks else 0
            total_bid_vol = sum(s for _, s in book_yes.bids[:3]) if book_yes.bids else 0
            if total_ask_vol > 0:
                self.vpin.update(market.yes_token_id, total_ask_vol, is_buy=False)
            if total_bid_vol > 0:
                self.vpin.update(market.yes_token_id, total_bid_vol, is_buy=True)

        # === Calculate OBI ===
        obi_yes = 0.0
        if book_yes:
            obi_yes = book_yes.order_book_imbalance()

        context = TradeContext(
            market=market,
            spot_price=spot_price,
            order_book=book_yes,
            order_book_no=book_no,
            open_exposure=self.portfolio.total_exposure,
            token_exposure={
                p.token_id: p.size_usd
                for p in self.portfolio.positions.values()
                if not p.closed
            },
            daily_pnl=self.portfolio.get_daily_performance().total_pnl,
            obi_yes=obi_yes,
        )
        asset_exposure_map = self.portfolio.get_exposure_by_asset()
        open_positions_assets = {
            pos.asset: pos.size_usd
            for pos in self.portfolio.positions.values()
            if not pos.closed
        }

        all_orders = []
        for strategy in self.strategies_list:
            # Check per-strategy circuit breaker
            sm = self._strategy_metrics.get(strategy.name)
            if sm and sm.disabled_until > now:
                continue  # Strategy in cooldown

            try:
                orders = await strategy.scan(context)
                if orders:
                    for o in orders:
                        o['strategy'] = strategy.name
                        o['_sniper_size_mult'] = sniper_size_mult
                        o['_asset_exposure_map'] = asset_exposure_map
                        o['_open_positions_assets'] = open_positions_assets
                        edge_estimate = float(o.get("edge", 0.0))
                        metrics.record_strategy_edge(strategy.name, market.asset, edge_estimate)
                        if self._research:
                            self._research.record_signal({
                                "strategy": strategy.name,
                                "asset": market.asset,
                                "condition_id": market.condition_id,
                                "token_id": o.get("token_id"),
                                "side": o.get("side"),
                                "price": o.get("price"),
                                "size": o.get("size"),
                                "edge": edge_estimate,
                                "reason": o.get("reason", ""),
                            })
                    all_orders.extend(orders)
            except Exception as e:
                logger.error(f"Strategy {strategy.name} failed", error=str(e))

        # === Detect arb legs and handle atomicity ===
        arb_orders = [o for o in all_orders if 'arb' in o.get('reason', '').lower()]
        non_arb_orders = [o for o in all_orders if 'arb' not in o.get('reason', '').lower()]

        # Process arb orders atomically (matched sizes)
        if len(arb_orders) >= 2:
            await self._execute_arb_atomic(arb_orders, market)

        # Process non-arb orders normally
        for order_params in non_arb_orders:
            await self.execute_order_params(order_params, market)

    async def _execute_arb_atomic(self, arb_orders: list[dict], market: Market) -> None:
        """Execute arb legs atomically with matched sizes."""
        # Match sizes: use minimum across all legs
        min_size = min(o.get('size', 5.0) for o in arb_orders)
        for o in arb_orders:
            o['size'] = min_size

        # Submit all legs concurrently
        tasks = []
        for params in arb_orders:
            tasks.append(self.execute_order_params(params, market))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log any failures
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "Arb leg failed",
                    leg=i,
                    error=str(result),
                )

    async def execute_order_params(self, params: dict, market: Market):
        """Execute trade with RiskGate validation, Kelly sizing, VPIN, and OBI."""
        price = params.get('price')
        if not price or price <= 0 or price >= 1:
            return

        strategy_name = params.get('strategy', 'ensemble')
        token_id = params.get('token_id', market.yes_token_id)
        sniper_size_mult = params.pop('_sniper_size_mult', 1.0)
        asset_exposure_map = params.pop('_asset_exposure_map', None)
        open_positions_assets = params.pop('_open_positions_assets', None)

        # === RiskGate validation ===
        signal = signal_from_order_params(params, strategy_id=strategy_name)
        signal.asset = market.asset
        signal.market_question = market.question

        gate_result = self.risk_gate.validate(signal)
        if not gate_result.passed:
            metrics.record_signal(market.asset, signal.side.value, passed=False,
                                  rejection_reason=gate_result.rejection_reason)
            return

        metrics.record_signal(market.asset, signal.side.value, passed=True)

        # Use validated size from RiskGate as base
        base_size = gate_result.intent.validated_size_usd

        # === VPIN toxicity filter ===
        vpin_mult = 1.0
        vpin_result = self.vpin.get_vpin(params.get('token_id', market.yes_token_id))
        if self.config.vpin.enabled:
            if self.config.vpin.require_reliable and not vpin_result.is_reliable:
                vpin_mult = 1.0
            else:
                if vpin_result.should_halt:
                    logger.warning(
                        "VPIN halt - skipping order",
                        token_id=params.get('token_id', '')[:16],
                        vpin=f"{vpin_result.vpin:.3f}",
                        toxicity=vpin_result.toxicity_level.value,
                    )
                    return
                vpin_mult = vpin_result.size_multiplier

        # === Per-strategy circuit breaker ===
        strategy_mult = 1.0
        sm = self._strategy_metrics.get(strategy_name)
        if sm:
            strategy_mult = sm.size_multiplier
            sm.signals_generated += 1

        # === OBI signal integration ===
        obi_mult = 1.0
        obi_confidence_boost = 0.0
        if self.config.obi.enabled:
            obi = params.get('_obi_yes', 0.0)
            is_buy = params.get('side', 'BUY') == 'BUY'
            # If OBI agrees with signal direction, boost confidence
            if (is_buy and obi > self.config.obi.weak_threshold) or \
               (not is_buy and obi < -self.config.obi.weak_threshold):
                obi_confidence_boost = 0.05  # +5% confidence
            # If OBI contradicts signal, reduce size
            elif (is_buy and obi < -self.config.obi.weak_threshold) or \
                 (not is_buy and obi > self.config.obi.weak_threshold):
                obi_mult = 0.8  # -20% size

        # === Kelly-optimal position size ===
        is_arb = 'arb' in strategy_name.lower()
        is_buy_yes = (params.get('side', 'BUY') == 'BUY' and
                      params.get('token_id', market.yes_token_id) == market.yes_token_id)

        current_drawdown = 0.0
        if self.portfolio.peak_capital > 0:
            current_drawdown = (
                (self.portfolio.peak_capital - self.portfolio.current_capital)
                / self.portfolio.peak_capital
            )

        if isinstance(asset_exposure_map, dict):
            current_asset_exposure = float(asset_exposure_map.get(market.asset, 0.0))
        else:
            current_asset_exposure = sum(
                pos.size_usd for pos in self.portfolio.positions.values()
                if pos.asset == market.asset and not pos.closed
            )

        if is_arb:
            win_prob = 0.98
            edge_confidence = 0.95
        else:
            edge_estimate = params.get('edge', 0.04)
            base_prob = price if is_buy_yes else (1.0 - price)
            win_prob = max(0.01, min(0.99, base_prob + edge_estimate))
            edge_confidence = min(1.0, 0.6 + obi_confidence_boost)

        current_vol = self.config.models.default_volatility.get(market.asset, 0.70)

        kelly_result = self.kelly_sizer.calculate(
            win_prob=win_prob,
            market_price=price,
            side="YES" if params.get('side', 'BUY') == 'BUY' else "NO",
            current_asset_exposure=current_asset_exposure,
            current_volatility=current_vol,
            current_drawdown_pct=current_drawdown,
            edge_confidence=edge_confidence,
        )

        kelly_size = kelly_result.recommended_size_usd
        strategy_size = params.get('size', kelly_size)
        final_size = min(base_size, strategy_size, kelly_size, self.config.trading.max_trade_usd)

        # Apply all multipliers
        final_size *= vpin_mult * sniper_size_mult * strategy_mult * obi_mult

        # Apply correlation adjustment
        if isinstance(open_positions_assets, dict):
            open_positions = open_positions_assets
        else:
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

        # Per-token concentration cap across strategies.
        current_token_exposure = 0.0
        token_pos = self.portfolio.positions.get(token_id)
        if token_pos and not token_pos.closed:
            current_token_exposure = token_pos.size_usd
        max_token_exposure = float(
            getattr(
                self.config.trading,
                "max_position_per_token_usd",
                self.config.trading.max_total_exposure_usd,
            )
        )
        remaining_token_capacity = max_token_exposure - current_token_exposure
        if remaining_token_capacity <= 0:
            return
        final_size = min(final_size, remaining_token_capacity)

        # Final size check
        if final_size < self.config.trading.min_order_size_usd:
            return

        # === Create and submit order ===
        order = self.order_manager.create_order(
            token_id=token_id,
            side=OrderSide.BUY if params.get('side', 'BUY') == 'BUY' else OrderSide.SELL,
            price=price,
            size=final_size,
            order_type=OrderType.GTC,
            strategy_id=strategy_name,
        )

        # Set leg context ID for LeggedHedge tracking
        leg_ctx_id = params.get('_leg_context_id')
        if leg_ctx_id:
            order._leg_context_id = leg_ctx_id

        success = await self.order_manager.submit_order(order)
        if success:
            self.metrics.orders_placed += 1
            if sm:
                sm.orders_placed += 1
            self._track_order_metadata(order, market, strategy_name)
            spread_maker = self.strategies.get("SpreadMaker")
            if (
                strategy_name == "SpreadMaker"
                and spread_maker
                and hasattr(spread_maker, "track_order")
            ):
                spread_maker.track_order(order.token_id, order.client_order_id)

            # Emit event
            await event_bus.publish(OrderEvent(
                event_type=EventType.ORDER_SUBMITTED,
                source=strategy_name,
                client_order_id=order.client_order_id,
                token_id=order.token_id,
                side=order.side.value,
                price=order.price,
                size=order.size,
            ))

            # Record metrics
            metrics.record_order(
                asset=market.asset,
                side=order.side.value,
                order_type=order.order_type.value,
                size_usd=final_size,
            )

            logger.info(
                f"Placed {strategy_name} order",
                asset=market.asset,
                price=price,
                size=f"${final_size:.2f}",
                kelly_fraction=f"{kelly_result.kelly_fraction:.4f}",
                vpin=f"{vpin_result.vpin:.3f}",
            )

    def _track_order_metadata(self, order: Order, market: Market, strategy_name: str = ""):
        """Track metadata for Portfolio fill handling."""
        side = PositionSide.LONG_YES if order.token_id == market.yes_token_id else PositionSide.LONG_NO

        self._pending_fills[order.client_order_id] = PendingFill(
            token_id=order.token_id,
            condition_id=market.condition_id,
            asset=market.asset,
            side=side,
            price=order.price,
            size_usd=order.size,
            market_question=market.question,
            expires_at=market.end_date.timestamp(),
            strategy_name=strategy_name,
        )

    def handle_fill(self, client_order_id: str, fill_size: float, fill_price: float) -> None:
        """
        Process fill and update portfolio.

        Args:
            client_order_id: The order that was filled
            fill_size: Size filled in SHARES (not USD)
            fill_price: Price per share at fill
        """
        with self._fill_lock:
            fill_key = f"{client_order_id}_{fill_size}_{fill_price}"
            if fill_key in self._processed_fill_ids:
                return
            self._processed_fill_ids.add(fill_key)
            self._processed_fill_fifo.append(fill_key)

            while len(self._processed_fill_fifo) > self._max_processed_fills:
                old_key = self._processed_fill_fifo.popleft()
                self._processed_fill_ids.discard(old_key)

            # Convert shares to USD for OrderManager
            fill_usd = float(fill_size) * float(fill_price)
            self.order_manager.handle_fill(client_order_id, fill_usd, fill_price)

            metadata = self._pending_fills.get(client_order_id)
            if metadata:
                self.portfolio.open_position(
                    token_id=metadata.token_id,
                    condition_id=metadata.condition_id,
                    asset=metadata.asset,
                    market_question=metadata.market_question,
                    side=metadata.side,
                    price=fill_price,
                    size_usd=fill_usd,
                    expires_at=metadata.expires_at,
                    strategy_id=metadata.strategy_name,
                )
                self.metrics.orders_filled += 1

                # Update per-strategy metrics
                sm = self._strategy_metrics.get(metadata.strategy_name)
                if sm:
                    sm.orders_filled += 1

                # Emit event (queued for background processing by event bus run loop)
                event_bus.publish_sync(OrderEvent(
                    event_type=EventType.ORDER_FILLED,
                    source="fill_handler",
                    client_order_id=client_order_id,
                    token_id=metadata.token_id,
                    side=metadata.side.value,
                    price=fill_price,
                    size=fill_usd,
                    filled_size=fill_usd,
                ))

                # Record metrics
                metrics.record_fill(
                    asset=metadata.asset,
                    side=metadata.side.value,
                    size_usd=fill_usd,
                )

                # Record execution latency
                order = self.order_manager.get_order(client_order_id)
                if order and order.submitted_at:
                    latency = time.time() - order.submitted_at
                    metrics.record_latency("order", latency)

                if order and order.state.is_terminal:
                    self._pending_fills.pop(client_order_id, None)

                if self._research:
                    self._research.record_fill({
                        "client_order_id": client_order_id,
                        "token_id": metadata.token_id,
                        "condition_id": metadata.condition_id,
                        "asset": metadata.asset,
                        "side": metadata.side.value,
                        "fill_price": fill_price,
                        "fill_usd": fill_usd,
                        "strategy": metadata.strategy_name,
                    })

    def stop(self) -> None:
        self._running = False

    def get_metrics(self) -> dict:
        return {
            "orders_placed": self.metrics.orders_placed,
            "orders_filled": self.metrics.orders_filled,
            "portfolio": self.portfolio.summary(),
            "circuit_breaker_tripped": not self.circuit_breaker.can_trade(),
            "strategy_metrics": {
                name: {
                    "signals": sm.signals_generated,
                    "orders": sm.orders_placed,
                    "fills": sm.orders_filled,
                    "consecutive_losses": sm.consecutive_losses,
                    "size_multiplier": sm.size_multiplier,
                }
                for name, sm in self._strategy_metrics.items()
            },
        }
