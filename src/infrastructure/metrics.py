"""
Prometheus metrics for observability.

Exposes metrics on /metrics endpoint for Prometheus scraping.

Metrics Categories:
- Trading: signals, orders, fills, PnL
- Latency: oracle, order placement, WebSocket
- Risk: drawdown, circuit breaker state
- Model: predictions, Brier score
"""

import time
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    start_http_server,
    REGISTRY,
)

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Trading Metrics
# =============================================================================

SIGNALS_TOTAL = Counter(
    "polymarket_signals_total",
    "Total trade signals generated",
    ["asset", "side"],
)

SIGNALS_PASSED_GATE = Counter(
    "polymarket_signals_passed_gate_total",
    "Signals that passed NO-TRADE gate",
    ["asset", "side"],
)

SIGNALS_REJECTED = Counter(
    "polymarket_signals_rejected_total",
    "Signals rejected by NO-TRADE gate",
    ["asset", "reason"],
)

ORDERS_PLACED = Counter(
    "polymarket_orders_placed_total",
    "Total orders placed",
    ["asset", "side", "order_type"],
)

ORDERS_FILLED = Counter(
    "polymarket_orders_filled_total",
    "Total orders filled",
    ["asset", "side"],
)

ORDERS_CANCELED = Counter(
    "polymarket_orders_canceled_total",
    "Total orders canceled",
    ["asset"],
)

ORDERS_REJECTED = Counter(
    "polymarket_orders_rejected_total",
    "Total orders rejected by exchange",
    ["asset", "reason"],
)

FILL_AMOUNT_USD = Counter(
    "polymarket_fill_amount_usd_total",
    "Total USD value of fills",
    ["asset", "side"],
)

# =============================================================================
# PnL Metrics
# =============================================================================

REALIZED_PNL = Gauge(
    "polymarket_realized_pnl_usd",
    "Realized PnL in USD",
)

UNREALIZED_PNL = Gauge(
    "polymarket_unrealized_pnl_usd",
    "Unrealized PnL in USD",
)

DAILY_PNL = Gauge(
    "polymarket_daily_pnl_usd",
    "Daily PnL in USD",
)

PORTFOLIO_VALUE = Gauge(
    "polymarket_portfolio_value_usd",
    "Current portfolio value in USD",
)

OPEN_POSITIONS = Gauge(
    "polymarket_open_positions",
    "Number of open positions",
)

EXPOSURE_USD = Gauge(
    "polymarket_exposure_usd",
    "Total USD exposure",
    ["asset"],
)

# =============================================================================
# Latency Metrics
# =============================================================================

ORACLE_LATENCY = Histogram(
    "polymarket_oracle_latency_seconds",
    "Oracle price fetch latency",
    ["source"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

ORDER_LATENCY = Histogram(
    "polymarket_order_latency_seconds",
    "Order placement latency",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)

WS_MESSAGE_LATENCY = Histogram(
    "polymarket_ws_message_latency_seconds",
    "WebSocket message processing latency",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1],
)

ORACLE_AGE = Gauge(
    "polymarket_oracle_age_seconds",
    "Age of last oracle price",
    ["asset"],
)

# =============================================================================
# Risk Metrics
# =============================================================================

DRAWDOWN_PCT = Gauge(
    "polymarket_drawdown_pct",
    "Current drawdown percentage",
)

CIRCUIT_BREAKER_STATE = Gauge(
    "polymarket_circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=soft, 2=hard)",
    ["breaker_type"],
)

RATE_LIMIT_USAGE = Gauge(
    "polymarket_rate_limit_usage_pct",
    "Rate limit usage percentage",
    ["bucket"],
)

KELLY_FRACTION = Gauge(
    "polymarket_kelly_fraction",
    "Current Kelly fraction being used",
)

# =============================================================================
# Model Metrics
# =============================================================================

MODEL_EDGE = Histogram(
    "polymarket_model_edge",
    "Model edge (model_prob - market_price)",
    ["asset"],
    buckets=[-0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.3],
)

MODEL_PROBABILITY = Histogram(
    "polymarket_model_probability",
    "Model probability predictions",
    ["asset"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
)

BRIER_SCORE = Gauge(
    "polymarket_brier_score",
    "Cumulative Brier score (lower is better)",
)

PREDICTIONS_TOTAL = Counter(
    "polymarket_predictions_total",
    "Total predictions made",
)

PREDICTIONS_CORRECT = Counter(
    "polymarket_predictions_correct_total",
    "Total correct predictions (> 50% prob and won)",
)

# =============================================================================
# System Metrics
# =============================================================================

BOT_INFO = Info(
    "polymarket_bot",
    "Bot information",
)

UPTIME_SECONDS = Gauge(
    "polymarket_uptime_seconds",
    "Bot uptime in seconds",
)

ITERATION_COUNT = Counter(
    "polymarket_iteration_count_total",
    "Total strategy iterations",
)


class MetricsCollector:
    """
    Collects and exposes Prometheus metrics.
    
    Usage:
        collector = MetricsCollector()
        collector.start_server(port=9090)
        
        # Record metrics
        collector.record_signal("BTC", "BUY_YES", passed=True)
        collector.record_order("BTC", "BUY", size=10.0)
        collector.update_pnl(realized=50.0, unrealized=10.0)
    """
    
    def __init__(self):
        self._start_time = time.time()
    
    def start_server(self, port: int = 9090) -> None:
        """Start the Prometheus metrics HTTP server."""
        try:
            start_http_server(port)
            logger.info("Metrics server started", port=port)
        except Exception as e:
            logger.error("Failed to start metrics server", error=str(e))
    
    def record_signal(
        self,
        asset: str,
        side: str,
        passed: bool,
        rejection_reason: str = "",
    ) -> None:
        """Record a trade signal."""
        SIGNALS_TOTAL.labels(asset=asset, side=side).inc()
        
        if passed:
            SIGNALS_PASSED_GATE.labels(asset=asset, side=side).inc()
        else:
            SIGNALS_REJECTED.labels(asset=asset, reason=rejection_reason).inc()
    
    def record_order(
        self,
        asset: str,
        side: str,
        order_type: str = "GTC",
        size_usd: float = 0,
    ) -> None:
        """Record an order placement."""
        ORDERS_PLACED.labels(asset=asset, side=side, order_type=order_type).inc()
    
    def record_fill(
        self,
        asset: str,
        side: str,
        size_usd: float,
    ) -> None:
        """Record an order fill."""
        ORDERS_FILLED.labels(asset=asset, side=side).inc()
        FILL_AMOUNT_USD.labels(asset=asset, side=side).inc(size_usd)
    
    def record_cancel(self, asset: str) -> None:
        """Record an order cancellation."""
        ORDERS_CANCELED.labels(asset=asset).inc()
    
    def record_rejection(self, asset: str, reason: str) -> None:
        """Record an order rejection."""
        ORDERS_REJECTED.labels(asset=asset, reason=reason).inc()
    
    def update_pnl(
        self,
        realized: float,
        unrealized: float,
        daily: float = 0,
    ) -> None:
        """Update PnL gauges."""
        REALIZED_PNL.set(realized)
        UNREALIZED_PNL.set(unrealized)
        DAILY_PNL.set(daily)
    
    def update_portfolio(
        self,
        value: float,
        open_positions: int,
        exposure_by_asset: dict[str, float],
    ) -> None:
        """Update portfolio gauges."""
        PORTFOLIO_VALUE.set(value)
        OPEN_POSITIONS.set(open_positions)
        
        for asset, exposure in exposure_by_asset.items():
            EXPOSURE_USD.labels(asset=asset).set(exposure)
    
    def record_latency(
        self,
        metric_type: str,
        latency_seconds: float,
        **labels,
    ) -> None:
        """Record a latency measurement."""
        if metric_type == "oracle":
            ORACLE_LATENCY.labels(**labels).observe(latency_seconds)
        elif metric_type == "order":
            ORDER_LATENCY.observe(latency_seconds)
        elif metric_type == "ws_message":
            WS_MESSAGE_LATENCY.observe(latency_seconds)
    
    def update_oracle_age(self, asset: str, age_seconds: float) -> None:
        """Update oracle age gauge."""
        ORACLE_AGE.labels(asset=asset).set(age_seconds)
    
    def update_risk(
        self,
        drawdown_pct: float,
        circuit_breaker_states: dict[str, int],
        rate_limit_usage: dict[str, float],
    ) -> None:
        """Update risk gauges."""
        DRAWDOWN_PCT.set(drawdown_pct)
        
        for breaker, state in circuit_breaker_states.items():
            CIRCUIT_BREAKER_STATE.labels(breaker_type=breaker).set(state)
        
        for bucket, usage in rate_limit_usage.items():
            RATE_LIMIT_USAGE.labels(bucket=bucket).set(usage)
    
    def record_model_prediction(
        self,
        asset: str,
        probability: float,
        edge: float,
    ) -> None:
        """Record a model prediction."""
        MODEL_PROBABILITY.labels(asset=asset).observe(probability)
        MODEL_EDGE.labels(asset=asset).observe(edge)
        PREDICTIONS_TOTAL.inc()
    
    def record_prediction_outcome(self, correct: bool) -> None:
        """Record prediction outcome."""
        if correct:
            PREDICTIONS_CORRECT.inc()
    
    def update_brier_score(self, score: float) -> None:
        """Update Brier score gauge."""
        BRIER_SCORE.set(score)
    
    def update_kelly(self, fraction: float) -> None:
        """Update Kelly fraction gauge."""
        KELLY_FRACTION.set(fraction)
    
    def set_bot_info(self, version: str, environment: str) -> None:
        """Set bot info labels."""
        BOT_INFO.info({
            "version": version,
            "environment": environment,
        })
    
    def increment_iteration(self) -> None:
        """Increment iteration counter."""
        ITERATION_COUNT.inc()
        UPTIME_SECONDS.set(time.time() - self._start_time)


# Pre-instantiated collector
metrics = MetricsCollector()
