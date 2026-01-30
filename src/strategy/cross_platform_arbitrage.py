"""
Cross-Platform Arbitrage Detection for prediction markets.

Detects price discrepancies between prediction market platforms:
- Polymarket vs Kalshi
- Polymarket vs PredictIt
- Different Polymarket markets for same event

Based on 2025 research showing:
- $40M+ extracted in risk-free profits from April 2024 to April 2025
- Top 3 arbitrage wallets made $4.2M combined

References:
- ChainCatcher 2025: Polymarket Six Major Profit Models
- DataWallet: Top 10 Polymarket Trading Strategies
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine

import aiohttp

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class Platform(Enum):
    """Supported prediction market platforms."""
    POLYMARKET = "polymarket"
    KALSHI = "kalshi"
    PREDICTIT = "predictit"
    METACULUS = "metaculus"


class ArbType(Enum):
    """Types of arbitrage opportunities."""
    BINARY_MISPRICING = "binary_mispricing"      # YES + NO ≠ 1 on same platform
    CROSS_PLATFORM = "cross_platform"            # Same event, different prices
    CORRELATED_EVENTS = "correlated_events"      # Logically related events
    TEMPORAL = "temporal"                        # Same event, different resolution times


@dataclass
class PlatformQuote:
    """Price quote from a prediction market platform."""
    platform: Platform
    market_id: str
    event_description: str
    yes_price: float
    no_price: float
    yes_liquidity_usd: float = 0.0
    no_liquidity_usd: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def implied_prob(self) -> float:
        """Implied probability from YES price."""
        return self.yes_price

    @property
    def spread(self) -> float:
        """Bid-ask spread approximation."""
        return abs((self.yes_price + self.no_price) - 1.0)

    @property
    def is_complete(self) -> bool:
        """Check if market is complete (YES + NO = 1)."""
        return abs(self.yes_price + self.no_price - 1.0) < 0.01


@dataclass
class CrossPlatformArbOpportunity:
    """Cross-platform arbitrage opportunity."""

    arb_type: ArbType
    quote_a: PlatformQuote
    quote_b: PlatformQuote

    # Profit analysis
    gross_profit_pct: float
    estimated_fees_pct: float
    net_profit_pct: float

    # Execution details
    action_a: str  # "BUY_YES" or "BUY_NO"
    action_b: str
    max_size_usd: float

    # Risk factors
    execution_risk: float = 0.0  # 0-1 scale
    timing_risk: float = 0.0     # Risk of price moving

    @property
    def is_profitable(self) -> bool:
        return self.net_profit_pct > 0

    @property
    def annualized_return(self) -> float:
        """Annualized return assuming daily turnover."""
        return self.net_profit_pct * 365 * 100  # Percentage


@dataclass
class EventMatch:
    """Match between events on different platforms."""
    polymarket_market_id: str
    other_platform: Platform
    other_market_id: str
    confidence: float  # 0-1, how confident we are these are same events
    resolution_time_diff_hours: float = 0.0


class CrossPlatformArbDetector:
    """
    Detects arbitrage opportunities across prediction market platforms.

    Strategy:
    1. Monitor similar events across platforms
    2. Detect price discrepancies exceeding fee thresholds
    3. Calculate risk-adjusted profit potential
    4. Generate execution signals

    Usage:
        detector = CrossPlatformArbDetector(
            polymarket_fee=0.02,
            kalshi_fee=0.01,
        )

        # Add quotes
        detector.add_quote(polymarket_quote)
        detector.add_quote(kalshi_quote)

        # Check for opportunities
        opps = detector.find_opportunities()
    """

    def __init__(
        self,
        polymarket_fee: float = 0.02,
        kalshi_fee: float = 0.01,
        predictit_fee: float = 0.10,  # PredictIt has 10% profit fee
        min_profit_threshold: float = 0.005,  # 0.5% minimum profit
        min_confidence: float = 0.9,  # 90% confidence events match
    ):
        self.fees = {
            Platform.POLYMARKET: polymarket_fee,
            Platform.KALSHI: kalshi_fee,
            Platform.PREDICTIT: predictit_fee,
        }
        self.min_profit_threshold = min_profit_threshold
        self.min_confidence = min_confidence

        # Store quotes by event hash
        self._quotes: dict[str, list[PlatformQuote]] = {}
        self._event_matches: list[EventMatch] = []

    def add_quote(self, quote: PlatformQuote, event_key: str | None = None) -> None:
        """
        Add a platform quote for analysis.

        Args:
            quote: Price quote from a platform
            event_key: Optional key to group same events across platforms
        """
        key = event_key or self._generate_event_key(quote.event_description)

        if key not in self._quotes:
            self._quotes[key] = []

        # Update or add quote for this platform
        existing_idx = None
        for i, q in enumerate(self._quotes[key]):
            if q.platform == quote.platform:
                existing_idx = i
                break

        if existing_idx is not None:
            self._quotes[key][existing_idx] = quote
        else:
            self._quotes[key].append(quote)

    def _generate_event_key(self, description: str) -> str:
        """Generate a normalized key for event matching."""
        # Simple normalization - in production, use NLP for better matching
        normalized = description.lower()
        normalized = ''.join(c for c in normalized if c.isalnum() or c.isspace())
        words = normalized.split()
        # Remove common words
        stopwords = {'will', 'the', 'be', 'to', 'in', 'on', 'by', 'at', 'for'}
        words = [w for w in words if w not in stopwords]
        return '_'.join(sorted(words[:5]))  # First 5 significant words

    def add_event_match(self, match: EventMatch) -> None:
        """Manually add a known event match across platforms."""
        self._event_matches.append(match)

    def find_opportunities(self) -> list[CrossPlatformArbOpportunity]:
        """
        Find all current arbitrage opportunities.

        Returns:
            List of profitable arbitrage opportunities
        """
        opportunities = []

        for event_key, quotes in self._quotes.items():
            if len(quotes) < 2:
                continue

            # Check all pairs
            for i in range(len(quotes)):
                for j in range(i + 1, len(quotes)):
                    opp = self._check_pair(quotes[i], quotes[j])
                    if opp and opp.is_profitable:
                        opportunities.append(opp)

        # Sort by profit potential
        opportunities.sort(key=lambda x: x.net_profit_pct, reverse=True)
        return opportunities

    def _check_pair(
        self,
        quote_a: PlatformQuote,
        quote_b: PlatformQuote,
    ) -> CrossPlatformArbOpportunity | None:
        """
        Check for arbitrage between two quotes.

        The key insight: if we can buy YES on one platform and NO on another
        such that total cost < 1.0, we have guaranteed profit.
        """
        fee_a = self.fees.get(quote_a.platform, 0.02)
        fee_b = self.fees.get(quote_b.platform, 0.02)

        # Strategy 1: Buy YES on cheaper platform, NO on more expensive
        # If yes_a + no_b < 1, profit exists
        cost_1 = quote_a.yes_price * (1 + fee_a) + quote_b.no_price * (1 + fee_b)
        profit_1 = 1.0 - cost_1

        # Strategy 2: Buy NO on platform A, YES on platform B
        cost_2 = quote_a.no_price * (1 + fee_a) + quote_b.yes_price * (1 + fee_b)
        profit_2 = 1.0 - cost_2

        # Choose better strategy
        if profit_1 >= profit_2 and profit_1 > self.min_profit_threshold:
            return CrossPlatformArbOpportunity(
                arb_type=ArbType.CROSS_PLATFORM,
                quote_a=quote_a,
                quote_b=quote_b,
                gross_profit_pct=profit_1 + fee_a + fee_b,
                estimated_fees_pct=fee_a + fee_b,
                net_profit_pct=profit_1,
                action_a="BUY_YES",
                action_b="BUY_NO",
                max_size_usd=min(quote_a.yes_liquidity_usd, quote_b.no_liquidity_usd),
                execution_risk=self._estimate_execution_risk(quote_a, quote_b),
            )
        elif profit_2 > self.min_profit_threshold:
            return CrossPlatformArbOpportunity(
                arb_type=ArbType.CROSS_PLATFORM,
                quote_a=quote_a,
                quote_b=quote_b,
                gross_profit_pct=profit_2 + fee_a + fee_b,
                estimated_fees_pct=fee_a + fee_b,
                net_profit_pct=profit_2,
                action_a="BUY_NO",
                action_b="BUY_YES",
                max_size_usd=min(quote_a.no_liquidity_usd, quote_b.yes_liquidity_usd),
                execution_risk=self._estimate_execution_risk(quote_a, quote_b),
            )

        return None

    def _estimate_execution_risk(
        self,
        quote_a: PlatformQuote,
        quote_b: PlatformQuote,
    ) -> float:
        """Estimate execution risk based on liquidity and staleness."""
        # Lower liquidity = higher risk
        min_liquidity = min(
            quote_a.yes_liquidity_usd + quote_a.no_liquidity_usd,
            quote_b.yes_liquidity_usd + quote_b.no_liquidity_usd,
        )
        liquidity_risk = max(0, 1.0 - min_liquidity / 10000)  # $10k = low risk

        # Stale quotes = higher risk
        now = datetime.now(timezone.utc)
        age_a = (now - quote_a.timestamp).total_seconds()
        age_b = (now - quote_b.timestamp).total_seconds()
        staleness_risk = min(1.0, max(age_a, age_b) / 60)  # 60s = high risk

        return (liquidity_risk + staleness_risk) / 2


# =============================================================================
# Market Regime Detection
# =============================================================================

class MarketRegime(Enum):
    """Market regime classifications."""
    BULL = "bull"           # Strong upward trend
    BEAR = "bear"           # Strong downward trend
    SIDEWAYS = "sideways"   # Range-bound, low trend
    HIGH_VOL = "high_vol"   # High volatility regardless of direction
    CRISIS = "crisis"       # Extreme moves, correlations spike


@dataclass
class RegimeIndicators:
    """Indicators used for regime detection."""

    # Trend indicators
    price_change_24h_pct: float = 0.0
    price_change_7d_pct: float = 0.0
    sma_20_vs_sma_50: float = 0.0  # Short MA vs Long MA

    # Volatility indicators
    realized_vol_24h: float = 0.6
    realized_vol_7d: float = 0.6
    vol_ratio: float = 1.0  # Short vol / long vol

    # Market breadth
    correlation_with_btc: float = 0.5

    # Sentiment proxies
    funding_rate: float = 0.0
    open_interest_change_pct: float = 0.0


@dataclass
class RegimeClassification:
    """Result of regime classification."""
    regime: MarketRegime
    confidence: float  # 0-1
    indicators: RegimeIndicators
    strategy_adjustments: dict[str, float] = field(default_factory=dict)

    @property
    def should_reduce_exposure(self) -> bool:
        """Whether to reduce position sizes in this regime."""
        return self.regime in (MarketRegime.HIGH_VOL, MarketRegime.CRISIS)


class RegimeDetector:
    """
    Detects current market regime for strategy adjustment.

    Different regimes require different trading approaches:
    - BULL: More aggressive, favor long positions
    - BEAR: More defensive, tighter stops
    - SIDEWAYS: Range trading, mean reversion
    - HIGH_VOL: Reduce size, wider stops
    - CRISIS: Halt or minimal trading

    Usage:
        detector = RegimeDetector()
        detector.update_price("BTC", 100000)
        regime = detector.classify("BTC")

        if regime.should_reduce_exposure:
            size *= 0.5
    """

    def __init__(
        self,
        vol_threshold_high: float = 1.0,    # 100% annualized
        vol_threshold_crisis: float = 2.0,  # 200% annualized
        trend_threshold: float = 0.05,      # 5% for trend classification
    ):
        self.vol_threshold_high = vol_threshold_high
        self.vol_threshold_crisis = vol_threshold_crisis
        self.trend_threshold = trend_threshold

        # Price history
        self._prices: dict[str, list[tuple[datetime, float]]] = {}

    def update_price(self, asset: str, price: float) -> None:
        """Add a price observation."""
        now = datetime.now(timezone.utc)

        if asset not in self._prices:
            self._prices[asset] = []

        self._prices[asset].append((now, price))

        # Keep 7 days of minute data max
        cutoff = now.timestamp() - 7 * 24 * 60 * 60
        self._prices[asset] = [
            (t, p) for t, p in self._prices[asset]
            if t.timestamp() > cutoff
        ]

    def classify(self, asset: str) -> RegimeClassification:
        """
        Classify current market regime for an asset.

        Args:
            asset: Asset symbol (e.g., "BTC")

        Returns:
            RegimeClassification with regime and adjustments
        """
        indicators = self._calculate_indicators(asset)

        # Classification logic
        regime = MarketRegime.SIDEWAYS
        confidence = 0.5

        # Check for crisis first (highest priority)
        if indicators.realized_vol_24h > self.vol_threshold_crisis:
            regime = MarketRegime.CRISIS
            confidence = min(1.0, indicators.realized_vol_24h / (self.vol_threshold_crisis * 1.5))

        # Check for high volatility
        elif indicators.realized_vol_24h > self.vol_threshold_high:
            regime = MarketRegime.HIGH_VOL
            confidence = (indicators.realized_vol_24h - self.vol_threshold_high) / \
                        (self.vol_threshold_crisis - self.vol_threshold_high)

        # Check for trend
        elif abs(indicators.price_change_24h_pct) > self.trend_threshold:
            if indicators.price_change_24h_pct > 0:
                regime = MarketRegime.BULL
            else:
                regime = MarketRegime.BEAR
            confidence = min(1.0, abs(indicators.price_change_24h_pct) / (self.trend_threshold * 2))

        # Default to sideways
        else:
            regime = MarketRegime.SIDEWAYS
            confidence = 1.0 - abs(indicators.price_change_24h_pct) / self.trend_threshold

        # Calculate strategy adjustments
        adjustments = self._calculate_adjustments(regime, indicators)

        return RegimeClassification(
            regime=regime,
            confidence=confidence,
            indicators=indicators,
            strategy_adjustments=adjustments,
        )

    def _calculate_indicators(self, asset: str) -> RegimeIndicators:
        """Calculate regime indicators from price history."""
        prices = self._prices.get(asset, [])

        if len(prices) < 2:
            return RegimeIndicators()

        current_price = prices[-1][1]

        # Find prices at different lookbacks
        now = datetime.now(timezone.utc)
        price_24h_ago = self._get_price_at_lookback(prices, 24 * 60)
        price_7d_ago = self._get_price_at_lookback(prices, 7 * 24 * 60)

        # Calculate changes
        change_24h = (current_price - price_24h_ago) / price_24h_ago if price_24h_ago > 0 else 0
        change_7d = (current_price - price_7d_ago) / price_7d_ago if price_7d_ago > 0 else 0

        # Calculate volatility
        vol_24h = self._calculate_volatility(prices, 24 * 60)
        vol_7d = self._calculate_volatility(prices, 7 * 24 * 60)

        return RegimeIndicators(
            price_change_24h_pct=change_24h,
            price_change_7d_pct=change_7d,
            realized_vol_24h=vol_24h,
            realized_vol_7d=vol_7d,
            vol_ratio=vol_24h / vol_7d if vol_7d > 0 else 1.0,
        )

    def _get_price_at_lookback(
        self,
        prices: list[tuple[datetime, float]],
        minutes_ago: int,
    ) -> float:
        """Get price at a specific lookback period."""
        if not prices:
            return 0.0

        target_time = datetime.now(timezone.utc).timestamp() - minutes_ago * 60

        # Find closest price
        closest = prices[0][1]
        closest_diff = float('inf')

        for t, p in prices:
            diff = abs(t.timestamp() - target_time)
            if diff < closest_diff:
                closest_diff = diff
                closest = p

        return closest

    def _calculate_volatility(
        self,
        prices: list[tuple[datetime, float]],
        lookback_minutes: int,
    ) -> float:
        """Calculate annualized volatility over lookback period."""
        import math

        cutoff = datetime.now(timezone.utc).timestamp() - lookback_minutes * 60
        recent_prices = [p for t, p in prices if t.timestamp() > cutoff]

        if len(recent_prices) < 2:
            return 0.6  # Default volatility

        # Calculate log returns
        returns = []
        for i in range(1, len(recent_prices)):
            if recent_prices[i-1] > 0 and recent_prices[i] > 0:
                returns.append(math.log(recent_prices[i] / recent_prices[i-1]))

        if len(returns) < 2:
            return 0.6

        # Standard deviation
        mean_ret = sum(returns) / len(returns)
        var = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
        std = math.sqrt(var)

        # Annualize (assuming minute data)
        annualized = std * math.sqrt(525600)  # Minutes in a year

        return annualized

    def _calculate_adjustments(
        self,
        regime: MarketRegime,
        indicators: RegimeIndicators,
    ) -> dict[str, float]:
        """Calculate strategy parameter adjustments for regime."""
        adjustments = {
            "position_size_mult": 1.0,
            "edge_threshold_mult": 1.0,
            "stop_loss_mult": 1.0,
            "take_profit_mult": 1.0,
        }

        if regime == MarketRegime.CRISIS:
            adjustments["position_size_mult"] = 0.25
            adjustments["edge_threshold_mult"] = 2.0
            adjustments["stop_loss_mult"] = 0.5

        elif regime == MarketRegime.HIGH_VOL:
            adjustments["position_size_mult"] = 0.5
            adjustments["edge_threshold_mult"] = 1.5
            adjustments["stop_loss_mult"] = 0.75

        elif regime == MarketRegime.BULL:
            adjustments["position_size_mult"] = 1.2
            adjustments["take_profit_mult"] = 1.5

        elif regime == MarketRegime.BEAR:
            adjustments["position_size_mult"] = 0.8
            adjustments["stop_loss_mult"] = 0.8

        return adjustments


# Pre-instantiated instances
cross_platform_arb_detector = CrossPlatformArbDetector()
regime_detector = RegimeDetector()
