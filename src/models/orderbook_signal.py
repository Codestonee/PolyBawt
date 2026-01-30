"""
Order Book Imbalance Signal Model.

Research finding: "Order book features (imbalance, spread depth) often beat
academic models, with 65-75% directional accuracy vs. 55-65% for JD."

This module provides a simple but effective signal based on order book
microstructure features that outperform traditional jump-diffusion models
for 15-minute crypto binary options.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OrderBookFeatures:
    """
    Features extracted from order book for signal generation.
    
    Research ranking of predictive features:
    1. Order imbalance (most important)
    2. Bid-ask spread
    3. Recent returns
    4. Time to expiry
    """
    
    # Order imbalance: (bid_vol - ask_vol) / total_vol, range [-1, 1]
    imbalance_100bp: float
    imbalance_200bp: float
    imbalance_500bp: float
    
    # Spread metrics
    spread_bps: float
    
    # Depth metrics
    bid_depth_100bp: float
    ask_depth_100bp: float
    total_depth_100bp: float
    
    # Price metrics
    mid_price: float
    best_bid: float
    best_ask: float
    
    # Recent returns (if available)
    return_1min: float = 0.0
    return_5min: float = 0.0
    
    # Time to expiry
    seconds_to_expiry: float = 900.0  # 15 min default
    
    # VPIN toxicity (if available)
    vpin: float = 0.5


class OrderBookSignalModel:
    """
    Simple heuristic model based on order book imbalance.
    
    Research: "Buy YES if bid depth > ask by 20%" yields 8-12% edge in backtests.
    
    This model provides probability adjustments based on order book state,
    designed to be used alongside (or instead of) academic pricing models.
    
    Usage:
        model = OrderBookSignalModel()
        features = model.extract_features(orderbook, market_state)
        adjustment = model.get_probability_adjustment(features)
        
        final_prob = base_model_prob + adjustment
    """
    
    def __init__(
        self,
        imbalance_threshold: float = 0.2,
        max_adjustment: float = 0.08,
        spread_penalty_factor: float = 0.5,
        thin_book_threshold: float = 1000.0,
    ):
        """
        Initialize the signal model.
        
        Args:
            imbalance_threshold: Minimum imbalance to trigger signal (default 20%)
            max_adjustment: Maximum probability adjustment (default ±8%)
            spread_penalty_factor: Reduce signal if spread is wide
            thin_book_threshold: Minimum depth (USDC) to trust imbalance signal
        """
        self.imbalance_threshold = imbalance_threshold
        self.max_adjustment = max_adjustment
        self.spread_penalty_factor = spread_penalty_factor
        self.thin_book_threshold = thin_book_threshold
    
    def extract_features(
        self,
        orderbook: Any,  # Can be OrderBook from websocket_client or dict
        market_state: dict[str, Any] | None = None,
    ) -> OrderBookFeatures:
        """
        Extract prediction features from order book.
        
        Args:
            orderbook: Order book object or dict with bids/asks
            market_state: Optional additional market state
        
        Returns:
            OrderBookFeatures dataclass
        """
        # Handle different input types
        if hasattr(orderbook, 'order_imbalance'):
            # It's our OrderBook class from websocket_client
            imbalance_100 = orderbook.order_imbalance(100)
            imbalance_200 = orderbook.order_imbalance(200)
            imbalance_500 = orderbook.order_imbalance(500)
            spread_bps = orderbook.spread_bps or 0.0
            bid_depth = orderbook.bid_depth(100)
            ask_depth = orderbook.ask_depth(100)
            mid_price = orderbook.mid_price or 0.5
            best_bid = orderbook.best_bid or 0.0
            best_ask = orderbook.best_ask or 1.0
        else:
            # It's a dict - calculate manually
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            
            bid_vol = sum(float(s) for _, s in bids[:10]) if bids else 0
            ask_vol = sum(float(s) for _, s in asks[:10]) if asks else 0
            total_vol = bid_vol + ask_vol
            
            imbalance_100 = (bid_vol - ask_vol) / total_vol if total_vol > 0 else 0
            imbalance_200 = imbalance_100  # Simplified
            imbalance_500 = imbalance_100
            
            best_bid = float(bids[0][0]) if bids else 0.0
            best_ask = float(asks[0][0]) if asks else 1.0
            mid_price = (best_bid + best_ask) / 2 if (best_bid and best_ask) else 0.5
            spread_bps = ((best_ask - best_bid) / mid_price * 10000) if mid_price > 0 else 0
            
            bid_depth = bid_vol
            ask_depth = ask_vol
        
        # Get additional features from market state
        return_1min = 0.0
        return_5min = 0.0
        seconds_to_expiry = 900.0
        vpin = 0.5
        
        if market_state:
            return_1min = market_state.get('return_1min', 0.0)
            return_5min = market_state.get('return_5min', 0.0)
            seconds_to_expiry = market_state.get('seconds_to_expiry', 900.0)
            vpin = market_state.get('vpin', 0.5)
        
        return OrderBookFeatures(
            imbalance_100bp=imbalance_100,
            imbalance_200bp=imbalance_200,
            imbalance_500bp=imbalance_500,
            spread_bps=spread_bps,
            bid_depth_100bp=bid_depth,
            ask_depth_100bp=ask_depth,
            total_depth_100bp=bid_depth + ask_depth,
            mid_price=mid_price,
            best_bid=best_bid,
            best_ask=best_ask,
            return_1min=return_1min,
            return_5min=return_5min,
            seconds_to_expiry=seconds_to_expiry,
            vpin=vpin,
        )
    
    def get_probability_adjustment(self, features: OrderBookFeatures) -> float:
        """
        Calculate probability adjustment based on order book features.
        
        Returns:
            Adjustment in range [-max_adjustment, +max_adjustment]
            Positive = bullish (increase YES probability)
            Negative = bearish (decrease YES probability)
        """
        # Base signal from imbalance
        imbalance = features.imbalance_100bp
        
        # Only act if imbalance exceeds threshold
        if abs(imbalance) < self.imbalance_threshold:
            logger.debug(
                "Imbalance below threshold",
                imbalance=imbalance,
                threshold=self.imbalance_threshold,
            )
            return 0.0
        
        # Calculate raw adjustment (linear scaling)
        # Imbalance of 1.0 → max_adjustment
        # Imbalance of threshold → 0
        effective_imbalance = imbalance - np.sign(imbalance) * self.imbalance_threshold
        adjustment = effective_imbalance * self.max_adjustment / (1 - self.imbalance_threshold)
        
        # Penalty #1: Wide spread reduces confidence
        if features.spread_bps > 100:
            spread_penalty = 1 - min((features.spread_bps - 100) / 400, 0.8)
            adjustment *= spread_penalty
            logger.debug("Applied spread penalty", spread_bps=features.spread_bps, penalty=spread_penalty)
        
        # Penalty #2: Thin book reduces confidence
        if features.total_depth_100bp < self.thin_book_threshold:
            depth_penalty = features.total_depth_100bp / self.thin_book_threshold
            adjustment *= depth_penalty
            logger.debug("Applied depth penalty", depth=features.total_depth_100bp, penalty=depth_penalty)
        
        # Penalty #3: High VPIN (toxic flow) reduces confidence
        if features.vpin > 0.6:
            vpin_penalty = 1 - min((features.vpin - 0.6) * 2, 0.8)
            adjustment *= vpin_penalty
            logger.debug("Applied VPIN penalty", vpin=features.vpin, penalty=vpin_penalty)
        
        # Clip to max adjustment
        adjustment = np.clip(adjustment, -self.max_adjustment, self.max_adjustment)
        
        logger.info(
            "Order book signal",
            imbalance=imbalance,
            adjustment=adjustment,
            spread_bps=features.spread_bps,
            depth=features.total_depth_100bp,
        )
        
        return float(adjustment)
    
    def get_directional_signal(self, features: OrderBookFeatures) -> str:
        """
        Get simple directional signal.
        
        Returns:
            'bullish', 'bearish', or 'neutral'
        """
        adjustment = self.get_probability_adjustment(features)
        
        if adjustment > 0.02:
            return 'bullish'
        elif adjustment < -0.02:
            return 'bearish'
        else:
            return 'neutral'
    
    def should_trade(self, features: OrderBookFeatures) -> bool:
        """
        Check if conditions are suitable for trading.
        
        Filters out markets with:
        - Thin liquidity
        - Wide spreads
        - Toxic flow
        """
        # Minimum depth requirement
        if features.total_depth_100bp < self.thin_book_threshold:
            logger.debug("Rejecting: thin book", depth=features.total_depth_100bp)
            return False
        
        # Maximum spread requirement
        if features.spread_bps > 500:  # 5% spread
            logger.debug("Rejecting: wide spread", spread_bps=features.spread_bps)
            return False
        
        # Toxicity check
        if features.vpin > 0.7:
            logger.debug("Rejecting: toxic flow", vpin=features.vpin)
            return False
        
        return True


# ========== Factory Function ==========

def create_orderbook_signal_model(
    imbalance_threshold: float = 0.2,
    max_adjustment: float = 0.08,
) -> OrderBookSignalModel:
    """
    Factory function to create order book signal model.
    
    Usage:
        model = create_orderbook_signal_model()
        features = model.extract_features(orderbook)
        adjustment = model.get_probability_adjustment(features)
    """
    return OrderBookSignalModel(
        imbalance_threshold=imbalance_threshold,
        max_adjustment=max_adjustment,
    )
