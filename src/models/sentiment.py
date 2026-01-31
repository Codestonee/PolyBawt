import numpy as np


class SentimentIntegrator:
    """Backward-compatible alias.

    Older code/tests import SentimentIntegrator; the implementation was renamed
    to SentimentAggregator.
    """

    def __init__(self):
        self._agg = SentimentAggregator()

    def composite_sentiment_score(self, market_state):
        return self._agg.composite_sentiment_score(market_state)

    def adjust_position_sizing(self, base_kelly_fraction, sentiment_score):
        return self._agg.adjust_position_sizing(base_kelly_fraction, sentiment_score)


class SentimentAggregator:
    """
    Combine multiple sentiment signals with optimal weighting
    """
    
    def __init__(self):
        self.weights = {
            'funding_rate': 0.6,
            'order_flow_imbalance': 0.25,
            'fear_greed_regime': 0.1,
            'social_sentiment': 0.05
        }
    
    def composite_sentiment_score(self, market_state):
        funding_signal = self.funding_rate_signal(market_state)
        # flow_signal = ... (needs VPIN/OrderBook integration)
        # regime_signal = ...
        
        # Simplified for example:
        return np.clip(funding_signal, -1, 1)

    def funding_rate_signal(self, market_state):
        """Extract funding signal. Assumes market_state has asset and funding_rate"""
        # -0.3% to +0.3% funding → -1 to +1 signal
        if hasattr(market_state, 'funding_rate'):
            signal = market_state.funding_rate / 0.003
            return np.clip(signal, -1, 1)
        return 0

    def adjust_position_sizing(self, base_kelly_fraction, sentiment_score):
        """
        Sentiment affects position size, NOT probability prediction
        """
        conviction = 1 + (sentiment_score * 0.5)
        adjusted_kelly = base_kelly_fraction * conviction
        max_kelly = 0.05
        return min(adjusted_kelly, max_kelly)
