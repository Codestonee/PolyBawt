# Ultimate Event Market Trading Strategy

Complete implementation of an automated event market trading system for Polymarket, covering Politics, Sports, Economics, and Pop Culture markets.

## Architecture

```
Event Market Discovery → Category Router → Probability Model → EV Calculator → NO-TRADE Gate → Kelly Sizer
                              ↓
                   External Data Sources
                  (Polls/Stats/Forecasts)
                              ↓
                   Brier Score Calibration
```

## Key Components

### Phase 1: External Data Infrastructure

**Files:** `src/ingestion/external_data/`

- `base_source.py` - Abstract base class for all data sources with caching, error handling, and status tracking
- `poll_aggregator.py` - Politics data from RCP, 538, PredictIt with weighted aggregation (recency, sample size, pollster rating)
- `sports_stats.py` - Sports data including ELO ratings, Vegas betting lines, injury tracking
- `economic_forecasts.py` - Fed futures (90%+ accuracy 30 days out), consensus forecasts, leading indicators
- `sentiment_analyzer.py` - Pop Culture sentiment from Metaculus, Manifold, Twitter/X, Reddit

### Phase 2: Category-Specific Probability Models

**Files:** `src/models/event_probability/`

- `base_estimator.py` - Abstract base with confidence levels, uncertainty bounds, and estimate combination
- `politics_model.py` - Poll aggregation + incumbency advantage (+3%) + economic correlation
- `sports_model.py` - ELO ratings + Vegas line comparison + recent form analysis
- `economics_model.py` - Fed futures weighted by time to event + consensus forecasts
- `pop_culture_model.py` - Cross-platform aggregation (Metaculus/Manifold) + social sentiment
- `event_ensemble.py` - Combines all models with dynamic weighting:
  - Category model: 60%
  - Order book signal: 20% (65-75% accuracy!)
  - Base rate: 10%
  - Agreement bonus: 10%

### Phase 3: Calibration & Validation

**Files:** `src/calibration/`

- `brier_tracker.py` - Brier score tracking with components (reliability, resolution, uncertainty)
- `calibration_database.py` - SQLite database for predictions and outcomes
- `walk_forward_validator.py` - Temporal validation (never k-fold!) with requirements:
  - 500+ resolved markets
  - 180+ day span
  - Brier < 0.22 (better than random 0.25)
  - < 10% calibration deviation

### Phase 4: Main Event Strategy

**File:** `src/strategy/event_betting.py`

Main strategy class `EventBettingStrategy` that:
- Integrates all probability models
- Reuses existing components (EV calculator, NO-TRADE gate, Kelly sizer, VPIN, OBI)
- Tracks Brier scores per category
- Supports position rebalancing on new information
- Records all predictions for calibration

### Phase 5: Configuration

**Updated:** `src/infrastructure/config.py`
- Added `EventTradingConfig` class with category-specific settings

**Updated:** `src/main.py`
- Added `run_event_strategy()` function
- Runs crypto and event strategies concurrently

**Created:** `config/event_trading.yaml`
- Complete configuration template

## Key Insights from Research

### OBI is King
Order Book Imbalance achieves **65-75% accuracy** vs 55-65% for traditional models. The edge comes from microstructure features.

### Fee Extremes
Same edge at p=0.05 costs **62x less in fees** than p=0.50:
- Fee at p=0.01 → $0.0025 per 100 shares
- Fee at p=0.50 → $1.56 per 100 shares

### VPIN Works Universally
Toxicity detection applies to all market types (crypto + events).

### Fed Futures Accuracy
90%+ accuracy 30 days out for Fed policy predictions.

### Walk-Forward Only
Never use random k-fold validation for time series. Walk-forward is the only valid method.

## Category-Specific Edge Thresholds

| Category | Min Edge | Rationale |
|----------|----------|-----------|
| Politics | 5% | Polls are fairly reliable |
| Sports | 6% | More variance, injury uncertainty |
| Economics | 4% | Fed futures very accurate |
| Pop Culture | 7% | Harder to predict, higher uncertainty |

## Usage

### Run with Event Markets
```bash
# Using event trading config
python -m src.main --config config/event_trading.yaml

# Enable event markets with crypto strategy
# (Set event_trading.enabled: true in config)
```

### Run Only Event Strategy
Modify config:
```yaml
event_trading:
  enabled: true
  categories: ["politics", "sports", "economics"]
```

### Paper Trading
```bash
python -m src.main --config config/event_trading.yaml
```

### Live Trading
```bash
python -m src.main --config config/event_trading.yaml --live
```

## Data Sources by Category

| Category | Primary Source | Secondary | Update Freq |
|----------|---------------|-----------|-------------|
| Politics | 538/RealClearPolitics | PredictIt | 1 hour |
| Sports | ESPN stats + ELO | Vegas lines | 15 min |
| Economics | CME Fed futures | FRED indicators | 15 min |
| Pop Culture | Metaculus | Twitter sentiment | 30 min |

## Calibration Targets

- **Brier Score:** < 0.22 (better than random guessing at 0.25)
- **Calibration:** Within 10% of perfect
- **Minimum History:** 500+ resolved markets
- **Time Span:** 180+ days (multiple regimes)

## Rebalancing

The strategy monitors active positions and rebalances when:
- New external data arrives (polls, injury reports, etc.)
- Probability estimate changes > 10%
- Edge disappears due to market movement

## File Structure

```
src/
├── ingestion/
│   ├── external_data/
│   │   ├── __init__.py
│   │   ├── base_source.py
│   │   ├── poll_aggregator.py
│   │   ├── sports_stats.py
│   │   ├── economic_forecasts.py
│   │   └── sentiment_analyzer.py
│   └── event_market_discovery.py
├── models/
│   └── event_probability/
│       ├── __init__.py
│       ├── base_estimator.py
│       ├── politics_model.py
│       ├── sports_model.py
│       ├── economics_model.py
│       ├── pop_culture_model.py
│       └── event_ensemble.py
├── calibration/
│   ├── __init__.py
│   ├── brier_tracker.py
│   ├── calibration_database.py
│   └── walk_forward_validator.py
└── strategy/
    └── event_betting.py

config/
└── event_trading.yaml
```

## Testing

Run the strategy in dry-run mode first:
```bash
python -m src.main --config config/event_trading.yaml --verbose
```

Check calibration after running:
```python
from src.calibration.brier_tracker import brier_tracker
report = brier_tracker.get_calibration_report()
print(report)
```

## Monitoring

Key metrics to watch:
1. **Brier Score by Category** - Are models well-calibrated?
2. **Edge Realization** - Are predicted edges actually profitable?
3. **VPIN Levels** - Are we avoiding toxic flow?
4. **Category PnL** - Which categories perform best?

## Future Enhancements

1. **News Integration** - NLP for breaking news impact
2. **Social Graph Analysis** - Track insider information flow
3. **Weather Models** - For outdoor sports markets
4. **Debate Performance** - Real-time sentiment during debates
5. **Injury Prediction** - ML models for injury likelihood

## Research References

- arXiv:2510.15205 - Perpetual Futures Pricing
- "Volume-Synchronized Probability of Informed Trading" - VPIN
- "Order Book Imbalance as a Predictor" - Market microstructure
- FiveThirtyEight ELO methodology
- CME FedWatch methodology
