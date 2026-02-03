# Grok-Native Polymarket Bot

Ultra-lean, high-frequency bot for Polymarket 15-minute markets. Rewritten for the "Grok Strategy Ensemble."

## Quick Start

```powershell
# 1. Start Paper Trading (Dry Run)
python -m src.main --config config/paper.yaml

# 2. Start Live Trading ($5 per bet, $20 cap)
python -m src.main --config config/live.yaml --live
```

## Strategy Ensemble (The "Grok-4")

1. **Arb Taker**: Snipes risk-free profit when YES + NO < 0.98.
2. **Latency Snipe**: Offensive trades on spot price moves > 2%.
3. **Spread Maker**: Passive market making on spreads > 5c.
4. **Legged Hedge**: Buys crashes (> 15% drop) and hedges opposite leg.

## Project Structure

```text
src/
├── strategy/       # Grok Ensemble & Strategy implementation
├── ingestion/      # Oracle feeds & Market discovery
├── execution/      # Order management & CLOB interaction
├── risk/           # Portfolio safety & Circuit breakers
└── infrastructure/ # Core logging and configuration
```

## Configuration

- `config/paper.yaml` - Real-time analysis, no capital risk.
- `config/live.yaml` - Live execution with $20 exposure cap.

## Safety Features

- Fractional Kelly sizing (1/4 Kelly)
- Daily loss limits with automatic halt
- Circuit breakers for volatility spikes
- Fail-closed error handling
- Rate limit protection

## License

Private - All Rights Reserved
