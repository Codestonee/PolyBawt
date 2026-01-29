# Polymarket 15m Trading Bot

Production-grade algorithmic trading bot for Polymarket's 15-minute BTC/ETH/SOL/XRP prediction markets.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure
cp config/example.yaml config/local.yaml
# Edit config/local.yaml with your settings

# Set environment variables
export POLYMARKET_PRIVATE_KEY="your_key"
export POLYMARKET_FUNDER_ADDRESS="your_address"

# Run paper trading
python -m src.main --config config/paper.yaml

# Run live (requires --live flag for safety)
python -m src.main --config config/production.yaml --live
```

## Project Structure

```
src/
├── ingestion/      # WebSocket, Oracle feeds, Market discovery
├── models/         # Pricing models (Jump-Diffusion, BSM)
├── strategy/       # Trading strategies
├── execution/      # Order management, Rate limiting
├── risk/           # Kelly sizing, Circuit breakers
├── portfolio/      # PnL tracking
└── infrastructure/ # Config, Logging, Metrics
```

## Configuration

- `config/local.yaml` - Local development (mock exchange)
- `config/paper.yaml` - Paper trading (real data, no orders)
- `config/production.yaml` - Live trading

## Safety Features

- Fractional Kelly sizing (1/4 Kelly)
- Daily loss limits with automatic halt
- Circuit breakers for volatility spikes
- Fail-closed error handling
- Rate limit protection

## License

Private - All Rights Reserved
