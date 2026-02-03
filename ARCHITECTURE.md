# Polymarket Trading Bot Architecture

## System Overview

The bot is a **modular, event-driven trading system** designed for 15-minute binary option markets on Polymarket. It prioritizes **safety** and **resilience** over raw speed, using a monolithic application structure with clear logical service boundaries.

## Logical Service Boundaries

### 1. Data Ingestion Layer (`src/ingestion`)

**Responsibilities**:

- Fetching market data from Gamma API.
- Normalizing API responses into domain objects (`Market`).
- Handling API rate limits and outages (Resilience).

**Key Components**:

- `MarketDiscovery`: Fetches active markets. Handles pagination and retries.

### 2. Strategy & Analysis Layer (`src/strategy`, `src/models`)

**Responsibilities**:

- Pricing options using quantitative models (`JumpDiffusionModel`).
- Detecting arbitrage and positive expected value (EV) opportunities.
- Filtering toxic order flow (`ToxicityFilter`).

**Key Components**:

- `ValueBettingStrategy`: Main orchestrator.
- `JumpDiffusionModel`: Core pricing logic.
- `ArbitrageDetector`: Risk-free profit finder.
- `EVCalculator`: Computes Edge and Kelly criteria.

### 3. Risk Management Layer (`src/strategy`, `src/models`)

**Responsibilities**:

- Protecting capital from ruin.
- Validating trade conditions before execution.
- Preventing trading during system instability.

**Key Components**:

- `CircuitBreaker`: Stops trading if PnL drops or errors spike.
- `NoTradeGate`: Validates oracle freshness, spread, and time-to-expiry.
- `KellySizer`: Calculates optimal position sizing.

### 4. Execution Layer (`src/execution`)

**Responsibilities**:

- interacting with the CLOB (Central Limit Order Book).
- Managing order lifecycle (placing, canceling, tracking).
- Managing local state persistence.

**Key Components**:

- `CLOBClient`: Wrapper for the exchange API.
- `OrderManager`: State machine for orders. Persists state to disk.

## Data Flow

1. **Discovery**: `MarketDiscovery.get_crypto_15m_markets()` -> List[`Market`]
2. **Analysis**:
   - `Oracle.get_price()` -> Spot Price
   - `PricingModel` -> Model Probability
   - `ToxicityFilter` -> Flow Check
   - `EVCalculator` -> `TradeSignal`
3. **Risk Check**:
   - `NoTradeGate.validate()` -> Pass/Fail
   - `KellySizer.size()` -> `SafeSize`
4. **Execution**:
   - `OrderManager.create_order()` -> `Order`
   - `CLOBClient.place_order()` -> `ExchangeID`
5. **Persistence**:
   - Updates saved to `orders.json` on every state change.

## Key Patterns

### Resilience

- **Exponential Backoff**: Used in API clients for temporary outages.
- **Circuit Breakers**: Halt trading on excessive consecutive losses or API errors.

### Performance

- **Caching**: Order books are cached per strategy iteration to minimize API calls.
- **AsyncIO**: Fully asynchronous I/O for concurrent market analysis.

### Safety

- **Fail-Safe Oracles**: Missing oracle data is treated as "Stale" (infinite age), blocking trades.
- **Overfill Protection**: Logic clamps fill amounts to remaining order size.
