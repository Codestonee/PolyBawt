# PolyBawt Senior Architecture & Trading Systems Code Review

**Review Date**: 2026-02-08  
**Reviewer**: Senior Trading Systems Architect  
**Codebase Version**: 1.1.0  

---

## 1. Graded Assessment by Subsystem

| Subsystem | Score | Weight | Weighted |
| :--- | :--- | :--- | :--- |
| Strategy Logic Quality | 6.5/10 | 12% | 0.78 |
| Alpha/Signal Integrity | 5.5/10 | 15% | 0.83 |
| Risk Management & Controls | 6.0/10 | 18% | 1.08 |
| Execution Reliability | 6.0/10 | 15% | 0.90 |
| Portfolio/Accounting | 5.5/10 | 12% | 0.66 |
| Async/Concurrency Safety | 5.0/10 | 10% | 0.50 |
| Observability/Operability | 6.5/10 | 8% | 0.52 |
| Security/Secrets Handling | 6.0/10 | 5% | 0.30 |
| Test Quality | 3.0/10 | 5% | 0.15 |
| **Production Readiness** | **5.7/10** | — | **5.72** |

### Weighting Rationale

- **Risk/Execution (33%)**: Trading systems fail at execution; risk controls are table stakes
- **Alpha/Strategy (27%)**: Useless without edge; strategy correctness is profit-critical
- **Portfolio (12%)**: Incorrect accounting = invisible P&L leaks
- **Concurrency (10%)**: Async bugs cause silent failures in production
- **Observability/Security/Tests (18%)**: Operational excellence enables trust

---

### Detailed Subsystem Analysis

#### Strategy Logic Quality: 6.5/10

**What's Good**:

- Clean strategy abstraction via `BaseStrategy` + `TradeContext`
- Per-strategy circuit breakers with size_multiplier degradation
- OBI/VPIN integration for signal confidence adjustment

**What Prevents 8+**:

- `LeggedHedge` state machine is incomplete (hedge price formula is naive: `1.0 - book.best_bid - 0.02`)
- Short arbitrage detected but not executed (dead code path)
- Hardcoded thresholds instead of config-driven parameters
- No backtesting validation of strategy edge

#### Alpha/Signal Integrity: 5.5/10

**What's Good**:

- Latency snipe concept is sound (spot→options price lag exploitation)
- Arb detection math is correct for long arb case

**What Prevents 8+**:

- **CRITICAL**: Settlement outcome hardcoded to `True` (line 318 `value_betting.py`)
- VPIN uses order book snapshots as trade flow proxy (fundamentally incorrect)
- No oracle freshness gates before signal generation
- Kelly sizing confidence boost from OBI is arbitrary (+5%)
- No signal decay / time-to-expiry scaling

#### Risk Management & Controls: 6.0/10

**What's Good**:

- Multi-tier circuit breakers (soft/hard trips)
- Correlation-adjusted sizing via `correlation_matrix`
- Sniper risk detection framework exists
- RiskGate validates all signals before execution

**What Prevents 8+**:

- Chainlink fallback to Binance defeats sniper protection purpose
- No per-token position limits (strategies can pile on same token)
- Daily loss and drawdown breakers never auto-reset (by design, but no manual reset API)
- Missing real-time margin/collateral validation
- No exposure concentration limits by asset class

#### Execution Reliability: 6.0/10

**What's Good**:

- Retry logic with exponential backoff for 500 errors
- OrderManager state machine with persistence
- Reconciler syncs with exchange every 30s
- Rate limiter with order/query separation

**What Prevents 8+**:

- No idempotency keys passed to exchange (duplicate risk on retry)
- Order size not rounded to exchange tick size (rejection risk)
- Cancel response handling assumes success on non-error
- No order acknowledgment timeout handling
- WebSocket fill handler has no sequence number validation

#### Portfolio/Accounting: 5.5/10

**What's Good**:

- Position accumulation with weighted average entry price
- Atomic file persistence with temp→rename pattern
- Per-strategy P&L attribution
- Win/loss streak tracking

**What Prevents 8+**:

- **CRITICAL**: Settlement uses placeholder `outcome=True`
- Unrealized P&L calculation requires manual price dict (not automated)
- No mark-to-market revaluation loop
- Closed positions array grows unbounded in memory
- No reconciliation with exchange position state

#### Async/Concurrency Safety: 5.0/10

**What's Good**:

- `asyncio.gather` for parallel order book fetches
- Background tasks for event bus and reconciler

**What Prevents 8+**:

- `publish_sync()` called from async functions (event loop blocking)
- No locks on shared state (`_pending_fills`, `_order_book_cache`, `positions`)
- `handle_fill()` is sync but called from async WebSocket handler
- Dedup set trimming is racey (`list(set)[-5000:]`)
- Order listener callback creates tasks without tracking

#### Observability/Operability: 6.5/10

**What's Good**:

- Structured logging with `structlog`
- Prometheus metrics for PnL, win rate, portfolio
- Event bus with audit trail
- API server for dashboard integration

**What Prevents 8+**:

- No `/health` or `/ready` endpoints
- Missing latency histograms for order lifecycle
- No alerting integration (Telegram disabled by default)
- Log messages lack correlation IDs for request tracing
- No runbook documentation for operational incidents

#### Security/Secrets Handling: 6.0/10

**What's Good**:

- Secrets loaded from environment variables only
- `SecretsConfig` separate from `AppConfig`
- Production requires explicit `--live` flag

**What Prevents 8+**:

- Private key stored in env (no HSM/vault integration)
- API credentials logged in error messages (risk of exposure)
- No secret rotation mechanism
- Missing audit log for sensitive operations

#### Test Quality: 3.0/10

**What's Good**:

- Basic unit test structure exists
- Chaos test file present (`test_chaos.py`)

**What Prevents 8+**:

- Integration tests may hit real APIs (no mocking)
- No deterministic replay tests
- No edge case coverage (settlement, reconciliation failures)
- No performance/load tests
- Test coverage likely <30%

---

### 2. Top Findings by Severity

### 🔴 CRITICAL (Must Fix Before Any Live Trading)

| ID | Finding | Location | Risk |
| :--- | :--- | :--- | :--- |
| C1 | **Settlement outcome hardcoded to `True`** | `value_betting.py:318-320` | All P&L calculations wrong; positions may show profit when they lost |
| C2 | **No Chainlink credentials = sniper protection disabled** | `oracle_feed.py:412-418` | HFT bots can frontrun stale orders; guaranteed adverse selection |
| C3 | **`publish_sync()` blocks event loop** | `value_betting.py:344,653,737` | Strategy loop hangs under load; missed trading opportunities |
| C4 | **No locks on shared state** | `value_betting.py:114-116` | Race conditions corrupt `_pending_fills` and `_order_book_cache` |
| C5 | **Dedup set grows unbounded** | `value_betting.py:710-711` | Memory exhaustion over time; OOM kills bot |

### 🟠 HIGH (Fix Before Production)

| ID | Finding | Location | Risk |
| :--- | :--- | :--- | :--- |
| H1 | VPIN uses order book as trade flow proxy | `value_betting.py:434-441` | Toxicity filter makes wrong decisions |
| H2 | Short arbitrage detected but not executed | `strategies.py:112-128` | Missing revenue; incomplete feature |
| H3 | Order size not rounded to tick size | `clob_client.py:190-193` | Order rejections from exchange |
| H4 | 60-second order book cache TTL | `value_betting.py:243-246` | Trading on stale prices |
| H5 | No idempotency keys | `clob_client.py:153` | Duplicate orders on retry |
| H6 | Cancel assumes success | `clob_client.py:407-411` | Ghost orders left on book |
| H7 | LeggedHedge hedge price naive | `strategies.py:495` | Unprofitable hedge execution |
| H8 | No per-token exposure limits | `gatekeeper.py` | Concentration risk |

### 🟡 MEDIUM (Fix for Reliability)

| ID | Finding | Location | Risk |
| :--- | :--- | :--- | :--- |
| M1 | Closed positions array unbounded | `tracker.py:288` | Memory growth |
| M2 | No market discovery retry logic | `market_discovery.py` | Missing markets |
| M3 | Hardcoded strategy thresholds | `strategies.py:*` | No A/B testing |
| M4 | No correlation ID in logs | `logging.py` | Debug difficulty |
| M5 | Missing health endpoints | `server.py` | K8s orchestration fails |
| M6 | Order listener creates untracked tasks | `value_betting.py:188` | Silent failures |
| M7 | Kelly confidence boost arbitrary | `value_betting.py:593` | Oversizing risk |
| M8 | No mark-to-market loop | `tracker.py` | Stale P&L display |

---

## 3. Phased Implementation Plan

### Phase 0: Blockers for Live Deployment (Week 1)

#### P0.1: Fix Settlement Logic

- **Objective**: Correct position settlement based on actual market outcome
- **Changes**:
  - Add `get_market_resolution(token_id)` to `MarketDiscovery`
  - Query Polymarket API for resolved markets
  - Replace `outcome = True` with actual resolution
- **Risk Reduced**: Model risk (incorrect P&L)
- **Complexity**: M (2-3 days)
- **Validation**: Unit tests with mock resolutions; integration test with historical settlements
- **Exit Criteria**: All historical settlements match exchange records

#### P0.2: Chainlink Integration

- **Objective**: Real settlement price source for sniper protection
- **Changes**:
  - Require `CHAINLINK_CLIENT_ID/SECRET` in live mode
  - Remove Binance fallback in `_fetch_chainlink_price()`
  - Add connection health monitoring
- **Risk Reduced**: Counterparty risk (adverse selection)
- **Complexity**: M (2 days)
- **Validation**: Compare Chainlink prices to settlement outcomes
- **Exit Criteria**: Sniper alerts fire on historical arbitrage events

#### P0.3: Async Event Bus

- **Objective**: Non-blocking event publishing
- **Changes**:
  - Replace `publish_sync()` with `await event_bus.publish()`
  - Make `EventBus.publish()` truly async with queue
- **Risk Reduced**: Operational risk (loop blocking)
- **Complexity**: S (1 day)
- **Validation**: Load test with 100 events/second
- **Exit Criteria**: No event loop blocking in flame graphs

#### P0.4: Concurrency Safety

- **Objective**: Thread-safe shared state
- **Changes**:
  - Add `asyncio.Lock()` for `_pending_fills`, `_order_book_cache`, `positions`
  - Use `async with lock:` for mutations
  - Replace `handle_fill()` with async version
- **Risk Reduced**: Operational risk (data corruption)
- **Complexity**: M (2 days)
- **Validation**: Concurrent stress test with 50 simultaneous fills
- **Exit Criteria**: No race condition failures in 10k runs

#### P0.5: Fix Dedup Memory Leak

- **Objective**: Bounded memory for fill deduplication
- **Changes**:
  - Replace set with `collections.OrderedDict` (LRU cache)
  - Cap at 1000 entries with FIFO eviction
- **Risk Reduced**: Operational risk (OOM)
- **Complexity**: S (0.5 days)
- **Validation**: Memory profiling over 24h simulated run
- **Exit Criteria**: Memory stable within 50MB variance

---

### Phase 1: Reliability Baseline to Reach 8 (Weeks 2-3)

#### P1.1: Order Tick Size Rounding

- **Objective**: Prevent order rejections
- **Changes**:
  - Add `round_to_tick(price, tick_size=0.01)` utility
  - Apply before `create_order()` in CLOB client
- **Risk Reduced**: Execution risk
- **Complexity**: S (0.5 days)
- **Exit Criteria**: Zero tick-size rejections in paper trading

#### P1.2: Idempotency Keys

- **Objective**: Prevent duplicate orders on retry
- **Changes**:
  - Generate UUID client_order_id before submission
  - Pass to Polymarket API
  - Check for existing order before retry
- **Risk Reduced**: Execution risk (double fills)
- **Complexity**: M (1 day)
- **Exit Criteria**: Retry test shows no duplicates

#### P1.3: VPIN with Real Trade Flow

- **Objective**: Correct toxicity detection
- **Changes**:
  - Subscribe to Polymarket trade WebSocket
  - Feed actual trade sizes to `VPINCalculator.update()`
  - Remove order book proxy logic
- **Risk Reduced**: Model risk (wrong toxicity signals)
- **Complexity**: M (2 days)
- **Exit Criteria**: VPIN correlates with actual adverse selection events

#### P1.4: Order Book Cache TTL

- **Objective**: Fresh prices for trading decisions
- **Changes**:
  - Reduce TTL from 60s to 5s
  - Add per-strategy cache refresh before `scan()`
  - Log stale cache hits as warnings
- **Risk Reduced**: Market risk (stale prices)
- **Complexity**: S (0.5 days)
- **Exit Criteria**: 95% of trades use <5s old data

#### P1.5: Cancel Verification

- **Objective**: Confirm order cancellation success
- **Changes**:
  - Query order status after cancel
  - Retry cancel if still active
  - Mark as "cancel_pending" locally
- **Risk Reduced**: Operational risk (ghost orders)
- **Complexity**: M (1 day)
- **Exit Criteria**: No orphaned orders after 1000 cancel cycles

#### P1.6: Per-Token Exposure Limits

- **Objective**: Prevent concentration risk
- **Changes**:
  - Add `max_per_token_usd: 10.0` to config
  - Check in `RiskGate.validate()` against portfolio
  - Reject or reduce size if exceeded
- **Risk Reduced**: Market risk (concentration)
- **Complexity**: S (1 day)
- **Exit Criteria**: No single token exceeds limit in backtest

#### P1.7: Health Endpoints

- **Objective**: Kubernetes readiness
- **Changes**:
  - Add `/health` (liveness) and `/ready` (dependencies)
  - Check: database, exchange connection, oracle freshness
  - Return structured health object
- **Risk Reduced**: Operational risk
- **Complexity**: S (0.5 days)
- **Exit Criteria**: Probes work in container deployment

#### P1.8: Integration Test Suite

- **Objective**: Automated quality gates
- **Changes**:
  - Mock CLOB client with recorded responses
  - Mock oracle feed with deterministic prices
  - Test full signal→execution→settlement flow
- **Risk Reduced**: Regression risk
- **Complexity**: L (3 days)
- **Exit Criteria**: 80% code coverage; CI blocks on failure

---

### Phase 2: Institutional-Grade Improvements to Reach 9+ (Weeks 4-6)

#### P2.1: LeggedHedge State Machine Rewrite

- **Objective**: Robust multi-leg execution
- **Changes**:
  - Add leg correlation tracking
  - Calculate hedge size from actual fill
  - Use market-aware pricing (mid + slippage buffer)
  - Add timeout and partial fill handling
- **Risk Reduced**: Model risk (unprofitable hedges)
- **Complexity**: L (3 days)
- **Exit Criteria**: Backtested hedge PnL positive

#### P2.2: Short Arbitrage Implementation

- **Objective**: Capture reverse arb opportunities
- **Changes**:
  - Check portfolio for existing positions
  - Execute sells with matched sizes
  - Add short arb to metrics
- **Risk Reduced**: Opportunity cost
- **Complexity**: M (2 days)
- **Exit Criteria**: Short arb profits in backtest

#### P2.3: Real-Time Mark-to-Market

- **Objective**: Accurate unrealized P&L
- **Changes**:
  - Add MTM loop to strategy (every iteration)
  - Fetch current prices for all positions
  - Update portfolio unrealized P&L
  - Push to metrics
- **Risk Reduced**: Reporting risk
- **Complexity**: M (1 day)
- **Exit Criteria**: Dashboard shows live P&L

#### P2.4: Position Reconciliation

- **Objective**: Sync with exchange state
- **Changes**:
  - Enhance `Reconciler` to compare position sizes
  - Auto-adjust local state on divergence
  - Emit `POSITION_DRIFT` event for audit
- **Risk Reduced**: Accounting risk
- **Complexity**: M (2 days)
- **Exit Criteria**: Zero drift alerts in 24h live paper

#### P2.5: Deterministic Backtest Framework

- **Objective**: Validate strategies historically
- **Changes**:
  - Integrate `backtesting/engine.py` with strategies
  - Add historical data loader
  - Replay order books and trades
  - Compare simulated vs actual fills
- **Risk Reduced**: Model risk
- **Complexity**: L (5 days)
- **Exit Criteria**: Backtest matches paper trading >95%

#### P2.6: Alert Integration

- **Objective**: Real-time incident response
- **Changes**:
  - Enable Telegram alerts
  - Add PagerDuty/Slack webhook support
  - Define alert severity taxonomy
  - Create on-call runbooks
- **Risk Reduced**: Operational risk
- **Complexity**: M (2 days)
- **Exit Criteria**: Alerts fire within 10s of trigger

#### P2.7: Chaos Testing Suite
- **Objective**: Resilience validation
- **Changes**:
  - Expand `test_chaos.py` with scenarios:
    - Oracle timeout
    - Exchange 500 burst
    - WebSocket disconnect
    - Rate limit saturation
  - Add fault injection hooks
- **Risk Reduced**: Operational risk
- **Complexity**: L (3 days)
- **Exit Criteria**: System recovers from all scenarios

---

### Phase 3: Stretch Goals for 10 (Weeks 7-10)

#### P3.1: HSM/Vault for Secrets

- Integrate with AWS Secrets Manager or HashiCorp Vault
- Rotate credentials automatically

#### P3.2: Multi-Region Failover

- Deploy to 2+ regions
- Automatic failover on health degradation

#### P3.3: ML Signal Refinement

- Train classifier on historical fills
- Predict adverse selection probability
- Adjust sizing dynamically

#### P3.4: Exchange-Native Idempotency

- Partner with Polymarket for proper idempotency key support
- Eliminate retry duplicate risk at source

#### P3.5: Formal Verification

- Model state machine in TLA+
- Prove no deadlocks or race conditions

---

## 4. Quantified Target-State Controls

### Data Freshness

| Metric | Threshold | Action |
| :--- | :--- | :--- |
| Oracle price age | >5s | Block new orders |
| Order book age | >3s | Refresh before trade |
| Chainlink divergence | >0.3% | Cancel all orders |

### Execution SLOs

| Metric | Target | Alert Threshold |
| :--- | :--- | :--- |
| Order reject rate | <1% | >2% |
| Fill latency (p99) | <500ms | >1s |
| Reconciliation drift | 0 | Any drift |
| Cancel success rate | >99% | <95% |

### Risk Limits

| Control | Value | Action |
| :--- | :--- | :--- |
| Daily loss hard | 5% of capital | Halt + flatten |
| Max drawdown | 10% | Kill switch |
| Per-token exposure | $10 | Reject order |
| Total exposure | $20 | Reject order |
| Consecutive losses | 5 | 10-min cooldown |

### Kill Switch Triggers

| Condition | Response Time |
| :--- | :--- |
| Manual trigger | Immediate |
| Drawdown breach | <1s |
| Exchange disconnect | 30s grace, then halt |
| Oracle stale | Immediate block |
| Error rate >10% | 1-min rolling window |

### Recovery Objectives

| Scenario | RTO | RPO |
| :--- | :--- | :--- |
| Bot restart | 30s | Last persisted state |
| Exchange outage | N/A (wait) | No trades |
| Oracle failover | 5s | Use stale + flag |

---

## 5. Testing Strategy Upgrade

### Test Matrix

| Layer | Type | Coverage Target | Tools |
| :--- | :--- | :--- | :--- |
| Unit | Pure functions | 90% | pytest |
| Integration | Module interaction | 70% | pytest + mocks |
| E2E | Full trading loop | 50% | pytest + replay |
| Chaos | Fault tolerance | 10 scenarios | toxiproxy, pytest |
| Performance | Latency/throughput | p99 targets | locust |
| Security | SAST/DAST | All secrets | bandit, trivy |

### Deterministic Backtest/Replay

```python
# Replay harness structure
class ReplayEngine:
    def __init__(self, historical_data: Path):
        self.oracle = MockOracle(data=historical_data)
        self.exchange = MockCLOB(fills=historical_data)
    
    async def run(self, strategy: EnsembleStrategy) -> BacktestResult:
        # Deterministic timestamp progression
        # Record all trades
        # Compare to expected outcomes
```

### Integration Test Design

- **Mocked components**: CLOB client, Oracle feed, WebSocket
- **Recorded fixtures**: Real order books, prices, fills
- **Assertions**: Signal generation, order lifecycle, P&L

### Stress/Load Tests

- 1000 markets/second discovery
- 100 concurrent order submissions
- 50 WebSocket fills/second
- Burst: 10x normal rate for 30s

### Fault Injection Scenarios

1. Oracle returns `None` for 60s
2. CLOB returns 500 for 10 consecutive calls
3. WebSocket disconnects mid-fill
4. Rate limit exhausted
5. Reconciler finds 50% drift
6. Kill switch file appears
7. Out-of-memory simulation
8. Clock skew (future timestamps)
9. Invalid price data (negative, >1.0)
10. Concurrent settlement + trade

### CI/CD Gates

```yaml
# Required checks before deploy
- unit_tests: 90% pass rate
- integration_tests: 100% pass rate
- coverage: >=70%
- lint: zero errors
- security: zero HIGH/CRITICAL
- chaos: all scenarios recover
- performance: p99 < 500ms
```

---

## 6. Path to 8/9/10 Summary

### Current State: **5.7/10**

- Core trading logic exists and is reasonably structured
- Critical bugs in settlement, concurrency, and memory management
- Insufficient testing and operational tooling

### Target: **8/10** (Production-Safe)

**Timeline**: 3 weeks (Phase 0 + Phase 1)

**Key Milestones**:

- Week 1: All Phase 0 blockers resolved
- Week 2: Execution reliability fixes (P1.1-P1.5)
- Week 3: Risk controls + integration tests (P1.6-P1.8)

**Exit Criteria**:

- Zero critical findings
- 80% test coverage
- 24h paper trading with no incidents
- All SLOs met

### Target: **9/10** (Institutional-Grade)

**Timeline**: 3 additional weeks (Phase 2)

**Key Changes**:

- Full backtest framework with validation
- Chaos testing suite
- Real-time alerting and runbooks
- Position reconciliation

**Exit Criteria**:

- Backtest accuracy >95%
- All chaos scenarios pass
- <1% order reject rate in paper
- Oncall runbooks documented

### Target: **10/10** (Best-in-Class)

**Timeline**: 4 additional weeks (Phase 3)

**Key Changes**:

- HSM/vault integration
- Multi-region deployment
- ML signal enhancement
- Formal verification

**Exit Criteria**:

- SOC2 Type II audit ready
- >99.9% uptime SLA
- Regulatory compliance (where applicable)

---

## Appendix: Quick Reference

### Critical Files Requiring Immediate Attention

1. `src/strategy/value_betting.py` - Settlement, concurrency, memory
2. `src/execution/clob_client.py` - Idempotency, tick size, cancel
3. `src/ingestion/oracle_feed.py` - Chainlink integration
4. `src/infrastructure/events.py` - Async publishing

### Recommended Reading Order for New Engineers

1. `ARCHITECTURE.md`
2. `src/main.py`
3. `src/strategy/base.py`
4. `src/risk/gatekeeper.py`
5. `src/execution/order_manager.py`

### Emergency Procedures
1. **Kill Switch**: Create `/tmp/kill_switch` file
2. **Manual Halt**: Call `circuit_breaker.manual_halt()`
3. **Cancel All**: `POST /api/cancel_all` or `order_manager.cancel_all_orders()`
4. **Restart**: `Ctrl+C` → wait for graceful shutdown → restart
