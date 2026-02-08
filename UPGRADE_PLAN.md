# PolyBawt: Institutional-Grade Upgrade Plan
## From 7.5/10 → 10/10 — A Unified Synthesis & Original Architecture

**Date:** February 8, 2026  
**Scope:** Polymarket 15-Minute Crypto Binary Markets (BTC, ETH, SOL, XRP)  
**Classification:** Principal Quantitative Research — Final Remediation Blueprint

---

## 1. Executive Summary

### What Must Change and Why

PolyBawt is an architecturally mature solo project — structured logging, event-driven async, Kelly sizing, VPIN framework, circuit breakers — that is fundamentally misaligned with the market it trades. Polymarket's January 2026 fee regime transformation (dynamic taker fees peaking at ~1.56–3.15% at p=0.50, 100% redistribution to makers via daily USDC rebates) has inverted the economics of every speed-dependent strategy in the system. Three of four strategies (ArbTaker, LatencySnipe, LeggedHedge) are now negative-EV or fragile. Only SpreadMaker has a path to durable edge — and only if it becomes toxicity-aware, fee-accurate, and replay-validated.

Simultaneously, the settlement oracle is Chainlink Data Streams, not Binance spot. The system currently treats exchange prices as settlement truth, creating systematic basis risk that corrupts alpha signals and PnL attribution.

The upgrade plan transforms PolyBawt from a multi-strategy prototype into a maker-centric, toxicity-gated, replay-validated trading system through three pillars:

1. **Kill negative-EV components immediately** — stop the bleeding before building.
2. **Harden the execution and state layer** — local orderbook, idempotency, backpressure, SLOs.
3. **Build the validation infrastructure** — deterministic replay, exchange simulator, PnL attribution.

### Consensus Across All Four Audits

All four research documents independently converge on the same conclusions:

- **ArbTaker: Decommission.** The fee curve makes two-leg taker arbs negative-EV unless combined bids exceed ~1.03–1.08 at mid-probability (a catastrophic market failure scenario).
- **LatencySnipe: Decommission or radically restructure.** Operating at 120s lookback against 10–200ms professional infrastructure is three orders of magnitude too slow. After taker fees, the strategy is a liquidity donation mechanism.
- **LeggedHedge: Decommission.** Binary gamma risk near strike is infinite. No vol normalization. Heuristic crash rules without statistical backing.
- **SpreadMaker: Elevate to primary.** The only strategy structurally aligned with maker-rebate economics. Spread + rebate revenue can exceed adverse selection if toxicity-gated.

### Current Score: 4.5/10 → Target: 9+/10 in 10–12 Weeks

---

## 2. Current System: Key Bottlenecks, Risks, and Failure Modes

### 2.1 Graded Subsystem Assessment (Reconciled Across All Audits)

| Subsystem | Score | Critical Gap |
|:---|:---:|:---|
| Strategy Logic | 4/10 | Three of four strategies are structurally negative-EV post-fee-change |
| Alpha Integrity | 3/10 | No deterministic backtest; edge assumptions predate fee + competition regime; oracle-vs-exchange basis unmodeled |
| Risk Controls | 6/10 | Good primitives (Kelly, VPIN, breakers, caps) but VPIN is unwired; toxicity detection is a logged feature, not a gate |
| Execution Reliability | 5/10 | Async design and CLOB client exist but no local orderbook with sequence tracking; no latency SLOs; no backpressure |
| Reconciliation / State | 5/10 | Reconciler exists but lacks idempotency keys, deterministic replay, crash-safe event persistence |
| Observability | 5.5/10 | Structured logging + metrics present, but no PnL attribution, fill-quality decomposition, or toxicity-vs-PnL dashboards |
| Testing Maturity | 3/10 | No exchange simulator; no deterministic replay harness; validation is "tested in production" |
| Production Readiness | 4/10 | Health endpoints and breakers exist but no alerting matrix, runbooks, chaos tests, or shadow-mode validation |

### 2.2 Top 7 Failure Modes (Ranked by Severity × Likelihood)

**F1. Oracle Basis Risk (Severity: Critical).** The system uses Binance spot as settlement truth, but contracts resolve on Chainlink Data Streams. In boundary conditions (price near strike at expiry), oracle-vs-exchange divergence creates false-positive signals and incorrect PnL attribution.

**F2. Fee-Model Mismatch (Severity: Critical).** Static ~2% fee assumptions vs. dynamic probability-dependent curve means most detected "arbs" are actually net-negative after real fees. The system systematically takes losing trades it believes are winners.

**F3. State Desynchronization (Severity: High).** Missing WebSocket sequence tracking means the bot can miss fills ("ghost orders"), believe it holds positions it doesn't, or breach exposure caps silently. Under high-throughput conditions, this is not rare.

**F4. Adverse Selection Blindness (Severity: High).** SpreadMaker runs without VPIN gating. When informed flow spikes (spot moves, news events, settlement boundary sniping), stale quotes become toxic fill magnets. A single undetected toxic regime can erase days of spread capture.

**F5. Backpressure Absence (Severity: High).** No priority queue or coalescing policy means fills and cancels compete with book deltas. Under load, critical events (fills, acks) can be delayed while processing stale market data.

**F6. Latency Opacity (Severity: Medium-High).** No p50/p95/p99 tracking for data→signal, signal→submit, submit→ack, or cancel→ack latency. Degradation is invisible until it manifests as PnL loss.

**F7. No Validation Pipeline (Severity: Medium-High).** Without deterministic replay and exchange simulation, every claim about edge is unfalsifiable. This blocks rational capital allocation and strategy improvement.

---

## 3. Target Architecture

### 3.1 Design Principles

1. **Maker-first, taker-selective.** Default to passive liquidity provision. Taker execution only for fee-favorable probability extremes or validated dislocations.
2. **Oracle-canonical.** Chainlink Data Streams is the settlement truth. Exchange feeds are features, not ground truth.
3. **Event-sourced, replay-deterministic.** Every decision the system makes must be reproducible from logged events under configurable latency/fee/queue assumptions.
4. **Fail-closed, risk-bounded.** Unknown state → cancel all orders. Breached invariant → halt trading. Stale data → disable quoting.
5. **Toxicity-gated.** No quote exists without a toxicity assessment. High toxicity → widen or withdraw; never quote blind.

### 3.2 Module Architecture (Clean Boundaries)

```text
┌─────────────────────────────────────────────────────────┐
│                    INGESTION LAYER                       │
│  ┌──────────┐ ┌──────────┐ ┌───────────┐ ┌──────────┐  │
│  │Polymarket│ │ Binance  │ │ Chainlink │ │ Coinbase │  │
│  │CLOB WSS  │ │ Spot WSS │ │DataStream │ │ Fallback │  │
│  └────┬─────┘ └────┬─────┘ └─────┬─────┘ └────┬─────┘  │
│       └──────┬─────┴──────┬──────┘─────────────┘        │
│              ▼            ▼                               │
│  ┌──────────────────────────────────┐                    │
│  │    EVENT BUS (Priority Queue)    │                    │
│  │  P0: fills, cancels, kill-switch │                    │
│  │  P1: order acks, active book     │                    │
│  │  P2: snapshots, background       │                    │
│  │  Coalescing: book deltas by      │                    │
│  │    token_id under backpressure   │                    │
│  └──────────────┬───────────────────┘                    │
└─────────────────┼───────────────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────────────┐
│                  BOOK + STATE LAYER                      │
│  ┌──────────────────┐  ┌──────────────────────────────┐ │
│  │ Local Orderbook   │  │  Order State Machine         │ │
│  │ (seq-validated,   │  │  NEW→SUBMITTED→ACKED→        │ │
│  │  snapshot-recover) │  │  {PARTIAL→FILLED,CANCELED,  │ │
│  │                    │  │   REJECTED,EXPIRED}          │ │
│  └────────┬───────────┘  └──────────┬───────────────── │ │
│           │                         │                    │
│  ┌────────┴───────────┐  ┌──────────┴───────────────┐  │
│  │ Microprice Engine  │  │  Portfolio Reconciler     │  │
│  │ (VWAP-weighted mid)│  │  (periodic + event-driven)│  │
│  └────────┬───────────┘  └──────────────────────────┘  │
└───────────┼─────────────────────────────────────────────┘
            ▼
┌─────────────────────────────────────────────────────────┐
│                  SIGNAL + RISK LAYER                     │
│  ┌─────────────────┐  ┌──────────────────────────────┐  │
│  │  Fee Model       │  │  Toxicity Panel              │  │
│  │  (dynamic curve, │  │  (VPIN + flow-imbalance +    │  │
│  │   rebate calc)   │  │   microprice-drift +         │  │
│  │                  │  │   realized-spread +           │  │
│  │                  │  │   time-to-expiry decay)       │  │
│  └─────────────────┘  └──────────────────────────────┘  │
│  ┌─────────────────┐  ┌──────────────────────────────┐  │
│  │  Risk Gate       │  │  Capital Allocator           │  │
│  │  (invariants,    │  │  (strategy-level Kelly,      │  │
│  │   drawdown,      │  │   regime-adjusted,           │  │
│  │   kill-switch)   │  │   exposure limits)            │  │
│  └─────────────────┘  └──────────────────────────────┘  │
└───────────┬─────────────────────────────────────────────┘
            ▼
┌─────────────────────────────────────────────────────────┐
│                  STRATEGY LAYER                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │  SpreadMaker (PRIMARY)                          │    │
│  │  • Toxicity-gated quoting                       │    │
│  │  • Microprice-anchored spread calculation       │    │
│  │  • Time-to-expiry-aware regime switching        │    │
│  │  • Inventory skew with mean-reversion target    │    │
│  │  • Post-only enforcement, fee-rebate-aware EV   │    │
│  └─────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────┐    │
│  │  StatArb Sleeve (SECONDARY — Phase 5)           │    │
│  │  • Cross-asset mean-reversion (BTC/ETH implied  │    │
│  │    vol divergence)                              │    │
│  │  • Probability-extreme taker arb (p<0.10 or     │    │
│  │    p>0.90 where fees ≈ 0)                       │    │
│  │  • Multi-window holding (hours, not seconds)    │    │
│  └─────────────────────────────────────────────────┘    │
└───────────┬─────────────────────────────────────────────┘
            ▼
┌─────────────────────────────────────────────────────────┐
│                  EXECUTION LAYER                        │
│  ┌──────────────────┐  ┌───────────────────────────┐   │
│  │  Order Manager    │  │  Idempotency Registry     │   │
│  │  (post-only,      │  │  (COID:{run}:{strat}:     │   │
│  │   cancel-before-  │  │   {token}:{ts}:{nonce})   │   │
│  │   requote,        │  └───────────────────────────┘   │
│  │   rate-limited)   │                                   │
│  └──────────────────┘                                    │
└───────────┬─────────────────────────────────────────────┘
            ▼
┌─────────────────────────────────────────────────────────┐
│              PERSISTENCE + OBSERVABILITY                 │
│  ┌──────────────┐ ┌────────────┐ ┌─────────────────┐   │
│  │ NDJSON Event │ │ Metrics    │ │ PnL Attribution │   │
│  │ Log (append- │ │ (Prometheus│ │ (by strategy,   │   │
│  │  only, replay│ │  /StatsD)  │ │  regime, token,  │   │
│  │  -grade)     │ │            │ │  fill-quality)   │   │
│  └──────────────┘ └────────────┘ └─────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Strategy & Alpha Layer Redesign

### 4.1 Strategy Disposition

| Strategy | Disposition | Rationale |
|:---|:---|:---|
| ArbTaker | **Decommission immediately** | Fee curve makes two-leg taker arb negative-EV in 35–65% probability zone. "Arbs" detected by static 0.98 threshold are fee-taxed losers. |
| LatencySnipe | **Decommission** | 120s lookback vs. 10–200ms professional infra is non-competitive. Taker fees at mid-probability eliminate residual edge. Adverse selection dominates fills. |
| LeggedHedge | **Decommission** | Binary gamma risk near strike is infinite. No vol normalization. Heuristic crash rules without statistical backing. Complexity unjustified for $5 notional. |
| SpreadMaker | **Elevate to primary** | Only strategy structurally aligned with maker-rebate economics. Spread + rebate revenue can exceed adverse selection if toxicity-gated. |

### 4.2 SpreadMaker: Why It Should Work in 15-Minute Binaries

**The economic argument:** Polymarket explicitly subsidizes makers through daily USDC rebates funded by taker fees. The rebate formula is "fee-curve weighted" — makers providing liquidity near p=0.50 receive maximum compensation, precisely because adverse selection risk is highest there. This is a structural edge: the platform pays you to stand in the line of fire, provided you can dodge the bullets.

**The microstructure argument:** With $5 max trade and $20 total exposure, PolyBawt is well below depth constraints (avg. book depth ~$2.1M). A small, intelligent maker doesn't move the market and doesn't need front-of-queue fill rates to be profitable. The edge comes from when you quote, not how fast you quote.

**How it can fail or decay:**

- **Spread compression:** As more professional MMs enter and spreads tighten from 1.2% toward the tick floor, gross spread capture shrinks. Monitor: effective spread per fill, trend over rolling 30 days.
- **Adverse selection escalation:** If toxic flow increases (more snipers, faster information propagation), fills become systematically unprofitable. Monitor: post-fill microprice move at +250ms/+1s/+5s.
- **Rebate dilution:** If maker volume grows faster than taker fees, per-unit rebate decreases. Monitor: daily rebate per $1000 of maker volume.
- **Regime shifts:** Volatility regime changes (low-vol = no fills; extreme-vol = all fills are toxic). Monitor: fill rate and adverse selection by realized volatility decile.

**Degradation response:**

- Adverse selection >50% of spread for 3 consecutive days → widen minimum spread by 1 tick.
- Daily net PnL negative for 5 consecutive days → reduce quoting to 50% of available windows.
- Effective spread per fill trending below 0.5¢ over 14 days → strategy review, potential pause.

### 4.3 Future Alpha Sleeves (Phase 5 Only)

**Probability-Extreme Taker Arb:** At p<0.10 or p>0.90, taker fees are negligible (<0.5%). Genuine mispricings at extremes — particularly around settlement boundaries — become viable for selective taker execution. This is the only regime where ArbTaker-like logic may survive post-fee.

**Cross-Asset Mean-Reversion:** BTC and ETH 15-minute binaries are correlated but imperfectly so. When implied probability divergences exceed historical norms (e.g., BTC binary implies 60% up-probability while ETH implies 45% despite 0.92 correlation), a pairs-like mean-reversion trade across contracts becomes plausible. Holding horizon: hours, not seconds. This exploits analytical sophistication, not speed.

**Implied Volatility Surface Arbitrage:** As market matures, cross-window (15-minute vs. 1-hour) probability structures may create vol surface inconsistencies. This requires substantial data collection before any trading.

---

## 5. Execution & Market Microstructure Layer

### 5.1 Maker vs. Taker Logic

**Default posture: Maker-only.** All SpreadMaker orders must use post-only flag where available, preventing accidental crossing into taker fills. The taker path requires explicit activation via a TakerGate that verifies:

- Probability is in extreme regime (p<0.10 or p>0.90 where fees <0.5%)
- Expected edge exceeds fee + estimated slippage + adverse selection buffer
- Strategy has been validated via replay with positive net EV in this regime

**Order lifecycle:** Cancel-before-requote pattern. Every requote cycle: (1) cancel existing orders, (2) wait for cancel ack, (3) evaluate toxicity and book state, (4) place new orders if conditions pass. Never have overlapping live orders for the same token and side.

### 5.2 Queue Position & Fill Probability

Queue position is the hidden variable that dominates maker PnL. First-in-queue gets 85–95% fill rate but maximum adverse selection exposure. Positions 6–15 get 35–55% fill rate with lower adverse selection.

**Estimation approach:** Track local book depth at your price level. Count orders ahead of yours by monitoring book updates after your order placement. Estimate fill probability using a simple Poisson model based on historical fill rate at that depth.

**Optimal positioning:** PolyBawt's small size ($5 max) is an advantage here — it doesn't need front-of-queue. Quoting 2–5 ticks behind the best price during toxic regimes and tightening during clean flow exploits the queue position vs. adverse selection tradeoff.

### 5.3 Slippage & Adverse Selection Measurement

Every fill must be tagged with post-fill metrics:

```yaml
fill_quality:
  realized_spread:  (fill_price - microprice_at_fill) × side_sign
  adverse_250ms:    microprice(t+250ms) - microprice(t_fill)
  adverse_1s:       microprice(t+1s) - microprice(t_fill)
  adverse_5s:       microprice(t+5s) - microprice(t_fill)
  regime:           {calm|trending|volatile|settlement_approach}
  vpin_at_fill:     current VPIN reading
  time_to_expiry:   seconds remaining in 15-min window
  fee_saved:        maker_rebate_estimate for this fill
```

This data feeds toxicity model calibration and strategy-level PnL attribution.

### 5.4 Cancellation Policy

**Mandatory cancel triggers (immediate, no delay):**

- Kill-switch event (any source)
- Oracle staleness exceeds 900ms
- Local book sequence gap detected
- Toxicity panel enters CRITICAL regime
- Daily loss limit hit

**Proactive cancel triggers (next requote cycle):**

- VPIN exceeds TOXIC threshold (configurable)
- Time-to-expiry < 60 seconds (settlement sniper avoidance)
- Inventory exceeds skew threshold and no favorable fills in 30s

### 5.5 Latency Sensitivity & Failure Modes

**Latency budget (SLOs):**

| Metric | p50 SLO | p99 SLO | Violation Action |
|:---|:---|:---|:---|
| Oracle staleness (Chainlink) | ≤250ms | ≤900ms | Disable quoting for affected token |
| Local book freshness | ≤250ms | ≤1.5s | Cancel quotes, re-snapshot, cooldown |
| Cancel-to-ack | ≤300ms | ≤1.5s | Alert if p99 >2s for 5 minutes |
| Fill detection | ≤150ms | ≤700ms | Alert; potential state desync |
| Data→signal | ≤100ms | ≤500ms | Performance review |
| Signal→submit | ≤50ms | ≤200ms | Event loop analysis |

### 5.6 Idempotency, Retries, and State Reconciliation

**Client Order ID format:** `COID:{run_id}:{strategy}:{token_id}:{intent_ts_ns}:{nonce}`

Every order intent generates a unique `intent_id`. All retries reference the same `intent_id`. The idempotency registry (in-memory set + periodic disk flush) prevents duplicate submissions.

**Reconciliation protocol:**

1. **Continuous:** Every fill/cancel ack updates order state machine; invalid transitions logged as bugs.
2. **Periodic (every 60s):** Cross-check internal position state against CLOB API reported positions. Mismatch > threshold → halt trading, alert, attempt reconciliation.
3. **On restart:** Load last checkpoint, replay unacknowledged intents, reconcile against CLOB, enter shadow mode until state confirmed clean.

---

## 6. Risk, Capital Allocation & Kill-Switch Framework

### 6.1 Layered Risk Architecture

```text
Layer 1: PER-ORDER LIMITS
  • Max order size: $5
  • Post-only enforcement
  • Fee-aware EV check (must be positive after dynamic fees)

Layer 2: PER-TOKEN LIMITS
  • Max inventory per token: configurable (default $10)
  • Max open orders per token: 4 (2 per side)

Layer 3: PER-STRATEGY LIMITS
  • Daily loss limit per strategy: -$X (configurable)
  • Strategy disabled on breach until next trading day
  • Trade frequency ceiling

Layer 4: PORTFOLIO LIMITS
  • Total exposure cap: $20
  • Total daily loss limit: -$Y (configurable, e.g., -$5)
  • Cross-token correlation-aware (BTC+ETH exposure combined)

Layer 5: SYSTEM KILL-SWITCHES
  • Manual kill (API endpoint + keyboard shortcut)
  • Auto-kill on: state desync, 3 consecutive failed reconciliations,
    latency SLO violation >5min, oracle staleness >5s, API error rate >30%
  • Kill = cancel all orders + disable new orders + alert
```

### 6.2 Drawdown Control

**Trailing drawdown stop:** If portfolio mark-to-market drops X% from peak equity (configurable, recommend 5–10% for small capital), enter "recovery mode":

- Reduce position sizing to 50%
- Widen minimum spreads by 1 tick
- Increase requote interval
- Alert operator

**Daily hard stop:** If daily realized loss exceeds threshold, halt all trading for remainder of day. No overrides except manual restart with explicit acknowledgment.

### 6.3 Tail-Risk Protection

**Settlement-boundary guard:** Pull all maker quotes 60 seconds before contract expiry. The "Chainlink sniper race" — where HFTs with oracle preview windows pick off stale quotes — is an existential risk for Python-speed makers. The rebate loss from missing final-minute flow is far outweighed by the adverse selection avoided.

**News/event guard:** If Chainlink-vs-Binance price divergence exceeds 3σ of recent distribution, or if spot volatility spikes >2x 15-minute rolling average, enter "wide-only" mode (minimum spread 3x normal) or pause entirely.

### 6.4 Regime Detection

Classify current regime using a simple decision tree updated every 5 seconds:

| Regime | Detection Criteria | SpreadMaker Response |
|:---|:---|:---|
| CALM | Realized vol < 30th percentile; VPIN < 0.3 | Normal spreads, full quoting |
| TRENDING | Realized vol 30-70th percentile; directional microprice drift | Skew quotes toward trend; moderate spreads |
| VOLATILE | Realized vol > 70th percentile; VPIN 0.3-0.6 | Wide spreads (2x); reduced size |
| TOXIC | VPIN > 0.6 OR flow-imbalance extreme OR time-to-expiry < 60s | Pull all quotes; cooldown |

### 6.5 Model Risk Management

VPIN is academically contested and may not reliably predict toxicity in this specific product. Therefore:

- **Never use VPIN as a sole gate.** Always combine with at least two other signals (flow-imbalance, microprice-drift, realized-spread deterioration).
- **Continuously calibrate.** Every 7 days, compute VPIN-decile vs. post-fill PnL. If VPIN has no predictive power (flat or inverted relationship), demote it and increase weight on alternatives.
- **Track model confidence.** Maintain a "toxicity model health" metric — if out-of-sample prediction accuracy drops below 55%, disable toxicity-gated quoting adjustments and revert to conservative wide spreads.

---

## 7. Research, Backtesting & Validation Pipeline

### 7.1 Deterministic Replay Harness (Must-Have)

**Purpose:** Given NDJSON event logs from live or historical capture, reproduce every decision and PnL outcome deterministically under configurable assumptions.

**Components:**

1. **EventLog (NDJSON):** Append-only, ordered by `ts_recv_ns` (monotonic). Separate source vs. receive timestamps. Event types: BOOK_DELTA, BOOK_SNAPSHOT, ORACLE_TICK, SIGNAL, ORDER_SUBMIT, ORDER_ACK, FILL, CANCEL_ACK, RESOLUTION, HEALTH.

2. **Replayer:** Reads EventLog, emits events into single-threaded async loop. Supports `speed=1.0` (real time) and `speed=∞` (AFAP). Injects configurable: ack_delay_ms, wss_jitter_ms, snapshot_gap_prob.

3. **ExchangeSimulator:** Maintains per-token limit orderbook with price-time priority. Enforces tick_size, min_order_size, post-only rejection, dynamic taker fee curve. Emits fills with fill_id, match_time, fee_paid, estimated_rebate.

4. **PnL Engine:** Marks inventory to current midpoint (intraday risk) and settlement outcome (final). Tracks: realized spread, mark-to-market, fee spend, rebate estimates, adverse selection decomposition.

**Exit criteria:** Same EventLog + same config → identical output hash. Replay of "wiretap mode" (observed exchange fills) produces slippage distribution matching empirical within tolerance.

### 7.2 Walk-Forward Testing

Standard time-series cross-validation adapted for market structure regime changes:

- **In-sample window:** 14 rolling days, step 7 days
- **Out-of-sample window:** 7 days following each in-sample period
- **Regime stratification:** Ensure each fold contains calm, volatile, and trending periods
- **Output:** Distribution of strategy Sharpe, max drawdown, and fill quality across folds

Critically: if any walk-forward fold shows negative net PnL, the strategy does not promote to live.

### 7.3 Monte Carlo & Stress Testing

**Bootstrap PnL simulation:** Resample daily PnL outcomes (with replacement) to generate 10,000 paths. Report: probability of ruin at various bankroll levels, probability of drawdown >X%, expected recovery time from max drawdown.

**Scenario stress tests:**

- Flash crash: BTC drops 5% in 30 seconds. Are quotes cancelled in time?
- Oracle outage: Chainlink stops updating for 30 seconds. Does system enter safe state?
- API degradation: CLOB ack latency increases 10x for 5 minutes. Is backpressure engaged?
- Fill storm: 10 fills arrive within 100ms. Are all processed without state corruption?

### 7.4 Ablation Studies

For every feature of SpreadMaker (VPIN gating, microprice anchoring, time-to-expiry adjustment, inventory skew), run the replay with that feature disabled. If removing a feature doesn't degrade risk-adjusted returns, the feature is noise and should be simplified out.

### 7.5 Overfitting Detection

- **Parameter count audit:** SpreadMaker should have ≤10 tunable parameters. More = overfit risk.
- **Sensitivity analysis:** Perturb each parameter ±20%. If performance degrades >30%, the parameter is fragile and the model is likely overfit to historical specifics.
- **In-sample vs. out-of-sample Sharpe ratio:** If IS Sharpe > 2x OOS Sharpe, the model is likely overfit.

### 7.6 Post-Trade Analytics and Feedback Loops

**Daily automated report:**

- Net PnL by strategy, token, and regime
- Fill quality decomposition (realized spread, adverse selection at 250ms/1s/5s)
- Fee spend vs. estimated rebate
- VPIN effectiveness (toxicity prediction accuracy)
- Latency histogram (data→signal, signal→submit, submit→ack, cancel→ack)
- Inventory statistics (max, mean, time-to-flat)
- Strategy-specific KPIs (fill rate, quote-to-trade ratio, effective spread)

**Weekly review checklist:**

- Is SpreadMaker net positive after all costs?
- Is adverse selection trending up or down?
- Have any SLOs been breached?
- Is VPIN still predictive of fill quality?
- Are there regime shifts that require parameter adjustment?

---

## 8. Observability, Metrics & Operations

### 8.1 Golden Signals (Must-Track)

| Signal | Metric | Alert Threshold |
|:---|:---|:---|
| **Throughput** | Orders/minute, fills/minute | < 50% of recent average for > 10 minutes |
| **Latency** | p50/p95/p99 for each pipeline stage | p99 > 2x SLO for > 5 minutes |
| **Error rate** | API errors/minute, state machine violations | > 5% of requests or any state violation |
| **PnL** | Rolling 1-hour net PnL | < -$2 in any hour (configurable) |
| **Saturation** | Event queue utilization % | > 80% for > 30 seconds |

### 8.2 Strategy-Specific Dashboards

**SpreadMaker Dashboard:**

- Real-time: current quotes (both sides), inventory, VPIN, regime classification, microprice
- Rolling 1-hour: fill rate, effective spread, adverse selection, net PnL
- Daily: cumulative PnL, rebate estimate, fee spend, Sharpe, max inventory

**System Health Dashboard:**

- WebSocket connection status (all feeds)
- Event queue depth and processing rate
- Latency histograms (real-time, last 1 hour, last 24 hours)
- Oracle staleness (Chainlink, Binance, Polymarket book)
- Reconciliation status (last check time, result)

### 8.3 Critical Logs & Traces

**Every order must produce a trace:**

```json
{
  "intent_id": "...",
  "client_order_id": "...",
  "strategy": "...",
  "token_id": "...",
  "side": "...",
  "price": 0.0,
  "size": 0.0,
  "signal_ts": "...",
  "submit_ts": "...",
  "ack_ts": "...",
  "fill_ts": "...",
  "fill_price": 0.0,
  "fill_size": 0.0,
  "fee_paid": 0.0,
  "rebate_est": 0.0,
  "microprice_at_signal": 0.0,
  "microprice_at_fill": 0.0,
  "vpin_at_signal": 0.0,
  "regime_at_signal": "...",
  "queue_position_est": 0,
  "adverse_250ms": 0.0,
  "adverse_1s": 0.0,
  "adverse_5s": 0.0
}
```

**Retention:** 90 days minimum for all structured logs. Event logs (NDJSON) retained indefinitely (they are the system of record for replay validation).

### 8.4 Debugging Bad Days

When daily PnL is significantly negative, the investigation playbook:

1. **Was there a regime shift?** Check realized vol, VPIN, flow-imbalance for the day. Compare to historical.
2. **Was there execution degradation?** Check latency histograms. Were SLOs breached? Were cancels delayed?
3. **Was there a data integrity issue?** Check sequence gaps in local book. Check oracle staleness events.
4. **Was adverse selection elevated?** Compute post-fill microprice moves. Were fills systematically on the wrong side?
5. **Was the fee model wrong?** Compare estimated fees to actual (if observable). Check that dynamic fee curve is still accurate.
6. **Replay the day.** Run deterministic replay with actual event logs. Does simulated PnL match realized PnL within tolerance? If not, there's a state consistency bug.

---

## 9. Codebase Refactor Plan

### 9.1 What to Delete

- `ArbTakerStrategy` and `ArbitrageDetector` — remove from active strategy registry; archive in `deprecated/` for reference.
- `LatencySnipeStrategy` — remove from active registry; archive.
- `LeggedHedgeStrategy` and `LegState`/`LegContext` — remove; archive.
- Any static fee constant (e.g., `FEE_RATE = 0.02`) — replace with dynamic FeeModel.
- Any reference to exchange spot as "settlement truth" — replace with oracle-canonical path.

### 9.2 What to Add

| New Module | Responsibility | Priority |
|:---|:---|:---|
| `src/market/fee_model.py` | Dynamic fee calculation using published curve parameters. Probability-dependent taker fee and rebate estimation. | P0 |
| `src/market/local_book.py` | Sequence-validated local orderbook with snapshot recovery, coalescing, and staleness detection. | P0 |
| `src/market/microprice.py` | Volume-weighted mid calculation from local book. Microprice used as anchor for all fair-value calculations. | P1 |
| `src/risk/toxicity_panel.py` | Multi-signal toxicity assessment (VPIN + flow-imbalance + microprice-drift + realized-spread). Returns regime classification. | P0 |
| `src/core/event_bus.py` | Priority-queue event dispatcher with backpressure, coalescing, and P0/P1/P2 classification. | P0 |
| `src/core/state_machine.py` | Formalized order and strategy state machines with explicit transition enforcement. | P1 |
| `src/core/idempotency.py` | Client order ID generation and duplicate detection registry. | P1 |
| `src/analytics/fill_quality.py` | Post-fill analysis: realized spread, adverse selection at multiple horizons, regime tagging. | P1 |
| `src/analytics/pnl_attribution.py` | Strategy-level PnL decomposition: gross spread, adverse selection, fees, rebates. | P2 |
| `src/replay/replayer.py` | Deterministic event replay engine. | P1 |
| `src/replay/exchange_sim.py` | Simulated exchange with price-time priority, fee curve, post-only behavior. | P1 |

### 9.3 What to Refactor

- **SpreadMaker:** Narrow quoting parameters (current `min_spread=0.05` is too wide for 1–2¢ effective spreads). Integrate toxicity_panel for dynamic spread adjustment. Add time-to-expiry-aware regime switching. Wire VPIN module into quoting decisions.
- **RiskGate:** Add portfolio-level invariant enforcement (total exposure, daily loss, cross-token correlation). Make all breaches logged with structured context.
- **Reconciler:** Add idempotency checking. Add scheduled reconciliation with CLOB API. Add mismatch → halt logic.
- **Metrics:** Add fill-quality metrics, latency histograms (p50/p95/p99), PnL attribution series, toxicity effectiveness tracking.

### 9.4 Performance & Safety

- **uvloop:** Replace default asyncio event loop with uvloop for 20–40% latency reduction on I/O operations.
- **WebSocket sequence tracking:** Track sequence numbers on all CLOB WebSocket messages. Detect gaps → trigger snapshot recovery.
- **Atomic config updates:** Strategy parameters loaded from versioned config. Changes require shadow-mode validation before live promotion.

---

## 10. Phased Implementation Roadmap

### Phase 1: Survive (Days 1–14) — Target: 6/10

**Theme:** Stop the bleeding. Kill negative-EV. Establish data collection.

| # | Task | Effort | EV Impact | Risk Impact |
|:---|:---|:---|:---|:---|
| 1 | Disable ArbTaker, LatencySnipe, LeggedHedge in config | Hours | High | High |
| 2 | Implement `FeeModel` with dynamic curve parameters | 2 days | High | Medium |
| 3 | Wire VPIN into SpreadMaker quoting (widen on VPIN > threshold) | 2 days | High | High |
| 4 | Implement priority event queue with P0/P1/P2 classification | 3 days | Medium | High |
| 5 | Add fill-quality logging (realized spread, adverse selection at +250ms/+1s/+5s) | 2 days | High (data) | Low |
| 6 | Add latency SLO tracking (p50/p95/p99 for all pipeline stages) | 2 days | Medium | Medium |
| 7 | Replace Binance-as-truth with Chainlink-as-truth for fair value | 2 days | High | High |
| 8 | Narrow SpreadMaker quote parameters for competitive flow capture | 1 day | Medium | Low |
| 9 | Implement settlement-boundary guard (pull quotes 60s before expiry) | 1 day | Low | High |
| 10 | Run 2-week shadow mode: full code path, no live orders, complete logging | Ongoing | Critical | Critical |

**Exit criteria:** 2 weeks of clean shadow logs. Fill-quality data collected. Latency baselines established. Zero state machine violations.

### Phase 2: Validate (Weeks 3–4) — Target: 7/10

**Theme:** Build the replay harness. Prove edge exists.

| # | Task | Effort | EV Impact | Risk Impact |
|:---|:---|:---|:---|:---|
| 1 | Build NDJSON event logger (append-only, replay-grade) | 3 days | Critical | Medium |
| 2 | Build ExchangeSimulator (price-time priority, fee curve, post-only) | 5 days | Critical | Medium |
| 3 | Build deterministic Replayer | 3 days | Critical | Medium |
| 4 | Replay shadow-mode data through simulator. Compare to observed fills. | 2 days | Critical | High |
| 5 | Implement local orderbook with sequence validation and snapshot recovery | 4 days | High | High |
| 6 | Add backpressure and coalescing (book deltas by token_id under load) | 2 days | Medium | High |
| 7 | Formalize idempotency keys and duplicate detection | 2 days | Medium | High |
| 8 | Calibrate toxicity panel: VPIN-decile vs. post-fill PnL analysis | 2 days | High | Medium |

**Exit criteria:** Replay harness produces deterministic outputs. Shadow PnL simulation shows positive net EV for SpreadMaker after fees. Toxicity panel shows predictive power.

### Phase 3: Harden (Weeks 5–6) — Target: 8/10

**Theme:** Make the system safe to run unattended.

| # | Task | Effort | EV Impact | Risk Impact |
|:---|:---|:---|:---|:---|
| 1 | Formalize order state machine (explicit transitions, violation logging) | 3 days | Low | High |
| 2 | Implement periodic CLOB reconciliation with halt-on-mismatch | 2 days | Low | Critical |
| 3 | Add alerting matrix (daily loss, latency SLO, state desync, API errors) | 2 days | Low | High |
| 4 | Implement graceful degradation (Chainlink stale → skip; Binance down → Coinbase fallback) | 3 days | Low | High |
| 5 | Chaos/restart testing (kill mid-session; verify reconciliation recovery) | 2 days | Low | Critical |
| 6 | Walk-forward testing (14-day rolling, 7-day OOS, regime-stratified) | 3 days | High | Medium |
| 7 | Deploy to live with minimal capital ($20–50) with all monitoring active | Ongoing | Medium | Medium |

**Exit criteria:** Comfortable running overnight with modest capital. All alerts tested with fault injection. Walk-forward shows positive OOS performance.

### Phase 4: Optimize (Weeks 7–10) — Target: 9/10

**Theme:** Squeeze performance. Add statistical edges.

| # | Task | Effort | EV Impact | Risk Impact |
|:---|:---|:---|:---|:---|
| 1 | Monte Carlo stress testing and parameter sensitivity analysis | 3 days | Medium | Medium |
| 2 | Ablation studies on all SpreadMaker features | 2 days | Medium | Low |
| 3 | Overfitting audit (IS vs. OOS Sharpe, parameter count, sensitivity) | 2 days | Medium | Medium |
| 4 | Implement PnL attribution (by strategy, regime, token, fill-quality) | 3 days | High | Low |
| 5 | Develop probability-extreme taker arb sleeve (p<0.10, p>0.90) | 5 days | Medium | Medium |
| 6 | Evaluate uvloop and latency-critical-path optimization | 2 days | Low | Low |
| 7 | Scale to live with target capital ($100–250) | Ongoing | Medium | Medium |

### Phase 5: Extend (Weeks 11–12+) — Target: 9.5/10

**Theme:** New alpha sources. Cross-asset expansion.

| # | Task | Effort | EV Impact | Risk Impact |
|:---|:---|:---|:---|:---|
| 1 | Cross-asset mean-reversion (BTC/ETH implied probability divergence) | 1 week | Medium | Medium |
| 2 | Time-to-expiry lifecycle optimization (different quoting per bucket) | 3 days | Medium | Low |
| 3 | Queue position estimation and fill probability modeling | 3 days | Medium | Low |
| 4 | SOL/XRP market expansion (if liquidity sufficient) | 3 days | Medium | Medium |
| 5 | Continuous model retraining pipeline with automated walk-forward | 1 week | High | Medium |

---

## 11. Final "7.5 → 10/10" Upgrade Checklist

### Strategy Layer

- [ ] ArbTaker decommissioned
- [ ] LatencySnipe decommissioned
- [ ] LeggedHedge decommissioned
- [ ] SpreadMaker elevated to primary with toxicity-aware quoting
- [ ] Dynamic FeeModel integrated into all EV calculations
- [ ] Chainlink Data Streams used as canonical settlement truth
- [ ] Time-to-expiry-aware regime switching (pull quotes 60s before expiry)
- [ ] Microprice-anchored spread calculation
- [ ] Edge validated through deterministic replay with positive OOS performance

### Execution Layer

- [ ] Local orderbook with sequence validation and snapshot recovery
- [ ] Backpressure with priority queue (P0/P1/P2) and coalescing
- [ ] Post-only enforcement on all maker orders
- [ ] Cancel-before-requote lifecycle
- [ ] Idempotency keys on all order intents
- [ ] Rate-limited requotes

### Risk Layer

- [ ] Toxicity panel (VPIN + flow-imbalance + microprice-drift + realized-spread)
- [ ] Per-order, per-token, per-strategy, portfolio-level limits
- [ ] Daily loss hard stop
- [ ] Trailing drawdown recovery mode
- [ ] Settlement-boundary guard (quote withdrawal)
- [ ] News/event volatility guard
- [ ] Kill-switch (manual + automated)
- [ ] Regime classification (CALM/TRENDING/VOLATILE/TOXIC)

### State & Consistency

- [ ] Formalized order state machine with transition enforcement
- [ ] Periodic CLOB reconciliation with halt-on-mismatch
- [ ] Crash-safe event persistence (NDJSON append-only)
- [ ] Restart recovery: load checkpoint, reconcile, shadow-validate

### Validation Pipeline

- [ ] Deterministic replay harness with exchange simulator
- [ ] Walk-forward testing (14-day rolling, regime-stratified)
- [ ] Monte Carlo stress testing
- [ ] Ablation studies on all strategy features
- [ ] Overfitting detection (IS vs. OOS Sharpe, parameter sensitivity)
- [ ] Post-trade analytics with automated daily reports

### Observability

- [ ] Golden signals: throughput, latency, error rate, PnL, saturation
- [ ] Fill-quality metrics: realized spread, adverse selection at multiple horizons
- [ ] Latency SLOs with alerts (p50/p95/p99 for all pipeline stages)
- [ ] PnL attribution by strategy, regime, token
- [ ] Strategy-specific and system health dashboards
- [ ] 90-day structured log retention
- [ ] Replay-grade event logs retained indefinitely

### Production Operations

- [ ] Shadow-mode validation before any live promotion
- [ ] Chaos/restart testing passed
- [ ] Alerting matrix with runbooks
- [ ] Graceful degradation on vendor outages
- [ ] Change control: versioned configs, shadow-validate before deploy
- [ ] Incident playbook for common failure modes

---

*This plan synthesizes and resolves the strongest recommendations from all four audit documents, adds original architectural proposals (priority event bus design, settlement-boundary guard, regime-aware queue positioning, cross-asset mean-reversion sleeve, automated walk-forward pipeline), and provides a concrete implementation path that prioritizes survival and validation over feature expansion. Every claim about edge should be considered provisional until validated through the replay harness with out-of-sample data.*
