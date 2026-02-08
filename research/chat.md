# PolyBawt Deep Research: Edge Reality in Polymarket 15‑Minute Crypto Binaries and an Institutional-Grade Remediation Plan

## Executive summary

**Date context:** This assessment reflects market-structure conditions observable as of **February 8, 2026** (Europe/Stockholm), including the **taker-fee + maker-rebate regime** introduced in early **January 2026** for 15‑minute crypto markets. citeturn12search3turn30search3turn30search1

**Settlement is oracle-defined, not “spot-defined.”** The 15‑minute “Up or Down” contracts resolve using **Chainlink Data Streams** (e.g., BTC/USD stream), comparing the oracle price at the start vs end of the window. A representative BTC market rule set states: resolve to “Up” if the **end-of-range** BTC price is **≥** the **start-of-range** price; otherwise “Down,” and explicitly notes the resolution source is the Chainlink BTC/USD data stream (not other spot sources). citeturn14search0turn15view0  
Implication: any strategy using **entity["company","Binance","crypto exchange"] spot** as *the* “truth” must explicitly quantify and manage **oracle-vs-exchange basis risk**.

**Fees now dominate short-horizon EV.** Polymarket applies taker fees **only** to 15‑minute crypto markets, with a curve that peaks around mid-probability. Public documentation states a **maximum effective rate ~1.56% at $0.50** for 100 shares, and decreases toward zero near 0 or 1. citeturn30search0turn30search1  
Polymarket’s docs also specify fee-curve parameters (fee rate bps and curve constants) and that clients must set fee parameters correctly or orders can be rejected. citeturn30search2turn30search1  
Implication: “taker-first” latency sniping and two-leg taker arbs are **structurally taxed**.

**Competition is explicitly being subsidized.** Polymarket created a **Maker Rebates Program** funded by taker fees and paid daily in USDC, with stated goals of deeper markets and tighter spreads. citeturn30search1turn30search2  
This is a strong signal that the equilibrium is intended to be dominated by professional/automated liquidity providers.

### Honest edge verdict (high-level)

- **ArbTaker (two-leg arb / “synthetic $1”):** *Edge sign:* **fragile / near-zero** net of fees unless (a) you are maker on at least one leg, (b) you enforce atomicity (or near-atomicity) of two-leg completion, and (c) you target fee-favorable probability regimes. Under pure two-leg *taker* execution, the fee curve alone can consume most “< $1” gaps. citeturn30search1turn30search2  
- **LatencySnipe (spot lead-lag taker offense):** *Edge sign:* **negative for non-colocated / non-professional infra** in 15‑minute markets; the strategy is highly likely to be “taxed” by faster participants (queue priority + stale-quote selection). Market-maker incentives intensify adverse-selection pressure on stale quotes. citeturn30search1turn28search10turn18search9  
- **SpreadMaker (passive MM / rebate harvesting):** *Edge sign:* **potentially positive but capacity-limited and execution-quality dominated.** The only strategy with a plausible *durable* edge is **maker-style capture** (spread + rebates), but only if you implement institutional-grade **orderbook sequencing, coalescing, and toxicity gating** and accept that capacity is small. citeturn30search2turn18search9turn28search10  
- **LeggedHedge (crash heuristic + opposite-leg hedge):** *Edge sign:* **negative / model-risky** as described; it is effectively a discretionary/tail bet compressed into an ultra-short digital-like payoff with limited time to mean-revert.

### What must change to be institutional-grade

1. **Edge must be proven via deterministic replay under realistic latency + fee + queue assumptions**, not via paper intuition. Build an event-sourced, deterministic replay harness with an exchange simulator (spec in Technical appendix).  
2. **Unify “truth prices”** around the **same oracle** the market resolves on (Chainlink Data Streams), and treat exchange prices as *features*, not settlement ground truth. citeturn14search0turn17search2turn11search0  
3. **Default posture should become maker-first, taker-selective**, because taker fees are explicitly designed to subsidize makers and penalize taker-heavy churn in 15‑minute crypto markets. citeturn30search3turn30search1  
4. **State consistency and idempotency must be hardened** (order lifecycle, reconciliation invariants, fill handling, restart semantics).  
5. **Toxicity modeling should be revalidated.** VPIN is debated in the literature and can become mechanically linked to volume/intensity; use it as a *feature*, not a gate, unless you validate it in this product. citeturn28search1turn28search2  

## Strategy-by-strategy edge scorecard

**Interpretation note:** Where public market-wide L2 trade/orderbook history is unavailable or inconsistent, I provide (i) *inferential method*, (ii) uncertainty, and (iii) the telemetry required to collapse uncertainty.

| Strategy | What it is (microstructure lens) | Likely EV sign | Edge half-life under competition | “Taxed” by faster participants? | Realistic net edge per trade (ranges) | Key failure modes |
|---|---|---:|---:|---:|---:|---|
| ArbTaker | Two-leg “complete-set” arb: buy Up + Down if cost < 1 (or sell both legs if sum bids > 1), but in a CLOB + fee curve world | **Fragile** | Seconds–minutes | **Yes** | **+$0.00 to +$0.05** on $5 “pair” attempts (often 0 after misses/partials) | Partial fills leaving directional exposure; fee curve makes many “paper arbs” negative; inventory caps bind at worst times citeturn30search2turn30search1 |
| LatencySnipe | Taker-style “stale quote picking” based on lead-lag between exchange mid and Polymarket implied prob | **Negative** | Sub-second | **Yes (strongly)** | **–$0.10 to +$0.05** per $5 attempt; mean negative once fees + adverse selection included | You only get filled when you’re wrong (classic adverse selection), plus taker fees peak near 0.5; cancel/replace races lost citeturn30search1turn28search10turn18search9 |
| SpreadMaker | Provide passive two-sided quotes; monetize spread + maker rebates; manage inventory/toxicity | **Potentially positive** (capacity-limited) | Hours–weeks (if continuously improved) | **Yes, but survivable** if you avoid being stale | **–$0.02 to +$0.08** per filled $5-equivalent notional (high variance by regime); can be positive if fill quality is good | Queue-position decay; toxic fills during jumps; inventory blowups near boundary; inadequate sequencing/local book causes stale quotes citeturn30search2turn28search10turn18search9 |
| LeggedHedge | Crash trigger + hedge opposite leg (effectively a short-horizon tail/mean-reversion bet) | **Negative** | N/A | **Yes** | **–$0.15 to +$0.05** per event when it triggers; mostly negative after costs | Model risk dominates; triggers enter after move; hedging logic can crystallize losses; limited time to realize hedge benefits |

### Fee reality check (why many “edges” disappear)

Polymarket’s published fee curve implies taker costs are **highest near p≈0.5** (the “most uncertain” state). citeturn30search1turn30search2  
At p=0.50, the help-center fee table shows **$0.78 fee per 100 shares**, i.e. **$0.0078 per share** (~0.78¢). citeturn30search0  
For a two-leg complete-set taker arb (Up+Down), that’s ~**1.56¢ per “paired share”** just in fees near mid—before spreads, slippage, partial fills, and opportunity cost. Therefore, a “YES+NO < 0.98” trigger can easily be **net ~0** unless you get unusually favorable prices and fills. citeturn30search0turn30search2

## Graded subsystem table

**Scoring rubric:** 1–3 hobby-grade, 4–6 competent retail/systematic prototype, 7–8 semi-institutional (robust eval + ops + controls), 9–10 institutional (replay + invariants + SLOs + incident response + change control).

| Subsystem | Grade (1–10) | Why (evidence-based) | What blocks “institutional” today |
|---|---:|---|---|
| Strategy logic | 6 | Ensemble decomposition is sensible (arb / latency / maker / hedge), and risk caps are explicit (per your spec). The product’s fee regime makes several legs structurally hard. citeturn30search3turn30search1 | Lacks demonstrated robustness under fee curve + adversarial selection + queue priority; weak “oracle basis” integration for alpha integrity |
| Alpha integrity | 4 | Alpha depends on very short-horizon microstructure; Polymarket explicitly incentivizes professional market making, compressing edge for slow takers. citeturn30search1turn28search10 | No deterministic replay proving edge; no measured lag/decay curves; no clean separation of signal vs execution drag |
| Risk controls | 6 | Circuit breakers align with microstructure reality; but toxicity measures (e.g., VPIN) are academically contested and must be validated per market. citeturn28search1turn28search2 | Need rigorous “risk invariants,” kill-switch semantics, and resolution/oracle staleness modeling tied to actual settlement rules citeturn14search0turn17search2 |
| Execution reliability | 5 | Basic async stack is plausible; however, Polymarket recommends maintaining a local book and tracking sequences; stale-book execution is existential for makers. citeturn18search9 | Missing (or unproven) snapshot+delta sequencing, coalescing, idempotent retries, and latency SLO enforcement |
| Reconciliation/state consistency | 5 | Periodic reconciliation is necessary; Polymarket has multiple APIs (CLOB/Gamma/Data) and onchain resolution paths. citeturn18search1turn18search9 | Need formal order/position state machines, hard invariants, and crash-safe event-sourced persistence |
| Observability | 6 | Metrics/health endpoints are directionally correct; Polymarket has had API component incidents (e.g., price-history) so vendor monitoring matters. citeturn33search7 | Need “golden signals,” latency histograms, and alertable SLOs; also need replay-grade structured logs |
| Testing maturity | 3 | Public endpoints exist for prices/books/history, but even official endpoints have had outages and inconsistent returns per community reports; testing must be simulation/replay-based. citeturn33search0turn33search7turn33reddit31 | No deterministic exchange simulator + replay harness (must-have) |
| Production readiness | 5 | Hard risk caps and “fail-closed” intent are good; market structure is adversarial and speed-sensitive. citeturn30search1turn28search10 | Must implement change control, incident playbooks, dependency fallbacks, and live shadow-mode validations |

## Critical findings

**Ranked by severity × likelihood, with remediation implications.**

1. **Oracle-defined settlement makes “exchange‑spot truth” an alpha integrity trap.** These markets resolve on Chainlink Data Streams; trading as if Binance is the settlement truth creates systematic basis risk and false positives, especially around boundary conditions and when chainlink vs exchange diverges. citeturn14search0turn17search2

2. **Taker fee curve (Jan 2026) structurally taxes your two highest-churn legs (LatencySnipe, taker-mode ArbTaker).** Taker fees peak near 0.5 probability and are only enabled on 15‑minute crypto markets; this directly penalizes the exact regime where lag-arb and frequent flipping tend to occur. citeturn12search3turn30search3turn30search0

3. **Maker rebates are an explicit invitation for professional market makers to compress spreads and punish stale liquidity.** If you are maker and slow, you become a transfer function paying alpha to faster takers. The docs explicitly recommend local orderbook maintenance with sequence tracking—i.e., this is not optional for maker-grade execution. citeturn30search1turn18search9turn28search10

4. **“Edge” is currently unproven in a scientifically defensible way.** Polymarket provides endpoints for orderbooks, price history, and market metadata, but public history endpoints can be unreliable in practice (documented incidents and community reports). This makes *your* own capture + deterministic replay the only credible validation path. citeturn19search3turn33search0turn33search7turn33reddit31

5. **VPIN-as-circuit-breaker is risky without market-specific calibration.** VPIN’s predictive value is contested; it can be mechanically linked to volume/intensity and not reliably predictive of “toxicity” depending on implementation. Use it as a feature, not a gate, until validated on your own data for this product. citeturn28search1turn28search2

6. **Queue position and cancel/replace races dominate maker PnL, and Python async needs explicit backpressure + coalescing policies.** In price-time priority markets, queue position affects fill probability and adverse selection and can “crowd out” later liquidity providers. If your event loop lags, your quotes turn stale and get picked off. citeturn28search10turn18search9

7. **Third-party data sources exist but must be treated as non-authoritative and audited.** Providers advertise historical orderbook snapshots and normalized data, but they introduce vendor risk, schema drift, and survivorship bias. Prefer self-collection as the system of record. citeturn33search4turn31search6

## Quantitative scenarios

This section gives **realistic ranges**, with **explicit assumptions** and **confidence bands**. All numbers are **net of taker fees** where applicable, but *not* net of external onramp/offramp costs (irrelevant for intra-day bot EV). citeturn30search3turn30search0

### Core assumptions (transparent)

- You cap max trade at $5 and max total exposure at $20 (per your constraints).  
- Your fills are a mix of maker and taker:
  - ArbTaker: often taker on both legs unless redesigned.
  - SpreadMaker: predominantly maker, sometimes taker defensively to flatten.
  - LatencySnipe: mostly taker (by design).
- **Fee curve applies to taker trades** in 15‑minute markets; max effective rate around **1.56%** at ~50c, declining toward extremes. citeturn30search0turn30search2
- “Sharpe-like” is estimated on *daily* PnL series; high uncertainty because edge is not yet proven via replay.

### Scenario set

I provide three scenarios:

- **S0 (Taxed / no true edge):** Expected value ≤ 0 after costs; bot behaves like a liquidity donor.
- **S1 (Maker-first micro-edge):** SpreadMaker dominates; Arb only when “fee-corrected” and near-atomic; minimal taker.
- **S2 (Strong execution + validated signals):** You achieve measurable lead-lag alpha decay capture and/or superior maker rebates without toxic fills (hard).

#### Expected net edge per $5 trade attempt (all strategies blended)

- **S0:** –$0.03 to –$0.10  
- **S1:** –$0.01 to +$0.05  
- **S2:** +$0.03 to +$0.12  

**Confidence:** low-to-medium, because public market-wide fill-quality distributions are not available; this must be measured from your telemetry + replay. citeturn18search9turn33search4

### Bankroll outcomes (with your hard exposure cap)

Because exposure is capped at $20, **returns do not scale linearly with bankroll** once bankroll > $20–$40. Your bankroll mostly affects your ability to survive variance, not to deploy more risk.

Assume 50–250 trade attempts/day across all assets (maker quote updates and occasional taker actions), but only a subset become fills.

| Bankroll | Scenario | Expected net $/day | 90% band ($/day) | Expected %/month (simple) | “Sharpe-like” daily (rough) |
|---:|---|---:|---:|---:|---:|
| $100 | S0 | –$0.50 | [–$2.00, +$0.20] | –15% | –0.3 to 0.0 |
| $100 | S1 | +$0.40 | [–$0.80, +$2.00] | +12% | 0.1 to 0.6 |
| $100 | S2 | +$1.50 | [–$0.50, +$5.00] | +45% | 0.4 to 1.2 |
| $250 | S0 | –$0.50 | [–$2.00, +$0.20] | –6% | –0.3 to 0.0 |
| $250 | S1 | +$0.50 | [–$0.80, +$2.20] | +6% | 0.1 to 0.6 |
| $250 | S2 | +$1.80 | [–$0.50, +$5.50] | +22% | 0.4 to 1.2 |
| $500 | S0 | –$0.50 | [–$2.00, +$0.20] | –3% | –0.3 to 0.0 |
| $500 | S1 | +$0.50 | [–$0.80, +$2.20] | +3% | 0.1 to 0.6 |
| $500 | S2 | +$1.80 | [–$0.50, +$5.50] | +11% | 0.4 to 1.2 |

**How to interpret:**  
- These distributions are **exposure-capped** and assume you do not scale max exposure with bankroll.  
- The large upside in S2 reflects “what’s possible” only if you become *execution-grade* and avoid toxic fills; public docs strongly suggest the platform is engineered to reward this tier. citeturn30search1turn18search9turn28search10

### What telemetry collapses the uncertainty fastest

To replace S0/S1/S2 with real distributions, you need to measure:

1. **Realized spread vs effective spread** on maker fills (classic MM performance decomposition).  
2. **Adverse selection**: mid/microprice move after your fill (e.g., +250ms/+1s/+5s), per token, per regime.  
3. **Lag-alpha decay**: probability update lag against the oracle feed (Chainlink stream) and exchange feed, and how quickly it mean-reverts. citeturn17search2turn18search9  

## Phased implementation roadmap

Targets: **≥8 in 4–6 weeks**, **≥9 in 8–12 weeks**, while preserving safety over throughput.

### Phase 1: Reach “8/10” robustness in 4–6 weeks

**Theme:** Turn a competent prototype into a replay-validated, state-correct, maker-first system.

| Task | Why it matters | Exact technical approach | EV impact | Risk impact | Effort | Validation & exit criteria |
|---|---|---|---:|---:|---:|---|
| Deterministic replay + simulator (MVP) | Without this, “edge” is belief-based | Implement event-sourced NDJSON log → deterministic replayer → exchange simulator with fee curve + price/time priority + post-only behavior; see appendix spec | High | High | High | Replay matches live fills for a canary day within tolerance; deterministic hash of run outputs |
| Local orderbook with sequencing | Maker execution dies if you quote on stale books | Maintain local book via CLOB WebSocket deltas + sequence tracking; snapshot recovery; coalescing on lag citeturn18search9 | High | High | Med | Inject dropped messages → system self-heals and pauses quoting; invariant: no negative depth; spread non-negative |
| Fee-corrected arb gating | “YES+NO < 1” is not enough in a fee-curve market | Replace “0.98” with fee-aware threshold: require (1 − (pU+pD)) ≥ fees(pU)+fees(pD)+slippage_budget; use published curve parameters citeturn30search2turn30search0 | Med | Med | Low | Backtest shows positive net realized for arb attempts; no partial-fill blowups |
| Maker-first posture by default | Taker in 15m markets is structurally taxed | Convert LatencySnipe into “maker leaning” (skew quotes, don’t cross) except in extreme dislocations; enforce post-only where applicable citeturn30search3turn12search3 | High | High | Med | Taker percentage < X% (target 10–20%); reduced fee spend per $ of volume |
| Risk invariants as code | Safety must be provable | Implement invariants: exposure caps, max open orders, max outstanding notional, per-token caps, daily loss stop; fail-closed | Low–Med | High | Med | Property tests: invariant violations impossible under random event orderings |
| Replace “VPIN gate” with “toxicity panel” | VPIN alone is not reliable without calibration | Log VPIN + alternatives (imbalance, microprice drift, realized spread, fill-to-move); do not hard-block unless validated citeturn28search1turn29search1 | Med | Med | Med | Offline analysis shows which signals predict adverse selection; only then promote to breaker |
| Oracle-truth unification | Settlement depends on Chainlink streams | All “fair value” should reference Chainlink stream; exchange feed used as feature/hedge; explicitly model basis citeturn14search0turn17search2 | High | High | Med | Post-trade: PnL attribution includes oracle basis term; fewer “wins” that lose at settlement |

### Phase 2: Reach “9/10” institutional behavior in 8–12 weeks

**Theme:** SLO-driven ops + full validation + controlled change process.

| Task | Why | Approach | EV impact | Risk impact | Effort | Exit criteria |
|---|---|---:|---:|---:|---:|---|
| Latency budgets + SLO alerts | You can’t manage what you don’t measure | Define SLOs for feed staleness, order ack, fill detection; histogram metrics; alert routing; see appendix citeturn18search9turn33search7 | Med | High | Med | SLO dashboards exist; paging thresholds tested with fault injection |
| Shadow-mode production validation | Prevent live regressions | Run strategies in shadow against live data; compare simulated fills to real book; promote only if drift small | High | High | High | Weekly release only if shadow KPIs pass |
| Strategy retirement + capital allocation | Concentrate on durable edge | Remove/disable negative-EV legs (likely LatencySnipe taker and LeggedHedge) unless proven in replay | High | High | Low | Strategy PnL attribution shows positive net after costs |
| Vendor / dependency controls | Polymarket endpoints can degrade | Build fallbacks for price-history, gamma outages, websocket disconnects; circuit-breakers on stale data citeturn33search7turn18search1 | Low | High | Med | Chaos testing: simulated outages produce safe halt and graceful recover |

## Technical appendix

### Deterministic replay/backtest harness with exchange simulator (spec)

**Goal:** Given NDJSON event logs from your live bot, reproduce decisions and PnL deterministically, under configurable latency, fee, and queue assumptions.

**Core components**

1. **EventLog (NDJSON)**
   - Append-only, strictly ordered by `ts_recv_ns` (monotonic) for replay determinism.
   - Separate “source timestamp” vs “receive timestamp.”

2. **Replayer**
   - Reads EventLog, emits events into a single-threaded asyncio loop.
   - Supports `speed=1.0` (real time) and `speed=∞` (as-fast-as-possible).
   - Can inject synthetic delays: `ack_delay_ms`, `wss_jitter_ms`, `snapshot_gap_prob`.

3. **ExchangeSimulator**
   - Maintains per-token limit order book with **price-time priority**.
   - Enforces:
     - `tick_size`, `min_order_size` (from /book) citeturn19search3
     - post-only rejection behavior (per Polymarket feature set) citeturn12search3
     - taker fees using fee curve parameters (fee_rate_bps, exponent) citeturn30search2turn30search1
   - Emits fills with `fill_id`, `match_time`, `fee_paid`.

4. **PnL Engine**
   - Marks inventory to:
     - current midpoint (for intraday risk), and
     - settlement outcome when available.
   - Tracks:
     - realized spread
     - mark-to-market
     - fee spend vs rebate receipts (if modeled).

**Exit criteria**
- Same input EventLog + same simulator config ⇒ identical output hash every run.
- When replaying in “wire-tap mode” (using observed exchange fills), simulator slippage distribution matches empirical within tolerance bands.

### Event schema + idempotency keys (concrete)

**Canonical envelope (NDJSON per line):**
```json
{
  "event_id": "sha256:…",
  "ts_recv_ns": 0,
  "ts_source_ns": 0,
  "source": "clob_wss|clob_rest|gamma|rtds_chainlink|rtds_binance|strategy|risk|exec",
  "type": "BOOK_DELTA|BOOK_SNAPSHOT|ORACLE_TICK|SIGNAL|ORDER_SUBMIT|ORDER_ACK|FILL|CANCEL_ACK|RESOLUTION|HEALTH",
  "key": {
    "condition_id": "...",
    "token_id": "...",
    "client_order_id": "..."
  },
  "payload": {}
}
```

**Idempotency rules**
- `event_id` = hash of (`source`, `type`, `ts_source_ns`, stable payload fields).
- `client_order_id` must be globally unique and deterministic across retries:
  - e.g. `COID:{run_id}:{strategy}:{token_id}:{intent_ts}:{nonce}`  
- Every order intent has an `intent_id`; all retries reference the same `intent_id`.

### Backpressure policy for event queue saturation

**Problem:** In maker regimes, you can receive book deltas faster than you can safely process, causing stale quotes and toxic fills.

**Policy (must-do)**
- Maintain two queues:
  1. **Critical queue (never drop):** fills, acks, cancels, kill-switch events, resolution events.
  2. **Market-data queue (droppable/coalescible):** book deltas, mid updates.

**Coalescing rule**
- If market-data queue length > `Q_HI`:
  - For each `token_id`, keep only the latest delta/snapshot; drop intermediate deltas.
  - Emit a `DATA_LAG` event; set `quoting_disabled[token_id]=True` for `cooldown_ms`.
- If lag persists > `T_LAG_MAX`, trigger global safety halt (new orders disabled; cancel outstanding).

This aligns with Polymarket’s explicit recommendation to maintain a local orderbook and handle sequencing; without it, maker fills become dominated by adverse selection. citeturn18search9turn28search10

### Latency budget and SLOs

Because 15‑minute markets are latency-sensitive and competitive, set SLOs around **staleness** and **time-to-cancel** (not just time-to-place).

**Proposed SLOs (initial, tighten after measurement)**

- **Oracle staleness (Chainlink stream):**
  - SLO: `now - ts_oracle <= 250ms` p50, `<= 900ms` p99 while quoting
  - If violated: disable quoting for affected token  
  Rationale: Chainlink Data Streams are designed for sub-second reporting and low latency. citeturn17search2turn11search0

- **CLOB book freshness (local book):**
  - SLO: `now - last_seq_applied_time <= 250ms` p50, `<= 1.5s` p99
  - If violated: cancel quotes, re-snapshot, cooldown

- **Order cancel-to-ack (risk-off latency):**
  - SLO: p50 <= 300ms, p99 <= 1500ms
  - Alert if p99 exceeds 2s for 5 minutes

- **Fill detection latency (exchange→bot):**
  - SLO: p50 <= 150ms, p99 <= 700ms

### Data source catalog (best-available, with caveats)

**Official / primary**
- **CLOB API** (prices, orderbooks, trading): base URL published by Polymarket docs. citeturn18search7turn19search3  
- **CLOB WebSocket** (real-time book + user updates): published endpoint. citeturn18search7turn18search9  
- **Gamma API** (market/event metadata, resolution source fields): published base URL and filters. citeturn18search1turn18search8  
- **Price history** endpoint exists (`/prices-history`) but has documented incidents and occasional empty responses in community reports—treat as “best-effort,” not a system of record. citeturn33search0turn33search7turn33reddit31  
- **RTDS** (real-time crypto prices and comments): published WebSocket URL; changelog indicates crypto prices are available from **two sources: Binance & Chainlink**. citeturn12search0turn12search3  

**Settlement / oracle**
- **Chainlink Data Streams**: official docs describe pull-based, low-latency, sub-second oracle reports (the resolution source for these markets). citeturn17search2turn14search0turn15view0  

**Third-party / commercial (use as supplemental, not authoritative)**
- **Orderbook history providers** can supply snapshots, but introduce vendor risk and must be validated against your own live capture. Example: Dome API advertises Polymarket orderbook history back to Oct 2025. citeturn33search4  
- **Aggregators/unified APIs** (e.g., Predexon) can accelerate research, but you must audit: timestamp fidelity, missingness, survivorship bias, and alignment to settlement oracle. citeturn31search6  

## Top 10 immediate actions

1. **Disable LatencySnipe taker mode by default** until you can prove (via replay) positive net EV after taker fees and stale-quote selection. citeturn30search0turn28search10  
2. **Replace the ArbTaker trigger with a fee-corrected threshold** derived from the published fee curve parameters (do not use “0.98” as a static constant). citeturn30search2turn30search0  
3. **Implement local orderbook maintenance with sequence validation and snapshot recovery** (maker survival requirement). citeturn18search9  
4. **Add strict backpressure + coalescing**: never drop fills/acks; coalesce book deltas by token_id under load; enter “risk-off cooldown” on lag.  
5. **Make Chainlink stream the canonical “truth” for alpha and settlement modeling**; treat exchange feeds as features and hedges. citeturn14search0turn17search2turn11search0  
6. **Build the deterministic replay + simulator MVP** (even if simplified): without it, you cannot honestly claim edge.  
7. **Instrument fill-quality metrics** (realized spread, adverse selection after 250ms/1s/5s, fee spend, maker rebate estimates) and enforce retention ≥ 90 days. citeturn30search1turn29search1  
8. **Demote VPIN from a hard breaker to a logged feature** until calibrated on this product; include alternative toxicity signals (imbalance, microprice drift). citeturn28search1turn29search1  
9. **Define SLOs and alerts for “risk-off latency” (cancel-to-ack) and feed staleness**, because vendor incidents happen (e.g., price-history component outages). citeturn33search7turn18search9  
10. **Run a two-week shadow-mode campaign**: same code path, no live orders, but full logging + simulated execution, then produce a post-trade report with PnL attribution by strategy and regime.

