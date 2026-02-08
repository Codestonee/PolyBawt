# PolyBawt Deep Research Report: Edge Assessment & Institutional Roadmap
**Principal Quantitative Research Analysis**  
**Date:** February 8, 2026  
**Repository:** https://github.com/Codestonee/PolyBawt.git  
**Analyst Role:** Principal Quantitative Researcher, Market Microstructure Specialist, Trading Systems Architect

---

## 1. Executive summary (one page max)

### Current state
PolyBawt is a Python 3.11+ asyncio, event‑driven execution stack for Polymarket’s CLOB 15‑minute crypto binaries. Architecturally, it is unusually mature for a solo project: clear separation of ingestion, strategy, risk, execution, portfolio, and infra; structured logging; metrics; and risk gates (Kelly sizing, VPIN, circuit breakers) are all present in the codebase.

However, **the core alpha assumptions are now misaligned with current Polymarket market structure**:

- Since late 2025–Jan 2026, Polymarket has:
  - Introduced **dynamic taker fees** explicitly aimed at suppressing latency arbitrage and short‑horizon mispricing capture in 15‑minute crypto markets.[cite:4][cite:5]
  - Tightened spreads and improved depth via maker rebates and professional MM programs.[cite:11][cite:15]
  - Hardened Chainlink‑based settlement for 15‑minute crypto markets, compressing outcome information arrival to seconds around expiry.[cite:16][cite:21]
- At the same time, HFT / professional bots now dominate short‑horizon opportunities on Polymarket.[cite:23][cite:28][cite:31]

Given this environment and the actual code in `src/strategy/`, the honest assessment is:

- **ArbTaker** (short/long rebalancing arb on YES+NO != 1):
  - Uses static fee assumptions that are no longer valid given Polymarket’s dynamic fee curve.
  - Under realistic fees, most “arb” trades are **negative EV**; the strategy is being directly “taxed” by the new fee model.
  - **Verdict:** Systematically negative edge; should be disabled before any live capital.

- **LatencySnipe** (spot–Polymarket lead–lag):
  - Uses a 120s lookback, while competitive bots operate on 10–200 ms horizons.[cite:23][cite:28]
  - By the time the strategy fires, Polymarket has usually already adjusted; remaining fills are likely adverse selection.
  - **Verdict:** Edge is at best fragile and most likely negative net of fees + slippage.

- **SpreadMaker** (passive MM on CLOB):
  - Conceptually aligned with how Polymarket rewards makers (rebates + tighter spreads).[cite:11][cite:12]
  - Implementation currently uses conservative, wide spreads and does not integrate the existing VPIN toxicity module, leaving it exposed in toxic regimes.
  - **Verdict:** Only component with realistic, durable edge potential if made toxicity‑aware and backtested.

- **LeggedHedge** (crash‑catch + later hedge):
  - Uses heuristic crash rules (>15% drop) without volatility normalization or robust statistical backing.
  - Leg lifecycle can leave the bot long the “falling knife” without a reliable hedge.
  - **Verdict:** High model risk, low evidence of edge; not suitable for institutional deployment in current form.

At the system level, engineering quality is **4/10 overall** with potential to reach **8/10 in 4–6 weeks** and **9/10 in 8–12 weeks**, assuming focus on:

- Decommissioning negative‑EV strategies (ArbTaker, current LatencySnipe, LeggedHedge).
- Making SpreadMaker toxicity‑aware (VPIN integration), fee‑accurate, and fully backtested.
- Building a deterministic replay/backtest harness with an exchange simulator and comprehensive telemetry.

### High‑level outcomes & expectations
Under **current code and current market microstructure**, realistic expectations are:

- **Per‑trade edge (net):**
  - ArbTaker: −0.5% to −1.5% (negative).
  - LatencySnipe: −0.3% to +0.2% (fragile; likely negative).
  - SpreadMaker: +0.1% to +0.8% (if tuned and toxicity‑aware).
  - LeggedHedge: −0.5% to +0.3% (dominated by model risk).
- **Bankroll trajectories (current mix, $5 max trade / $20 exposure limits):**
  - $100: roughly −2% to −8% per month expected.
  - $250: roughly −2% to −8% per month.
  - $500: roughly −2% to −8% per month.
- **Sharpe‑like profile:**
  - Current mix: 0.0 to 0.3 (sub‑institutional; noise‑dominated).
  - SpreadMaker‑only after remediation: 0.8–1.3 plausible (small‑capital MM), if backtests and live telemetry agree.

The actionable plan is:

1. **Immediately disable ArbTaker and LeggedHedge.**
2. **Treat LatencySnipe as experimental** and either radically shorten horizons or retire it.
3. **Elevate SpreadMaker to primary alpha source**, add toxicity gating and proper fee modeling.
4. **Invest heavily in backtesting and telemetry** rather than new strategies for the next 4–6 weeks.

---

## 2. Strategy‑by‑strategy edge scorecard

Scores: −2 (strongly negative EV), −1 (likely negative), 0 (uncertain/fragile), +1 (mildly positive), +2 (strong, likely durable).

### 2.1 ArbTaker

**Description:**
- Looks for markets where YES_ask + NO_ask < long threshold or YES_bid + NO_bid > short threshold, then buys/sells both legs.
- Uses fee‑adjusted net profit check and basic liquidity checks.

**Implementation anchors:**
- `ArbTakerStrategy` in `src/strategy/strategies.py`.
- `ArbitrageDetector` in `src/strategy/arbitrage_detector.py`.[cite:19][cite:20]

**Edge analysis:**

- **Fee model mismatch:**
  - Code assumes a flat ~2% per trade, 4% round trip.[cite:19]
  - Actual dynamic taker fees peak near 50% probability; effective fee for typical short‑dated crypto binaries is ≈1.5–1.6% per leg (≈3.1% round trip) at the exact region where YES+NO mispricings are most common.[cite:4][cite:5][cite:7]
  - Many apparent 2–3% “gross” arbs are actually **non‑profitable after real fees**.

- **Slippage & competition:**
  - HFT bots and professional MMs aggressively arbitrage YES+NO deviations; observed mispricings are typically <1% and live only for milliseconds–seconds.[cite:23][cite:28][cite:31]
  - With Python asyncio + public WS + REST, PolyBawt is realistically in the 100–500 ms latency band (best case), far behind co‑located infra.

- **Inventory & exposure caps:**
  - Max $5 per leg and $20 total exposure limit absolute PnL, which is good for safety but means even rare positive arbs contribute little.

**Verdict:**
- **Likely EV sign:** Negative.
- **Edge half‑life:** Already expired post‑fee‑change; remaining edge is illusory/fee‑taxed.
- **Taxed by faster participants?** Yes—prof bots capture true arbs; remaining “arbs” are effectively traps created by dynamic fees and slippage.

**Score:** **−2 (strongly negative EV under current conditions).**

---

### 2.2 LatencySnipe

**Description:**
- Watches Binance WS spot feed; if spot moves >2% in 2 minutes and Polymarket prices lag, it “snipes” the old YES/NO prices on 15‑minute markets.
- Window: 120 s; trigger: |Δspot| >2%; expectation that 15‑minute market should have repriced.

**Implementation anchors:**
- `LatencySnipeStrategy` in `src/strategy/strategies.py`.[cite:20]

**Edge analysis:**

- **Horizon mismatch:**
  - Professional crypto/derivatives bots act on 10–200 ms horizons.[cite:22][cite:23] Your 120 s window puts you literally **three orders of magnitude slower**.
  - By the time your condition is met, Polymarket’s YES/NO book has almost always been updated by faster participants.

- **Signal definition:**
  - Triggering on 2% spot moves over 2 minutes is coarse; many such moves are already fully priced into the Polymarket 15‑minute binary.
  - The mapping of spot move → binary fair price change is simplistic (YES ~ half of spot move), and not empirically calibrated.

- **Adverse selection:**
  - You end up trading into an already repriced book, where informed flows have either taken the good side or widened spreads.
  - Without slippage accounting and PnL attribution, adverse selection is invisible but likely material.

**Verdict:**
- **Likely EV sign:** Slightly negative.
- **Half‑life:** Short—once any measurable edge appears, faster bots will compress it further.
- **Taxed by faster participants?** Yes; you act as a slow taker in a highly informed environment.

**Score:** **−1 (likely negative; at best fragile and fee‑sensitive).**

---

### 2.3 SpreadMaker

**Description:**
- Passive MM quoting on YES side with cancel‑before‑quote lifecycle, min spread, inventory skew, rate‑limited requotes.

**Implementation anchors:**
- `SpreadMakerStrategy` in `src/strategy/strategies.py`.[cite:20]
- VPIN implementation in `src/risk/vpin.py` (currently unused by strategy).[cite:34]

**Edge analysis:**

- **Alignment with Polymarket economics:**
  - Polymarket’s shift to taker fees and maker rebates makes well‑behaved passive liquidity an actual source of returns.[cite:11][cite:12][cite:15]
  - Reported MM PnLs on Polymarket show sustainable returns for liquidity always present within tight spreads, with rebates materially compensating for adverse selection.[cite:12][cite:18]

- **Current quoting parameters:**
  - `min_spread = 0.05` (5 cents) and `quote_offset = 0.01` are very conservative.[cite:20]
  - With many 15‑minute markets now trading at 1–2 cent effective spreads[cite:5][cite:6], you are often not at the top of book and may rarely get filled unless the market is wide.

- **Toxicity handling:**
  - You’ve implemented VPIN but do not connect it to quoting behavior, so you do not widen spreads or reduce size in toxic regimes.[cite:27][cite:34]
  - This leaves you vulnerable when informed traders dominate short‑term flow.

- **Inventory control:**
  - Basic skewing exists and you cap per‑side size at $5 with 15 s requote cadence; this is conservative but good for a small bankroll.

**Verdict:**
- **Likely EV sign:** Mildly positive **if**:
  - Quotes are narrowed to compete for flow.
  - VPIN and microprice/mid‑price dynamics are used to avoid quoting aggressively in toxic periods.
  - Maker rebates are correctly modeled and included in PnL.
- **Half‑life:** Medium (months), since MM is structurally rewarded under the new dynamics; edge will erode only gradually as MMs compete on tighter spreads/latency.
- **Taxed by faster participants?** Somewhat—but as a maker, you are being compensated via spread + rebates. The key is to not quote too aggressively in high‑toxicity regimes.

**Score:** **+1 (real but modest, improvable edge potential).**

---

### 2.4 LeggedHedge

**Description:**
- “Crash & catch” mechanism:
  - Leg 1: Buy the “crashed” side on a >15% book drop.
  - Leg 2: Hedge with the opposite side when some sum‑of‑legs condition is met (<0.95 in spec).

**Implementation anchors:**
- `LeggedHedgeStrategy` and `LegState`/`LegContext` FSM in `src/strategy/strategies.py`.[cite:20]

**Edge analysis:**

- **Heuristic, not model‑driven:**
  - No volatility normalization (15% drop in a calm BTC regime is not the same as 15% during an event, nor is it comparable across assets).
  - No explicit mapping to fair value or expected outcome probability.

- **Execution fragility:**
  - Leg 1 can fill into a structural crash (e.g., oracle or exchange event), leaving you long the wrong side with no guarantee Leg 2 hedge will be executed at a sane price.
  - No explicit timeout or abandonment logic if Leg 2 cannot be executed under safe conditions.

- **Complexity vs benefit:**
  - For $5 notional per leg, the complexity and potential tail risk are hard to justify.

**Verdict:**
- **Likely EV sign:** Slightly negative or at best very fragile.
- **Half‑life:** N/A; no clear sustainable alpha story.
- **Taxed by faster participants?** Indirectly—your Leg 1 fills occur when others may be offloading risk or exiting; you’re catching their dump without a robust model.

**Score:** **−1 (unproven & risky; should be disabled for now).**

---

## 3. Graded subsystem table (1–10)

| Subsystem                         | Score | Rationale |
|-----------------------------------|:-----:|-----------|
| **Strategy logic**                | 3/10  | Three of four strategies are either mis‑specified (fees), structurally too slow, or heuristic. SpreadMaker is the only fundamentally aligned one. |
| **Alpha integrity**               | 2/10  | No full‑history backtests; edge assumptions pre‑date recent fee + competition regime; no out‑of‑sample validation. |
| **Risk controls**                 | 6/10  | Good ingredients (Kelly, VPIN, circuit breakers, exposure caps) but partial wiring; some critical gates unused (VPIN in quoting). |
| **Execution reliability**         | 5/10  | Async design, CLOB client, reconciler, rate limiting exist, but no explicit latency SLOs, no slippage monitoring, no queue backpressure policy. |
| **Reconciliation/state consistency** | 5/10 | Portfolio tracking and persistence exist but lack idempotency keys and deterministic event replay; recovery from partial failures is underspecified. |
| **Observability**                 | 6/10  | Structured logging + metrics are present, but no standard PnL attribution by strategy, fill‑quality metrics, or toxicity vs PnL correlation dashboards. |
| **Testing maturity**              | 3/10  | Tests folder present but no full exchange simulator or deterministic replay harness; most logic is only “tested in production” in spirit. |
| **Production readiness**          | 4/10  | Health endpoints and some circuit breakers exist, but no full alerting matrix, no runbooks, no chaos or restart‑recovery tests. |

Overall: **4/10** today with a clear path to **8/10 in 4–6 weeks** and **9/10 in 8–12 weeks** if roadmap is executed.

---

## 4. Critical findings (ranked, with rationale)

### 4.1 P0 – Strategy/alpha correctness

1. **ArbTaker fee model is structurally wrong under 2026 Polymarket fees.**  
   - **Why it matters:** Mispricing YES+NO sums by ignoring the nonlinear fee curve means you will systematically take what appear to be 2–4% arbitrage opportunities that are actually **fee‑taxed into negative EV**.[cite:4][cite:5][cite:7]
   - **Impact:** Every such “arb” trade loses around 0.5–1.5% expected, compounded by your small bankroll.
   - **Fix:** Implement a dynamic fee model consistent with official docs and/or Polymarket’s own examples, and feed it everywhere EV is computed.

2. **LatencySnipe operates far outside the microstructure edge window.**  
   - **Why it matters:** Microsecond–millisecond scale bots capture the true lead‑lag edge; your 120 s window is effectively a slow, noisy momentum bet in a highly competitive marketplace.[cite:23][cite:28]
   - **Impact:** You buy/sell after the move has been priced in. Fills are more likely to be on the wrong side of microprice dynamics.
   - **Fix:** Either radically shorten horizon and accept you are still second‑tier even with async Python, or repurpose the idea into slower, mean‑reversion or correlation‑based models.

3. **LeggedHedge is heuristic, not statistically justified.**  
   - **Why it matters:** Crash‑catch strategies can carry substantial left‑tail risk; doing them without robust calibration and monitoring is dangerous.
   - **Impact:** Although notional per trade is capped, repeated “catching falling knives” can bleed the account and create psychologically hard drawdowns.
   - **Fix:** Disable until you have a historical regime analysis showing clear reversion behavior after similar drops.

### 4.2 P1 – Market microstructure & competition

4. **Market structure changes in late 2025/early 2026 directly target your original alpha hypotheses.**  
   - **Why it matters:** Dynamic taker fees + maker rebates + increased HFT presence turn simple latency/arb into a losing or neutral game.
   - **Impact:** Strategies designed for a zero/flat‑fee, low‑competition landscape no longer apply; continuing to run them is fighting the last war.

5. **You are systematically slower than the dominant participants.**  
   - **Why it matters:** For anything relying on being first to mispricings, you are guaranteed to be second or third; best case, you end up warehousing risk the HFTs did not want.
   - **Impact:** Persistent adverse selection and slippage.

### 4.3 P1 – Execution/ops

6. **No idempotency / replay semantics on events.**  
   - **Why it matters:** Network glitches, retries, or restarts can cause duplicate order submissions or state divergence.
   - **Impact:** Positions may not match intended risk; can breach your exposure caps silently if state tracking fails.

7. **No systematic latency and slippage monitoring.**  
   - **Why it matters:** Without measuring (p50/p95/p99) order ack latencies and slippage, you can’t tell if you’re being picked off or if infra changes help.
   - **Impact:** Undetected degradation of execution quality.

### 4.4 P2 – Risk, infra, dependencies

8. **VPIN and toxicity framework present but unused in real decisions.**  
   - **Why it matters:** This is exactly the tool you need to avoid being a sitting duck as passive liquidity.
   - **Impact:** You run SpreadMaker with blindfolds on regarding informed flow concentration.

9. **Polymarket + Chainlink dependency surface not fully modeled.**  
   - **Why it matters:** Chainlink/Data Streams outages, stale oracles, and CLOB downtime create edge cases where you may trade on stale or invalid prices.[cite:21][cite:24][cite:29]
   - **Impact:** Occasional but potentially expensive “black swan” trades.

10. **Regulatory/platform risk not tracked.**  
    - **Why it matters:** Polymarket’s terms, geo‑restrictions and regulatory posture have evolved; future constraints may impact bot operation.[cite:12][cite:32]
    - **Impact:** Operational sudden‑stop risk rather than PnL risk.

---

## 5. Quantitative scenarios (bankroll outcomes)

All numbers below are **approximate ranges**, meant to set realistic expectations and design telemetry. Actual figures should be refined by your backtest/replay framework.

### 5.1 Assumptions

- Average trade size: $3–5 (capped at $5 by config).
- Total exposure: capped at $20.
- Active window: ~8 hours/day, 5 days/week of meaningful 15‑minute BTC/ETH markets.
- Per‑trade edge estimates:
  - ArbTaker: −1% net.
  - LatencySnipe: −0.5% net.
  - SpreadMaker (improved): +0.4% net.
  - LeggedHedge: −0.2% net.
- Volatility of individual trade PnL is high relative to edge (binary outcomes), so Sharpe estimates are rough.

### 5.2 Scenario A – Current mix, no fixes (ArbTaker + LatencySnipe + SpreadMaker + LeggedHedge)

Rough mix (by trade count): 40% ArbTaker, 30% LatencySnipe, 25% SpreadMaker, 5% LeggedHedge.

- **Expected per‑trade edge:**
  - E[edge] ≈ 0.4×(−1%) + 0.3×(−0.5%) + 0.25×(+0.4%) + 0.05×(−0.2%) 
  ≈ −0.4% −0.15% +0.10% −0.01% ≈ **−0.46%**.

For 10–20 trades/day, that’s −0.046×10 to −0.046×20 = −0.46% to −0.92% of capital “at risk” per day if fully utilized. With caps and realized turnover, you get something like:

- **$100 bankroll:** −$3 to −$8/month (−3% to −8%).
- **$250 bankroll:** −$8 to −$20/month.
- **$500 bankroll:** −$15 to −$40/month.
- **Approximate monthly Sharpe:** −0.2 to 0.1 (loss‑drift with high noise).

### 5.3 Scenario B – Kill ArbTaker & LeggedHedge, keep LatencySnipe + SpreadMaker (no VPIN yet)

Mix: 65–75% SpreadMaker, 25–35% LatencySnipe.

Assume:
- SpreadMaker: +0.4% per trade net.
- LatencySnipe: −0.2% per trade (after tuning/removal of worst behavior).

E[edge] ≈ 0.7×0.4% + 0.3×(−0.2%) ≈ 0.28% − 0.06% ≈ **+0.22%**.

With 10–20 trades/day (lower if purely passive):
- Daily drift: ≈ +0.22% × effective turnover fraction.
- With small bankroll and caps, realistic realized drift might be **+0.5% to +2% per month**.

So:
- **$100 bankroll:** −$1 to +$5/month (breakeven to slightly positive).
- **$250 bankroll:** −$3 to +$12/month.
- **$500 bankroll:** −$5 to +$25/month.
- **Monthly Sharpe:** 0.2–0.5.

### 5.4 Scenario C – Fully remediated SpreadMaker + small statistical arb sleeve

Mix: 80–90% SpreadMaker, 10–20% slow statistical arb (e.g. mean‑reversion or correlation trades with holding horizon of multiple 15‑minute buckets, not milliseconds).

Assume after backtesting and tuning:
- SpreadMaker: +0.5% to +0.8% per round‑trip.
- Stat arb: +1% to +2% per trade, but lower frequency.

On small capital with your caps, realized monthly drift could be:
- **$100 bankroll:** +$8 to +$15/month (8–15%).
- **$250 bankroll:** +$20 to +$40/month.
- **$500 bankroll:** +$40 to +$80/month.
- **Monthly Sharpe:** 0.8–1.3.

These are **ambitious but plausible** for a well‑designed MM + small arb bot in a niche venue, provided that:
- You invest heavily in backtesting & telemetry.
- You accept slower scale‑up of bankroll.
- You routinely kill or down‑weight strategies whose live performance diverges from backtest.

---

## 6. Phased implementation roadmap (8/9/10 target path)

### 6.1 Target scores by phase

- **Now:** 4/10.
- **4–6 weeks:** 8/10 (robust infra, profitable SpreadMaker, backtesting live).
- **8–12 weeks:** 9/10 (statistical arb layer, hardened production ops).

### 6.2 Phase 0 (Week 0–1): Safety & de‑risking – “No more blind trading”

**Objectives:**
- Ensure the bot cannot catastrophically lose money in the current negative‑EV configuration.

**Tasks (must‑do before live):**

1. **Disable ArbTaker and LeggedHedge in the strategy ensemble.**
   - *Why it matters:* Both are either fee‑taxed into negative EV or materially model‑risky.
   - *Technical approach:* Remove from the ensemble instantiation; keep the code, but mark them as `EXPERIMENTAL_DISABLED` in config.
   - *Impact:* Removes the largest consistent negative contributors.
   - *Effort:* Very low.
   - *Validation:* Confirm no orders from these strategies appear in logs in a 1‑day dry‑run.

2. **Introduce a global “paper trading” mode (no‑op OrderManager).**
   - *Why:* Validate signal logic, event flow, and risk gating without financial exposure.
   - *Approach:* Implement a `PaperOrderManager` that pretends to submit orders and simulates fills at current top‑of‑book, logging as if real.
   - *Impact:* Allows you to see would‑be PnL and fill quality using live data.
   - *Effort:* Medium.
   - *Validation:* Compare behavior of paper vs live order objects; ensure RiskGate sees them identically.

3. **Implement basic dynamic fee helper consistent with public docs.**
   - *Why:* All EV estimates and risk checks must use actual 2026 fee curves to be meaningful.
   - *Approach:* Centralize a `FeeModel` component with methods like `estimate_taker_fee(prob, size)` and `estimate_maker_rebate(prob, size)` calibrated from docs and small live tests.[cite:4][cite:5][cite:11]
   - *Impact:* Makes it much harder to accidentally run an EV‑negative strategy under fee misassumptions.
   - *Effort:* Low–medium (depending on how precise you want to be); unit‑testable.
   - *Validation:* Numerical tests against Polymarket examples; smoke tests in paper mode.

4. **Tighten global risk caps further (while in development).**
   - *Why:* Even with small bankroll, early mistakes can hurt psychologically and bias your evaluation.
   - *Approach:* Temporarily set: max trade $2, max total exposure $10, daily loss stop 3–5%.
   - *Impact:* Contains drawdowns during early experiments.
   - *Effort:* Minimal.
   - *Validation:* RiskGate unit tests; manual config inspection.

**Exit criteria to move past Phase 0:**
- At least one week of paper trading shows non‑catastrophic behavior (spread PnL not obviously neg).
- No live capital at risk during this phase.

---

### 6.3 Phase 1 (Weeks 1–2): Core infra – idempotency, backpressure, latency SLOs (target: 5/10)

**Goals:**
- Make event processing safe and measurable.

**Key tasks:**

1. **Event schema + idempotency keys.**
   - *Why:* Avoid double‑processing events and duplicate orders, and enable deterministic replay.
   - *Approach:* Introduce a `BaseEvent` dataclass with `event_id: UUID`, `timestamp`, and `source`. Require every event on the bus to carry an `event_id`.
   - *Implementation:* Maintain an in‑memory `processed_event_ids` LRU + periodic persistence; ignore duplicates on receipt.
   - *Impact:* Prevents state divergence during retries/replays.
   - *Effort:* Medium.
   - *Validation:* Unit tests with duplicate events; chaos test that replays the last N events.

2. **Backpressure policy for event queue saturation.**
   - *Why:* Unbounded queues in async systems can lead to OOM, lag, and unpredictable behavior.
   - *Approach:* Use bounded `asyncio.Queue` with size thresholds:
     - <70%: normal operation.
     - 70–90%: drop non‑critical events (e.g., redundant snapshots) or downsample them.
     - >90%: switch to “survival mode” (only order, cancel, and reconciliation events processed; drop low‑priority data).
   - *Impact:* Keeps latency bounded and avoids catastrophic slowdown in stress.
   - *Effort:* Medium.
   - *Validation:* Synthetic load test where ingestion floods the queue; check that critical events still get processed.

3. **Latency budget and SLOs.**
   - *Why:* Even though you can’t beat HFT speeds, you must ensure your **own** latency is stable and within your design envelope.
   - *Approach:* For each hop:
     - `spot_tick → strategy_scan → order_submit` (data → decision → action).
     - `order_submit → order_ack`.
   - Track p50/p95/p99, define SLOs (e.g., p95 < 500 ms for ack; p95 < 300 ms for strategy path).
   - *Impact:* Lets you detect infra regressions and justify assumptions in your backtests.
   - *Effort:* Low–medium.
   - *Validation:* Metrics dashboard, alert if breached for >5 minutes.

4. **State/reconciliation baseline.**
   - *Why:* Divergence between bot’s view of positions and CLOB reality is unacceptable.
   - *Approach:* Every N seconds (e.g. 60), pull current positions/open orders from CLOB and reconcile to local state; if mismatch >$1 or >5%, halt trading and alert.
   - *Impact:* Prevents slow drift or silent bugs from compounding risk.
   - *Effort:* Medium.
   - *Validation:* Tests that introduce fake fills or missed events and ensure reconciliation catches them.

**Exit criteria:**
- Idempotent event consumption.
- Latency metrics live with SLOs.
- Reconciliation loop active with alerts on divergence.

---

### 6.4 Phase 2 (Weeks 3–4): SpreadMaker upgrade + toxicity awareness (target: 6–7/10)

**Goals:**
- Turn SpreadMaker into a genuinely positive‑EV, modest‑capacity MM engine.

**Tasks:**

1. **Integrate VPIN & order‑flow toxicity into SpreadMaker.**
   - *Why:* VPIN (volume‑synchronized probability of informed trading) is a natural filter for when you should quote tight vs wide.[cite:27][cite:30]
   - *Approach:* Feed VPIN into SpreadMaker:
     - For VPIN < 0.4: normal/tight quoting.
     - 0.4–0.6: slightly wider spreads, smaller size.
     - >0.6: either stand down or quote very wide with tiny size.
   - *Impact:* Reduces adverse selection and improves realized spread.
   - *Effort:* Medium.
   - *Validation:* Compare adverse vs favorable fill proportions before/after.

2. **Calibrate spreads to actual market microstructure.**
   - *Why:* 5‑cent minimum spreads are too wide for today’s tighter CLOBs; you’ll get starved of flow.
   - *Approach:* Use historical L2 snapshots to measure the empirical distribution of spreads for targeted markets and adjust `min_spread` and `quote_offset` to be competitive but safe.
   - *Impact:* Raises fill rate while maintaining margin.
   - *Effort:* Medium (requires some data analysis).
   - *Validation:* Increased maker fill counts without deterioration in realized PnL per share.

3. **Incorporate real fee/rebate economics.**
   - *Why:* Maker rebates are a non‑trivial part of MM PnL on Polymarket.[cite:11][cite:12]
   - *Approach:* Track maker vs taker fills explicitly; attribute synthetic “rebate” PnL to maker fills based on official schedule.
   - *Impact:* More accurate PnL attribution and better spread/size tuning decisions.
   - *Effort:* Low.
   - *Validation:* PnL attribution (spread capture vs rebates vs adverse selection) visible in analytics.

4. **Add basic microprice awareness.**
   - *Why:* Quoting symmetrically around the mid is suboptimal if the microprice (mid weighted by queue) is skewed.[cite:22]
   - *Approach:* Use L2 to compute microprice and skew quotes slightly away from it to avoid being at the “wrong” edge.
   - *Impact:* Reduces chance of being picked off on the informed side.
   - *Effort:* Medium.
   - *Validation:* Compare fill quality vs microprice direction.

**Exit criteria:**
- SpreadMaker alone (in paper trading or replay backtest) shows consistent positive PnL across multiple days/weeks.
- Adverse selection metrics improved vs baseline.

---

### 6.5 Phase 3 (Weeks 5–6): Deterministic replay/backtest harness (target: 7–8/10)

**Goals:**
- Be able to say “this strategy has positive EV on historical Polymarket data under realistic execution assumptions.”

**Components (specs):**

#### 6.5.1 Exchange simulator

- **Inputs:**
  - Time‑ordered sequence of book snapshots and trades (from Polymarket’s CLOB timeseries APIs or third‑party datasets).[cite:35][cite:41][cite:47]
  - Strategy orders (price, size, side, time).
- **Behavior:**
  - Match marketable orders against historical book at simulated time.
  - For limit orders:
    - Add to a simulated book with queue priority based on arrival time.
    - Fill when historical prices cross through your price, accounting for queue position where possible.
  - Apply dynamic taker fees, potential maker rebates.
- **Outputs:**
  - Synthetic fills, cancels, position/PnL over time.

#### 6.5.2 Replay harness

- **Mechanism:**
  - Replace live ingestion with a historical event stream.
  - Run your actual strategy code (signals, RiskGate, OrderManager) against the simulator instead of live CLOB.
  - Keep everything else identical (configs, risk caps) to maximize realism.

- **Idempotency:**
  - Use the same event_id semantics as live to ensure deterministic behavior across runs.

- **Performance:**
  - Aim for at least 1–2× realtime (e.g., one trading day replayed in less than one hour) to make parameter sweeps feasible.

**Validation methodology:**
- Start with a single BTC 15‑minute market over a week, compare simulated book behaviour visually with real trade tape.
- Check that a trivial reference strategy (e.g., buy‑and‑hold or random taker) reproduces approximate expected PnL/margins.

---

### 6.6 Phase 4 (Weeks 7–8): Production hardening (target: 8/10)

**Goals:**
- Make the system safe to run unattended for multi‑day periods with clear alerting and recovery paths.

**Key tasks:**

1. **Formalize all state machines.**
   - For orders, strategies, connectivity (CLOB/WS/oracle), and circuit breakers.
   - Enforce allowed transitions in code; log violations.

2. **Add full alerting + runbook.**
   - Alerts on: daily loss limit hit, position divergence, latency SLO breaches, queue saturation, API error spikes.
   - Runbooks describing what to do for each alert, including “safe shutdown” procedure.

3. **Implement graceful degradation.**
   - If Chainlink is stale: **skip** market evaluation.
   - If Binance WS down: automatically switch to Coinbase fallback with degraded confidence.
   - If CLOB lagging/unstable: reduce order rate or pause new entries.

4. **Chaos and restart tests.**
   - Kill the process mid‑session; on restart, ensure reconciliation brings system back to correct state.
   - Induce artificial connection drops and verify degradation mode engages.

**Exit criteria:**
- You would be comfortable leaving the bot running overnight with modest capital.

---

### 6.7 Phase 5 (Weeks 9–12): Alpha layer (target: 9/10)

Once infra + MM are solid, add **slow** statistical alpha where HFT is less dominant:

- Mean‑reversion on mispriced 15‑minute binaries vs an implied volatility/realized volatility model.
- Cross‑asset or cross‑venue spreads (BTC vs ETH binaries, or Polymarket vs large CEX perpetuals) with holding times of hours, not seconds.

These should be developed **only via the replay harness**, then paper traded, and only then gently rolled into live with very small size sleeves.

---

## 7. Technical appendix (state machines, schemas, SLOs/alerts, data sources)

### 7.1 State machines (order + strategy)

**Order lifecycle FSM (minimal institutional spec):**

- States: `NEW → SUBMITTED → ACKED → {PARTIALLY_FILLED → FILLED, CANCELED, REJECTED, EXPIRED}`.
- Transitions must be explicitly coded; invalid transitions should be logged as bugs.

**Strategy FSMs:**

- For LeggedHedge, a proper FSM (SEARCHING → LEG1_OPEN → LEG1_FILLED → HEDGING → COMPLETE/ABORTED) with explicit timeouts, cancels, and hedging limits is required if it ever returns.

### 7.2 Event schema + idempotency keys

**Base:**

- `event_id: UUID` (idempotency key).
- `timestamp: float or ISO8601`.
- `event_type: str` (`market_data`, `strategy_signal`, `order_update`, `risk_event`, etc.).
- `source: str` (`binance_ws`, `chainlink`, `polymarket_clob`, `strategy_engine`).

**Idempotency:**

- Maintain a rolling in‑memory set of processed `event_id`s plus periodic persistence.
- On restart, load recent IDs to prevent re‑processing.
- For replay, you can either reuse IDs (to test idempotency) or annotate scenarios.

### 7.3 Backpressure policy

- Use `asyncio.Queue(maxsize=N)`.
- Define priority classes for events:
  - P0: fills, cancels, circuit breaker triggers.
  - P1: order acks, market data for active markets.
  - P2: background snapshots, low‑priority logs.
- When queue usage >80%, drop or coalesce P2; >95%: treat only P0/P1.

### 7.4 Latency budgets & SLOs

**Metrics to track:**

- `t_data_to_signal_ms`: time from market data event to strategy signal emission.
- `t_signal_to_submit_ms`: time from signal to order submit call.
- `t_submit_to_ack_ms`: time from submit to ack/first fill.

**SLOs:**

- `p95(t_submit_to_ack_ms) < 500 ms`.
- `p95(t_data_to_signal_ms) < 200 ms`.
- Alert if SLO violated for >5 consecutive minutes.

### 7.5 Data source catalog

- **Live:** Polymarket CLOB API, Binance WS, Coinbase REST fallback, Chainlink/Polygon on‑chain resolution (as used now).[cite:14][cite:16][cite:24]
- **Historical:**
  - Commercial: Telonex, Polymarketdata.co (recommended for serious backtesting).[cite:35][cite:41]
  - Community: GitHub `poly_data`, Kaggle dumps (good for prototyping, less so for production calibration).[cite:39][cite:42]

---

## 8. Top 10 immediate actions checklist

**Must‑do before any live real‑money trading:**

1. Disable **ArbTaker** in the ensemble.
2. Disable **LeggedHedge** in the ensemble.
3. Implement a **central FeeModel** using Polymarket’s dynamic fee function and wire it into all EV calculations.
4. Introduce a **global paper‑trading mode** and run the full stack with no live orders for at least one week.
5. Add **idempotency keys (UUID)** to all events and prevent duplicate order submissions.
6. Implement a **bounded event queue with backpressure policy** and log dropped/coalesced events.
7. Start logging **full execution telemetry**: expected vs actual fill, slippage, classification (favorable/neutral/adverse), and VPIN at fill.
8. Establish **latency SLO metrics and alerts** (data→signal, signal→submit, submit→ack).
9. Harden **reconciliation**: periodic cross‑check between internal positions and CLOB; halt trading on mismatch.
10. Move **SpreadMaker** to primary focus, integrating VPIN and fee/rebate modeling; treat all other strategies as experimental until they pass backtest + paper‑trade gates.

---

### Final perspective

PolyBawt’s **architecture is “senior‑level” for a personal project**, but its current strategies are tuned for a market regime that no longer exists. Polymarket’s introduction of dynamic taker fees and the influx of dedicated HFT/market‑making firms mean that:

- Simple arb and slow lead‑lag ideas are now a **negative‑sum game for non‑co‑located, Python‑based bots**.
- The viable institutional path is: **robust, toxicity‑aware market making + slower statistical edges** validated over serious historical data.

If you follow the roadmap above—especially killing negative‑EV components early, implementing a replay harness, and investing in SpreadMaker + telemetry—you can realistically grow PolyBawt into a 9/10‑grade system over the next 2–3 months.
