

# PolyBawt Institutional-Grade Assessment & Remediation Plan

## 1. Executive Summary

### 1.1 Core Verdict on Edge Sustainability

PolyBawt's four-strategy ensemble faces **fundamental viability crisis** following Polymarket's January 2026 market structure transformation. The platform's introduction of **dynamic taker fees peaking at 3.15% for 50% probability contracts**—explicitly designed to eliminate latency-based arbitrage—renders **LatencySnipe economically non-viable** and **ArbTaker marginally viable only at probability extremes**. **SpreadMaker emerges as the sole durable edge source**, capitalizing on maker rebates and fee avoidance, though profitability depends critically on **adverse selection detection capabilities that remain unvalidated**. **LeggedHedge carries unquantified model risk** with correlation breakdown potential that likely dominates any theoretical edge.

The competitive landscape has intensified dramatically. Documented professional implementations—including gabagool222's arbitrage bot and 0xalberto's $764 daily return on $200 capital—demonstrate infrastructure advantages (C++ cores, colocated servers, sub-10ms RPC latency) that **Python asyncio cannot match** for latency-sensitive strategies . Post-fee regime, survival requires strategic pivot from speed-dependent to **prediction-dependent, maker-focused approaches** where analytical sophistication offsets infrastructure disadvantage.

### 1.2 Critical Market Structure Changes (January 2026 Fee Regime)

Polymarket's **January 7, 2026 fee implementation** represents the most significant structural shift in platform history. The dynamic fee formula **fee = C × 10% × (p·(1-p))²** creates probability-dependent costs ranging from negligible at extremes to **3.15% maximum at 50% probability**—precisely where latency arbitrage concentrated . This design explicitly targets "infrastructure-driven arbitrage" with platform-stated objective to render such strategies "unprofitable at scale" .

The fee redistribution mechanism—**100% of taker fees to liquidity providers through daily USDC rebates**—creates asymmetric opportunity for maker strategies while eliminating taker economics . Empirical impact validates design intent: bid-ask spreads compressed from **4.5% (2023) to 1.2% (2025)**, order book depth expanded to **$2.1 million average**, and wash trading declined from **25% to 5% of volume** .

| Metric | Pre-January 2026 | Post-January 2026 | Implication |
|:---|:---|:---|:---|
| Taker fee at 50% probability | 0% | **3.15%** | Latency arbitrage elimination |
| Maker rebate | None | **100% of taker fees** | SpreadMaker viability |
| Typical bid-ask spread | 4.5% | **1.2%** | Edge compression for all strategies |
| Average order book depth | ~$500K | **$2.1M** | Professional MM entry |
| Wash trading volume | 25% | **5%** | Market quality improvement |

*Table 1: Polymarket 15-minute crypto market structural transformation*

### 1.3 Institutional-Readiness Gap Summary

PolyBawt's current architecture scores **4.5/10** for institutional readiness, with **critical deficiencies** across multiple dimensions:

| Subsystem | Score | Critical Gap |
|:---|:---|:---|
| Strategy Logic | 4/10 | LatencySnipe obsolete; ArbTaker fragile; toxicity detection unvalidated |
| Alpha Integrity | 3/10 | No deterministic backtest; edge estimates theoretical; no live validation |
| Risk Controls | 6/10 | Kelly sizing present; dynamic fee integration incomplete; toxicity adjustment absent |
| Execution Reliability | 5/10 | Python asyncio latency variance uncharacterized; no SLO enforcement |
| Reconciliation/State Consistency | 5/10 | Reconciler present; formal state machines absent; idempotency unvalidated |
| Observability | 5/10 | Structured logging, metrics, event bus; queue drop counter added; latency monitoring insufficient |
| Testing Maturity | 3/10 | **No deterministic replay**; no exchange simulator; paper trading unvalidated |
| Production Readiness | 4/10 | Safety-first architecture appropriate; critical validation gaps before live deployment |

*Table 2: PolyBawt institutional-readiness assessment*

The **most severe gap** is absence of deterministic backtesting infrastructure—strategy validation depends entirely on forward testing with unacceptable capital risk. Secondary gaps include: sub-50ms latency monitoring with alerting; validated adverse selection detection beyond basic VPIN; and formalized state machines with exactly-once processing semantics.

### 1.4 Resource Requirements & Timeline

| Phase | Duration | Target Score | Engineering FTE | Infrastructure Cost | Key Deliverables |
|:---|:---|:---|:---|:---|:---|
| 1: Survival & Validation | Weeks 1–2 | ≥6 | 1.5 | $500-1,000 | LatencySnipe deprecation; SpreadMaker MVP; telemetry expansion |
| 2: Fee-Aware Optimization | Weeks 3–4 | ≥7 | 2 | $1,000-2,000 | Dynamic fee integration; VPIN enhancement; microstructure capture |
| 3: Institutional Hardening | Weeks 5–6 | ≥8 | 2.5 | $2,000-3,500 | Deterministic replay; state machine formalization; latency SLOs |
| 4: Competitive Positioning | Weeks 7–10 | ≥9 | 3 | $3,500-6,000 | VPS/colocation evaluation; ML toxicity prediction; multi-strategy allocation |
| 5: Scale & Diversification | Weeks 11–12 | ≥9.5 | 3 | $5,000-10,000 | Cross-market arbitrage; predictive settlement; institutional capital structure |

*Table 3: Phased implementation resource requirements*

**Total 12-week investment**: $12,000-22,500 infrastructure + ~$75,000-125,000 engineering (assuming $150-200/hr blended rate). This investment is justified only with validated edge demonstration; conservative path executes Phase 1-3 only ($3,500-6,500 + ~$35,000-50,000 engineering) with live trading validation before Phase 4-5 commitment.

---

## 2. Market Structure Intelligence: 15-Minute Crypto Binary Markets

### 2.1 Settlement & Timing Dynamics

#### 2.1.1 Chainlink Oracle Update Latency vs. Economic Tradability

The settlement mechanism for Polymarket's 15-minute crypto binaries hinges on **Chainlink Data Streams** providing authoritative price benchmarks at contract expiration. The critical, uncharacterized question is **temporal sequencing**: whether Chainlink updates propagate before, simultaneously with, or after economic resolution becomes deterministic through spot market price action.

Chainlink's decentralized oracle network typically achieves updates within **1-3 block confirmations on Polygon (2-6 seconds under normal conditions)**, with compression to **15-30 seconds during elevated volatility** as deviation thresholds trigger more frequent updates . However, the **economically tradable window**—when position expected value exceeds transaction costs—depends on whether oracle latency creates exploitable information asymmetry.

Two competing hypotheses require empirical resolution:

| Hypothesis | Mechanism | Strategic Implication |
|:---|:---|:---|
| **Oracle preview window** | Chainlink aggregation rounds progress through preliminary states before final commitment; premium data access enables 500ms-2s prediction advantage | Sniper race at settlement boundaries; infrastructure-dependent edge |
| **Spot-led discovery** | Binance/Coinbase price moves precede Chainlink updates; Polymarket CLOB lags both | Latency arbitrage on spot-Polymarket divergence; fee regime eliminates viability |

*Table 4: Settlement timing hypotheses and strategic implications*

PolyBawt's current Chainlink integration for "settlement/sniper checks" suggests recognition of this dimension, but lacks explicit characterization of: (a) typical Chainlink update latency relative to 15-minute boundary; (b) variance under network congestion; (c) historical incidence of oracle-spot divergence; and (d) platform resolution policy for edge cases. The **authoritative settlement lookup path with heuristic fallback disabled** prioritizes safety over speed—appropriate for risk management but potentially missing edge opportunities where heuristic prediction exceeds oracle latency.

#### 2.1.2 Settlement Boundary "Sniper Race" Evidence

Direct evidence of sniper race behavior remains **limited in public sources**, but platform fee structure design implies internal analysis identified concentrated latency-sensitive activity at settlement boundaries. The **0xalberto case**—$764 daily return on $200 deposit (December 21, 2025)—demonstrates that profitability was achievable under zero-fee conditions with "stable, always-on infrastructure and the ability to continuously update strategies" . This performance **predates fee implementation and is structurally irreproducible** under current economics.

Post-fee, residual sniper opportunities—if any—concentrate in: (a) **probability extremes** (p<0.10 or p>0.90) where fees approach zero; (b) **maker-rebate-capturing passive orders** that benefit from settlement-boundary volume without taker fee burden; or (c) **oracle update prediction** faster than market consensus. Each hypothesis requires empirical validation through telemetry collection that PolyBawt's current architecture does not support.

The **gabagool222 arbitrage bot**—publicly available on GitHub—provides competitive intelligence on execution patterns, though its pre-fee optimization limits direct applicability . Analysis of on-chain transaction timing around settlement boundaries would enable direct sniper race detection; this analysis is **recommended priority for Phase 4 competitive positioning**.

#### 2.1.3 Information Asymmetry Windows

Information asymmetry in 15-minute crypto binaries manifests through **three channels with differential persistence and exploitability**:

| Channel | Description | Persistence | PolyBawt Accessibility |
|:---|:---|:---|:---|
| **Cross-venue spot lag** | Binance/Coinbase price moves precede Polymarket CLOB adjustment | 100-500ms, compressing | **Eliminated by fee regime**; Python asyncio disadvantageous |
| **Order book microstructure** | Queue position dynamics, flow imbalance, cancellation patterns not visible in coarse price feeds | Seconds to minutes | **Viable** with microprice calculation and queue position estimation |
| **Settlement prediction** | Oracle update timing or outcome prediction from spot microstructure | Seconds | **Unvalidated**; requires Chainlink Data Streams premium access |

*Table 5: Information asymmetry channels and strategic accessibility*

The **compression of bid-ask spreads to 1.2%** indicates substantial reduction in naive information asymmetry, with professional market makers rapidly incorporating public signals . Residual edge concentrates in microstructure signals and predictive modeling—domains where Python's machine learning ecosystem provides potential advantage over C++ competitors.

### 2.2 Fee Economics: Post-January 2026 Regime

#### 2.2.1 Dynamic Taker Fee Formula: fee = C × 10% × (p·(1-p))²

The fee formula's mathematical properties create **non-linear cost dynamics that fundamentally reshape strategy economics**. With **feeRate = 10% (1000 basis points)** and **exponent = 2**, the probability-dependent component generates:

| Probability (p) | (p·(1-p))² | Effective Fee Rate | Strategic Regime |
|:---|:---|:---|:---|
| 0.50 | 0.0625 | **3.15%** | Latency arbitrage **prohibited** |
| 0.60 | 0.0576 | 2.90% | High-frequency taking **severely taxed** |
| 0.70 | 0.0441 | 2.22% | Selective opportunities **marginal** |
| 0.80 | 0.0256 | 1.29% | Moderate fee burden |
| 0.90 | 0.0081 | **0.41%** | Near-negligible fee impact |
| 0.95 | 0.0025 | **0.13%** | **Extreme-probability arbitrage viable** |
| 0.99 | 0.0001 | ~0% | Fee-irrelevant |

*Table 6: Dynamic fee schedule and strategic segmentation*

The **documented maximum of ~3.15% at 50% probability** reflects rounding mechanics: fees computed to six decimal places, rounded to four, with minimum non-zero fee of 0.0001 USDC . The constant **C ≈ 0.5** for share-count-based calculation aligns with documentation examples (100 shares at $0.50 generating ~$1.56 fee) .

This fee structure creates **strategic imperative for probability-regime-aware operation**: strategies must dynamically adjust thresholds, or restrict activity to low-fee regimes, or abandon taker execution entirely for maker status.

#### 2.2.2 Probability-Regime Fee Sensitivity (0.5% to 3.15% Peak)

The **quadratic probability dependence** creates distinct operational regimes with dramatically different economics:

**Extreme Probability Zone (p < 0.10 or p > 0.90):** Fees below 0.5% enable **residual latency-sensitive strategies**, but opportunity frequency is constrained by market conditions. Directional conviction trades dominate flow, with **intensified adverse selection risk**: trades at 95% probability may reflect informed conviction or manipulation, with limited liquidity for exit.

**Mid-Probability Zone (0.40 < p < 0.60):** **Fee maximization zone** where latency arbitrage is explicitly uneconomical. Maker strategies avoid fees entirely, capturing rebates while providing liquidity. **Competition concentrates here**, with professional market makers deploying sophisticated queue position optimization.

**Transition Zone (0.20 < p < 0.40 or 0.60 < p < 0.80):** Moderate fees (1-3%) create complex EV calculations where edge magnitude must exceed fee plus slippage plus adverse selection. **Dynamic fee-aware EV estimation**—partially implemented in PolyBawt—becomes critical for viability.

#### 2.2.3 Maker Rebate Redistribution Mechanics

All collected taker fees flow **daily into Maker Rebates Program**, distributed in USDC proportional to "liquidity shares that are actually traded" . This creates **zero-sum transfer from active to passive strategies**, with platform capturing no direct revenue. The "fee-curve weighted" redistribution aligns rewards with value generated .

| Rebate Component | Calculation | Strategic Implication |
|:---|:---|:---|
| Base rebate | Proportional to maker volume executed | Volume maximization incentive |
| Fee-curve weighting | Higher weight for high-fee-regime liquidity provision | Compensation for adverse selection risk |
| Daily distribution | USDC to wallet, immediately available | Working capital efficiency |

*Table 7: Maker rebate mechanics and strategic optimization*

For SpreadMaker, successful operation requires: **(a) consistent queue position near best bid/offer**; **(b) inventory management avoiding excessive directional exposure**; **(c) requote cadence maintaining position without excessive cancellation**; and **(d) toxicity detection withdrawing liquidity when adverse selection risk exceeds rebate income**. The empirical impact—spread compression and depth expansion—validates incentive design but implies **reduced edge magnitude for all participants**, with survival contingent on superior execution.

#### 2.2.4 Net EV Threshold: Required Spread >2.5-3% Post-Fees

Platform documentation and market analysis converge on critical threshold: **arbitrage strategies now require spreads of 2.5-3% for viability** after accounting for fees . This represents **substantial elevation from historical ~1.5% thresholds** under zero-fee conditions.

For ArbTaker's "short arb" logic—selling both legs when combined bid exceeds threshold—the implication is severe:

| Cost Component | Mid-Probability (p=0.50) | Extreme Probability (p=0.90) |
|:---|:---|:---|
| Gross spread threshold (historical) | 1.5% | 1.5% |
| Dynamic taker fee (both legs) | **6.30%** | 0.82% |
| **Effective threshold required** | **7.8-9.3%** | **2.3-3.8%** |
| Historical frequency of such spreads | <1% | ~5-10% |

*Table 8: ArbTaker viability threshold elevation under dynamic fees*

The **"fee-aware EV and slippage haircut"** currently implemented requires critical validation: does it correctly model **probability-dependent fees**? Does it incorporate **maker rebate opportunity cost** for taker strategies? The "edge realism is unknown" acknowledgment signals **unvalidated assumptions requiring empirical resolution**.

### 2.3 Liquidity & Depth Profile

#### 2.3.1 Typical Order Book Depth by Time-to-Expiry

Order book depth dynamics exhibit **predictable lifecycle patterns** tied to contract maturity:

| Time-to-Expiry | Typical Depth | Spread Characteristics | Strategic Implication |
|:---|:---|:---|:---|
| 13-15 minutes (initiation) | Shallow ($200-500K) | Wide, establishing | Limited size; price discovery active |
| 5-12 minutes (mid-contract) | **Deep ($1.5-2.5M)** | Tight, competitive | Optimal execution window |
| 2-5 minutes (approach) | Variable ($800K-2M) | Widening, selective | Toxicity detection critical |
| <2 minutes (settlement) | **Concentrated or evacuated** | Highly variable | Sniper activity or liquidity flight |

*Table 9: Order book depth by time-to-expiry*

The documented **$2.1M average depth (Q3 2025)** masks substantial variance across buckets . PolyBawt's **$5 max trade and $20 total exposure** are well below depth constraints, but scaling requires **depth-aware position sizing**. Current implementation lacks explicit time-to-expiry bucketing for execution optimization.

#### 2.3.2 BTC/ETH 15m Contract Microstructure

BTC and ETH contracts exhibit **distinct microstructure characteristics**:

| Characteristic | BTC-USD 15m | ETH-USD 15m |
|:---|:---|:---|
| Absolute price level | Higher (~$95K) | Lower (~$2.6K) |
| Volatility (annualized) | ~45-55% | ~55-70% |
| Tick size impact | Smaller relative to price | Larger relative to price |
| Queue position value | Higher (more active flow) | Moderate |
| Correlation with spot | ~0.92-0.96 | ~0.88-0.94 |
| Professional MM presence | **Intense** | Moderate |

*Table 10: BTC vs. ETH 15-minute contract microstructure*

Cross-correlation between BTC and ETH contracts creates **portfolio-level risk concentration** that LeggedHedge attempts to address—though likely inadequately given correlation instability during stress periods.

#### 2.3.3 Queue Position Value & Adverse Selection Risk

In maker-rebate-driven markets, **queue position carries substantial economic value**:

| Queue Position | Fill Probability | Adverse Selection Exposure | Optimal Strategy |
|:---|:---|:---|:---|
| First-in-queue | **85-95%** | **Highest** (toxic flow targets front) | Tight spread, rapid cancellation on toxicity signal |
| Positions 2-5 | 60-80% | High | Moderate spread, selective participation |
| Positions 6-15 | 35-55% | Moderate | Wider spread, patience for uninformed flow |
| Positions 16+ | 15-30% | Lower (filtered by front queue) | **Uncertain fill; may indicate stale quote** |

*Table 11: Queue position value and adverse selection tradeoffs*

PolyBawt's current architecture **lacks explicit queue position estimation**. SpreadMaker's "order-book-derived side selection" provides coarse directionality but not queue depth awareness. Enhancement requires: **WebSocket sequence number tracking** for order book reconstruction; **queue length estimation** from visible depth; and **fill probability modeling** based on position and flow characteristics.

### 2.4 Competitive Landscape

#### 2.4.1 HFT/Professional MM Signatures

Professional participation exhibits **multiple observable signatures**:

| Signature | Detection Method | PolyBawt Implication |
|:---|:---|:---|
| Sub-10ms order response latency | Timestamp correlation analysis | **Unachievable with Python asyncio** |
| Quote refresh >10 Hz during active periods | WebSocket message frequency | Competitive pressure on requote cadence |
| Inventory skew near-neutral | Position tracking via on-chain analysis | Benchmark for SpreadMaker optimization |
| Anticipatory cancellation before spot moves | Cancellation-to-trade ratio, timing | **Toxicity detection requirement** |
| Correlation with spot order flow imbalance | Cross-venue flow analysis | Information disadvantage for retail |

*Table 12: Professional MM signatures and competitive implications*

The platform's evolution toward **CFTC licensing, maker rebates, and spread compression** reflects successful attraction of professional liquidity providers .

#### 2.4.2 Latency Arbitrage Bot Ecosystem (gabagool222, 0xalberto cases)

| Case | Date | Performance | Architecture | Post-Fee Viability |
|:---|:---|:---|:---|:---|
| **gabagool222** | Ongoing (GitHub) | Undisclosed live | Python, WebSocket, FOK orders | **Marginal** (maker adaptation required) |
| **0xalberto** | December 21, 2025 | **$764/day on $200** (382% daily) | Undisclosed; "stable infrastructure" | **Eliminated** (fee burden exceeds edge) |
| **0x8dxd** | Pre-January 2026 | **$515K/month**, 99% win rate, 7,300 trades | C++, colocation, parallel scanning | **Eliminated** (platform explicitly targeted) |

*Table 13: Documented latency arbitrage cases and post-fee viability*

The **0xalberto and 0x8dxd cases** demonstrate that **extraordinary profitability attracted platform intervention**—the fee regime change was responsive to detected arbitrage activity, not anticipatory . Post-fee, competitive landscape has **bifurcated**: latency-sensitive strategies migrated to probability extremes or exited; maker strategies compete on **queue position and toxicity detection**.

#### 2.4.3 Retail Bot Displacement Mechanisms

Retail-scale bots face **systematic displacement through four reinforcing mechanisms**:

| Mechanism | Manifestation | Mitigation for PolyBawt |
|:---|:---|:---|
| **Fee economics** | Edge elimination in accessible probability regimes | **Exclusive maker operation**; probability-extreme filtering |
| **Infrastructure disadvantage** | 10-100x latency gap vs. C++ competitors | **Abandon latency-sensitive strategies**; focus on prediction quality |
| **Information disadvantage** | Inferior flow analysis and toxicity detection | **ML investment** for adverse selection modeling |
| **Adverse selection concentration** | Toxic flow targets predictable retail patterns | **Dynamic spread adjustment**; selective participation |

*Table 14: Retail bot displacement mechanisms and mitigation strategies*

The documented **80% net loser rate** among Polymarket participants—with only **0.51% of wallets achieving >$1,000 profits**—reflects this displacement dynamic .

#### 2.4.4 Infrastructure Arms Race: Colocation, C++, Sub-10ms RPC

Competitive infrastructure deployment has **intensified beyond Python feasibility**:

| Component | Professional Standard | PolyBawt Current | Gap |
|:---|:---|:---|:---|
| Language/runtime | **C++/Rust**, <100μs critical path | Python 3.11 asyncio, 2-5ms context switch | **20-50x** |
| Network stack | **Kernel bypass (DPDK/RDMA)**, <50μs | Standard Linux networking, 100-500μs | **10-100x** |
| Infrastructure | **Colocated/VPS optimization**, <10ms RPC | Consumer cloud, 50-200ms | **5-20x** |
| Market scanning | **Parallel 100+ contracts** | Sequential or limited parallel | **Throughput disadvantage** |

*Table 15: Infrastructure competitive gap analysis*

QuantVPS documentation emphasizes that **"even the smallest delays can erase your trading edge"** with arbitrage opportunities "closing within seconds" . The LinkedIn-documented arbitrage system uses **C++ for speed, WebSockets for real-time data, and AWS regional optimization** for georestriction compliance .

**Strategic implication**: PolyBawt must **abandon direct latency competition** and exploit Python's **analytical ecosystem advantage** (ML libraries, rapid development) for prediction-dependent, maker-focused strategies.

### 2.5 Data Availability & Sources

#### 2.5.1 Historical Trades: Polymarket API, Dune Analytics, Flipside

| Source | URL | Data Type | Latency | Historical Depth | Cost | Quality Notes |
|:---|:---|:---|:---|:---|:---|:---|
| **Polymarket API** | https://docs.polymarket.com/ | Trades, orders, markets | <50ms WebSocket | 90 days detailed, 2 years aggregated | Free | Official; schema evolution risk |
| **Dune Analytics** | https://dune.com/polymarket | Aggregated statistics, wallet analysis | 1-6 hours | Full history | Free/premium | Community-maintained; SQL interface |
| **Flipside Crypto** | https://flipsidecrypto.xyz/ | Blockchain-derived metrics | 1-6 hours | Full history | Free/premium | Cross-chain context; alternative indexing |

*Table 16: Historical trade data sources*

For **institutional-grade backtesting**, direct API capture with local storage is preferred over third-party aggregation due to latency and schema control requirements.

#### 2.5.2 Order Book Snapshots: WebSocket Capture, QuantVPS, Custom Infrastructure

| Approach | Provider/Cost | Latency | Historical Depth | Recommendation |
|:---|:---|:---|:---|:---|
| Self-capture (WebSocket) | Infrastructure: $500-2,000/month | Configurable (target <10ms) | Unlimited (self-managed) | **Preferred for control** |
| QuantVPS hosted | $200-1,000/month | Optimized (<10ms claimed) | Provider-dependent | Evaluate for rapid deployment |
| Commercial (Amberdata) | $5,000+/month | <100ms | Full reconstruction | Excessive for current scale |

*Table 17: Order book capture infrastructure options*

Critical requirements: **message timestamping with microsecond precision**; **sequence gap detection and recovery**; **snapshot-plus-incremental reconstruction** for efficient storage.

#### 2.5.3 Market Metadata/Resolutions: Polymarket Subgraph, Chainlink Data Streams

| Source | URL | Purpose | Update Frequency | Reliability Notes |
|:---|:---|:---|:---|:---|
| **Chainlink Data Streams** | https://docs.chain.link/data-streams | Settlement prices, oracle updates | Deviation-triggered + heartbeat | **Critical dependency**; fallback required |
| **Polymarket Resolution Subgraph** | https://thegraph.com/hosted-service/subgraph/polymarket/matic-markets | Market resolutions, outcomes | Event-driven (1-5 min indexing) | Blockchain-final; query flexibility |

*Table 18: Settlement and resolution data sources*

**Resolution timing relative to economic outcome determination** remains critical uncertainty requiring empirical characterization through telemetry collection.

#### 2.5.4 Commercial Providers: Amberdata, Nansen, Arkham Intelligence

| Provider | Specialization | Polymarket Coverage | Use Case | Cost |
|:---|:---|:---|:---|:---|
| **Amberdata** | Institutional crypto data | Full market data, order book | Production data feed, historical analysis | $1,000-5,000/month |
| **Nansen** | On-chain analytics, wallet labeling | Wallet flow, smart money tracking | Competitive intelligence, toxicity estimation | $500-2,000/month |
| **Arkham Intelligence** | Entity resolution, transaction tracing | Exchange wallets, bot identification | Competitive landscape mapping | $500-2,000/month |

*Table 19: Commercial data provider evaluation*

For PolyBawt's **current $20 exposure scale**, commercial provider costs likely exceed justified investment. Evaluation becomes relevant at **10x+ scale** or for **competitive intelligence requirements** in Phase 4-5.

---

## 3. Strategy-by-Strategy Edge Scorecard

### 3.1 ArbTaker (Short Arbitrage)

#### 3.1.1 EV Sign Assessment: Fragile-to-Negative Post-Fee

ArbTaker's core logic—**selling both legs when combined bid exceeds threshold**—faces **fundamental economic pressure** under dynamic fee regime. The strategy's viability depends on **probability regime**:

| Regime | Fee Burden | Required Gross Spread | Historical Frequency | EV Assessment |
|:---|:---|:---|:---|:---|
| p ≈ 0.50 (mid) | **6.30%** (both legs) | **7.8-9.3%** | <1% | **Negative** (fee exceeds typical edge) |
| p ≈ 0.70-0.80 | 4.4-2.6% | 5.0-6.5% | ~2-5% | **Fragile** (marginal viability) |
| p ≈ 0.90+ (extreme) | **<1%** | 2.5-3.5% | ~5-10% | **Fragile-positive** (conditional viability) |

*Table 20: ArbTaker EV assessment by probability regime*

The **"fee-aware EV and slippage haircut"** implementation requires critical validation: does it **dynamically adjust threshold by real-time probability estimation**? Does it incorporate **maker rebate opportunity cost** for taker execution? The "edge realism is unknown" acknowledgment signals **unvalidated assumptions requiring empirical resolution**.

**Realistic EV estimate**: **-50 to +30 bps per trade** (net, post-slippage), with **positive expectation only at probability extremes** where opportunity frequency is constrained and adverse selection intensified.

#### 3.1.2 Half-Life Under Competition: <30 Days

Any residual edge in extreme-probability arbitrage faces **rapid competitive decay**:

| Decay Mechanism | Timeline | Impact |
|:---|:---|:---|
| Detection by competing bots | 1-2 weeks | Opportunity frequency reduction |
| Infrastructure investment (colocation, C++) | 2-4 weeks | Queue position disadvantage |
| Platform fee adjustment if arbitrage persists | 4-8 weeks | Structural elimination |

*Table 21: ArbTaker competitive decay dynamics*

The **30-day half-life estimate** reflects accelerated crypto market competitive cycles; **infrastructure-matched competitors** (C++, colocated) may compress this to **<14 days**.

#### 3.1.3 Taxation by Faster Participants: Severe (C++ Bots, Colocation)

Taxation intensity is **severe and structural**:

| Taxation Mechanism | Manifestation | Magnitude |
|:---|:---|:---|
| Sniping (faster execution on identified opportunity) | Missed fills, degraded execution prices | 60-80% of theoretical edge |
| Queue position dominance | Systematic fill at worse-than-intended prices | 10-30 bps additional slippage |
| Spread compression | Competitive quoting reduces available edge | 20-50% edge reduction |

*Table 22: ArbTaker taxation by faster participants*

Python asyncio's **inherent latency variance** (garbage collection, GIL contention, event loop scheduling) creates **systematic disadvantage that infrastructure investment cannot fully overcome**.

#### 3.1.4 Required Modifications for Viability

| Modification | Implementation | Expected Impact | Priority |
|:---|:---|:---|:---|
| **Probability-extreme filtering** | Restrict execution to p<0.15 or p>0.85 | Eliminates fee-prohibitive regimes | Critical |
| **Dynamic threshold adjustment** | Real-time fee calculation with EV threshold | Prevents negative-EV execution | Critical |
| **Maker transition evaluation** | Post-only order preference where possible | Fee avoidance, rebate capture | High |
| **Queue position estimation** | Sequence number tracking, fill probability model | Improved execution quality | Medium |

*Table 23: ArbTaker viability modifications*

**Even with modifications, expected edge is modest (20-50 bps)** with **low opportunity frequency**. Recommendation: **deprioritize relative to SpreadMaker**; maintain as **conditional, low-allocation component**.

### 3.2 LatencySnipe

#### 3.2.1 EV Sign Assessment: Negative at 50% Probability (3.15% Fee)

LatencySnipe's core mechanism—**triggering on spot material movement with Polymarket lag exploitation**—targets **exactly the probability regime where fees maximize**:

| Scenario | Pre-Fee (Zero) | Post-Fee (3.15%) | EV Change |
|:---|:---|:---|:---|
| Typical gross edge | 150-300 bps | 150-300 bps | Unchanged |
| Taker fee | 0 bps | **315 bps** | **-315 bps** |
| Net pre-slippage | 150-300 bps | **-165 to -15 bps** | **Eliminated** |

*Table 24: LatencySnipe EV transformation under fee regime*

The **0xalberto case**—$764 daily on $200 deposit—demonstrates **historical profitability that is structurally irreproducible** . The platform's **explicit design objective** to "curb latency-based arbitrage strategies" and render them "unprofitable at scale" has been achieved .

**EV sign: Negative across all probability regimes** where typical gross edge (<300 bps) fails to overcome fee burden (50-315 bps) plus slippage (10-50 bps) plus adverse selection (20-100 bps).

#### 3.2.2 Half-Life Under Competition: Already Expired (Fee Regime Change)

Unlike gradual competitive decay, LatencySnipe experienced **instantaneous edge expiration** through **exogenous policy change**:

| Date | Event | Impact |
|:---|:---|:---|
| December 21, 2025 | 0xalberto $764/day documented | Peak profitability demonstration |
| January 7, 2026 | **Dynamic fee implementation** | **Structural elimination** |
| January 2026 onward | Competitive adaptation to maker strategies | No recovery path for latency-taking |

*Table 25: LatencySnipe edge expiration timeline*

The **half-life concept is inapplicable**—edge did not decay, it was **confiscated by platform design**. Any historical backtests or paper trading from pre-January 2026 are **structurally non-representative**.

#### 3.2.3 Taxation by Faster Participants: Complete (Fee Designed to Eliminate)

The fee structure's **quadratic probability dependence** achieves **complete taxation through platform design**:

| Competitive Dimension | Pre-Fee | Post-Fee |
|:---|:---|:---|
| Speed advantage | Determined queue position, fill priority | **Irrelevant** (fee exceeds edge) |
| Signal quality | Predictive accuracy for lag exploitation | **Irrelevant** (no viable execution path) |
| Infrastructure investment | Colocation, C++, kernel bypass | **Wasted** (cannot overcome 315 bps fee) |

*Table 26: Competitive dimension irrelevance under fee regime*

**No infrastructure investment enables viability** against fee burden exceeding gross edge. The fee is **not a competitive disadvantage to overcome** but **market structure elimination of the strategy class**.

#### 3.2.4 Strategic Pivot Requirements

| Option | Description | Viability | Recommendation |
|:---|:---|:---|:---|
| **A. Deprecation** | Complete removal from strategy ensemble | Certain | **Preferred** |
| **B. Signal repurposing** | Transform spot momentum detection into SpreadMaker trigger | Moderate | Evaluate if signal quality validated |
| **C. Extreme-probability operation** | Restrict to p<0.10 or p>0.90 with minimal fees | Marginal | Insufficient opportunity frequency |
| **D. Maker transition** | Post-only liquidity placement on momentum signals | Moderate | Converges with SpreadMaker |

*Table 27: LatencySnipe strategic pivot options*

**Recommended path: Option A (deprecation)** with **signal generation evaluation for SpreadMaker integration** (Option B/D convergence). Engineering resources **immediately redirected** to SpreadMaker enhancement.

### 3.3 SpreadMaker

#### 3.3.1 EV Sign Assessment: Positive (Maker Rebates, Fee Avoidance)

SpreadMaker—**passive quoting with minimum spread threshold, requote cadence, inventory skew**—is **sole structurally viable strategy** under current market conditions:

| Revenue Component | Magnitude | Dependency |
|:---|:---|:---|
| Bid-ask spread capture | 50-120 bps | Spread threshold, fill rate |
| **Maker rebate** | 10-50 bps (variable) | Volume share, fee regime |
| **Fee avoidance** | 50-315 bps (vs. taker) | Maker status maintenance |
| **Total gross advantage** | **110-485 bps** | Execution quality, toxicity management |

*Table 28: SpreadMaker revenue decomposition*

**Net EV after adverse selection**: **+20 to +120 bps per trade** (realistic), with **wide variance** depending on toxicity detection effectiveness.

Critical parameters requiring optimization:

| Parameter | Too Aggressive | Optimal | Too Conservative |
|:---|:---|:---|:---|
| Spread threshold | Excessive adverse selection | **Balance fill rate and selection** | Missed opportunities, low fill rate |
| Requote cadence | High cancellation, gas cost, queue position loss | **Maintain freshness without churn** | Stale quotes, poor queue position |
| Inventory skew | Over-adjustment, whipsaw losses | **Compensate for predicted flow** | Neutral exposure, missed edge |

*Table 29: SpreadMaker parameter optimization tradeoffs*

#### 3.3.2 Half-Life Under Competition: 60-90 Days

SpreadMaker edge decays through **slower mechanisms than latency strategies**:

| Decay Mechanism | Timeline | Mitigation |
|:---|:---|:---|
| Professional MM entry with superior toxicity detection | 30-60 days | **Continuous model refinement**, ML integration |
| Rebate rate adjustment by platform | 60-180 days (unpredictable) | Diversified strategy ensemble |
| Adverse selection model improvement by competitors | Ongoing | **Data accumulation advantage**, proprietary signals |
| Spread compression from competitive quoting | 30-90 days | **Inventory management differentiation** |

*Table 30: SpreadMaker competitive decay dynamics*

The **60-90 day half-life**—**3-6x longer than latency strategies**—justifies development investment and enables **meaningful return accumulation** before competitive convergence.

#### 3.3.3 Taxation by Faster Participants: Moderate (Toxic Flow, Adverse Selection)

Taxation manifests through **adverse selection rather than latency disadvantage**:

| Taxation Mechanism | Detection | Mitigation |
|:---|:---|:---|
| Informed flow targeting mispriced quotes | VPIN elevation, flow imbalance | **Dynamic spread widening**, liquidity withdrawal |
| Anticipatory cancellation before adverse moves | Cancellation pattern analysis | **Faster toxicity signals**, predictive models |
| Queue position manipulation | Sequence analysis | **Acceptable queue depth evaluation** |

*Table 31: SpreadMaker taxation mechanisms and mitigation*

The **moderate classification** reflects **manageable taxation through detection and response**—unlike latency strategies where taxation is complete and unmitigable.

#### 3.3.4 Toxicity/Adverse Selection Validation Needs

| Validation Component | Current State | Required Enhancement | Timeline |
|:---|:---|:---|:---|
| VPIN implementation | Basic, uncalibrated | **Real-time update, regime-specific thresholds** | 1-2 weeks |
| Post-fill PnL analysis | Absent | **Automated attribution by toxicity regime** | 2-3 weeks |
| Flow classification | Absent | **ML-based informed/uninformed detection** | 4-6 weeks |
| Dynamic spread adjustment | Fixed thresholds | **Toxicity-responsive, inventory-aware** | 2-3 weeks |

*Table 32: Toxicity detection validation requirements*

**Critical unmet need**: "toxicity/adverse-selection validation under real flow" acknowledged in project overview. **Deployment without this validation risks rapid capital erosion** through uninformed quoting in toxic environments.

### 3.4 LeggedHedge

#### 3.4.1 EV Sign Assessment: Negative (Model Risk Dominates)

LeggedHedge—**crash detection + hedge flow**—carries **unquantified model risk that likely dominates any theoretical edge**:

| Risk Component | Manifestation | Likelihood | Impact |
|:---|:---|:---|:---|
| False positive rate | Unnecessary hedge execution, transaction cost accumulation | **High** (rare events, imbalanced data) | Continuous drag |
| Correlation breakdown | Hedge fails when most needed (stress periods) | **Very high** (empirically validated) | Catastrophic protection failure |
| Execution slippage | Hedge execution at degraded prices during volatility | High | Reduced or negative hedge effectiveness |
| Signal latency | Detection lag relative to crash initiation | Moderate | Partial protection, whipsaw entry |

*Table 33: LeggedHedge model risk decomposition*

The project overview's accurate assessment—**"likely weakest and most model-risky component"**—should be **operationalized as deprecation recommendation**.

#### 3.4.2 Half-Life Under Competition: N/A (Fundamentally Flawed)

The **half-life concept is inapplicable**: LeggedHedge's limitations are **structural rather than competitive**. No infrastructure investment, information advantage, or model refinement addresses **core correlation instability** that invalidates hedge construction.

#### 3.4.3 Taxation by Faster Participants: High (Hedge Slippage, Correlation Breakdown)

If operated, taxation occurs through:

| Mechanism | Timing | Magnitude |
|:---|:---|:---|
| Hedge execution slippage | Stress periods, liquidity evacuation | 50-200 bps |
| Correlation inversion | Precise moment of maximum protection need | **Complete hedge failure** |
| Anticipatory flow front-running | Predictable hedge patterns detected by competitors | 10-50 bps |

*Table 34: LeggedHedge taxation mechanisms*

#### 3.4.4 Deprecation vs. Redesign Decision

| Option | Description | Recommendation |
|:---|:---|:---|
| **A. Deprecation** | Complete removal from strategy ensemble | **Strongly preferred** |
| **B. Delta-neutral inventory targeting** | Replace crash detection with continuous neutralization | Evaluate as SpreadMaker enhancement |
| **C. Options-style protection** | Purchase of binary options for tail risk (if available) | Platform limitation: unavailable |
| **D. Volatility scaling** | Position size reduction in elevated volatility regimes | Integrate with RiskGate |

*Table 35: LeggedHedge disposition options*

**Recommendation: Option A (deprecation)** with **risk management function transferred to position sizing and circuit breakers** (Option D integration).

---

## 4. Quantified Performance Scenarios

### 4.1 Edge Per Trade Estimates

#### 4.1.1 Gross Edge Ranges by Strategy

| Strategy | Probability Regime | Gross Edge (bps) | Fee (bps) | Net Pre-Slippage |
|:---|:---|:---|:---|:---|
| ArbTaker (extreme) | p<0.15 or p>0.85 | 80-150 | 5-15 | **65-145** |
| ArbTaker (mid) | p≈0.50 | 100-200 | **315** | **-215 to -115** |
| LatencySnipe | Any (post-fee) | 150-300 | 50-315 | **-165 to -15** |
| **SpreadMaker (base)** | All (maker) | 50-120 spread + rebate | **0** | **60-170** |
| **SpreadMaker (optimized)** | All (maker, toxicity-managed) | 50-120 spread + rebate | **0** | **80-200** |
| LeggedHedge | Stress periods | Unquantified | 0-315 | **Negative expected** |

*Table 36: Gross and net edge estimates by strategy*

#### 4.1.2 Net Edge Post-Fee, Post-Slippage

| Strategy | Optimistic | Realistic | Pessimistic | Confidence |
|:---|:---|:---|:---|:---|
| ArbTaker (extreme) | 80 bps | **30 bps** | -30 bps | Low (opportunity frequency) |
| **SpreadMaker (base)** | 120 bps | **50 bps** | -10 bps | Medium (unvalidated toxicity) |
| **SpreadMaker (optimized)** | 150 bps | **90 bps** | 20 bps | Medium (post-validation) |

*Table 37: Net edge estimates with confidence levels*

#### 4.1.3 Confidence Intervals & Uncertainty Sources

| Uncertainty Source | Magnitude | Resolution Path |
|:---|:---|:---|
| Adverse selection intensity | ±40 bps impact | Post-fill PnL analysis, 500+ trades |
| Slippage by market condition | ±15 bps impact | Latency decomposition, venue analysis |
| Rebate rate variability | ±10 bps impact | Daily tracking, platform monitoring |
| Correlation spot-Polymarket | ±20 bps impact (ArbTaker) | Lead-lag analysis at multiple horizons |

*Table 38: Uncertainty sources and resolution paths*

### 4.2 Sharpe-Like Profile Estimation

#### 4.2.1 Return Volatility Assumptions

| Parameter | Assumption | Basis |
|:---|:---|:---|
| Per-trade return volatility | 2.0-3.5% | Adverse selection variance, trade clustering |
| Daily trade count (SpreadMaker) | 15-35 | Opportunity frequency, fill rate |
| Serial correlation (daily returns) | 0.15-0.30 | Adverse selection regime persistence |

*Table 39: Volatility assumption parameters*

#### 4.2.2 Drawdown Expectations

| Scenario | Max Drawdown (95%) | Recovery Time (50%) |
|:---|:---|:---|
| Base SpreadMaker, half-Kelly sizing | 15-25% | 20-40 days |
| Optimized SpreadMaker, dynamic sizing | 10-18% | 15-30 days |
| Stress (adverse selection regime) | 25-40% | 40-80 days |

*Table 40: Drawdown expectations by scenario*

#### 4.2.3 Kelly Criterion-Adjusted Sizing

| Bankroll | Full-Kelly Trade Size | Recommended (Half-Kelly) | Current Limit | Gap |
|:---|:---|:---|:---|:---|
| $100 | $8.75 (8.75%) | $4.38 (4.4%) | $5 (5.0%) | **Near-optimal** |
| $250 | $21.88 (8.75%) | $10.94 (4.4%) | $5 (2.0%) | **Conservative** |
| $500 | $43.75 (8.75%) | $21.88 (4.4%) | $5 (1.0%) | **Very conservative** |

*Table 41: Kelly-optimal sizing vs. current limits*

Current **$5 max trade is appropriate for $100 bankroll, increasingly conservative for larger capital**. Scaling requires **dynamic sizing with edge uncertainty penalty** (fractional Kelly 0.3-0.5).

### 4.3 Bankroll-Specific Projections

#### 4.3.1 $100 Bankroll: Daily/Monthly Expectations

| Metric | Optimistic | Realistic | Pessimistic |
|:---|:---|:---|:---|
| Daily trades | 10-15 | 5-10 | 3-5 |
| Avg net edge/trade | 80 bps | **40 bps** | 0 bps |
| Daily return | 0.8-1.2% | **0.2-0.4%** | 0% |
| Monthly return (20 days) | 16-24% | **4-8%** | 0% |
| Monthly $ profit | $16-24 | **$4-8** | $0 |
| **Ruin probability (90 days)** | 10% | **20%** | 35% |

*Table 42: $100 bankroll projections*

**Operational constraints**: Limited diversification; excessive sensitivity to individual outcomes; **minimum viable for learning, not income generation**.

#### 4.3.2 $250 Bankroll: Daily/Monthly Expectations

| Metric | Optimistic | Realistic | Pessimistic |
|:---|:---|:---|:---|
| Daily return | 0.8-1.2% | **0.25-0.5%** | 0-0.1% |
| Monthly return | 16-24% | **5-10%** | 0-2% |
| Monthly $ profit | $40-60 | **$12.50-25** | $0-5 |
| **Ruin probability (90 days)** | 8% | **15%** | 28% |

*Table 43: $250 bankroll projections*

**Practical minimum for live deployment**: Enables modest diversification; **$12-25/month realistic expectation** before operational costs.

#### 4.3.3 $500 Bankroll: Daily/Monthly Expectations

| Metric | Optimistic | Realistic | Pessimistic |
|:---|:---|:---|:---|
| Daily return | 0.9-1.3% | **0.3-0.6%** | 0-0.15% |
| Monthly return | 18-26% | **6-12%** | 0-3% |
| Monthly $ profit | $90-130 | **$30-60** | $0-15 |
| **Ruin probability (90 days)** | 5% | **12%** | 22% |

*Table 44: $500 bankroll projections*

**Recommended deployment scale**: Improved risk-adjusted returns; **$30-60/month realistic expectation**; foundation for scaling validation.

#### 4.3.4 Ruin Probability Estimates

| Factor | Impact on Ruin Probability | Mitigation |
|:---|:---|:---|
| Edge uncertainty (unvalidated toxicity) | +5-10% | **Phase 1-2 validation before scale** |
| Operational failure (connectivity, API) | +3-7% | Redundancy, circuit breakers, runbooks |
| Adverse selection clustering | +5-15% | **Dynamic sizing, toxicity-triggered halt** |
| Platform parameter change | +2-5% | Monitoring, multi-venue capability |

*Table 45: Ruin probability factors and mitigation*

### 4.4 Inferential Methods & Telemetry Gaps

#### 4.4.1 Assumptions Requiring Validation

| Priority | Assumption | Current Basis | Validation Method | Timeline |
|:---|:---|:---|:---|:---|
| 1 | SpreadMaker toxicity detection effectiveness | Theoretical VPIN | Post-fill PnL by regime | 2-3 weeks |
| 2 | Adverse selection magnitude by time-of-day | Industry benchmarks | Telemetry aggregation | 1-2 weeks |
| 3 | Slippage by order size and market condition | Fixed estimate | Latency decomposition | 2-3 weeks |
| 4 | Lead-lag correlation stability | Historical pattern | Real-time cross-correlation | 3-4 weeks |
| 5 | Chainlink oracle timing vs. economic resolution | Undocumented | Timestamp correlation | 2-3 weeks |

*Table 46: Priority validation requirements*

#### 4.4.2 Required Data Collection for Uncertainty Reduction

| Telemetry Component | Fields | Frequency | Retention | Use Case |
|:---|:---|:---|:---|:---|
| **Fill quality** | Intended price, arrival mid, fill price, fill time, queue position | 100% | 90 days | Slippage decomposition, adverse selection |
| **Latency decomposition** | Signal generation, order prep, submission, ack, fill notification | 100% | 90 days | Bottleneck identification, SLO validation |
| **Toxicity regime** | VPIN, flow imbalance, cancellation rate, trade size distribution | Real-time | 90 days | Dynamic spread adjustment, selective participation |
| **Lead-lag dynamics** | Spot returns, Polymarket price changes, at 10ms-5s lags | 10% sample | 90 days | Signal validation, alpha decay monitoring |
| **Inventory performance** | Position entry, holding period return, exit reason | 100% | 2 years | Strategy refinement, Kelly calibration |

*Table 47: Required telemetry specification*

---

## 5. Risk Surface Analysis

### 5.1 Market Microstructure Risk

#### 5.1.1 Fee Regime Evolution Risk

| Scenario | Probability | Impact | Mitigation |
|:---|:---|:---|:---|
| Fee rate increase (feeRate >10%) | 15-25% | **Severe**: Eliminates remaining ArbTaker viability | Real-time monitoring, strategy disable triggers |
| Exponent increase (exponent >2) | 10-20% | **Severe**: Expands high-fee regime | Probability filtering, maker preference |
| Maker rebate reduction (<100%) | 20-30% | **Moderate**: Compresses SpreadMaker economics | Multi-venue capability, cost structure adaptation |
| Minimum fee implementation | 10-15% | **Moderate**: Affects extreme-probability strategies | Dynamic threshold adjustment |

*Table 48: Fee regime evolution risk scenarios*

#### 5.1.2 Liquidity Evaporation Risk

| Trigger | Manifestation | Mitigation |
|:---|:---|:---|
| High volatility event | Order book depth -50-80%, spread +200-500% | **Real-time depth monitoring**, position size reduction |
| Settlement approach (<2 min) | Concentrated liquidity flight or sniper activity | **Time-to-expiry bucketing**, early position closure |
| Correlated market stress | Cross-venue liquidity contraction | **Correlation monitoring**, portfolio-level exposure caps |

*Table 49: Liquidity evaporation risk and mitigation*

#### 5.1.3 Adverse Selection/Toxic Flow Risk

| Detection Signal | Response | Implementation |
|:---|:---|:---|
| VPIN elevation above threshold | Spread widening, reduced size | Real-time calculation, automatic adjustment |
| Flow imbalance anomaly | Temporary liquidity withdrawal | Pattern recognition, 30-60 second pause |
| Post-fill negative autocorrelation | Inventory skew adjustment, strategy pause | Automated PnL attribution, feedback loop |

*Table 50: Adverse selection detection and response*

### 5.2 Model Risk

#### 5.2.1 Lead-Lag Model Decay

| Decay Mechanism | Detection | Response |
|:---|:---|:---|
| Competitive latency reduction | Fill rate decline, slippage increase | **Strategy deprecation** (LatencySnipe) |
| Platform infrastructure improvement | Reduced spot-Polymarket divergence | Signal quality monitoring, automatic disable |
| Market efficiency improvement | Correlation breakdown at short lags | Longer-horizon signal adaptation |

*Table 51: Lead-lag model decay management*

#### 5.2.2 Correlation Assumption Failure (LeggedHedge)

| Failure Mode | Historical Evidence | Mitigation |
|:---|:---|:---|
| Correlation convergence to 1.0 in stress | March 2020, May 2021, Nov 2022 crypto crashes | **Strategy deprecation** |
| Correlation inversion (hedge amplifies) | DeFi liquidation cascades | Not mitigable; avoid hedge construction |
| Regime-dependent correlation instability | Volatility clustering, tail events | Stochastic correlation modeling (complex, unvalidated) |

*Table 52: LeggedHedge correlation risk and resolution*

**Recommended resolution: Deprecation** (Section 3.4.4).

#### 5.2.3 Probability Model Misspecification

| Misspecification | Impact | Detection | Correction |
|:---|:---|:---|:---|
| Systematic bias in probability estimation | Consistent inventory accumulation against direction | PnL attribution by position direction | Bayesian updating, recalibration |
| Volatility regime blindness | Spread inappropriate for current conditions | Realized volatility vs. implied comparison | Regime-switching spread model |
| Behavioral bias (favorite-longshot, recency) | Suboptimal queue positioning | Historical calibration vs. outcomes | Debiasing transformations |

*Table 53: Probability model misspecification management*

### 5.3 Execution Risk

#### 5.3.1 Latency Variance in Python asyncio

| Source | Magnitude | Mitigation |
|:---|:---|:---|
| Garbage collection pauses | 10-100 ms | **PyPy or Cython evaluation**; pre-allocated object pools |
| GIL contention (CPU-bound operations) | 5-50 ms | Process isolation for CPU-intensive tasks |
| Event loop blocking (synchronous calls) | 10-500 ms | **Strict async/await discipline**, timeout enforcement |
| Network stack variance | 20-200 ms | **VPS colocation evaluation**, connection pooling |

*Table 54: Python asyncio latency sources and mitigation*

**Strategic response**: Accept latency variance for maker strategies; **evaluate C++ critical path** if any latency-sensitive components retained.

#### 5.3.2 Order Book Stale Data Exposure

| Failure Mode | Detection | Response |
|:---|:---|:---|
| WebSocket disconnection | Heartbeat timeout, sequence gap | Automatic reconnection with state resync |
| Message loss (sequence gap) | Sequence number validation | **Strategy halt** until gap resolution |
| Processing delay (backpressure) | Queue depth threshold | Shedding with priority preservation |
| Timestamp staleness | Maximum age enforcement | **Data rejection**, conservative fallback |

*Table 55: Stale data exposure management*

#### 5.3.3 Partial Fill/Unhedged Leg Risk

| Scenario | Current Handling | Required Enhancement |
|:---|:---|:---|
| ArbTaker: one leg fills, other rejected | Unspecified | **Immediate cancellation**, position limit enforcement |
| ArbTaker: partial fill on both legs | Unspecified | **Atomic execution requirement** or explicit unhedged limit |
| SpreadMaker: partial fill, inventory drift | Inventory skew adjustment | **Real-time reconciliation**, automatic rebalancing |

*Table 56: Partial fill risk enhancement requirements*

### 5.4 State/Reconciliation Risk

#### 5.4.1 Position Tracking Drift

| Drift Source | Detection | Correction |
|:---|:---|:---|
| Missed fill notification | Reconciler periodic check | Manual reconciliation, position reset |
| Duplicate fill counting | Idempotency key validation | Duplicate rejection, audit log |
| Order status interpretation error | State machine validation | Explicit state transitions, invariant checks |

*Table 57: Position drift management*

#### 5.4.2 PnL Attribution Errors

| Error Source | Impact | Prevention |
|:---|:---|:---|
| Fill-to-strategy misassignment | Incorrect strategy evaluation | Explicit strategy tagging, validation |
| Fee miscalculation | EV estimation error | Real-time fee query, post-trade verification |
| Timing mismatch (mark-to-market) | Intraday PnL distortion | Consistent timestamp convention, snapshot protocol |

*Table 58: PnL attribution accuracy*

#### 5.4.3 Settlement Resolution Mismatch

| Scenario | Current Handling | Enhancement |
|:---|:---|:---|
| Oracle delay beyond expiry | Heuristic fallback disabled | **Graceful degradation** with uncertainty quantification |
| Significant oracle-spot divergence | Authoritative resolution path | Dispute monitoring, automatic position protection |
| Platform resolution override | Undocumented | **Resolution policy monitoring**, manual intervention protocol |

*Table 59: Settlement resolution risk management*

### 5.5 Infrastructure/Operations Risk

#### 5.5.1 WebSocket Disconnection/Backpressure

| Condition | Current | Required |
|:---|:---|:---|
| Disconnection detection | Implicit (message timeout) | **Explicit heartbeat, state machine** |
| Reconnection logic | Unspecified | Exponential backoff, state resync |
| Backpressure handling | "Event queue dropped counter" | **Shedding policy, priority preservation, recovery** |
| Message persistence | Unspecified | **At-least-once delivery guarantee** |

*Table 60: WebSocket resilience enhancement*

#### 5.5.2 Chainlink Oracle Unavailability

| Scenario | Impact | Mitigation |
|:---|:---|:---|
| Temporary unavailability (<5 min) | Strategy halt, opportunity cost | **Fallback to heuristic mode** with explicit uncertainty |
| Extended unavailability (>5 min) | Complete strategy disable | Manual override, position flattening protocol |
| Data quality degradation (stale, anomalous) | Erroneous signal generation | **Multi-source validation**, anomaly detection |

*Table 61: Chainlink dependency risk management*

#### 5.5.3 Rate Limit Breach Cascades

| Limit Type | Current Handling | Enhancement |
|:---|:---|:---|
| Polymarket API (requests/minute) | Unspecified | **Token bucket with pre-emptive throttling** |
| Polymarket API (orders/minute) | Unspecified | **Order queue with priority and batching** |
| Binance WebSocket (connection) | Unspecified | **Connection pool, automatic failover** |
| Chainlink (query frequency) | Unspecified | **Caching, subscription optimization** |

*Table 62: Rate limit management enhancement*

### 5.6 Dependency/Vendor Risk

#### 5.6.1 Polymarket API Instability

| Risk | Monitoring | Response |
|:---|:---|:---|
| Undocumented schema changes | Version pinning, diff monitoring | **Graceful degradation**, manual adaptation |
| Endpoint deprecation | Documentation monitoring, community channels | **Alternative endpoint preparation** |
| Availability degradation | Health endpoint, synthetic testing | **Circuit breaker**, manual trading capability |

*Table 63: Polymarket API risk management*

#### 5.6.2 Binance/Coinbase Feed Disruption

| Scenario | Fallback | Enhancement |
|:---|:---|:---|
| Binance WebSocket failure | Coinbase REST | **Coinbase WebSocket primary**, additional venues |
| Both primary sources failure | Chainlink (delayed) | **Manual signal generation capability** |
| Systematic venue anomaly | Cross-venue validation | **Anomaly detection, automatic disable** |

*Table 64: Data feed redundancy enhancement*

#### 5.6.3 Polygon Network Congestion

| Impact | Detection | Response |
|:---|:---|:---|
| Transaction submission delay | Mempool monitoring, gas price tracking | **Dynamic gas pricing**, submission retry |
| Transaction failure (out of gas) | Receipt validation | **Automatic resubmission**, position reconciliation |
| Extended congestion (>10 min) | Block time monitoring | **Strategy pause**, manual intervention |

*Table 65: Polygon network risk management*

### 5.7 Legal/Regulatory/Platform Risk

#### 5.7.1 Polymarket Georestriction Enforcement

| Jurisdiction | Restriction | PolyBawt Exposure |
|:---|:---|:---|
| United States (CFTC) | Platform geoblocking, user prohibition | **Depends on operator location** |
| Canada, EU | Regional access, KYC requirements | VPS location optimization required  |
| Other jurisdictions | Varying | **Legal review recommended** |

*Table 66: Georestriction risk assessment*

#### 5.7.2 CFTC/SEC Prediction Market Jurisdiction

| Development | Implication | Monitoring |
|:---|:---|:---|
| CFTC licensing (2024-2025) | Platform legitimacy, regulatory engagement | **Ongoing enforcement posture** |
| Election market controversy (2024) | Political sensitivity, potential restriction | **Platform policy changes** |
| SEC classification debate | Security vs. commodity uncertainty | **Regulatory guidance, legal review** |

*Table 67: Regulatory jurisdiction risk*

#### 5.7.3 Platform Terms of Service Violation Risk

| Activity | ToS Status | Mitigation |
|:---|:---|:---|
| Automated trading | **Permitted with rate limits** | Compliance monitoring, limit adherence |
| Multiple account coordination | Prohibited (wash trading) | **Single account operation** |
| API abuse (excessive polling) | Prohibited | **Efficient subscription, caching** |

*Table 68: Terms of service compliance*

---

## 6. Graded Subsystem Assessment (1–10 Scale)

| Subsystem | Score | Assessment | Critical Gap | Improvement Priority |
|:---|:---|:---|:---|:---|
| **6.1 Strategy Logic** | **4/10** | LatencySnipe obsolete; ArbTaker fragile; SpreadMaker viable but unvalidated; LeggedHedge flawed | **Toxicity detection unvalidated**; dynamic fee integration incomplete | Immediate (Weeks 1-2) |
| **6.2 Alpha Integrity** | **3/10** | No deterministic backtest; edge estimates theoretical; no live validation | **Absence of replay infrastructure**; no empirical Sharpe estimation | Critical (Weeks 3-6) |
| **6.3 Risk Controls** | **6/10** | Kelly sizing present; circuit breakers functional; VPIN basic | **Dynamic fee integration incomplete**; toxicity-adjusted sizing absent | High (Weeks 2-4) |
| **6.4 Execution Reliability** | **5/10** | asyncio foundation sound; OrderManager, Reconciler present | **Latency variance uncharacterized**; no SLO enforcement | High (Weeks 4-6) |
| **6.5 Reconciliation/State Consistency** | **5/10** | Reconciler component; position tracking; event bus | **Formal state machines absent**; idempotency unvalidated | High (Weeks 5-6) |
| **6.6 Observability** | **6/10** | Structured logging (NDJSON), metrics, health endpoint; queue drop counter added | **Latency monitoring insufficient**; microstructure data capture absent | Medium (Weeks 2-4) |
| **6.7 Testing Maturity** | **3/10** | Unit tests implied; no systematic integration testing | **No deterministic replay**; no exchange simulator; paper trading unvalidated | Critical (Weeks 3-6) |
| **6.8 Production Readiness** | **4/10** | Safety-first architecture appropriate; pre-launch phase acknowledged | **Critical validation gaps** before live deployment; single-strategy dependency risk | Critical (Weeks 1-6) |

*Table 69: Comprehensive subsystem assessment*

**Composite Score: 4.5/10** — Pre-production prototype with foundational elements but **critical gaps in validation, latency optimization, and edge verification** before institutional-grade deployment.

---

## 7. Phased Implementation Roadmap

### 7.1 Phase 1: Survival & Validation (Weeks 1–2) → Target Score ≥6

#### 7.1.1 LatencySnipe Deprecation/Transformation

| Aspect | Specification |
|:---|:---|
| **Why it matters** | LatencySnipe has **negative expected value** under current fee regime; continued development wastes resources and risks losses |
| **Technical approach** | Halt all development; **transform signal generation into SpreadMaker trigger** for liquidity placement timing; archive code with obsolescence documentation |
| **Expected impact** | Eliminates negative-EV strategy; preserves signal investment; redirects resources to viable edge |
| **Complexity/effort** | Low (2-3 days) |
| **Validation methodology** | Code removal confirmation; signal integration test with paper trading |
| **Exit criteria** | Zero LatencySnipe orders in live or paper trading; signal generation integrated with SpreadMaker |

#### 7.1.2 SpreadMaker-First Deployment

| Aspect | Specification |
|:---|:---|
| **Why it matters** | SpreadMaker is **sole structurally viable strategy**; focused deployment enables rapid validation and data collection |
| **Technical approach** | Deploy with **conservative parameters**: single market (BTC-USD 15m), **maker-only posting**, $5 position, **2% minimum spread**, slow requote (5-10 second cadence) |
| **Expected impact** | Establishes baseline maker performance; generates data for toxicity calibration; validates rebate capture |
| **Complexity/effort** | Medium (1 week) |
| **Validation methodology** | 100+ fills with full telemetry; preliminary profitability assessment; fill rate, spread capture, inventory distribution analysis |
| **Exit criteria** | **100+ fills with telemetry**; positive realized PnL over 48-hour minimum; fill rate >50%; no single-day drawdown >2% |

#### 7.1.3 Minimum Viable Telemetry Expansion

| Aspect | Specification |
|:---|:---|
| **Why it matters** | Current NDJSON + metrics insufficient for edge validation and optimization |
| **Technical approach** | Implement **core logging schema** (Section 9.1): market data events, order events, fill events with slippage decomposition, risk events, strategy events; establish **90-day retention** with query access |
| **Expected impact** | Enables post-trade analysis, adverse selection estimation, strategy refinement |
| **Complexity/effort** | Medium (1 week) |
| **Validation methodology** | Log completeness verification (100% order/fill capture); query functionality test; 7+ days operational retention |
| **Exit criteria** | All critical events logged with specified fields; query response <5 seconds for 7-day filter; no data loss in 48-hour stress test |

### 7.2 Phase 2: Fee-Aware Optimization (Weeks 3–4) → Target Score ≥7

#### 7.2.1 Dynamic Fee Integration & Real-Time EV Calculation

| Aspect | Specification |
|:---|:---|
| **Why it matters** | Fee-aware EV estimation exists but **validation and integration incomplete**; prevents fee-prohibitive execution |
| **Technical approach** | **Real-time fee calculator**: probability estimation from order book mid, dynamic fee lookup, net EV computation with slippage and adverse selection haircut; **strategy gate** preventing execution when net EV < threshold |
| **Expected impact** | Eliminates negative-EV trades; enables probability-regime-aware strategy selection |
| **Complexity/effort** | Medium (3-5 days) |
| **Validation methodology** | Unit test with known probability/fee pairs; integration test with simulated market conditions; backtest comparison pre/post integration |
| **Exit criteria** | <5% fee estimate error vs. actual; 100% of trades with positive ex-ante EV; automatic suppression at fee-prohibitive probabilities |

#### 7.2.2 Adverse Selection Detection (VPIN Enhancement)

| Aspect | Specification |
|:---|:---|
| **Why it matters** | **Critical unmet need** acknowledged in project overview; SpreadMaker viability depends on toxicity management |
| **Technical approach** | **VPIN enhancement**: real-time calculation with time-to-expiry weighting; volume-synchronized measurement; **regime-specific thresholds** (normal, elevated, toxic); **dynamic spread adjustment** widening 0.5-2x based on toxicity |
| **Expected impact** | 20-40% reduction in adverse selection costs; improved risk-adjusted returns |
| **Complexity/effort** | Medium-High (1-2 weeks) |
| **Validation methodology** | Post-fill PnL by VPIN regime; toxic flow <25% in high-VPIN periods; correlation VPIN → subsequent returns |
| **Exit criteria** | VPIN calculation <100ms latency; threshold calibration with 500+ fill sample; demonstrable PnL improvement in high-toxicity regimes |

#### 7.2.3 Order Book Microstructure Data Capture

| Aspect | Specification |
|:---|:---|
| **Why it matters** | Queue position value and microprice dynamics **unobserved**; critical for spread optimization |
| **Technical approach** | **Order book snapshots** at quote submission: bid/ask levels with size, **microprice calculation** (volume-weighted or tick imbalance), **queue position estimation** (price level + time priority), fill probability model |
| **Expected impact** | Improved fill rates, reduced adverse selection, optimized spread positioning |
| **Complexity/effort** | Medium (1 week) |
| **Validation methodology** | Fill rate by estimated queue position; microprice vs. subsequent returns correlation; queue depth predictability |
| **Exit criteria** | >60% fill rate for top-quartile queue positions; microprice signal with >55% directional accuracy; <10% queue position estimate error |

### 7.3 Phase 3: Institutional Hardening (Weeks 5–6) → Target Score ≥8

#### 7.3.1 Deterministic Replay/Backtest Harness

| Aspect | Specification |
|:---|:---|
| **Why it matters** | **Most critical gap**: no strategy validation without live capital risk; no regression testing for strategy changes |
| **Technical approach** | **Exchange simulator**: Polymarket CLOB matching engine replication, dynamic fee model, latency injection with configurable distribution; **event log ingestion** with state reconstruction; **48-hour minimum replay window** |
| **Expected impact** | Enables systematic strategy validation, regression testing, competitive analysis; reduces live testing risk |
| **Complexity/effort** | High (2 weeks) |
| **Validation methodology** | Capture-replay test: 24-hour live operation replayed, state comparison; known-opportunity validation; discrepancy investigation |
| **Exit criteria** | **<5% PnL divergence replay vs. live**; 100% state coverage in tests; zero invariant violations; 48-hour replay in <4 hours wall time |

#### 7.3.2 State Machine Formalization & Idempotency

| Aspect | Specification |
|:---|:---|
| **Why it matters** | Position drift, duplicate execution, state inconsistency create **unquantified operational risk** |
| **Technical approach** | **Formal state machines**: Order lifecycle (Created→Pending→Open→PartiallyFilled→Filled/Cancelled/Rejected), Strategy state (Idle→Armed→Active→Cooldown→Disabled), Risk state (Normal→Caution→Restricted→Halt); **UUID7 idempotency keys** with deterministic hash fallback; **exactly-once processing** with deduplication |
| **Expected impact** | Near-zero state-related incidents; deterministic recovery from failures; audit trail completeness |
| **Complexity/effort** | Medium-High (1.5 weeks) |
| **Validation methodology** | Model checking; fuzz testing with state transition injection; chaos testing with failure simulation |
| **Exit criteria** | 100% state transition coverage; zero invariant violations in 10,000+ test scenarios; <1 second recovery from any state |

#### 7.3.3 Latency Budget Enforcement & SLO Monitoring

| Aspect | Specification |
|:---|:---|
| **Why it matters** | Python asyncio latency variance **uncharacterized**; no enforcement of performance requirements |
| **Technical approach** | **End-to-end instrumentation**: signal generation, order preparation, submission, acknowledgment, fill notification; **histogram aggregation** with p50/p99/p99.9; **alerting thresholds** at 50ms (warning), 100ms (critical); **automatic strategy degradation** on SLO breach |
| **Expected impact** | Predictable execution performance; bottleneck identification; competitive positioning assessment |
| **Complexity/effort** | Medium (1 week) |
| **Validation methodology** | Synthetic benchmark with known latency injection; alert firing verification; 7-day operational latency distribution |
| **Exit criteria** | **95% of orders <50ms acknowledgment**; 99% <100ms; alerting <5 second detection; automatic degradation <10 second response |

### 7.4 Phase 4: Competitive Positioning (Weeks 7–10) → Target Score ≥9

#### 7.4.1 Infrastructure Latency Optimization (VPS/Colocation Evaluation)

| Aspect | Specification |
|:---|:---|
| **Why it matters** | Python asyncio **10-100x latency disadvantage** vs. C++ competitors; infrastructure investment may be justified for specific components |
| **Technical approach** | **VPS benchmarking**: QuantVPS, AWS regional optimization (Canada Central 1, EU West 1/2 for georestriction compliance); **C++ critical path prototype** for order submission; **cost-benefit analysis** with explicit performance thresholds |
| **Expected impact** | 10-50ms latency reduction if justified; informed architecture decision; potential competitive parity in specific dimensions |
| **Complexity/effort** | Medium-High (2 weeks evaluation, 4+ weeks implementation if pursued) |
| **Validation methodology** | Latency measurement across configurations; C++ prototype benchmark; ROI analysis with edge improvement assumptions |
| **Exit criteria** | **<10ms end-to-end with optimized configuration** (if pursued); explicit decision document with cost-benefit; implementation plan if justified |

#### 7.4.2 Multi-Strategy Ensemble with Dynamic Allocation

| Aspect | Specification |
|:---|:---|
| **Why it matters** | Single-strategy dependency creates concentration risk; dynamic allocation optimizes capital deployment |
| **Technical approach** | **Regime detection**: volatility, volume, time-to-expiry, toxicity; **strategy performance tracking**: realized Sharpe, drawdown, correlation; **Kelly-based dynamic allocation** with uncertainty penalty; **70% SpreadMaker, 20% ArbTaker (extreme-probability), 10% experimental** initial weighting |
| **Expected impact** | 10-20% improvement in risk-adjusted returns; reduced single-strategy risk; adaptive market condition response |
| **Complexity/effort** | High (2-3 weeks) |
| **Validation methodology** | Out-of-sample performance vs. static allocation; regime detection accuracy; allocation stability |
| **Exit criteria** | Sharpe improvement >0.2 vs. SpreadMaker-only; <20% allocation volatility; positive contribution from non-SpreadMaker components |

#### 7.4.3 Machine Learning Integration for Toxicity Prediction

| Aspect | Specification |
|:---|:---|
| **Why it matters** | **ML toxicity detection extends competitive half-life** beyond rule-based approaches; Python ecosystem advantage |
| **Technical approach** | **Feature engineering**: order book imbalance, flow characteristics, trade size distribution, time-of-day, inventory level; **gradient-boosted or neural model** with online learning; **real-time inference** <50ms; **A/B testing** vs. VPIN baseline |
| **Expected impact** | 30-50% reduction in toxic flow fills; extended competitive durability; potential edge differentiation |
| **Complexity/effort** | Very High (3-4 weeks) |
| **Validation methodology** | AUC-ROC >0.7 on validation set; live PnL improvement vs. baseline; feature importance stability |
| **Exit criteria** | **AUC-ROC >0.75**; toxic flow <15% with ML filter; <5% performance degradation 30 days post-deployment |

### 7.5 Phase 5: Scale & Diversification (Weeks 11–12) → Target Score ≥9.5

#### 7.5.1 Cross-Market Arbitrage (Kalshi, Other Venues)

| Aspect | Specification |
|:---|:---|
| **Why it matters** | Fee regime diversification; regulatory jurisdiction spreading; expanded opportunity set |
| **Technical approach** | **Kalshi API integration** (CFTC-regulated, U.S.-accessible); **unified risk management** across venues; **fee structure comparison** for optimal execution routing; **correlation monitoring** for portfolio construction |
| **Expected impact** | 20-40% increase in trade opportunities; reduced single-venue risk; regulatory optionality |
| **Complexity/effort** | High (2-3 weeks) |
| **Validation methodology** | Positive EV verification on ≥2 venues; execution quality comparison; correlation stability |
| **Exit criteria** | **Profitable operation on ≥2 venues**; <30% correlation between venue returns; regulatory compliance verified |

#### 7.5.2 Predictive Settlement Modeling

| Aspect | Specification |
|:---|:---|
| **Why it matters** | Residual edge in settlement timing; information advantage through data integration |
| **Technical approach** | **Oracle update prediction**: Chainlink aggregation round monitoring, preliminary state analysis; **settlement outcome modeling**: spot microstructure, cross-venue information, historical resolution patterns; **maker-based execution** capturing settlement-boundary volume |
| **Expected impact** | Early-mover advantage in settlement trading; rebate capture from elevated boundary volume |
| **Complexity/effort** | Very High (3-4 weeks) |
| **Validation methodology** | Prediction accuracy vs. market consensus; PnL attribution to settlement-specific trades; model calibration stability |
| **Exit criteria** | **>55% accuracy on resolution timing**; positive PnL from settlement-specific strategy; model confidence calibration |

#### 7.5.3 Institutional Capital Structure

| Aspect | Specification |
|:---|:---|
| **Why it matters** | Scaling beyond personal capital requires external capital access with appropriate structure |
| **Technical approach** | **Legal entity formation** (LLC, LP as appropriate); **investor agreements** with fee structure, liquidity terms; **audited track record** preparation; **third-party audit** of systems and performance |
| **Expected impact** | Capital access for strategy capacity; fee income from management/performance; institutional credibility |
| **Complexity/effort** | Very High (4-6 weeks, parallel with technical development) |
| **Validation methodology** | Legal review; investor due diligence; audit completion |
| **Exit criteria** | **$100K+ committed capital**; legal structure operational; audit opinion unqualified; investor reporting infrastructure |

---

## 8. Technical Specifications

### 8.1 Deterministic Replay/Backtest Harness

#### 8.1.1 Exchange Simulator Architecture

```
┌─────────────────────────────────────────┐
│         Replay Controller               │
│  - Event log ingestion                  │
│  - Time-scaled or accelerated playback  │
│  - State checkpoint/restore             │
└─────────────────┬───────────────────────┘
                  │
    ┌─────────────┼─────────────┐
    ▼             ▼             ▼
┌────────┐   ┌────────┐   ┌────────┐
│ Market │   │ Order  │   │ Fill   │
│  Data  │   │  Book  │   │ Engine │
│  Feed  │   │        │   │        │
└────────┘   └────────┘   └────────┘
    │             │             │
    └─────────────┴─────────────┘
                  │
                  ▼
         ┌─────────────┐
         │   Strategy  │
         │   Under Test│
         └─────────────┘
```

**Core components**: (1) Event log parser with schema validation; (2) Market data reconstruction with microprice calculation; (3) Order book simulation with price-time priority; (4) Fill engine with latency modeling; (5) Strategy interface matching production.

#### 8.1.2 Event Log Format & Ingestion

**NDJSON format with mandatory fields**:

```json
{
  "timestamp_ns": 1707494400000000000,
  "event_type": "market_data|order|fill|risk|strategy",
  "source": "polymarket_ws|binance_ws|coinbase_rest|chainlink",
  "payload": {...},
  "sequence_number": 1234567,
  "checksum": "sha256:..."
}
```

**Ingestion rate target**: 100,000 events/second for historical replay.

#### 8.1.3 State Reconstruction Guarantees

| Guarantee | Mechanism | Validation |
|:---|:---|:---|
| Determinism | Identical initial state + event sequence → identical final state | Bit-identical comparison across replays |
| Recoverability | 1-minute interval checkpoints enable mid-replay restart | Checkpoint restore test |
| Divergence detection | Automatic flagging of strategy behavior changes between versions | Version A/B comparison |

*Table 70: State reconstruction guarantees*

### 8.2 Event Schema & Idempotency

#### 8.2.1 Core Event Types

| Event Type | Key Fields | Retention |
|:---|:---|:---|
| **MarketData** | timestamp, source, symbol, bids[], asks[], trade_seq | 90 days full, 2 years aggregated |
| **Order** | order_id, client_order_id, side, size, price, strategy_id, idempotency_key | 2 years |
| **Fill** | fill_id, order_id, fill_price, fill_size, slippage_metrics | 2 years |
| **RiskEvent** | event_type, threshold, action, position_snapshot | 2 years |
| **StrategyEvent** | strategy_id, transition, signal, confidence | 90 days |

*Table 71: Core event types and retention*

#### 8.2.2 Idempotency Key Generation

| Method | Application | Format |
|:---|:---|:---|
| **UUID7** (time-ordered) | Primary keys | Standard UUID7 encoding |
| **Deterministic hash** | Duplicate detection | SHA-256(strategy_id, timestamp, symbol, side, price, size) |

*Table 72: Idempotency key methods*

#### 8.2.3 Exactly-Once Processing Semantics

| Component | Implementation |
|:---|:---|
| Deduplication | Idempotency key database with 24-hour TTL |
| Delivery | At-least-once with duplicate detection |
| Outcome logging | Acknowledgment with retry on failure |
| Reconciliation | Orphan detection loop for unacknowledged operations |

*Table 73: Exactly-once processing components*

### 8.3 Backpressure Policy

#### 8.3.1 Queue Saturation Detection

| Metric | Threshold | Action |
|:---|:---|:---|
| Event queue depth | >1,000 | Alert; reduce non-critical logging |
| Event queue depth | >5,000 | Shed by strategy priority |
| Processing latency p99 | >100ms | Degrade to conservative sizing |
| Memory utilization | >80% | Emergency position flattening |

*Table 74: Queue saturation detection thresholds*

#### 8.3.2 Shedding Strategies (Priority Order)

| Priority | Strategy | Rationale |
|:---|:---|:---|
| 1 (preserve) | Risk events, order status | Safety-critical |
| 2 (preserve) | SpreadMaker signals | Primary revenue source |
| 3 (sample) | Market data | 10% sampling, age-based shedding |
| 4 (shed) | ArbTaker signals (extreme-probability only) | Conditional viability |
| 5 (shed) | LatencySnipe (deprecated) | Non-viable |

*Table 75: Strategy-aware shedding priority*

#### 8.3.3 Recovery & Catch-Up Mechanisms

| Phase | Action |
|:---|:---|
| Saturation resolution | Resume from checkpoint |
| Accelerated replay | 2-5x speed for buffered events |
| Selective skip | Non-critical historical data omitted |
| Gradual re-enablement | Health verification before full activation |

*Table 76: Recovery and catch-up mechanisms*

### 8.4 Latency Budget & SLOs

| Metric | Target | Alert | Critical | Action |
|:---|:---|:---|:---|:---|
| **Order acknowledgment** | <50ms | >100ms | >250ms | Circuit break |
| **Fill notification** | <100ms | >250ms | >500ms | Position verification |
| **Data freshness** | <25ms | >75ms | >150ms | Feed failover |
| **End-to-end strategy** | <150ms | >300ms | >500ms | Strategy pause |

*Table 77: Latency SLO specification*

### 8.5 State Machines

#### 8.5.1 Order Lifecycle

```
Created → Pending → Open → PartiallyFilled → Filled
   ↓         ↓        ↓          ↓
Rejected  Cancelled  Expired    (terminal)
```

**Transitions triggered by**: exchange response, fill notification, timeout, risk event, manual intervention.

#### 8.5.2 Strategy State

```
Idle → Armed → Active → Cooldown ─→ Disabled
        ↓        ↓         ↓           ↑
     (signal)  (entry)  (exit/timeout) (manual/risk)
```

#### 8.5.3 Risk State

```
Normal → Caution → Restricted → Halt
   ↑        ↓          ↓
 (recovery) (improvement)   (manual)
```

---

## 9. Data Collection Plan

### 9.1 Core Logging Schema (NDJSON)

#### 9.1.1 Market Data Events

```json
{
  "timestamp_ns": 1707494400000000000,
  "source": "binance_ws",
  "symbol": "BTCUSDT",
  "event_type": "order_book_update",
  "bids": [[50000.00, 1.5], [49999.50, 2.0]],
  "asks": [[50000.50, 1.2], [50001.00, 3.5]],
  "microprice": 50000.23,
  "spread_bps": 10.0,
  "queue_position_estimate": null,
  "sequence": 123456789
}
```

#### 9.1.2 Order Events

```json
{
  "timestamp_ns": 1707494400000000000,
  "order_id": "pm_abc123",
  "client_order_id": "polybawt_20260209_123456_abc",
  "strategy_id": "spread_maker_btc_15m",
  "side": "sell",
  "size": 5.00,
  "price": 0.55,
  "order_type": "LIMIT",
  "time_in_force": "GTC",
  "post_only": true,
  "idempotency_key": "sha256:..."
}
```

#### 9.1.3 Fill Events

```json
{
  "timestamp_ns": 1707494400000000000,
  "fill_id": "fill_xyz789",
  "order_id": "pm_abc123",
  "fill_price": 0.5505,
  "fill_size": 5.00,
  "fill_time_ns": 1707494400050000000,
  "slippage_vs_intended_bps": 5.0,
  "slippage_vs_arrival_bps": 2.0,
  "venue_latency_us": 5000,
  "queue_position_at_submit": 3
}
```

#### 9.1.4 Risk Events

```json
{
  "timestamp_ns": 1707494400000000000,
  "event_type": "daily_loss_limit_approaching",
  "threshold": 0.04,
  "current": 0.038,
  "action": "position_reduction",
  "position_snapshot": {...},
  "pnl_snapshot": {...}
}
```

#### 9.1.5 Strategy Events

```json
{
  "timestamp_ns": 1707494400000000000,
  "strategy_id": "spread_maker_btc_15m",
  "event_type": "signal_generated",
  "signal": "quote_update",
  "confidence": 0.75,
  "signal_latency_us": 15000,
  "market_state": {...}
}
```

### 9.2 Retention & Sampling

| Data Category | Retention | Sampling | Rationale |
|:---|:---|:---|:---|
| Orders, fills | 2 years | 100% | Complete audit trail, regulatory |
| Risk events | 2 years | 100% | Safety-critical, incident analysis |
| Strategy events | 90 days | 100% | Strategy refinement, short-term |
| Market data | 90 days full, 2 years aggregated | 10% time-weighted | Storage efficiency, trend analysis |
| Telemetry (latency, etc.) | 90 days | 100% | Performance optimization |

*Table 78: Data retention and sampling specification*

### 9.3 Derived Metrics Estimation

#### 9.3.1 Adverse Selection

$$\text{AS} = \frac{(P_{t+\Delta t} - P_{\text{fill}}) \times Q - E[P_{t+\Delta t} - P_{\text{fill}} | \mathcal{I}] \times Q}{\text{Trade Size}}$$

where $\Delta t \in \{1\text{s}, 5\text{s}, 15\text{s}, 60\text{s}, 300\text{s}\}$ enables decay profile estimation. Aggregated by strategy, time-to-expiry bucket, toxicity regime.

#### 9.3.2 Slippage Decomposition

$$\text{Fill Slippage} = \underbrace{(P_{\text{fill}} - P_{\text{signal}})}_{\text{execution delay}} + \underbrace{(P_{\text{signal}} - P_{\text{arrival mid}})}_{\text{signal latency}} + \underbrace{(P_{\text{arrival mid}} - P_{\text{intended}})}_{\text{market impact}}$$

#### 9.3.3 Lead-Lag Alpha Decay

$$\rho(\tau) = \text{Corr}(r_{\text{spot}}(t), r_{\text{polymarket}}(t+\tau))$$

for $\tau \in \{10\text{ms}, 50\text{ms}, 100\text{ms}, 500\text{ms}, 1000\text{ms}, 5000\text{ms}\}$. Peak correlation and decay rate inform strategy timing.

#### 9.3.4 Fill Quality by Strategy

| Metric | Calculation | Target |
|:---|:---|:---|
| Fill rate | Fills / Quotes submitted | >60% SpreadMaker, >30% ArbTaker |
| Partial fill rate | Partial fills / Total fills | <10% |
| Cancellation rate | Cancellations / Quotes submitted | <50% (platform limit awareness) |
| Time-to-fill | Fill time - Submit time | <30s median for competitive quotes |

*Table 79: Fill quality metrics and targets*

---

## 10. Critical Findings (Ranked)

### 10.1 LatencySnipe Strategy Non-Viability Under New Fee Regime

**Severity: CRITICAL | Confidence: HIGH**

The January 7, 2026 dynamic fee implementation with **3.15% peak taker fee at 50% probability structurally eliminates LatencySnipe economics** . Historical profitability (e.g., 0xalberto $764/day on $200) is **irreproducible**; continued development represents **misallocated resources and unquantified loss risk**. **Immediate deprecation required**.

### 10.2 Python asyncio Latency Disadvantage vs. C++ Competitors

**Severity: CRITICAL | Confidence: HIGH**

Industry analysis explicitly identifies Python as **"likely to be too slow"** for competitive arbitrage . **10-100x latency gap** vs. C++/colocated infrastructure is **structural, not addressable through optimization**. Strategic response: **abandon latency competition; exploit Python's analytical ecosystem for prediction-dependent, maker-focused strategies**.

### 10.3 SpreadMaker as Sole Durable Edge Source

**Severity: HIGH | Confidence: MEDIUM-HIGH**

**Only SpreadMaker retains plausible positive EV** through maker rebates and fee avoidance. **Viability contingent on validated toxicity detection**—currently unmet critical need. **Unvalidated deployment risks rapid capital erosion** through adverse selection.

### 10.4 Absence of Deterministic Backtesting Infrastructure

**Severity: HIGH | Confidence: HIGH**

**Most critical capability gap**: no replay harness, no exchange simulator, no systematic validation. **Strategy development depends on live testing with unacceptable capital risk**. **Phase 3 (Weeks 5-6) critical path priority**.

### 10.5 Insufficient Adverse Selection Detection

**Severity: HIGH | Confidence: HIGH**

Project overview **explicitly acknowledges** "toxicity/adverse-selection validation under real flow" as unmet need. **VPIN implementation basic, uncalibrated**. **Deployment without enhancement invites systematic losses**.

### 10.6 LeggedHedge Model Risk Unquantified

**Severity: MEDIUM-HIGH | Confidence: HIGH**

**"Likely weakest and most model-risky component"** per project overview. Correlation breakdown during stress periods **well-documented, likely dominates any theoretical edge**. **Deprecation recommended**.

### 10.7 Chainlink Oracle Dependency as Single Point of Failure

**Severity: MEDIUM | Confidence: MEDIUM-HIGH**

Live mode **requires Chainlink availability** with no documented fallback. Unavailability scenarios (network congestion, node failure, platform integration issue) **disable trading during potentially profitable periods**. **Graceful degradation enhancement required**.

---

## 11. Top 10 Immediate Actions

| Priority | Action | Owner | Deadline | Validation |
|:---|:---|:---|:---|:---|
| **1** | **Halt LatencySnipe development; document obsolescence** | Strategy Lead | Day 2 | Code review, team notification |
| **2** | **Implement real-time dynamic fee calculator** | Quant Dev | Day 5 | Unit test, integration test |
| **3** | **Deploy minimum viable SpreadMaker with maker-only posting** | Strategy Lead | Day 7 | 48-hour paper trading, 100+ fills |
| **4** | **Establish sub-50ms latency monitoring & alerting** | Infra Lead | Day 10 | Synthetic benchmark, alert test |
| **5** | **Capture order book microprice & queue position estimates** | Quant Dev | Day 14 | Data quality review |
| **6** | **Build deterministic event replay for 48-hour windows** | Platform Eng | Week 3 | Known-opportunity validation |
| **7** | **Formalize idempotency keys for all order operations** | Platform Eng | Week 3 | Fuzz test, duplicate injection |
| **8** | **Implement backpressure with strategy-aware shedding** | Infra Lead | Week 4 | Load test, graceful degradation |
| **9** | **Establish 90-day raw log retention with query access** | Data Eng | Week 4 | Query performance test |
| **10** | **Evaluate C++ core/latency-critical path vs. full Python retention** | Architecture | Week 6 | Prototype benchmark, ROI analysis |

*Table 80: Top 10 immediate actions checklist*

---

## 12. Data Source Catalog

### 12.1 Primary Market Data

| Source | URL | Data Type | Latency | Notes |
|:---|:---|:---|:---|:---|
| **Polymarket CLOB WebSocket/API** | https://docs.polymarket.com/ | Real-time order book, trades | <50ms | Official; schema evolution risk |
| **Binance Spot WebSocket** | https://binance-docs.github.io/apidocs/spot/en/#websocket-market-streams | Spot prices, trades | <100ms | Reliable; georestriction considerations |
| **Coinbase REST** | https://docs.cdp.coinbase.com/exchange/reference/exchangerestapi_getproducts | Spot prices, trades | 200-500ms | Fallback; rate limits apply |

*Table 81: Primary market data sources*

### 12.2 Oracle & Settlement

| Source | URL | Purpose | Update Frequency |
|:---|:---|:---|:---|
| **Chainlink Data Streams** | https://docs.chain.link/data-streams | Settlement prices, oracle updates | Deviation-triggered + heartbeat |
| **Polymarket Resolution Subgraph** | https://thegraph.com/hosted-service/subgraph/polymarket/matic-markets | Market resolutions, outcomes | Event-driven (1-5 min indexing) |

*Table 82: Oracle and settlement data sources*

### 12.3 Historical & Analytics

| Source | URL | Specialization | Access |
|:---|:---|:---|:---|
| **Dune Analytics** | https://dune.com/polymarket | Aggregated statistics, wallet analysis | SQL query; community dashboards |
| **Flipside Crypto** | https://flipsidecrypto.xyz/ | Blockchain-derived metrics | SQL query; API available |
| **Amberdata** | https://amberdata.io/ | Institutional-grade market data | Commercial API; $500-5000/month |
| **QuantVPS Analysis** | https://www.quantvps.com/blog/polymarket-hft-traders-use-ai-arbitrage-mispricing | Competitive intelligence, infrastructure guidance | Free; vendor perspective |

*Table 83: Historical and analytics sources*

### 12.4 Competitive Intelligence

| Source | URL | Relevance | Notes |
|:---|:---|:---|:---|
| **gabagool222 Arbitrage Bot** | https://github.com/gabagool222/15min-btc-polymarket-trading-bot | Open-source competitive implementation | Pre-fee era; architecture reference |
| **0xalberto Case Study** | On-chain analysis (Dune/Flipside) | Documented successful latency arbitrage | $764 daily on $200; **pre-fee, non-reproducible** |

*Table 84: Competitive intelligence sources*

---

## 13. Assumptions & Uncertainty Register

### 13.1 High-Confidence Assumptions (Multiple Source Validation)

| Assumption | Evidence | Implication if Invalid |
|:---|:---|:---|
| Dynamic fee formula: fee = C × 10% × (p·(1-p))² |  | Fundamental strategy economics invalid |
| Maximum fee ~3.15% at p=0.50 | Consistent across sources | LatencySnipe EV assessment requires revision |
| Fee applicability: takers only, 15-min crypto markets |  | SpreadMaker viability contingent |
| Maker rebate 100% redistribution |  | SpreadMaker economics require revision |

*Table 85: High-confidence assumptions*

### 13.2 Medium-Confidence Assumptions (Single Source or Inference)

| Assumption | Source | Uncertainty Reduction |
|:---|:---|:---|
| Python asyncio 10-100x latency disadvantage vs. C++ |  | Benchmark measurement with PolyBawt implementation |
| SpreadMaker 60-90 day competitive half-life | Inference from  | Systematic tracking of spread compression, rebate yield |
| 2.5-3% minimum arbitrage spread post-fees |  | Transaction cost analysis with actual gas, slippage |

*Table 86: Medium-confidence assumptions*

### 13.3 Low-Confidence Assumptions (Speculative, Require Telemetry)

| Assumption | Rationale | Required Telemetry |
|:---|:---|:---|
| VPIN toxicity indicator effectiveness | Theoretical foundation; unvalidated implementation | Labeled flow analysis: inventory PnL by VPIN decile |
| Chainlink update latency <5 seconds typical | Documentation; unmeasured in production | Timestamp correlation: Chainlink vs. spot vs. economic tradability |
| Order book depth $2.1M average sustained | Q3 2025 measurement  | Real-time depth monitoring with regime identification |

*Table 87: Low-confidence assumptions*

### 13.4 Telemetry Collection Priorities for Uncertainty Reduction

| Priority | Uncertainty | Telemetry | Timeline |
|:---|:---|:---|:---|
| 1 | Adverse selection magnitude | Post-fill PnL by regime, 500+ trades | 2-3 weeks |
| 2 | Slippage decomposition | Latency decomposition by component | 2-3 weeks |
| 3 | Lead-lag correlation stability | Cross-correlation at multiple horizons | 3-4 weeks |
| 4 | Chainlink-spot timing | Timestamp correlation analysis | 2-3 weeks |
| 5 | Queue position value | Fill rate by estimated position | 3-4 weeks |

*Table 88: Telemetry collection priorities*

