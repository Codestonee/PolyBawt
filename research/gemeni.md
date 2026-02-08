# **PolyBawt: Institutional Grade Audit, Microstructure Analysis, and Strategic Remediation Report**

**Date:** February 8, 2026

**Subject:** Operational Viability and Architectural Remediation of "PolyBawt" in Polymarket 15-Minute Crypto Binary Markets

**To:** Lead Architect, PolyBawt Project

## ---

**1\. Executive Summary**

This comprehensive audit evaluates the "PolyBawt" asynchronous trading system against the evolved market microstructure of Polymarket's 15-minute cryptocurrency binary contracts as of February 2026\. The central finding of this research is that the system's foundational thesis—specifically the **ArbTaker** and **LatencySnipe** strategies—faces an existential threat from the "dynamic fee" regime introduced in January 2026\.1 The shift from a zero-fee environment to a taker-fee model peaking at **1.56%** (at 50% probability) has structurally inverted the Expected Value (EV) of aggression-based strategies. The system, in its current state, is paying a "retail tax" to sophisticated market makers who now operate under a protective umbrella of fee-curve weighted rebates.2

However, the research identifies a clear, albeit challenging, path to institutional-grade viability. The "edge" in this market has migrated from taking liquidity to providing it. To survive, PolyBawt must pivot its core logic from identifying arbitrage to capturing rebates through a **SpreadMaker** strategy. This transition requires a sophisticated upgrade to the risk engine, specifically the integration of **VPIN (Volume-Synchronized Probability of Informed Trading)** to detect and evade "toxic flow" within the high-volatility "swing zones" of 15-minute markets.3

Technically, while the Python asyncio stack remains viable, the current implementation exhibits critical latency bottlenecks in event loop management and data ingestion. The audit reveals that without immediate remediation—specifically the adoption of uvloop, backpressure conflation, and sequence-based message tracking—the system is susceptible to "ghost fills" and state desynchronization during the sub-second volatility windows characteristic of Chainlink-settled markets.4

The remediation plan outlined herein targets a migration to a **Rebate-Capture Market Making** model. By leveraging the new rebate economics and implementing "news guard" logic derived from Chainlink Data Stream signals, the system can target a Sharpe ratio \>2.5.6 This report details the mathematical impossibility of the current taker strategies, the rigorous implementation details required for a maker-centric pivot, and the architectural hardening necessary to compete with professional firms in the 2026 regime.

## ---

**2\. Market Microstructure Analysis (The 2026 Regime)**

To accurately assess the viability of PolyBawt, one must first dissect the hostile and highly specific environment of the 15-minute crypto binary market. As of early 2026, this market has matured into a complex ecosystem defined by dynamic fee structures, sophisticated settlement mechanics, and aggressive adverse selection.

### **2.1 The Fee Cliff: Dynamic Taker Economics**

The most significant structural alteration to the Polymarket landscape is the transition from a growth-focused low-fee model to a revenue-generating dynamic fee model. Introduced on January 5, 2026, this regime specifically targets 15-minute crypto markets, intending to curb latency arbitrage by taxing the "taker" flow that previously exploited stale quotes.1

#### **2.1.1 The Mathematical Reality of 1.56%**

Unlike traditional perpetual futures where fees are typically flat basis points (e.g., 2–5 bps), Polymarket’s fee structure is a function of probability. The fee peaks when the outcome uncertainty is highest.

**The Fee Function:** The taker fee is dynamic and scales with the price (probability) of the outcome token. At a price of **$0.50** (50% probability), the fee hits its ceiling of **1.56%** of the notional value.7

![][image1]  
For a standard "ArbTaker" trade, which implies a round-trip execution (opening and closing a position, or buying both legs), the friction is immense. Consider a trade executed near the money (![][image2]):

* **Entry Fee:** \~1.56%  
* **Exit Fee:** \~1.56%  
* **Total Friction:** \>3.12%

In the context of a 15-minute binary option, the underlying asset (e.g., Bitcoin) rarely moves enough in seconds to justify paying a 3.12% premium purely for execution. This fee structure effectively kills "scratch" trading or thin-edge arbitrage. A strategy must now capture an edge significantly greater than 3% to be net profitable, which is a rarity in efficient markets.

#### **2.1.2 Implications for PolyBawt**

The ArbTaker component of PolyBawt, which scans for misalignments where Bid\_YES \+ Bid\_NO \> 1.00, acts as a taker on both legs. Under the old regime, a combined bid of ![][image3] yielded a risk-free ![][image4] profit. Under the 2026 regime:

* Buy YES at ![][image5] (Fee: ![][image6]) ![][image7] Cost: ![][image8]  
* Buy NO at ![][image5] (Fee: ![][image6]) ![][image7] Cost: ![][image8]  
* **Total Cost:** ![][image9]  
* **Payout:** ![][image10]  
* **Net Loss:** ![][image11]

The strategy effectively locks in a 2.5% loss on every "arbitrage" opportunity detected, unless the misalignment is massive (e.g., combined bid \> ![][image12]), which implies a catastrophic failure of market makers or extreme toxicity that PolyBawt is likely ill-equipped to handle.

### **2.2 The Rebate Opportunity: Fee-Curve Weighting**

Conversely, the market structure now heavily subsidizes liquidity provision through a "fee-curve weighted" rebate program.2 This is the inverse of the taker tax and represents the primary revenue source for institutional participants.

**The Rebate Formula:** Since January 19, 2026, rebates are calculated using a specific "fee-equivalent" formula that mirrors the taker fee curve.2

![][image13]  
This formula has profound strategic implications:

1. **The "Liquidity Smile":** The rebate is maximized when ![][image14]. The term ![][image15] creates a bell curve of profitability. Providing liquidity at ![][image16] or ![][image17] yields negligible rebates.  
2. **Incentivized Risk:** The market explicitly pays makers to stand in the "line of fire"—the 50/50 probability zone where uncertainty and volatility are highest. This is where the price swings are most violent, and where adverse selection is most acute.

**Strategic Pivot:** The "edge" for PolyBawt lies in providing liquidity in the ![][image18] **probability range**. Here, the system can capture the spread *plus* the maximum rebate tier. However, this zone is also the "kill zone" for informed traders. Success depends entirely on the ability to differentiate between "uninformed noise" (which we want to trade against) and "informed flow" (which we must avoid).

### **2.3 Settlement Mechanics: The Chainlink "Pull" Race**

The mechanism of settlement in 15-minute markets has evolved to utilize **Chainlink Data Streams** and **Automation**.6 This is a critical microstructure detail that differentiates Polymarket from centralized exchanges.

#### **2.3.1 The "Pull" Architecture**

Unlike a "Push" oracle that updates on-chain at fixed intervals (heartbeats), Chainlink Data Streams utilize a "Pull" model:

1. **Off-Chain Observation:** A Decentralized Oracle Network (DON) observes Binance/Coinbase prices in real-time.  
2. **Signature Generation:** The DON signs a report with the current price and timestamp.  
3. **On-Chain Trigger:** At the exact second of market expiry (e.g., 10:15:00), a Chainlink Automation node (or any incentivized keeper) submits this signed report to the Polymarket Exchange contract to resolve the market.

#### **2.3.2 The Sniper Race**

This architecture creates a measurable latency gap—a "sniper race."

* **The Gap:** There is a delta between the *observation* of the price (off-chain) and the *resolution* (on-chain).  
* **The Threat:** Sophisticated High-Frequency Trading (HFT) firms monitor the individual Chainlink nodes and the Binance order book directly. They know the settlement price *milliseconds to seconds* before the on-chain transaction confirms.  
* **PolyBawt's Vulnerability:** If PolyBawt relies on the standard crypto\_prices\_chainlink WebSocket feed 9, it is receiving data that has already been aggregated and broadcast. HFTs running local Chainlink nodes or sniffing the DON's P2P traffic will have "lookahead" information. In the final seconds of a 15-minute contract, if the price is near the strike, these snipers will aggressively pick off any stale maker quotes that are on the wrong side of the looming settlement tick.

### **2.4 Liquidity and Depth Analysis**

Liquidity in these markets is not uniform. Understanding the depth profile is essential for sizing positions and estimating slippage.

* **BTC/ETH 15m Contracts:** Typical depth is heavily concentrated at the Best Bid/Offer (BBO).  
* **The "Microprice" Effect:** Research indicates that in high-frequency order books, the "Microprice" (a volume-weighted adjustment of the mid-price) is a better predictor of future price movement than the simple mid-price.10  
* **Depth Decay:** Liquidity tends to evaporate in the final 60 seconds before expiry as makers pull quotes to avoid the "Chainlink Sniper" risk described above.  
* **Competition:** Evidence suggests the presence of professional market makers (Wintermute, etc.) who utilize "Builder Program" tiers with higher rate limits (Partner/Enterprise) to update quotes at sub-second frequencies.11 PolyBawt, operating likely at a lower tier, faces a "rate limit latency" disadvantage.

## ---

**3\. Strategy Audit & Edge Verdicts**

This section rigorously evaluates the four component strategies of the PolyBawt ensemble against the 2026 microstructure reality.

### **3.1 Strategy 1: ArbTaker (Short Arb)**

* **Logic:** Monitor the CLOB for situations where the cost of buying both outcome tokens (YES \+ NO) is less than the payout ($1.00). Specifically, Ask\_YES \+ Ask\_NO \< 1.00 (for long arb) or selling into a combined bid \> 1.00.  
* **Status:** **CRITICAL FAILURE / OBSOLETE**  
* **Edge Verdict:** **Negative EV (-100% Reliability)**  
* **Half-Life:** Expired (Jan 5, 2026).  
* **Analysis:** As detailed in Section 2.1, the fee hurdle (\~3.1% round trip) makes this strategy mathematically impossible in normal market conditions. "Combined Bid \> 1.00" anomalies usually only occur during extreme volatility events (flash crashes) where liquidity on one side is pulled, leaving a "stale" bid.  
* **The "Legging" Risk:** Attempting to hit a stale bid during a crash often results in a partial fill. The bot gets filled on the "toxic" side (the side crashing) but fails to execute on the hedge side. The result is a naked, losing position in a crashing market.  
* **Taxation:** This strategy is heavily "taxed" by the exchange fees. It effectively acts as a donation mechanism to the rebate pool.  
* **Recommendation:** **Decommission immediately.**

### **3.2 Strategy 2: LatencySnipe (Spot-Future Arb)**

* **Logic:** Listen to Binance Spot WebSocket. If Spot moves \> ![][image19], buy the corresponding Polymarket outcome before the CLOB updates.  
* **Status:** **FRAGILE / TAXED**  
* **Edge Verdict:** **Negative Net EV (Gross Positive, Net Negative)**  
* **Half-Life:** \< 50ms.  
* **Analysis:**  
  * **The Latency War:** You are competing against firms with collocated servers in Tokyo (AWS Northeast) for Binance and potentially direct fiber connections to Polygon RPC nodes. A Python bot over the public internet (even fiber) typically has a "tick-to-trade" latency of **20–100ms**.4 HFTs operate in the **1–10ms** range (or microseconds for FPGA setups).  
  * **The Fee Hurdle:** To profit, the spot price must move enough to cover the **1.56% taker fee** *plus* the spread. A 1.5% move in BTC in a single second is a "tail event" (rare).  
  * **Adverse Selection:** When a move is large enough to cover the fee (e.g., a CPI print), the makers will have already pulled or widened their quotes (using "News Guard" logic). The only quotes remaining will likely be "traps" or too small to cover the gas/fee costs.  
* **Recommendation:** Only viable during extreme, scheduled macro news events (CPI, FOMC), but likely rate-limited by the exchange during those exact moments. Likely not worth the infrastructure cost for a retail-sized bot.

### **3.3 Strategy 3: SpreadMaker (Passive Quoting)**

* **Logic:** Post limit orders on both sides (Bid/Ask) to capture the spread.  
* **Status:** **HIGHEST POTENTIAL (Conditional)**  
* **Edge Verdict:** **Positive EV (Conditional on VPIN integration)**  
* **Half-Life:** Durable (Structural Edge).  
* **Analysis:**  
  * **Revenue Source:** Spread capture \+ **Maker Rebates**.2 The rebates alone (up to \~0.5-0.75% effective yield per trade) can constitute the primary profit driver.  
  * **Risk:** Adverse Selection (Toxic Flow). If BTC jumps $500, snipers will hit your stale quotes before you can cancel.  
  * **Requirement:** The bot needs a "toxicity filter" (VPIN) to widen spreads or pull quotes *before* the toxic flow arrives, or immediately upon detecting a microstructure imbalance.  
* **Recommendation:** This must become the core engine of PolyBawt. It transforms the bot from a "gambler" into a "casino" (liquidity provider).

### **3.4 Strategy 4: LeggedHedge (Crash Detection)**

* **Logic:** Enter a position and hedge via another instrument (Perps) or the opposite leg later.  
* **Status:** **MODEL RISK HIGH**  
* **Edge Verdict:** **Fragile / Unknown**  
* **Analysis:**  
  * **Gamma Risk:** Hedging a binary option with a linear instrument (Perps) is mathematically treacherous. As expiry nears and the price approaches the strike, the "Gamma" (rate of change of Delta) approaches infinity. A small move in Spot can flip the binary Delta from 0 to 1 instantly. A linear hedge cannot react fast enough, leaving the portfolio exposed to massive variance.  
* **Recommendation:** Too complex for the current risk engine maturity. Deprioritize until the SpreadMaker engine is robust.

## ---

**4\. Quantitative Scenarios & Bankroll Projections**

Given the fee structure, we model the **SpreadMaker** strategy's potential as the sole viable path.

**Assumptions:**

* **Market:** BTC 15m Binary.  
* **Trade Frequency:** 20 round trips per hour (high turnover due to 15m expiry cycles).  
* **Size:** $20 exposure (Max constraint per user specs).  
* **Rebate:** \~0.5% effective (conservative estimate based on fee-curve weights in the 35-65% prob range).  
* **Spread Captured:** $0.01 (1 cent) on average.

### **4.1 Unit Economics (Per Trade)**

| Metric | Value | Notes |
| :---- | :---- | :---- |
| **Gross Spread** | $0.20 | $20 size \* $0.01 spread |
| **Rebate Income** | $0.10 | $20 size \* 0.5% rebate |
| **Adverse Selection Loss** | \-$0.15 | Estimated 50% of spread lost to informed traders (without VPIN) |
| **Net EV (Pre-Optimization)** | **$0.15** | Per successful round trip |

### **4.2 Bankroll Growth Scenarios (Daily)**

The following scenarios assume a highly active bot participating in multiple 15-minute windows throughout the day.

| Bankroll | Trades/Day | Gross Spread | Rebates | Adv. Select Loss | Net Daily EV | Daily ROI |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **$100** | 50 | $10.00 | $5.00 | \-$7.50 | **$2.50** | **2.5%** |
| **$250** | 100 | $20.00 | $10.00 | \-$15.00 | **$5.00** | **2.0%** |
| **$500** | 200 | $40.00 | $20.00 | \-$30.00 | **$10.00** | **2.0%** |

**Risk Context & Confidence Bands:**

* **High Confidence:** The Rebate income is deterministic based on volume and the published formula.  
* **Low Confidence:** The "Adverse Selection Loss." This is the variable that kills market makers. Without a robust VPIN filter (see Section 5), this loss could easily double to **\-$0.30 per trade**, rendering the strategy negative EV (-$0.00 per trade).  
* **Scaling Limit:** The ROI diminishes with scale due to "queue position." With a $20 max trade, PolyBawt is a "small fish" and can likely get filled. As size increases, the bot must sit in the queue longer, increasing the probability of being picked off.

## ---

**5\. Risk Surface & VPIN Integration Plan**

The "SpreadMaker" strategy survives only if it avoids toxic flow. We introduce the **VPIN (Volume-Synchronized Probability of Informed Trading)** metric as the primary risk control mechanism.

### **5.1 The Risk Hierarchy**

| Rank | Risk Type | Severity (1-10) | Description |
| :---- | :---- | :---- | :---- |
| **1** | **Microstructure (Adverse Selection)** | **10** | Getting "run over" by informed traders (snipers) milliseconds before a price move. |
| **2** | **Execution (State Desync)** | **9** | Believing the bot is flat when it is actually long, due to missed WebSocket messages or sequence gaps. |
| **3** | **Regulatory/Platform** | **7** | Polymarket changing fee tiers or banning API access for aggressive quoting/cancellations. |
| **4** | **Model Risk (Inventory)** | **6** | Accumulating a large position in a losing outcome due to mean reversion failure. |
| **5** | **Infra/Ops** | **5** | Cloudflare bans or VPS latency spikes. |

### **5.2 Implementing VPIN for 15m Markets**

VPIN estimates the toxicity of order flow. High VPIN means informed traders are active; the bot should **widen spreads** or **stop trading**.

**The Algorithm for PolyBawt:**

1. **Volume Bucketing:** Do not sample by time (which is noisy). Sample by "Volume Buckets" (e.g., every $5,000 traded).  
2. **Trade Classification (Bulk Volume):**  
   For each bucket, classify volume as Buy (![][image20]) or Sell (![][image21]) initiated.  
   * *Method:* Use the standard **Tick Rule** (if price \> prev\_price, it's a Buy; if price \< prev\_price, it's a Sell).  
   * *Refinement:* Use Polymarket's maker field in trade data. If Maker \== PolyBawt, the other side is the Aggressor.  
3. **Calculate VPIN:**  
   The VPIN metric is calculated over a rolling window of ![][image22] buckets (e.g., ![][image23], approximating 1 hour of volume).  
   ![][image24]  
4. **Signal Generation (The Actionable Logic):**  
   * **Normal Regime (![][image25]):** Informed trading is low. Quote standard spreads (e.g., ![][image4]).  
   * **Toxic Regime (![][image26]):** Informed trading is rising. Widen spreads by 2x (e.g., ![][image27]) to compensate for higher risk.  
   * **Critical Regime (![][image28]):** High probability of a directional breakout or crash. **Pull all quotes immediately** and wait for volatility to subside.

**Theoretical Basis:** The implementation of VPIN is derived from the "Toward Black Scholes for Prediction Markets" research 3, which explicitly links VPIN spikes to "swing zone" toxicity in binary markets.

## ---

**6\. Technical Remediation Roadmap**

To achieve "Institutional-Grade" status (score \>= 9/10), the Python architecture requires significant hardening. The current asyncio implementation is likely suffering from "event loop blocking" during high-traffic moments.

### **Phase 1: The "Must-Dos" (Weeks 1-4)**

*Goal: System Stability & Data Integrity (Score 6 \-\> 8\)*

#### **Task 1.1: Event Loop Optimization (uvloop)**

* **Why:** The standard Python asyncio loop has high overhead for IO-bound tasks.  
* **Approach:** Replace the default loop with uvloop, a Cython wrapper over libuv (used by Node.js).  
* **Impact:** Reduces internal event loop latency by \~20-40% and increases throughput.12  
* **Implementation:**  
  Python  
  import asyncio  
  import uvloop  
  asyncio.set\_event\_loop\_policy(uvloop.EventLoopPolicy())

#### **Task 1.2: WebSocket Sequence Tracking**

* **Why:** "Ghost orders" occur when the bot misses a "Filled" message due to network jitter. The bot thinks it still has an open order, but it has actually been filled.  
* **Approach:**  
  * In the on\_message handler for the Polymarket WebSocket (wss://ws-subscriptions-clob.polymarket.com/ws/market).  
  * Track the sequence field included in every message.  
  * **Logic:** if new\_seq \> last\_seq \+ 1: trigger\_resync().  
* **Validation:** Intentionally disconnect the network for 2 seconds in a test environment, then verify that the bot logs "Gap Detected" and triggers an automatic REST API snapshot fetch to reconcile state.13

#### **Task 1.3: Backpressure Policy (Conflation)**

* **Why:** During market crashes, the WebSocket emits thousands of updates per second. An unbounded asyncio.Queue will consume all RAM or delay processing of the *latest* price, causing the bot to trade on stale data.5  
* **Approach:** Implement a **LIFO (Last-In-First-Out) Conflating Queue** for price updates.  
  * If the consumer is busy and new prices arrive, overwrite the old "pending" price with the new one. The bot only cares about the *current* price, not the history of the last 50ms.  
  * *Note:* Critical events like OrderUpdate (Fills) must remain FIFO (First-In-First-Out) and never be dropped.

### **Phase 2: Professionalization (Weeks 5-8)**

*Goal: Competitive Edge & Alpha Integrity (Score 8 \-\> 9+)*

#### **Task 2.1: Deterministic Replay Harness**

* **Why:** You cannot debug a race condition in production. You must be able to replay a crash to see how the bot reacted.  
* **Approach:**  
  * Log all incoming WebSocket frames (raw JSON) with receive timestamps to a journal.ndjson file.  
  * Create a ExchangeSimulator class that mocks the CLOB API.  
  * Feed the journal.ndjson into the bot at 10x speed.  
  * **Exit Criteria:** The simulator must accurately reproduce the bot's PnL and state changes given historical input data.

#### **Task 2.2: Idempotency Keys**

* **Why:** Retrying a "Place Order" HTTP request on a timeout can lead to double fills if the first request actually succeeded but the response was lost.  
* **Approach:**  
  * Polymarket API uses signed orders. The nonce acts as a unique ID.  
  * **Strict Rule:** Never re-sign an order with a new nonce if the previous status is unknown.  
  * **Logic:** If a request times out, query GET /order/{id} (using the computed OrderID derived from the signature) to check status before retrying.14

## ---

**7\. Data Collection Plan (Practical)**

To validate the "Edge Verdict" and fine-tune the VPIN thresholds, the bot must log high-fidelity post-trade data.

**Schema: post\_trade\_analysis.csv**

| Field | Description | Rationale |
| :---- | :---- | :---- |
| trade\_id | Unique Trade ID | Reconciliation with exchange records. |
| strategy\_tag | e.g., "SpreadMaker\_v2" | Attribution of PnL to specific logic versions. |
| entry\_timestamp | Unix micros | Precise latency calculation. |
| arrival\_mid\_price | Mid price *at moment of decision* | Critical for calculating **Slippage** (Difference between decision price and fill price). |
| exec\_price | Actual fill price | Cost analysis. |
| slippage\_bps | (exec \- arrival) / arrival | Key metric for execution quality. High slippage \= slow bot. |
| vpin\_1m | VPIN value at entry | Did we trade during a toxic period? Used to calibrate filters. |
| microprice\_5s | Imbalance-adjusted price 5s later | **Mark-out PnL.** |
| pnl\_realized | Net USD | Bottom line performance. |
| fee\_paid | Taker fee (or negative for rebate) | Validation of fee curve assumptions. |

**The "Mark-out PnL" Metric:**

The most important metric for a market maker is the "Mark-out." Calculate the theoretical PnL of the trade 5 seconds, 15 seconds, and 60 seconds *after* execution.

* If PnL\_5s is consistently negative, it means you are being adversely selected (likely by latency arbitrageurs).  
* If PnL\_5s is flat or positive, your queue position is healthy, and you are capturing the spread effectively.

## ---

**8\. "Top 10 Immediate Actions" Checklist**

1. \*\*\*\* **Disable ArbTaker logic immediately.** It cannot survive the 1.56% fee regime.  
2. **\[Infra\]** **Install uvloop** and replace the default asyncio event loop policy.  
3. \*\*\*\* **Hard-code a "News Guard":** Stop trading 2 minutes before and 2 minutes after major macro events (CPI, FOMC) until VPIN is fully calibrated.  
4. **\[Execution\]** **Implement nonce tracking** in a persistent store (Redis or local LMDB) to prevent double-ordering on bot restarts.  
5. \*\*\*\* **Enable sequence checking** on the WebSocket market channel to detect gaps.  
6. \*\*\*\* **Build the ExchangeSimulator** using recorded NDJSON data to enable backtesting.  
7. **\[Account\]** **Apply for the "Builder Program"** (Unverified Tier initially) to ensure correct rate limits are applied to your API keys.15  
8. **\[Code\]** **Split Queues:** Separate MarketDataQueue (Conflated/LIFO) from OrderUpdateQueue (FIFO) to prevent price updates from blocking trade confirmations.  
9. **\[Monitoring\]** **Add a gap\_detected metric** to your observability stack (Prometheus/Grafana) to alert on data loss.  
10. **\[Analysis\]** **Start logging arrival\_mid\_price** immediately to measure slippage and validate execution speed.

## ---

**Technical Appendix**

### **A. Sequence Tracking State Machine (Python Pseudo-code)**

Python

class SequenceMonitor:  
    def \_\_init\_\_(self):  
        self.last\_seq \= \-1  
        self.gap\_detected \= False

    def process\_packet(self, packet: dict):  
        current\_seq \= packet.get('sequence')  
        \# Keepalives or non-sequenced messages  
        if current\_seq is None:   
            return   
          
        \# Check for gap  
        if self.last\_seq\!= \-1 and current\_seq\!= self.last\_seq \+ 1:  
            self.gap\_detected \= True  
            logger.critical(f"GAP DETECTED: Expected {self.last\_seq \+ 1}, Got {current\_seq}")  
            \# Trigger State Reconciliation (REST Snapshot)  
            raise StreamGapError("Resync Required")  
              
        self.last\_seq \= current\_seq

### **B. VPIN Calculation (Simplified Logic)**

Python

def calculate\_vpin(trades\_df, bucket\_volume=1000):  
    """  
    trades\_df: DataFrame with \['price', 'size', 'side'\]  
    bucket\_volume: Volume per bucket (e.g., $1000)  
    """  
    \# 1\. Assign volume to buckets  
    trades\_df\['cum\_vol'\] \= trades\_df\['size'\].cumsum()  
    trades\_df\['bucket\_id'\] \= (trades\_df\['cum\_vol'\] // bucket\_volume).astype(int)  
      
    \# 2\. Aggregates per bucket  
    buckets \= trades\_df.groupby('bucket\_id').apply(  
        lambda x: pd.Series({  
            'buy\_vol': x.loc\[x\['side'\] \== 'BUY', 'size'\].sum(),  
            'sell\_vol': x.loc\[x\['side'\] \== 'SELL', 'size'\].sum()  
        })  
    )  
      
    \# 3\. Calculate VPIN over rolling window  
    \# Window of 50 buckets approx. equal to 1 hour of trading  
    buckets\['imbalance'\] \= (buckets\['buy\_vol'\] \- buckets\['sell\_vol'\]).abs()  
    buckets\['vpin'\] \= buckets\['imbalance'\].rolling(50).sum() / (50 \* bucket\_volume)  
      
    return buckets\['vpin'\].iloc\[-1\]

#### **Citerade verk**

1. Polymarket Introduces Dynamic Fees to Curb Latency Arbitrage in Short-Term Crypto Markets \- TradingView, hämtad februari 8, 2026, [https://www.tradingview.com/news/financemagnates:ab852684e094b:0-polymarket-introduces-dynamic-fees-to-curb-latency-arbitrage-in-short-term-crypto-markets/](https://www.tradingview.com/news/financemagnates:ab852684e094b:0-polymarket-introduces-dynamic-fees-to-curb-latency-arbitrage-in-short-term-crypto-markets/)  
2. Maker Rebates Program \- Polymarket Documentation, hämtad februari 8, 2026, [https://docs.polymarket.com/polymarket-learn/trading/maker-rebates-program](https://docs.polymarket.com/polymarket-learn/trading/maker-rebates-program)  
3. Toward Black Scholes for Prediction Markets: A Unified Kernel and Market Maker's Handbook \- ResearchGate, hämtad februari 8, 2026, [https://www.researchgate.net/publication/396693524\_Toward\_Black\_Scholes\_for\_Prediction\_Markets\_A\_Unified\_Kernel\_and\_Market\_Maker's\_Handbook](https://www.researchgate.net/publication/396693524_Toward_Black_Scholes_for_Prediction_Markets_A_Unified_Kernel_and_Market_Maker's_Handbook)  
4. How Latency Impacts Polymarket Bot Performance (And How to Reduce It) \- QuantVPS, hämtad februari 8, 2026, [https://www.quantvps.com/blog/how-latency-impacts-polymarket-trading-performance](https://www.quantvps.com/blog/how-latency-impacts-polymarket-trading-performance)  
5. Tackling Backpressure in Real-Time Trading Systems — Problems & Practical Solutions | by rupesh gupta | Medium, hämtad februari 8, 2026, [https://medium.com/@rupeshgupta0912/tackling-backpressure-in-real-time-trading-systems-problems-practical-solutions-3281ce19ce12](https://medium.com/@rupeshgupta0912/tackling-backpressure-in-real-time-trading-systems-problems-practical-solutions-3281ce19ce12)  
6. Polymarket turns to Chainlink oracles for resolution of price-focused bets | The Block, hämtad februari 8, 2026, [https://www.theblock.co/post/370444/polymarket-turns-to-chainlink-oracles-for-resolution-of-price-focused-bets](https://www.theblock.co/post/370444/polymarket-turns-to-chainlink-oracles-for-resolution-of-price-focused-bets)  
7. Polymarket Changelog, hämtad februari 8, 2026, [https://docs.polymarket.com/changelog/changelog](https://docs.polymarket.com/changelog/changelog)  
8. Data Streams: Low-Latency Market Data for DeFi & Beyond | Chainlink, hämtad februari 8, 2026, [https://chain.link/data-streams](https://chain.link/data-streams)  
9. RTDS Crypto Prices \- Polymarket Documentation, hämtad februari 8, 2026, [https://docs.polymarket.com/developers/RTDS/RTDS-crypto-prices](https://docs.polymarket.com/developers/RTDS/RTDS-crypto-prices)  
10. Overestimated effective spreads: Implications for investors \- Federation of European Securities Exchanges (FESE), hämtad februari 8, 2026, [https://www.fese.eu/app/uploads/2025/01/overestimated-effective-spreads.pdf](https://www.fese.eu/app/uploads/2025/01/overestimated-effective-spreads.pdf)  
11. Builder Tiers \- Polymarket Documentation, hämtad februari 8, 2026, [https://docs.polymarket.com/developers/builders/builder-tiers](https://docs.polymarket.com/developers/builders/builder-tiers)  
12. nexustrader \- PyPI, hämtad februari 8, 2026, [https://pypi.org/project/nexustrader/](https://pypi.org/project/nexustrader/)  
13. Data Feeds \- Polymarket Documentation, hämtad februari 8, 2026, [https://docs.polymarket.com/developers/market-makers/data-feeds](https://docs.polymarket.com/developers/market-makers/data-feeds)  
14. Make Your API Idempotent, Avoid Ruining Clients Lives, hämtad februari 8, 2026, [https://apisyouwonthate.com/blog/idemptoency-keys/](https://apisyouwonthate.com/blog/idemptoency-keys/)  
15. Builder Profile & Keys \- Polymarket Documentation, hämtad februari 8, 2026, [https://docs.polymarket.com/developers/builders/builder-profile](https://docs.polymarket.com/developers/builders/builder-profile)

[image1]: <data:image/png;base64,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>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEsAAAAZCAYAAAB5CNMWAAADG0lEQVR4Xu2Wy6vNURTHl0TkUR7Jq1ySIkUJkYGEkkd5FRm4I0pKJkQpJf+AjEgykMjAiBidUhITA0Yo5FHKgJDI4/u1fuv81m+d/fudc8+pe7v1+9S3e8/a+6yz99prrb1FampqaoYvm6AL0CXoDfQk+5826gg0pTl7aJgGrYJmQCPDWIoR0ExpnTtO0nvh/I78L4R2QVeh11B/9pk6B/2EvkGbs/mDTR/0VPTg7kPPC6NpxkMN6Ct0S/Tw6YN72ZZP+08f9ECK/pf6CRFzvi/YyTLRYP0VzcLB5DD0J9h4uOdFs6EM7ue25NWxVTSrIuZ/vbPR/yep8D8P+gAtiQNgh2igfkFrwliKUdDoaAxwIZXpLvkBcuEelsozaGqwe/hdZhQPugzvnwEy6J8VVuqfqcmATAp2boqnyLFHkq53zwLRudQ7aG9xuMls6HQ0BuwAuXAPN8lMbxeIdsHy/hkgw4JY+t2zohuMzIfei9Y6U7kKzn0JLYbmQAdFF7LIT8pgtrIEqrDyTwWLa61ajwVrhWivZe9dLcVs9v5TwUr6twhzAbwNqbfQb+gYNDGfWslJSacus+2e5Bn3WTrrfVws53cbrLvQTmfrF/3eGdGK8f47DpaVIPtALzBQpU1RtIQ3iva0TmBAuw1WCsskJsJcKfrvOFhWgjfjQBcwzennoej7rCwrGbjkYhy9lGEKqyC7/QZchjbAH2eP6QX6ugFdF33XvBAtudStx4Uej8ZAWYOfAH2Hlge756joLXfA2eyWs0CXNXj653urxb99od2Pd8Ie0TeNwZJcJ3qL+iY/GboDbXC2FLyZH0Nfgp0bewXNCnZPQzQobPL2jLFMsieQ9++fTBbUFv8XRZ3yAcfM6AXeOmOiMYMLOCVanpyXyrYUdhtbn+MBnIBWNmcosffwIBrQ9OzzWNH2wEvLv+DN/+Xss/lnUJswkowof8Qr9codapid10Qzl4fLKogXyQ/RDTMohOOHRJv5FclbwpZszEP/H6Xof3dhxjCDt+j+7O9ADpQXyXZorZRnPaHPbvzX1NTU1NTU1Hj+AV2RvJB0uyilAAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACIAAAAXCAYAAABu8J3cAAABKElEQVR4XmNgGAWjgHTACMQq6IJEAA4g1gFiHnQJNMAKxAoMEHuwApCEFhBvBOIHqFJ4ATMQ5wHxWyBeCKU7GSAWogOQ2lNAfIABh4N1gfgvEH8D4v9A/BBVGi/IYIDoc4LyNYD4GRBXwlUwMLhAxf4xQMw/wIDDITBgDMRfGYh3iB8DxOBJaOLlUHFsgCYOaWWAGAyyGBn4QsWxAZo4BJQm8DkEm2U0cchWhkHikAMMg8QhgyZqcCXWIKg4tkKLJg6JZoAYPAdNnObZdxoDJDr4oXxFIH4CxGuAmAWmiAFSrvxE4iMDvA6RZIAoQMcHGFA1XAHiG0CshiTGDcTzgfgpEKcB8R0g3gnEQkhqYB5ENx9XqJENQGkB5DhQVIFobGljFIyC4QEA+V1mzeBhcmwAAAAASUVORK5CYII=>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACIAAAAXCAYAAABu8J3cAAABiUlEQVR4Xu2VvytGYRTHv0IR5UcZLGIRZTMog0EWA8Wk7DIZyR9gsBpMJINFRgaL7ihWYjBQ2CyKsuB7Os+9Hee9z+t97yTdT3263fOc5z7f++N9XqCkpD466Tjt8wM10EJHaLsfcDTTftrg6hmH9J7u0mM6/WM0TiNdoS90Pxw3oQt6pPeCJogEbqWXtCecS9pXOpZ1xFmm73QynA/RZ7qedQBTofZJvxAJIovuQZstCfTuhl3dMgu98Jarr4V6HtEgXdCnMerq8phl0oyrWzagPbKwRebUHaSXPiAexC9iifWkQSoWQ5Ug8ujlFRQJcoL8nkJBJMBbOFpqCZIgv6dQkD/zamIf6wF00ryrW2Ifq8yRet6mFQ3SRI9QuYElyH9llkXohXdcvdDPV1iAXtRyTc9ph6ltQ19HWhugj9AbkRtKkX3lw5xbqgYRbugZNJA88lXolmy5ord00NTaoBviE12id/SUdpue9AchIbwVyH/DBDTInBv7DfkWJJzMlWPet1FS8j/4BggCbYxuRLA8AAAAAElFTkSuQmCC>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAWCAYAAAC7ZX7KAAACX0lEQVR4Xu2WTYiOURTHj1BEJMpHMhYsRLFBLH0UyZqyVOwsKLKzsfCVkgWiydQsxE5KsZiQhI3FNCWzGLFAFqahkI//7z33znOf+7zP9GSajZ5//Zr3nnvuveee595zx6xVq/9Lc8W6QFPNEfNyo3W3zRQrxNa8I2ixmJ7ZaC/LbB1dEqPiVuCMmF3y6K694o/5mGvipngjtqRO0j7xWQyIG+KRWJn0kyz6fohn5nM9FGOit3BzMfnpzMYGvoe+iUTABMgCZ8Vmq2ZpvnghFiS2HeK3OBXaMWA2zFwHxcLQVxHBHchsh80zl28kFwEP5MZMBHdHzEhsa8wzHjcSA2a+CVXnGD81n4VzWqcmAZ8wz1yqpWLE/Biut/o4KooDc8cYMH341Am/r2K/6BFHxE/zCxZFsHUBswZzxIDfiavmm3gq3ooNPsQ12YA3iudJe5r5hT2W2JoEzLg+cTz8RqvFRzEU2h1NNuBuYiyLUKZQk4C7KWYdn3FNRcBk/Vv4i/41YDLdb+4Ts97RdfOqkOqouSMVpE5xU6UMSLutuEyxfU/MGvfwPnwGxSIrMvnKinLGZefS5/PbLquWLwKlTtKHGHxRPBDLg41F34vzoR3F5u9b8fBwNOrq8GXz7JG0l2J74sNG2BB+JXFWhq388tC+a8WiZClm80KwUQmuiJOhjXrEa7EpsaFf5q9dVK/4JNaGNv5PxKrQZhNcQNbjIamIEsKLdSjAp1iS9HPmmPCL2JnYecVuW/Es0888uagcnGu+JCWQ27+t5GG2R3wwn++xeWbPWbN/EVq1ajUV+gsRN5VpTX2NPgAAAABJRU5ErkJggg==>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADYAAAAWCAYAAACL6W/rAAACyUlEQVR4Xu2WT6hNURTGP6GI8jd/SjxCiTIQL0oMDCiMiDIxYiJmxMDkZaAMEDIgGSkpKYqSbhRiYELPgEL+ZCARivz7vrfOemed/fa5uXdgdL76de/Ze5/v7LX3OmsfoFGjRv9T48kKMjPt+AeNIovI2LQjKPqPTPqkPWR02lgj3T8P5tlWF8kLcpZcJesqvfUaTnaTD+R88XsYQyc+EVV/jUv1h/wir8mrhDtkChlG1pP35Erxe4NMQ0Zryc6k7Qm5jvYrOJX0k2uwHXMdJx/Dtfx/h2tpMob6nySbClaT6WQzeURmFGMukH3Ff9cc8hbJZmgFzpE1sZFqwVZ1QdIetRG2ygokSg9Wu+T+aWBK2eg/giwtuwekYO6R3tDWIidgvi4twEuyIbRhAnlIlsRGWFppcpXBiQ7BxqQrqHs8MPf/WnYPKvprx9P35TTZi2oQN2H3HUS528qIN2ShD5K0Ylq5usDSSUcpBXNjPDDtivvXBZbe65pLlqeNsHf3COxe8QWWOTH4ASkgPbSbwFrIj4mBuX8ngWmSR1FfYVVVb5GfMA8Vl1WVEdRi8hndBeZpkY6Jgbl/J4HNhlXGnHRcvCPbYanrO6ditawcVr543QRWNyYG5v6dBLYL1pdTP9mPMvW0U89g4y/7IEnV6BKGnlst5FM0ahvM8EzSHqui+6cTVdA5f1VC7VZuISQtkhYr1QFYX0VbYZOM0jl2n4wrrvV7CrZL3uYpo4krAJfK//dwLf80MJ1j0d/V7p2Ucq+Ndk8fBSpmFU0it1F9yA+yJVx7egkv0TLsg02it2jrIc9hZ5dL/g9Q9Zd39Hcpc/SMusDko8KhLxmX3jsVkJWhbVDKXd2gndP5pPNDn0uuWeQxeUrmh/YxsCB0juyA5bs+ceKDJZX96P8NVX+XPhR0mGtncuohd8kncgzlh0T6KjVq1KhR5/oLSkG3mxWTyI0AAAAASUVORK5CYII=>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAYCAYAAAAYl8YPAAAAfUlEQVR4XmNgGAWjYHACWSDuBmIOdAlyQTkUUwWIAfF+IDZDlyAXgAw6AsQq6BI8QCxJBg4G4kdAzMmABCqggqTiZ0D8H4jjGSgE3EC8EIj70CVIBa5AvJoBzXvkABYGiIs80CXIAdJAvBmIRdAlyAGsQCwExIzoEqNggAEAkekYp+CjMnEAAAAASUVORK5CYII=>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADYAAAAWCAYAAACL6W/rAAACxklEQVR4Xu2Wy+tNURTHv8JAlGdeeSUUBpQyEJlIFAYmFIPfzB+giIGJDMSE5FdKMlISKc9IFyWPKZGoH3mEEDHw9v3etfc9e6977v113V9G51ufTmftvc/a6+y1195ApUqV/qdGkSVkmm9oo7FkuLMNJpOdLWoQmQXrU6ahZDZZCuvbSrGf5txWJ0kfOUrOkdVZa2ttJ7/IHXKEXCXfSW/aKWgeOQvzMylvqmsNeUNekA/kCVme9bBgYz99S8/LZGLaKWoV2eJsD8glMszZvRTYBXIY9o2peXNd+rsTyCLylTxDc2Dyc5HMSWyHyB/k8zgB85lqJnkFtxj6A8fIitRI1ch7MtfZveTEO2qldoHpXUEcT2zrgy2dRw0WcJqmGqtvrk1sGE3uwZymkgN9NOtcooEKTCsif1sTm3zLpjFxfkp12XahWEVl3EsyP7zXpT+hP9IqsP4mrfbX5DRZQK6Tpyhf6XaBlUkpqDnsRrFCSuv9wS6+kHVJe0PRWTeBqWjIoaRK9ZbcbfQo1Glgv2ErNNLZVbWvkZ+wOT5Hc5Gp/+XP+PfAvEbA9oHGenUS2GJYMfFB6ThShvTASn1cuY+wMQ3FjTdQgQ0hp9BdYNPJYzSfj9JDsgNF6mmldCzI35nYSYoT8edWDeUpmkpB64PvUJTpblZMxUBprYnHA1zFTft3SnhvNXYnrC3TRrLJ2XSO3UaRCnrqrNJKRpvG9ZHNKP7gONjYb+E9VX+BKSgVBt1a1C50DF2BBSiVbRv53kvOO3v9WnQDeT7r9rAheY+lV8QjYDy5SVaGdznoCX16gy2VyvIPWHFZ6No0Nn7fo4xSZkkqSiocY8K7pH2nArIssTWk3NUArdwesg35fU55f588Qn470ARl0zWsBrte7UN+Y4n72E+4Bkvd2Me3Rw6GPtIMcot8IgdQXCT8VqpUqVKlzvUXWsm6UCC0QLkAAAAASUVORK5CYII=>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADYAAAAWCAYAAACL6W/rAAACyUlEQVR4Xu2WS6hOURTH/0IREcmj5CIpMpEQKRPKhJSJIhPFRIbE6JYMqFuSKHlkJI+JgYEYfBlIMSQSRXlEIQMieazft/a+Z51zz/3ccwdG51f/vm+vvc/ea+299jpHamlp+d+MMS2qGkfABNMy0+RqR2Cuab1pjmlspQ9mydePTDVNqdgyjF1jmpn+10LHUtMN08tyV09wcL/po+lS+j1mGh8HGStNb00fTH9MXzTUmVem96YrprNyPz6ZlocxGTbouXzNZ6YzGrpm18BOrzB9lS8wEtjhJ6ab8hPLnDR9Tv9x/pTpYdHdZZ/8WebIPJCPJajjqnFUHiRznzONk58+/rJZe8O4Ek0D2yKfkEAiB5Md2LBOasc05QR/mzYEW0fuaC8IKD7Hxm019cvTtpamgR2VO0wgkc3JDix8IrXZ4QxrYWNspqN/B/bO9Fp+X0dM08BIwV6BDVdIcnpeVTnduCtot6nP9F1+EnEM874w3ZYXubXy1CR961K3S9PAOhpdYKvl6+B8hHu4KrQpQsyzM9hok4qbgi1nzvZgK9E0sDtqHhjBcCqPqh015Hnuq7g/tKupmO80GVRL08AotU0Cwzk2465pdqWvjnWmn/JXyJJkY97Hphl5kIr1hvW7aWA75BNSqSKxKmZ433EPJgbbBXl1BIoGzwwU3YMOUzAWJhv/8S8WGco843hd1NIrMHb7tPyUcloskKfFdZUrHuX/R2hTLA6Y5skdyrqnwmHmvKxyimWHsef5+R9PEA7Lx/UH2yAszoXk6Pk6oB3Ju4dyiWbMEflmUBBgvrxqXUxt2Gb6peL5LHZ4WhiH05NCm34qXvzy2Cif65B8fTaE1Obe9oVxXXIaVNVRcU94iAv/1LQ42QBHCOKNaY/8M+eWaXrqjy/oqqpfLFRBNuW8PDOYK6dqhmB2mb6Zrsl94gsmnmBLS0vL6PgLXVq8yJe4EVQAAAAASUVORK5CYII=>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACIAAAAXCAYAAABu8J3cAAABm0lEQVR4Xu2VTygFURTGzwulkEhKNkhCRMlCWUl2LJRSFnayspWdjWRrYYGSvY2Fhd0rZWNroVCUP6VsFCX5833unevMfe/NvCmvqPnq18w998w933v3zhmRVKmSKwPa/WARqgQ9oNqfUOLaQ6AJlHlzTkzqBvvgKjwVKS64AB7Brr2ugQqdBLWAY7AJjsA56NcJVC94By/gE1yHpyM1L+a5ETvuBHdgyWWINIJTsGfHNL8KLl2GpwHwLMUbmRBjfN2LL9o4xa3KivmnuoIEMdvDOg0q5pTUyIqYgiysNW7jVBu4F7MmiwcKDLJmjpIa4ZmIMsJies18Rpibo6RGDiTeyCj4kBIbyUq8keC+pEb+zNYUOqyTNp6Rwoe1Rkw/GVQxp6RGZsQU3Pbi+vWtAyfgCfS5jJ/Xt1nFnOKMbIjZjlo7bgU3YhpVeZAkpq+8qvEyeAPDKsaewt6in/t2x1/gk5Xwt4Pd8Qx0qFgV2AG3YA5cgENQr3Iodt4HMA22xHTjqVDGL4hngea4VbxynE80PQvG7H2qVP9PX+n+caLkfb8HAAAAAElFTkSuQmCC>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEUAAAAWCAYAAACWl1FwAAADDklEQVR4Xu2WS6iNURTHlzwi8hZKLikliihKxIBiwMCEYkgmjIQYMTAQA0QkjwwUkRjII+VGScyUlBiQCKGUEnmsn7XXuets3zn33HtOh8H3q3/3nrUf395rr732EikpKSn5P5momq8anjc0AGNmq/rnDYGBYn1GqfpkbTA0NyhjVINyY4JvLZDerbdbWOBqVafqpOqp6k7sUIdxqkuqt6ozqg+qTaq+oQ+L35raPqp+qZ6pFoU+sD3Zmee46rzYWqbETmJz843PYn1fq/ZK/QPpEcNU91UPM/sS1a7MlrNGbINrg41ouKp6pZqsmibmsFNiDgQO4bDYWKLGwSmHxByyWzU1tDnbVD9Uq9LvOaovYnOt8E7NwuZ/qi5mdjaDo0Zk9gib/y4WwhE25s5anv5HbNphU9ji2Nhei/eqx6rR6TfRsUG1QzXYOzULC2FxhGFkvFh4zszskRdip8RpRXzOE2LOfSN2uitDH06VPjjNacQpjDkrxTmpZeCMWk7pLiRxSD2ndGZ2Z4hYG336BTvjbquWqTpU+8RyhzuA6GDMI9URsQRMLiRa90sLc8q/cMo6sfZbmZ0rRY5z2GSMMF/TE9XYZMOpXH36LU22CniTQY2K04J2O2Wu6pPqhmpk1lYE81wXiwpfU3599iR7nhf/bPJlD7TZhrXdKZzyMalde+QwDy/ZBLGkz++dVT26vkcCbgn+OvCSREiwMcsXQfjzcvGCRUiwzLkx2HiOb0r1CXOQ7lDf2OJKq4HtuXRdF37zukVwEvb8OvYaPsbpFdUp1BK+Ce439cysSg9LgiymqE7hisxINvIEtvVSfYUXpr/A4byTv5Ml88d1MC/XJCZoj/ZGXq+G2SKWqBwWdlo1PdiIHD58WTUg2QhpHEr16wlyntiVOiC2EezXxMYWyXObOy5GEvmHirYj2I6Kzc93YJJYJBElMUk3DU6gTCZhUaVeETu1CIv/KtVXAogcSvMHYhHDSeJQL6S8HqmlCE6+K1bRXlB9E9t0hI2fE/vOQbFa6Z7Y2JKSkpKSdvEb3iXUgdijNKkAAAAASUVORK5CYII=>

[image12]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACIAAAAXCAYAAABu8J3cAAABcElEQVR4Xu2UvyuFURjHH6EURQZlVJL8GGQwWZTBwGC9u0xGMssgZTCYlOwWhZLBzd9gMBgQNhvKIL7fznOucx7nXu+tM+n91qdzz/Oc85zv+57nviKlSjWvFjBogwXUAcZAl00kxDMqYM4mKCZHwDG4i1MN1QpWwAs41HELtIeLjKbAG1iziXHwCd7BF7iP0w21LG7fjM6HwTNYr62I1Q2uxJ3zy4jXpDinRY0siCu4a+I8gPGUzsC2ZDayKemC8xq34tuYlh+jdl9NzRphT6QKeiNh47IHV3XMbuRU0gVTRibApf7ObqQq6YLWCMcTMKvz7EaKXM0ouAB9QT67kXrNuqhx3w+v4CGAc+Y57uieSM0a4deRBfdN3D8xxbfSb9jQPMceXRfpLyN74q6Df0NqADyCI9DmF4n7rnwEc6u6V0OXTFiqEnf+NbgBQ0GsExyAJ7AEbsE56A3WeLGWPYMPlk3sBZrjVXHkvFSp/6lv8JJwEr9wSbkAAAAASUVORK5CYII=>

[image13]: <data:image/png;base64,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>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEsAAAAZCAYAAAB5CNMWAAACp0lEQVR4Xu2WzYvNURjHH4nIS4k02AzZWIgSMllITGlQ3kpZmBUlJRtlrfkHJguRZCEbCytlNzVlwcZirGYWSJSyICTy8v16zjP3Oeee372/udPcO1PnU99mfuflued8z3nOOSKFQqGwcDkC3YbuQm+hl+F/llFXoLXTrXvDemgftAFanNTlWARtlOa2KyQ/F7avFX8bdBp6AL2BhsM3NQr9hL5BQ6F9t+mHJkQXbhyajGrzrITGoK/QY9HFZwzO5Xij2X/6oWcSx9/pG6RY8HNJOdklatZf0V3YTS5Df5IyLu5N0d1QBefzRBrZcUx0V6VY/EOujPE/SYv4W6AP0I60ApwUNeoXtD+pm0tsATlwD1PlFbQuKfewL3cUF7oKH58GGYzPDKuMz61JQ9Yk5XSXq8i655LP97nCFpAD93CS3OntjGhnlo9PgwwzsbLviKghKVuh96K5zq3cTSz9c2ZxrK3GY2btET1refYOSHx4+/g5s7LxzWEOgLch9Q76DV2DVjeatmSvNC6GOtqu3SrhYDmmTs16Cp1yZcOi/W6IZoyPX9ssS0GeA/MJXiadmpXDdhI3wmaJ49c2y1LwUVrRY2aThjksg+z2m3EaWgV//GJc1XOqDvhV0Hdod1LuuSp6y11wZXbLmdFVBzzj873VFN86tPvxOtwSHUhdXddulfBmfgF9Sco5sdfQpqTcMyb6Gzzkl4Yy20n2BPLx/ZPJTG2Kf0c0KB9w3GXzDbuNl4RvHsw0mZeJh3PwO+SwqGF94Xu56DHDS8u/4C3+vfBt8WnqNHSSjqarnXvl9pqD0EPorOjiMgs4Kc8P0QnTFML6S6KH+X1oCvoMHQ11Hsb/KHH8M1GLBcYgdD78ncmC8hF9AjoALYurIhizk/iFQqFQKBQKnn94NrP0uLzr8wAAAABJRU5ErkJggg==>

[image15]: <data:image/png;base64,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>

[image16]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEsAAAAZCAYAAAB5CNMWAAACx0lEQVR4Xu2Xz6tNURTHv5KB/Hh+PZKeJBNKBn6lZCBEop4UZcCMgTIjf4ABZSKDF0kGUjKUjCRmGBiQAa8eiSiUIpEf329rr3P33fec+46rd+9V+1Pf7j1777PP2Wvttdc6QCaTyfy/7KAuUJeoV9Tj8F9t0jFqbjG6N8ynNlALqclJ33jMohZTU9IOWNsCalLSPpBcFyyn9lJXqZfUoXAtnaO+U1+onWF8t1lCPYE57j71vKm3mjnUdWqMukl9gG2MGBlfa34Hm/8KbPzHaEwL06m71IGkXayGGes3Wh820RylfiVtcu55tO6GmO2w+w5HbfOop9Rtampok7EewuY7Q21G+Q5sYin1llqVdpA9MEP9oDYmfROJO1A7IkYL1KK1+Couw4y1JWqL55PBheZSm35rsxtmkNlJu7wnq6vvAbp7drkDFSYxWrR2unZ8FdotZWMUZlrLrnDdkbFOwSZJWUa9gZ1b/oBu4eFfZqx4wWXonnbGOhGu3VhyjM4sheIKtAlx96AmUTaUXlM/qePUzMbQtqxHIzHU0Uq7rRIZQ+/UibFkqLrGekStC9cy0ukwphQPQZ0D/YSSSafG+ox6xirDnVRaPngI3kg7ekw3wrAMJTElM08CBZ4hNEGcZvuBqgN+BvWVWpu0x5Qd8Aox1ZJaqzK8SgQlL53JW6Nx7qTU0MULjffwOozAXqSuTtptlSgza9EKqRidM2PUoqQ9RlGiZ8R1oW8MN4QXpBp3tjGsCEPZpomLoeMWbLJ+w7OxF4raHTKykkmM1qCFewmgEkelzj00zp59sKyuX+cIdY2aFq4HYQ76VIyAFZ/yWOptv6mfUFWtBe2HOVdRkKb2b7BC1CtzoTPnGXUHdi7rPmX3+NtSTlD2G4WdZ++pF/j3KOsp26iD4fdvHCpjbKKG0b6gHoJ96q1Bjc+dTCaTyWQymQr+AJVjt8w3HHtbAAAAAElFTkSuQmCC>

[image17]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEsAAAAZCAYAAAB5CNMWAAAC6UlEQVR4Xu2XTahNURTH/0KR7+8kkUwoGfhKiRIiMZCBMmDGQJkRvYGJAWUiA5H0BiIxNWFwywzFgImPeiSiUIpEPv7/1l737rvPPfce7+Xeq/av/t231977nLPX2nuv9YBMJpP5f9lGXaAuUa+oR+Fv2aTD1Iz66N4wm1pLzaVGJ32dmErNo8amHTDbHGpUYp+StOssoXZTV6iX1P7Qls5S36kv1PYwvtsspB7DAneXetbUW8506jr1iXoIm7ehaYQ5X2t+B3v+IDVEfYzGFJhI1ai9iV2sgDnrN2wXdpND1K/EpuCeQ3E3xJyAfe/xyKZ5H6ir1Jhgk7Puw553mtqI1juwiUXUW2p52kF2wV78g1qX9P1LPIBaYIwW+ISamdhj5AB9cxx830VDsGPptlr4rcxO2MOnJXZFT15X3z109+7yAGqBMXKidrp2fBmao2/eEdncWdqpmyJbLfxW5iTs4SmLqTeweyt+cTfw49/KWakjUto5S3adFrfVYIHRnaWjuBRtjrhHUA9RNpReUz+pI9TkxtC2rEEjMVTRMptWihaqbxqOs46ieGdthe0q2dUv5KwH1OrQlpNOhTEt8SOoe6CfUDIZrrOU+u9QT6kFoX0NlvU090BjaAEPUsvywY/gjbSjx4zkGAqVDsp8X2EbQZf9C9jcdlldSUzJTNmzCc84nbzdC8ou+EkwB6xK7GWMgx0vv7OUXeUIlQhKXrqTN9dHN4JUSCD+QX/z8jLOw5xeVcdsWinKzCoBPid2LXoIjfTfivEwByhBOVqf1ul1Vnzhn4nG+TGUb5q4GDpuwXZZv+HZ2AtF7RA5WckkRmvQwr0E8NrwJsxxusBVlSthxZnuIMx5E0J7FixAqvrrqPhUxNJo+6R+QlW1FrQHFlztjjS1f6MuwxwjdDnfhv17pJJAcwZQrM7VVvbTXTZIvaeeY+SnrKdsofaF36oBlSPWwy72TsX0fNi4lSg6NJPJZDKZTKYqfwAbfLg4AnxXugAAAABJRU5ErkJggg==>

[image18]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAF0AAAAWCAYAAACi7pBsAAADsUlEQVR4Xu2YWaiNURTHl1BEhsyljCnxJuTB8ECR4UGKUh4USoYi45P74IFIhgeh5EGI8iCliFsk8SBFylDIkCQRMmRYv7v2umd/+55zz3HuuYf0/epf51t7f9+319rDWt8RycnJycn5X+muGhv0pwwK6pU2BLqlBqWfqmtqrAMdxcbaX9UhaauELmIxIl4x+NgjsUFPKW6XfaoPqlNBO6SygNCHvtxzWvVNdT/Tw/ileqQ6pjqkeqF6oBoRd6oDU1UPVbeD7qnGZ3qUxn39rLqseq1aKYWJmyvmJ74RD3zF5/eqSaFPMxi2JzYm4WtoK8UA1WOxF3UKNlbQU7H7Y24FGwFvkOpWWFs5oXqlGh7ZRqvehrbW6Kz6pBoX2fA7thF0grxfzM+JYruqKARjcWJbIfbQdDJi+qjuSGVBZ9b/NgSXye8d2fqKrXYmozVmqM6LHS3OFtUysQkBgt4oLY+dFtChUeyGGN8ql6T4eewQ+CnRNbPO7M+MbPAvBB1/GiUbFPeftlKwK4+qtqYNCRUH3VdmqaDTRp9ycLZdVX1UbUjaAPsVsckYovquWiX1PWaqDbofQeySzWJHBkp9IGYsOHLFItWa0GeXFHZDE7UK+gKxlfBEbBumsGPI4s5Z1Q/VvMjW3lQbdN+9FyRbXDAJb1RjwjUJ+abYogImg8TLs9cHWxO1CrozUvVSNTttSNgk9vzUkRgC4qVoJSq3a9oa9PR4OSPl857HMVPR1TroOH5cdVesDi7FfLHnP1cNTtqc1apnf6ByZ2m1QafaIdEy5hjyFPelCTaG1U+JiTIcEatWYtaJPTCtQmK2ifXhmIhhRfxUTQ/XrGr6TPMOUng+JSelZz3gO4RKhYrF8bKXM7sUnkjTGLG48IH4Ab/R0uYeIrOCjXdnILmlW4RgEzivQqhg9qguSmFl0ocH0s/xlc4KHhZsvHijZJOJ33tAyh8LtYKjrFSdTpszVXVDsuMqFqNGMd99B/BRRNKMc5eX3vHzm2CLMdtDIxvX56Rw3vqMod3BNlmsKlkbroGkQnKhfnUYxELJOvFO7It0SGRrb0ja7LgtYmNB/MZGHQ4sLpI+fo4KNigWI855PoJ8MR0Ue55/EOEbPuLrhGDLcF3sa2p5EC8eGLVzrl8T+6T1ATLoOWIrpUEsU9OO0tVLVqekZJD8XcD7hsYd6gBjorTlfD0cxO8loc1hdX6RljmCMT9V7VTtVZ2U7KrmN76Rz/CTOBDTSv9myMnJycnJySnHb77m+AQAalwNAAAAAElFTkSuQmCC>

[image19]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABIAAAAZCAYAAAA8CX6UAAAA8UlEQVR4XmNgGAWkAlsgXgPEs5DwVCBWh8rPRZMD8bECaSAOAeJqIP4LxIeBOBiIuaHy3UD8D4i/AfEMII6AiuMFIEXvgVgHSew0EMsj8YkC4kD8H4gboHwXBjIMgQGQQdeB2A+Ib6HJkQR+MkAMew3EZmhyJIE9DBCD0MOKJMAPxBOB+AYDaliRBFgZINELoicxIMKKJADS3MAAMQgELBkgYQVKP0QDRiAuA+LdDBCvgQAnEO9ggLiKBSqGE4BcMZ0BojiZAWIgDIDYxkD8Foi7gJgDSQ4FeDJADEDGRVA5NiDegEUeFJuwbDMKRgHVAACPkDRrH97rqwAAAABJRU5ErkJggg==>

[image20]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAYCAYAAADpnJ2CAAABgklEQVR4XtWVvytGURjHv4pSLFJkkyxCKfkPKCUGBorBxkySySC7pJQYTFL+B282kZhYDAyUYhAy4fvtnHvvuU/vO6jz3no/9RnO85zbOec5Py5Qw7zQX/pBH73XdIrWBf2iMUtfaU8QG4ebxEIQi8Y+LdFm39aqdugT7faxlBW6Z2zL9QCO6IHPybF8Ghd0O2iv0yvkV5xjE275+rDF5IRmfkrnUX5PnmlX0J6m33QkiOVQnTXgA+0wObFIj2mDTXhKyMopBuknXQ1iOZINVid1ttzQXhv0qCJhOcUc/aEzJp6SzEidhk1Oq1o2sRB9O2Fib3QXlSuCdnoPt8qwDJ30PGhHQ/UvwQ2oIy40O53INd+OziHcgCe+rTKd0da0R2SWkJ1U1X8L5a9ANJKT+g53H+3lj84Q/YIbtOKFjUlyUm9twqPy6gW5Q/ZHkJdhp/9QT0dpv014JuH2to9uwL1IsiqHSpMZQHaRNXhhNMLtd2HoZ9tkg9VED0ShFLq62uIPdhBJHMUeQHYAAAAASUVORK5CYII=>

[image21]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABoAAAAZCAYAAAAv3j5gAAABr0lEQVR4XtWVvyuGURTHj0KEEItSnmQgJAlRNoqBQZTBxICSwYLJZDCI1+hnRmW1GAyUQUpZTfwDSlEG8f127306z1VKnXfwrU/Pe8/39Jz73HPvfUXsVQeq/O9ibVipFGyCKbALjsBJJsNAbeABNKrYKFhRYxPxpR9gQMW6QLsaSwFYBXsRazoJmlEeWVJen7hCn2ASVCovFQv1gyfwBW7ErXNmNn5M/x7Mghbl8R0L4N3nvIAe5We0IS7pHJREHpWAzjgo7oWJGheCM/llIyyLK3QNKiKPM97xzzieE9ePIE6Sk51TsYzYUBbiEvIsaLHJl1GMqga3YFrFhsEVqFGxjLrFrfEr6FBxNvYCDKlYEL+EHuHZmfC/E5XzQ/yKsCH4dVQruANNIclC7Av7w0LhoJ2C+TTDSKGJLMRloDguTzMMxS3JQnzWg96sbScuGQtx1/BSjLezmUbEFeJVMhZ5puJ2fQPHoCjyqDKwDZ4jDuWP/z38TxkHtbHhtQUWQQNYF3ckiPmGafZPTmhfG/kS77GDOGitcDvzEs6r2JNHMBgb1uLZ4nWVtzP2P/QNOTFMyzFvIkYAAAAASUVORK5CYII=>

[image22]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAYCAYAAADOMhxqAAAAtklEQVR4XmNgGAWDDYgBMTMSXwCLGBjwA/EWIPYF4v9AfAaIl0HlhIH4HBBfhfLBwAWIlwNxOgNEwx4GiCEgwAPEB4D4IZQPBp5QvAaI/wGxB5KcEhA/Z0CzAQZ+A/FWIOaA8lkYIDajGwIHIOeUI/EVgfgJEF8HYnEgjgFidpgkyDSQDTYwAQaEn1qh/MlAzAiTNAbi+cgCQKAPxK+BeD8QX0QSBwNQWLOiCzJA/CMJpUcB7QAAc/QdgqUT/vMAAAAASUVORK5CYII=>

[image23]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADkAAAAZCAYAAACLtIazAAACEklEQVR4Xu2WyytFURSHlzzzyDOSgZgZSR5lJCWRKDEjhkoyMCAG/gNTAwYyMjEwkUJRJsoA5TGQopQkiqSQx+9399lZ53hcDI472F99dc7ae9+71zlr731EHA6H4+fkwbhALBEWBmIkHtbCfPk4JqY5hXdwFU7BTfgAx3UnUAePxPTZhvuw2tcjhjkQM3E6CdP8zZIA5+A5LFXxMnjltcc8O7AyGFTYZLZgtoqzzPk2deIRWMusa0vWJ7GwiZZkK3yF6zBdxXnNWLOKSRU8hnuwWEyNn8FreCOBziHCJGtgi5hyZVLceCwj8n2SfTbQIKauGeAALvJMr8125gbwHdzNOn8pJx6NFdih7uvhE1wWM7doSbI9At8SnYcvsMk2iKlpLmrWdyzAo4MP3M7zx0lamMgJLFKxNjE/MqNi/0kG3BAzpzH5Q5Isg0WY4t3b7Tn4dsOC/8+ds0vF7OSZ2KxE33jY7oOddeYlYjafQ1gAu2Gyatekihn/Gy8jI7+Gpcl+CzDJi3Ee3CAZH4Ll8FZMFfLYsNh+PGJ8wWDHCTE/xqeRCytUW1gswRx1z48CzmlY3j/duKSe4ah3zzivGfPBs2hG/N98fEp82mtwV8XDZABeiClNJvgIB8V/dnPO/fBezK497V33qD4ROEifPxauT5aNXaf/AausXczatEfbZ7DaemGjfPz8czgcDofjL7wB79Z+ZZLQDTQAAAAASUVORK5CYII=>

[image24]: <data:image/png;base64,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>

[image25]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAHcAAAAZCAYAAAALx7GgAAAFDUlEQVR4Xu2ZW8hmUxjH/3KIkBg5hBwiCUNkJqcih0jkNJGpSc3FzMVEEVKTpuRGKSTKoQkXIjkkh8zUvKFGuKAcStRHDlEocYEcnp9nP83a61378M68fTezfvW/ePdae+2117Oew9qvVKlUKpVKZTG5zfRopoNaPaRnTE80beiydrNOMD3ctG02fW16qfkduti0e9yQsI/pXk3P4Ya0k3Gp6V352OiqdrP2ND2o9jwfMB2V9FlsTjMdp/J7d7GLaYnpLNORWRvsZjpQ3i+FZxyaXqADg7xv+tf0nWmtpidzjekTeZ+n5IOn8PtK0x2mv0z/mK5NtFp+L885vLkn2EtuuLfkfb40rdL0i7GBGGuDfPzPTAcn7cz5AtNdpt9NNzW/83dZDA4xvWB60vSy6SfTulaPbtjAP5q+Mf0t37B7J+0Y8CvTb3JHYhNzzx/yd59ijXxhuall/QYM/qyGF+oe+Tiv5g1yw/c9gwXAaBfmDQl454umS+RjvSHfHClsgi3ZtXlQmnOJ6+VzW5lcY96sydHJtRIYkoiUeuUv8vFwEGAen2pbdFqhtvGnuFw+ADv+9KwNPjKdmF/M4AHsJMbBkDnPa5tnph4X0LZgOiy7noLXT+R92AgIQ6ewqDxrHrDIS02vyyPFEGFEotc5WRuGSw2eE+uXb/635WvDOxGSaftQZTsVoSOGLXkO3nprdq3EMabvVX4xWJBP8iFN5wtIX6AL5kb+p89Efk/8DlhEItGOsqvpA9NWeeri9xARMktOwoZ/PLuWwprcZ3pTXocEE/l7Yng2wMzGDcMwyNVZ23VqP6yLK9TtmUycts81nUthf3n7LXlDBmE/okKEP8LWSc01xtmkGV68AAtICKT+OD5rGyKcpMu4eDXe3QXrlG+iqHXYtBDGXSYvbClkibydKRPjTeSDxO6iMzH9zuZ3H+n9JPqoaH+WFwXvaHrSKYQrXiIv1FLYgOzqtE/kcYoPwGM3qhwZ+iAXct/NGshfAwwZd6JxjhJQNbN+FLTBvvINnF47Xx4xWZ8iVHYRGgFPpILlAUNQxIwpiEpgCBZ2KCQzn+fU7oNRMCzzjnH68lrOcvmCEFGIBDvKPI0bzkXBNLRZIx2w/kWi0iWBU7i8Zzq31aMbFpR7F9RfEJXAE/HaoTxZyqW8NDmcZxNCt8g32lgw7IJ84/RFlrHMy7jMhYjJkWgMeHMUXkXIdzSyAx4x3a/hHRMQysPr+7yvBMUXIeWMvCGDfFXqc6b8jPeY6TWNW7wUNgVnUnJsfqyalaGCiug4BGt+u/yEcmxz7QDT2fK1vVseJdMIlabFInEc+lX+sSH/StVHfATJPWsM3MO9ffkW8MpSH16YipkxoujYHjAMi7Ze41JRiTjOlNITDjBmfW6URxQ+hATYJjYGm4d35QvgHs01ClgK2U7jphXzLN5H9cc932q26pIdeorcY7i/K0qQe/BY8m1ehQf7yefOO8wDnkkO5nzLObdrbiVIaZyJqVcCcjvenI6DsSmWzkuucVJhLUrCwEAhxbzwZojcTB88vkhYn4mNAcPg5fkkuEZbH7Ehcj2tdu6L406qLu98Rf3HjO2B8+1W+Xk3/57ex6mmL+Shk3THcW1jq4d0kfwUER8s0tCaKw3zbJB1ph/k3vyx6U/559bOugFv5WvPyXlD5f+INCZfpuBRGJfv4UdkbfMAZ+RTJX/IELkqlUqlUqlUKpVKZWfnPziCQAxTPbjdAAAAAElFTkSuQmCC>

[image26]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAHcAAAAZCAYAAAALx7GgAAAEt0lEQVR4Xu2YW8guUxjH/0IROUQOoY2SnEWIEDknciqi3LjgQhQhErvkAjfIphza4UIkh9gItb9QxJUL7RL1kUMUSijJ4fl55rHXrHfNmnm/77OL1q/+vb2z1jwzs571HGakRqPRaDQam5IbTA9n2qU3Q3rK9Fg3hs7uD+sA04Pd2Jumz00vdP9Dp5u2jBMStjXdrdl7uDSdZJxlek9uG53fH9ZWpvvVv8/7THsnczY1R5j2U/m5p7LatHt+sGMHVexvZjrO9IHpT9NXpqs0O/lC00fyOU+Ydu4P//3/PNNNpt9Mf5guSnSF/Fyus2d3TrC13HFvyed8arrctCqdJN9A2Fott7/BtGsyzj2fYrrN9LPpmu5//iybgt1Mz5keN71o+s50dW/GNLh3nuXI7HjY/0Z9+5unk4Ir5Qv7mcq7BIc/rfGFulNuZ10+IHd87RrcIE47NR9IIDqfN50pt/WafHOksAnWZ8eGOEzjzzQvl8jv7bLkGPfNmuyTHBvjINPHmnVuzf4XybF/OEd+Qm4o+FB+sRrbyFMydnBkzrPaGJlpxAWMLZr2yI6nEPUL8jlsBISjU3horjWFW+QLcpdpp2xsKcQik72Oz8YoGalDalCqXjZdq1mf1OyzhjNwMkZKkcPOvj47VmJf09cqXxgW5Rd/QF4OchjDKVvkAwncG/WfOQvyc+J/wEOSiabCpqRsUJL2z8bmhYxEZsodAmz4R7NjJVibG01rTHtp1lbNftG54RgGL8jGLpbvpDHO1XBkcsOMkWZWZWOwo3z8unwgg7QfWSHS0w+mg7tj2HlDsw8+BeoVfcW78j6kWL9GiCAZWnyijuiucbTpbXmWYrPktmr2WY8Z+zhvQT4Yu4uIpeO8uftfIz3/J23saL83/W56R/XFIl2xsHmjlsIGfF39OfFAUWuI2LUqZ4Z5oT7SB5Aaie4pjDl3QfVA4b7v6H5hKc4t2qezi9QIRCId7JRaRBMzpSEqwYPgkLGUzP08o/4cHIBjue+wM7WuTYFnpx7Tld6TjZVYrnOPUX+9V8y50elGSnjfdEJvxjAsKOcuqt4QlSASidqxOlmqpTiUGs61qZfr5RttJdnOdLt8QcdYjnNxKpkpZcWcS71jkIL9kOleTU9vpPKI+lr0laD5ogk7Kh/IWKfynGNNv5oeMb2igYdbImvkGelWuZPHGGuoyI5D8MZCCYuShmjyWNdvu/9kxZp95haJ16Ef5R8b8q9UNeIjSB5ZU4h37Fq9BaKyNIfNRMeMDaJ7ubChD5R/JOBjSv4eXSNeB0vliQCorQ+NEJsj1RmmX7pf/jOnZn/QuWnHPE/0cUHO+VLzvUqwiHxEiN05lCVo7IhY6m3ehQfby++dZ1gKXIPu+1XToRq+lylQ0jbI+5WAWkq0pXZxBpF6UnIshbkny53Lb5yb2ue5IeyTbYuwcLzGcOIUcAxRjmNScYyxGrEhcj2pflcdrzuphqLzJRVeA0Yg0uiGaZhIwSvF4aZP5L0I5Y7XtbW9GdJp8reI0te6SLG5yK4Q9umLUvuDXT3RyteeQ/KB/zHsfj4YTKmn80I2wLl8D+djxEqD/RP179lvNBqNRqPRaDQajf8afwEps0KsgTVdlQAAAABJRU5ErkJggg==>

[image27]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACIAAAAXCAYAAABu8J3cAAAB9UlEQVR4Xu2Uv0tcQRDHR1RQVPyFhRgCAREFOwtBggGxEYyQNAbSRysbQfEPsBCsLCwk4QjBJqSLgoLIlUIqQTGEFAaSgEUSCESw8Mf3e7O7b27vHnexsJD3hQ+PnZndnZ2dtyKZMv2f2sAIeBg7qlADGATNscOoHjwGnaAm8gW9B6fgDdgCE0XedNWCOfALvHXfFdFNvbjpJDgDv8E1+AqemJiCGsEn0OXGnPgXDIeIdM2CczDmxv3gJ1gKESJT4BD0iq49DS7BHxNTcOTAuDVCedHTDUR2K27A061F9kVnp7rBNzdmxbyeO1tYv120GkPe4MRJDHwa2a2WRWO4sRXn+ERY7Y9uPB8ikpiwr884LZF4E6u0GL+Jb1z2UUviLmhGNKbHG1gaXsFtEtmW8jFxIrFYpR3RmPD3MIF/7mtVTSJ5KR9TKRE26xXYs8a7uBqrL+AdaIodac26KboYuztNac3q/4hQdqdWsCrJG1MHOryTgw9S+oDlpfyVWb0U3fB1ZLe/rxf7YkO0cb1YhAdmLC9EF7U6Bgeip/BaF70Ob3sEvosehAfy4rtyYcasAJNgNdgKHr5dTKZIJ2BfNCGWfEGKs6eOwGfQZ2y86xz4AV6JPt27kpTcPmjlKBGzHhVN5FnkqyT2ApPjXH7j3siU6f7oBqSrcqIAMa0eAAAAAElFTkSuQmCC>

[image28]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAHcAAAAZCAYAAAALx7GgAAAE5ElEQVR4Xu2YW8imUxTH/zKKyCHTIORQyDhmkkZIOUQip3IqN5O4EEVMI+FmEm6QQzk0ITkkhzQINV8IccOFlKiPHHKBEheSw/pZ7/LtZ737ObzffH0X0/Ov/8Wz197r2XuvvQ57SyNGjBgxYsSI5cRNxkcSVzV6SM8YH5/I4DlNsQ43PjSRvW38xvjy5Dt4pnGHGFBgF+Pdmp7D5WUnw9nGD+W64QVNsXY03q/mPO8zHlj0WW4cZzxE9XX3gTGM3T0LCiBr1b+d8UTjx8Z/jN8br9F054uMn8n7PGlc2RT/932+cb3xT+PfxosLrpOP5T/7TcYEdpIb7h15n6+MVxoPKDvJDxC67pDr/9y4VyFnzqcZbzP+brxu8p3XshzY2/ii8QnjK8afjNc2erRje3nfX+X7hU3uUnMdof9HNfUzdgpXyzf2a+M+SQYw+HPq36iNcj2bs0Bu+K5/MEGMdnoWFMA7XzKeJdf1hvxwlOAQbEltbThG/WuaFZfK53ZF0ca82ZODirYacLa/jBcWbRxU9J07+e7S/23R9j8YyAAUrUky8KnxiNyYsLM8JKMHQ2a8oAXPLD0ugGzeuG9qL4HXz8n7cBAghi7BovnXENwi3xA8Y88kWwxik4leJyUZKaM0SA3sMRGyjIxXGTfI9xd06WcPp4BBMWzNczjZN6a2Gg42/qD6j8G8/OcPyE9oBjKMsiILCjA38j995uRj4jvAIolEQ8GmkTYIf4cl2awgIhGZak7CgX8stWXQ52nV9yfQpb9q3DAMwjIkgEvkRU8fzlO7ZzJZZF9oOpeCPeTyG7IggbAfUSHC0y/GIydt6HlL0wsfAvIVXvOBvA6p5q8ehJO0bT5eh3fXEF5/p7wG+USedy9Tcy5d+tmPKf0Yb04ujNOFx1JxEhL6UI7/TQsV7c/yHPKeujeLcJXDUQYH8E01+8SCItfgsZvUffKHgvxIHXC9FkJiH/qMO6d2Rwmvz4Ui0Yw9PGPy3aWfvajqp7KL0AjwRCrYIbmIImZIQVQDhsAgfSGZ+TyvZh8MgGGZd+jpy2uzgLWTj6lK70myGpbCuDksR5Ea+9Olv9W4oeRdeeHykfHkRo92sKGMnVd3QVQDnojX9uXJWi5lE8jh/Jt8uUV+0JYSuxpvl29oH7bGuKQUrj4UeSXCaBHZuvS3Gpd8h5DT87DxXg0Pb4Ty8nTNAoovirDjsyBhs+p91hr/MD5qfE0ti1skHpRHpFvlRu5DX0FFdGxD5NyoKQJhNPT26adfFXEdistzfqXqQjyCZM8aAsYwtivfAryy1ofDRMWMDrx7a8GBXi1/JOAxJd+juxDXwVp6wgH69gcD5TXgyawNvX36W41bVsyzeB8njjHfabarBJvIIwJXEMa3RQkKOzyWfFsWGiV2k8+dNSwG/IPq+3Xj0WqfyxCQ0iiKqFcCJ8i9rdSLMSiUTi3aIqfSP8DtA4OyRlDqj7bQT7Stgo1DEQOHAMPg5RimJG3IuhAHIvMpNavquO6UzCc78Koq14Ae4AlUwxRMhOClwrHGL+W1COmO69qmRg+vfrlF5Ne6Z+X9eRfnzf19TT/Zhn7qolJ/a1XPqeG156gs2IbB6b9Zw/LprCAaYFzew/dPsi7g3YfKDzbj266Q6D9Fs+sfMWLEiBEjRowYMWLEiG0V/wIt70jyjS/vtQAAAABJRU5ErkJggg==>