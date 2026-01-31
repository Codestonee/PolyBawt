
import asyncio
import json
import logging
import re
import math
from datetime import datetime, timezone
import aiohttp
from typing import List, Optional, Tuple

# Fix imports to match project structure
from src.execution.clob_client import CLOBClient
from src.infrastructure.config import AppConfig

logger = logging.getLogger("strategy.btc_15m")

# Constants
BUY = "BUY"
SELL = "SELL"
VOLATILITY_BTC_15M = 1.00  # 100% Annualized Vol

class BTC15mStrategy:
    """
    Scalping Strategy for 15-minute BTC markets.
    """
    def __init__(self, config: AppConfig, clob_client: CLOBClient, risk_gatekeeper=None):
        self.config = config
        self.clob = clob_client
        self.risk_gatekeeper = risk_gatekeeper
        self.dry_run = config.dry_run
        
        # Pull config from YAML if available, else defaults
        # We need to add this section to AppConfig eventually
        self.min_edge = 0.02
        self.max_position_size = 20.0
        self.running = False

        self.target_market_keyword = "BTC"
        self.strike_pattern = re.compile(r"BTC.*?(above|below|>|<).*?\$?([\d,]+\.?\d*)", re.IGNORECASE)
        self.time_pattern = re.compile(r"at (\d{1,2}):(\d{2})\s*(AM|PM)?", re.IGNORECASE)
        
        logger.info(f"BTC15m Strategy initialized: dry_run={self.dry_run}, min_edge={self.min_edge}")

    async def setup(self):
        logger.info("BTC 15m Strategy Setup.")
        self.running = True

    def stop(self):
        self.running = False
        logger.info("BTC 15m Strategy Stopping.")

    async def get_btc_price(self) -> float:
        timeout = aiohttp.ClientTimeout(total=5)
        
        # 1. Binance (Primary)
        try:
            url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return float(data['price'])
        except Exception as e:
            logger.warning(f"Oracle Fail (Binance): {e}")

        # 2. Coinbase (Backup)
        try:
            url = "https://api.coinbase.com/v2/prices/BTC-USD/spot"
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        price = float(data['data']['amount'])
                        return price
        except Exception as e:
            logger.warning(f"Oracle Fail (Coinbase): {e}")

        return 0.0

    async def fetch_markets_direct(self) -> List[dict]:
        """
        Fetch BTC markets.
        """
        timeout = aiohttp.ClientTimeout(total=15)
        try:
            url = "https://gamma-api.polymarket.com/markets"
            params = {"limit": 100, "closed": "false", "active": "true"}
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        markets = await resp.json()
                        btc_markets = [m for m in markets if 'BTC' in m.get('question', '').upper()
                                       or 'BITCOIN' in m.get('question', '').upper()]
                        return btc_markets
        except Exception as e:
            logger.error(f"Gamma API lookup failed: {e}")
        return []

    def parse_market(self, question: str) -> Tuple[Optional[float], Optional[str], Optional[datetime]]:
        match_s = self.strike_pattern.search(question)
        strike = None
        direction = None
        if match_s:
            direction_raw = match_s.group(1).lower()
            strike_raw = match_s.group(2).replace(",", "")
            try:
                strike = float(strike_raw)
                direction = "above" if direction_raw in [">", "above"] else "below"
            except (ValueError, TypeError):
                pass
        
        if not strike: return None, None, None
        return strike, direction, None

    def norm_cdf(self, x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    def calculate_probability(self, spot, strike, time_remaining_years, direction):
        if time_remaining_years <= 0:
            if direction == "above": return 1.0 if spot > strike else 0.0
            else: return 1.0 if spot < strike else 0.0
            
        vol = VOLATILITY_BTC_15M
        d2 = (math.log(spot / strike) - 0.5 * vol**2 * time_remaining_years) / (vol * math.sqrt(time_remaining_years))
        prob_above = self.norm_cdf(d2)
        return prob_above if direction == "above" else (1.0 - prob_above)

    async def scan_and_execute(self):
        if not self.running: return
        try:
            btc_price = await self.get_btc_price()
            if btc_price == 0: return
            
            data = await self.fetch_markets_direct()
            if not data: return
            
            for m in data:
                await self.process_market(m, btc_price)
                
        except Exception as e:
            logger.error(f"Scan Loop Error: {e}")

    async def process_market(self, market: dict, spot_price: float):
        question = market.get('question', '')
        strike, direction, _ = self.parse_market(question)
        if not strike: return

        end_date_iso = market.get('end_date_iso') or market.get('endDate')
        if not end_date_iso: return

        try:
            if isinstance(end_date_iso, str):
                if end_date_iso.endswith('Z'):
                    expiry = datetime.fromisoformat(end_date_iso.replace('Z', '+00:00'))
                else:
                    expiry = datetime.fromisoformat(end_date_iso)
            else:
                return

            now = datetime.now(timezone.utc)
            delta = expiry - now
            seconds_remaining = delta.total_seconds()

            if seconds_remaining < 60 or seconds_remaining > 86400: return
            years_remaining = seconds_remaining / 31536000

        except Exception:
            return

        my_prob = self.calculate_probability(spot_price, strike, years_remaining, direction)

        tokens = market.get('tokens', [])
        if not tokens: return # Simplify for now
        
        yes_token = tokens[0]
        token_id = yes_token.get('token_id')
        curr_yes_price = float(yes_token.get('price', 0.5))

        if not token_id or curr_yes_price <= 0: return

        edge = my_prob - curr_yes_price

        if edge > self.min_edge:
            logger.info(f"OPPORTUNITY: {question} | Edge: {edge:.4f}")
            await self.execute_trade(token_id, curr_yes_price, question, edge)

    async def execute_trade(self, token_id, price, question: str, edge: float):
        size_usd = 10.0 # Hardcoded safe size for now
        
        if self.dry_run:
            logger.info(f"[DRY RUN] Would BUY {token_id} @ {price} for ${size_usd}")
            return

        try:
            # Cross spread by 1 tick
            exec_price = round(price + 0.01, 2)
            if exec_price >= 1.0: exec_price = 0.99
            
            resp = await self.clob.place_order(
                token_id=token_id,
                price=exec_price,
                side=BUY,
                size=size_usd,
            )
            logger.info(f"Order Placed: {resp}")
        except Exception as e:
            logger.error(f"Order Failed: {e}")

    async def run(self):
        await self.setup()
        while self.running:
            try:
                await self.scan_and_execute()
                await asyncio.sleep(15) 
            except Exception as e:
                logger.error(f"Error in BTC15m loop: {e}")
                await asyncio.sleep(5)
