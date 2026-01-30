"""
Test script: Place $5 bets on the FAVORITE (highest probability) option for each active market.

This script will:
1. Discover all active 15-minute crypto markets (BTC, ETH, SOL, XRP)
2. For each market, determine which option (YES/NO) has the highest price (favorite)
3. Place a $5 BUY order on the favorite option

WARNING: This places REAL money bets!
"""

from dotenv import load_dotenv
load_dotenv()

import asyncio
from src.infrastructure.config import init_config
from src.infrastructure.logging import configure_logging, get_logger
from src.ingestion.market_discovery import MarketDiscovery
from src.execution.clob_client import CLOBClient
from src.execution.rate_limiter import RateLimiter


BET_AMOUNT_USD = 5.0  # $5 per bet


async def main():
    # Load config
    config, secrets = init_config("config/paper.yaml")
    
    # Configure logging
    configure_logging(log_level="INFO", log_format="text")
    logger = get_logger(__name__)
    
    logger.info("=" * 60)
    logger.info("TEST SCRIPT: Place $5 bets on favorites")
    logger.info("=" * 60)
    
    # Initialize components
    market_discovery = MarketDiscovery()
    rate_limiter = RateLimiter()
    
    # LIVE trading - not dry run!
    clob_client = CLOBClient(dry_run=False, rate_limiter=rate_limiter)
    
    logger.info("Initializing CLOB client for LIVE trading...")
    success = await clob_client.initialize()
    if not success:
        logger.error("Failed to initialize CLOB client. Check your .env credentials.")
        return
    
    logger.info("CLOB client initialized successfully")
    
    try:
        # Discover markets
        logger.info("Discovering active 15-minute crypto markets...")
        markets = await market_discovery.get_crypto_15m_markets(
            assets=["BTC", "ETH", "SOL", "XRP"]
        )
        
        if not markets:
            logger.warning("No active markets found!")
            return
        
        logger.info(f"Found {len(markets)} active markets")
        
        # Track results
        orders_placed = 0
        orders_failed = 0
        total_spent = 0.0
        
        for market in markets:
            logger.info("-" * 40)
            logger.info(f"Market: {market.question[:60]}...")
            logger.info(f"  Asset: {market.asset}")
            logger.info(f"  Expires in: {market.minutes_to_expiry:.1f} minutes")
            logger.info(f"  YES price: {market.yes_price:.3f}")
            logger.info(f"  NO price: {market.no_price:.3f}")
            
            # Determine favorite (highest price = most likely to win)
            if market.yes_price >= market.no_price:
                # YES is the favorite
                token_id = market.yes_token_id
                side = "YES"
                price = market.yes_price
            else:
                # NO is the favorite
                token_id = market.no_token_id
                side = "NO"
                price = market.no_price
            
            logger.info(f"  Favorite: {side} @ {price:.3f}")
            
            # Skip if price is too extreme (avoid buying at 0.95+)
            if price > 0.95:
                logger.warning(f"  SKIPPING: Price too high ({price:.3f} > 0.95)")
                continue
            
            # Skip if market expires in less than 2 minutes
            if market.minutes_to_expiry < 2:
                logger.warning(f"  SKIPPING: Market expires too soon ({market.minutes_to_expiry:.1f}m)")
                continue
            
            # Place the order
            logger.info(f"  PLACING ORDER: BUY ${BET_AMOUNT_USD} of {side} @ {price:.3f}")
            
            response = await clob_client.place_order(
                token_id=token_id,
                side="BUY",
                price=price,
                size=BET_AMOUNT_USD,  # $5 USD
            )
            
            if response.success:
                orders_placed += 1
                total_spent += BET_AMOUNT_USD
                logger.info(f"  [OK] ORDER PLACED: {response.exchange_order_id}")
            else:
                orders_failed += 1
                logger.error(f"  [FAIL] ORDER FAILED: {response.error_message}")
        
        # Summary
        logger.info("=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Markets found: {len(markets)}")
        logger.info(f"Orders placed: {orders_placed}")
        logger.info(f"Orders failed: {orders_failed}")
        logger.info(f"Total spent: ${total_spent:.2f}")
        
    finally:
        await market_discovery.close()


if __name__ == "__main__":
    asyncio.run(main())
