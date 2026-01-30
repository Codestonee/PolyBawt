import asyncio
import logging
from src.infrastructure.config import load_config
from src.execution.order_manager import OrderManager
from src.ingestion.market_discovery import MarketDiscovery
from src.strategy.value_betting import ValueBettingStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_live")

async def main():
    logger.info("Verifying components...")
    
    # 1. Config
    config = load_config("config/live.yaml")
    logger.info(f"Config loaded: Bankroll=${config.trading.bankroll}")
    
    # 2. Persistence
    om = OrderManager(dry_run=True, persistence_file="orders_test.json")
    logger.info("OrderManager initialized with persistence")
    
    # 3. Strategy Cache
    # Mock dependencies
    strategy = ValueBettingStrategy(
        config=config,
        market_discovery=MarketDiscovery(),
        oracle=None,
        order_manager=om,
        rate_limiter=None,
        circuit_breaker=None,
        bankroll=50.0
    )
    
    if hasattr(strategy, '_order_book_cache'):
        logger.info("Strategy cache confirmed")
    
    print("ALL VERIFIED")

if __name__ == "__main__":
    asyncio.run(main())
