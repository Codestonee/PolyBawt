"""
Polymarket 15m Trading Bot - Main Entry Point

Usage:
    python -m src.main --config config/paper.yaml
    python -m src.main --config config/production.yaml --live
"""

import argparse
import asyncio
import signal
import sys
from pathlib import Path

from src.infrastructure.config import init_config, get_config, get_secrets, AppConfig
from src.infrastructure.logging import configure_logging, get_logger, bind_context
from src.ingestion.ws_client import WebSocketClient, ReconnectPolicy
from src.ingestion.market_discovery import MarketDiscovery
from src.ingestion.oracle_feed import OracleFeed
from src.execution.order_manager import OrderManager
from src.execution.rate_limiter import RateLimiter
from src.risk.circuit_breaker import CircuitBreaker
from src.strategy.value_betting import ValueBettingStrategy


async def run_strategy(config: AppConfig) -> None:
    """Initialize and run the value betting strategy."""
    logger = get_logger(__name__)
    
    logger.info(
        "Initializing components",
        environment=config.environment,
        dry_run=config.dry_run,
    )
    
    # Initialize components
    market_discovery = MarketDiscovery(base_url=config.api.gamma_base_url)
    oracle = OracleFeed()
    rate_limiter = RateLimiter()
    
    order_manager = OrderManager(dry_run=config.dry_run)
    circuit_breaker = CircuitBreaker(starting_capital=100)  # TODO: Make configurable
    
    # Create strategy
    strategy = ValueBettingStrategy(
        config=config,
        market_discovery=market_discovery,
        oracle=oracle,
        order_manager=order_manager,
        rate_limiter=rate_limiter,
        circuit_breaker=circuit_breaker,
        bankroll=100,  # TODO: Make configurable
    )
    
    # Set up shutdown handler
    def handle_shutdown():
        logger.info("Shutdown requested")
        strategy.stop()
    
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, handle_shutdown)
        except NotImplementedError:
            pass  # Windows
    
    try:
        logger.info("Starting strategy")
        await strategy.run()
    finally:
        # Cleanup
        await market_discovery.close()
        await oracle.close()
        
        # Log final metrics
        metrics = strategy.get_metrics()
        logger.info(
            "Strategy completed",
            signals=metrics["signals_generated"],
            passed_gate=metrics["signals_passed_gate"],
            orders=metrics["orders_placed"],
        )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Polymarket 15m Trading Bot"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/paper.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live trading (required for production)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Run for N iterations then stop (default: run forever)",
    )
    return parser.parse_args()


async def async_main() -> None:
    """Async entry point."""
    args = parse_args()
    
    # Load configuration
    config, secrets = init_config(args.config)
    
    # Safety check for production
    if config.environment == "production" and not args.live:
        print("ERROR: Production config requires --live flag")
        print("This is a safety measure to prevent accidental live trading")
        sys.exit(1)
    
    # Override dry_run if not --live
    if not args.live:
        config.dry_run = True
    
    # Configure logging
    configure_logging(
        log_level=config.observability.log_level,
        log_format=config.observability.log_format,
    )
    
    logger = get_logger(__name__)
    logger.info(
        "Polymarket 15m Trading Bot starting",
        environment=config.environment,
        dry_run=config.dry_run,
        config_file=args.config,
    )
    
    await run_strategy(config)


def main() -> None:
    """Synchronous entry point."""
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
