"""
Polymarket 15m Trading Bot - Main Entry Point

Usage:
    python -m src.main --config config/paper.yaml
    python -m src.main --config config/production.yaml --live
"""

# Load .env FIRST before any other imports
from dotenv import load_dotenv
load_dotenv()

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
from src.execution.clob_client import CLOBClient
from src.risk.circuit_breaker import CircuitBreaker
from src.strategy.value_betting import ValueBettingStrategy
from src.portfolio.tracker import Portfolio


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

    # FIX #3: Initialize CLOB client for order book pricing AND order execution
    clob_client = CLOBClient(
        dry_run=config.dry_run,
        rate_limiter=rate_limiter,
    )
    await clob_client.initialize()

    # Create submit callback that uses CLOBClient for live orders
    async def submit_order_callback(order):
        """Submit order to exchange via CLOB client."""
        from src.execution.order_manager import OrderState
        response = await clob_client.place_order(
            token_id=order.token_id,
            side=order.side.value,
            price=order.price,
            size=order.size,
            client_order_id=order.client_order_id,
        )
        if response.success:
            order.exchange_order_id = response.exchange_order_id
            order.update_state(OrderState.NEW)
            logger.info("Order submitted to exchange", 
                       exchange_order_id=response.exchange_order_id,
                       client_order_id=order.client_order_id)
        else:
            order.update_state(OrderState.REJECTED)
            logger.error("Order rejected by exchange",
                        error_code=response.error_code,
                        error_message=response.error_message)

    order_manager = OrderManager(
        dry_run=config.dry_run, 
        rate_limiter=rate_limiter,
        submit_callback=submit_order_callback if not config.dry_run else None,
    )

    # FIX #4: Initialize Portfolio for fill-based tracking
    bankroll = config.trading.bankroll  # Now configurable from YAML

    # Circuit breaker MUST use same starting capital as Portfolio!
    circuit_breaker = CircuitBreaker(starting_capital=bankroll)

    portfolio = Portfolio(starting_capital=bankroll)
    
    # Sync to global state for API
    from src.api.state import get_state
    get_state().portfolio = portfolio

    # Create strategy
    strategy = ValueBettingStrategy(
        config=config,
        market_discovery=market_discovery,
        oracle=oracle,
        order_manager=order_manager,
        rate_limiter=rate_limiter,
        circuit_breaker=circuit_breaker,
        clob_client=clob_client,  # FIX #3: Pass CLOB client for order books
        portfolio=portfolio,  # FIX #4: Pass Portfolio for fill-based tracking
        bankroll=bankroll,
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
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Use clean, minimal log format for easier terminal reading",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose/debug logging",
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
    log_format = "clean" if args.clean else config.observability.log_format
    log_level = "DEBUG" if args.verbose else config.observability.log_level
    
    configure_logging(
        log_level=log_level,
        log_format=log_format,
    )
    
    logger = get_logger(__name__)
    logger.info(
        "Polymarket 15m Trading Bot starting",
        environment=config.environment,
        dry_run=config.dry_run,
        config_file=args.config,
    )

    # Start API server in background
    import uvicorn
    from src.api.server import app
    from src.api.state import get_state

    api_config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="error")
    server = uvicorn.Server(api_config)
    
    # Run API server concurrently with strategy
    api_task = asyncio.create_task(server.serve())
    get_state().is_running = True
    
    try:
        await run_strategy(config)
    finally:
        get_state().is_running = False
        api_task.cancel()
        try:
            await api_task
        except asyncio.CancelledError:
            pass


def main() -> None:
    """Synchronous entry point."""
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
