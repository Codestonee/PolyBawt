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
import random
import signal
import sys
from pathlib import Path

from src.api.state import get_state
from src.execution.clob_client import CLOBClient
from src.execution.order_manager import OrderManager
from src.execution.rate_limiter import RateLimiter, rate_limiter
from src.infrastructure.config import init_config, AppConfig, load_config
from src.infrastructure.kill_switch import kill_switch, KillSwitchEvent, KillSwitchType
from src.infrastructure.logging import bind_context, get_logger, configure_logging
from src.ingestion.market_discovery import MarketDiscovery
from src.ingestion.oracle_feed import OracleFeed
from src.portfolio.tracker import Portfolio
from src.risk.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from src.strategy.value_betting import EnsembleStrategy


async def run_crypto_strategy(config: AppConfig) -> None:
    """Initialize and run the value betting strategy."""
    logger = get_logger(__name__)
    
    logger.info(
        "Initializing components",
        environment=config.environment,
        dry_run=config.dry_run,
    )
    
    # Initialize components
    market_discovery = MarketDiscovery(base_url=config.api.gamma_base_url)
    oracle = OracleFeed(use_websocket=True)
    rate_limiter = RateLimiter()

    # Start WebSocket for real-time price feeds
    ws_started = await oracle.start_websocket()
    if ws_started:
        logger.info("Real-time WebSocket price feed active")
    else:
        logger.info("Using REST API price polling (WebSocket unavailable)")

    # FIX #3: Initialize CLOB client for order book pricing AND order execution
    clob_client = CLOBClient(
        dry_run=config.dry_run,
        rate_limiter=rate_limiter,
    )
    await clob_client.initialize()

    # Create submit callback that uses CLOBClient for live orders
    async def submit_order_callback(order):
        """Submit order to exchange via CLOB client."""
        return await clob_client.place_order(
            token_id=order.token_id,
            side=order.side.value,
            price=order.price,
            size=order.size,
            client_order_id=order.client_order_id,
        )

    order_manager = OrderManager(
        dry_run=config.dry_run, 
        rate_limiter=rate_limiter,
        submit_callback=submit_order_callback if not config.dry_run else None,
    )

    # Values for initialization
    bankroll = config.trading.bankroll

    cb_config = CircuitBreakerConfig(
        daily_loss_soft_pct=config.risk.daily_loss_soft_limit_pct,
        daily_loss_hard_pct=config.risk.daily_loss_hard_limit_pct,
        max_drawdown_hard_pct=config.risk.max_drawdown_pct,
        max_drawdown_soft_pct=config.risk.max_drawdown_pct * 0.8,
        volatility_hard=config.risk.volatility_pause_threshold,
    )
    circuit_breaker = CircuitBreaker(starting_capital=bankroll, config=cb_config)
    portfolio = Portfolio(starting_capital=bankroll)
    portfolio.load_state()

    # SYNC: Fetch real balance
    try:
        real_balance = await clob_client.get_account_balance()
        if real_balance > 0:
            logger.info(f"Synced real balance from Polymarket: ${real_balance:.2f}")
            portfolio.current_capital = real_balance
            portfolio.peak_capital = max(portfolio.peak_capital, real_balance)
            circuit_breaker.update_drawdown(real_balance)
    except Exception as e:
        logger.warning(f"Failed to sync real balance (using config default): {e}")

    # Sync to global state
    get_state().portfolio = portfolio

    # Instantiate Strategy
    logger.info("Instantiating EnsembleStrategy...")
    try:
        strategy = EnsembleStrategy(
            config=config,
            market_discovery=market_discovery,
            oracle=oracle,
            order_manager=order_manager,
            rate_limiter=rate_limiter,
            circuit_breaker=circuit_breaker,
            clob_client=clob_client,
            portfolio=portfolio,
            bankroll=bankroll,
        )
    except Exception as e:
        logger.exception(f"CRITICAL: Failed to create ensemble: {e}")
        return

    logger.info("Grok Ensemble ready. Starting main loop...")
    
    try:
        await strategy.run()
    except asyncio.CancelledError:
        logger.info("Strategy loop stopped.")
    except Exception as e:
        logger.exception(f"Strategy loop crashed: {e}")
    finally:
        await market_discovery.close()
        await oracle.close()
        logger.info("Strategy cleanup complete.")





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
    
    # Override dry_run depending on --live flag
    if args.live:
        config.dry_run = False  # Force live
    else:
        config.dry_run = True   # Force paper
    
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

    api_config = uvicorn.Config(app, host="127.0.0.1", port=config.observability.metrics_port, log_level="error")
    server = uvicorn.Server(api_config)
    
    # Run API server concurrently with strategy
    api_task = asyncio.create_task(server.serve())
    get_state().is_running = True
    get_state().is_live = not config.dry_run

    # Populate active strategies for Dashboard & Terminal
    strategies = {
        "Arb Taker": True,
        "Latency Snipe": True,
        "Spread Maker": True,
        "Legged Hedge": True,
        "Circuit Breaker": True,
        "VPIN Filter": True,
    }
    get_state().active_strategies = strategies

    # Print Easy-to-Read Terminal Output
    print("\n" + "="*60)
    print(f"🤖 POLYMARKET BOT - {'LIVE TRADING 🔴' if args.live else 'PAPER TRADING ⚪'}")
    print("="*60)
    print(f"{'STRATEGY':<25} | {'STATUS':<10} | {'MODE':<10}")
    print("-" * 50)
    for name, enabled in strategies.items():
        status = "✅ ON" if enabled else "❌ OFF"
        mode = "Auto" if name != "Market Making" else "Manual"
        print(f"{name:<25} | {status:<10} | {mode:<10}")
    print("="*60 + "\n")

    # Set up kill switch monitoring
    active_tasks = []

    async def on_kill_switch(event: KillSwitchEvent):
        """Handle kill switch trigger."""
        logger.critical("Kill switch activated")
        for t in active_tasks:
            if not t.done():
                t.cancel()
    
    kill_switch.register_callback(on_kill_switch)
    await kill_switch.start_monitoring()
    
    try:
        # Run strategy concurrently
        crypto_task = asyncio.create_task(run_crypto_strategy(config))
        active_tasks.append(crypto_task)
        
        # Monitor for kill switch in parallel
        async def kill_switch_monitor():
            while True:
                if await kill_switch.check():
                    logger.critical("Kill switch detected - stopping strategies")
                    crypto_task.cancel()
                    break
                await asyncio.sleep(1.0)
        
        monitor_task = asyncio.create_task(kill_switch_monitor())
        active_tasks.append(monitor_task)
        
        # Use asyncio.wait to exit if ANY task finishes (e.g. crash)
        done, pending = await asyncio.wait(
            [crypto_task, monitor_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Check for exceptions in done tasks
        for task in done:
            if task.exception():
                logger.error(f"Task failed with exception: {task.exception()}")
            else:
                logger.info(f"Task completed: {task}")
        
        # Cancel remaining tasks
        for task in pending:
            task.cancel()
        
    except asyncio.CancelledError:
        logger.info("Main loop cancelled")
    except Exception as e:
        logger.exception("Main loop crashed", error=str(e))
    finally:
        await kill_switch.stop_monitoring()
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
