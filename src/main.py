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
from src.execution.reconciler import Reconciler
from src.infrastructure.config import init_config, AppConfig, load_config, load_secrets
from src.infrastructure.events import event_bus, event_logger, EventType
from src.infrastructure.kill_switch import kill_switch, KillSwitchEvent, KillSwitchType
from src.infrastructure.logging import bind_context, get_logger, configure_logging
from src.infrastructure.metrics import metrics
from src.ingestion.market_discovery import MarketDiscovery
from src.ingestion.oracle_feed import OracleFeed
from src.ingestion.ws_client import WebSocketClient
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
    oracle = OracleFeed(
        use_websocket=True,
        require_chainlink_in_live=not config.dry_run,
    )
    rate_limiter = RateLimiter()

    # Start WebSocket for real-time price feeds
    ws_started = await oracle.start_websocket()
    if ws_started:
        logger.info("Real-time WebSocket price feed active")
    else:
        logger.info("Using REST API price polling (WebSocket unavailable)")

    # Initialize CLOB client
    clob_client = CLOBClient(
        dry_run=config.dry_run,
        rate_limiter=rate_limiter,
    )
    await clob_client.initialize()

    # Create submit callback
    async def submit_order_callback(order):
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

    # === Wire EventBus ===
    event_bus.subscribe_all(event_logger.handle)
    event_bus_task = asyncio.create_task(event_bus.run())
    logger.info("Event bus started")

    # === Wire MetricsCollector ===
    try:
        metrics.start_server(port=config.observability.metrics_port)
        metrics.set_bot_info(version="1.1.0", environment=config.environment)
        logger.info("Metrics server started", port=config.observability.metrics_port)
    except Exception as e:
        logger.warning(f"Metrics server failed to start: {e}")

    # === Wire Reconciler ===
    reconciler = Reconciler(
        order_manager=order_manager,
        clob_client=clob_client,
        portfolio=portfolio,
    )
    await reconciler.start_periodic(interval_seconds=30)
    logger.info("Reconciler started (30s interval)")

    # === Wire WebSocket fill listener ===
    ws_fill_client = None
    secrets = load_secrets()
    if secrets.polymarket_api_key and not config.dry_run:
        try:
            ws_fill_client = WebSocketClient(
                url=config.api.ws_url,
                api_key=secrets.polymarket_api_key,
                api_secret=secrets.polymarket_api_secret,
                api_passphrase=secrets.polymarket_passphrase,
            )
            await ws_fill_client.connect()
            await ws_fill_client.subscribe_user()
            logger.info("WebSocket fill listener connected")
        except Exception as e:
            logger.warning(f"WebSocket fill listener failed: {e}")
            ws_fill_client = None

    # Sync to global state
    get_state().portfolio = portfolio
    get_state().order_manager = order_manager

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

    # Register WebSocket fill handler
    if ws_fill_client:
        async def ws_fill_handler(msg):
            """Route WebSocket fills to strategy."""
            data = msg.data
            client_order_id = data.get("client_order_id") or data.get("order_id", "")
            fill_size = float(data.get("size", 0))
            fill_price = float(data.get("price", 0))
            if client_order_id and fill_size > 0 and fill_price > 0:
                strategy.handle_fill(client_order_id, fill_size, fill_price)

        ws_fill_client.on("trade", ws_fill_handler)
        ws_fill_client.on("last_trade_price", ws_fill_handler)

    logger.info("Grok Ensemble ready. Starting main loop...")

    try:
        await strategy.run()
    except asyncio.CancelledError:
        logger.info("Strategy loop stopped.")
    except Exception as e:
        logger.exception(f"Strategy loop crashed: {e}")
    finally:
        # === Graceful shutdown ===
        logger.info("Beginning graceful shutdown...")

        # 1. Stop strategy
        strategy.stop()

        # 2. Cancel all open orders
        try:
            cancelled = await order_manager.cancel_all_orders()
            logger.info(f"Cancelled {cancelled} open orders on shutdown")
        except Exception as e:
            logger.error(f"Failed to cancel orders on shutdown: {e}")

        # 3. Cleanup SpreadMaker orders
        spread_maker = strategy.strategies.get("SpreadMaker")
        if spread_maker and hasattr(spread_maker, 'cleanup'):
            try:
                await spread_maker.cleanup()
            except Exception:
                pass

        # 4. Save portfolio state
        portfolio.save_state()
        logger.info("Portfolio state saved")

        # 5. Log final summary
        summary = portfolio.summary()
        logger.info(
            "Final portfolio summary",
            capital=f"${summary['current_capital']:.2f}",
            pnl=f"${summary['realized_pnl']:.2f}",
            trades=summary['total_trades'],
            win_rate=f"{summary['win_rate']}%",
        )

        # 6. Stop reconciler
        await reconciler.stop()

        # 7. Stop event bus
        event_bus.stop()
        event_bus_task.cancel()
        try:
            await event_bus_task
        except asyncio.CancelledError:
            pass

        # 8. Close WebSocket clients
        if ws_fill_client:
            await ws_fill_client.close()

        # 9. Close other connections
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
        config.dry_run = False
    else:
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
        "RiskGate": True,
        "Sniper Protection": True,
        "Reconciler": True,
    }
    get_state().active_strategies = strategies

    # Print terminal output
    print("\n" + "=" * 60)
    print(f"POLYMARKET BOT - {'LIVE TRADING' if args.live else 'PAPER TRADING'}")
    print("=" * 60)
    print(f"{'STRATEGY':<25} | {'STATUS':<10} | {'MODE':<10}")
    print("-" * 50)
    for name, enabled in strategies.items():
        status = "ON" if enabled else "OFF"
        mode = "Auto"
        print(f"{name:<25} | {status:<10} | {mode:<10}")
    print("=" * 60 + "\n")

    # Set up kill switch monitoring
    active_tasks = []

    async def on_kill_switch(event: KillSwitchEvent):
        """Handle kill switch trigger."""
        logger.critical("Kill switch activated")

        state = get_state()
        if state.order_manager:
            try:
                cancelled = await state.order_manager.cancel_all_orders()
                logger.critical(f"Kill switch: cancelled {cancelled} open orders")
            except Exception as e:
                logger.error(f"Failed to cancel orders on kill switch: {e}")

        for t in active_tasks:
            if not t.done():
                t.cancel()

    kill_switch.register_callback(on_kill_switch)
    await kill_switch.start_monitoring()

    try:
        crypto_task = asyncio.create_task(run_crypto_strategy(config))
        active_tasks.append(crypto_task)

        async def kill_switch_monitor():
            while True:
                if await kill_switch.check():
                    logger.critical("Kill switch detected - stopping strategies")
                    crypto_task.cancel()
                    break
                await asyncio.sleep(1.0)

        monitor_task = asyncio.create_task(kill_switch_monitor())
        active_tasks.append(monitor_task)

        done, pending = await asyncio.wait(
            [crypto_task, monitor_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in done:
            if task.exception():
                logger.error(f"Task failed with exception: {task.exception()}")
            else:
                logger.info(f"Task completed: {task}")

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
