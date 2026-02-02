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
from src.infrastructure.kill_switch import kill_switch, KillSwitchEvent
from src.ingestion.ws_client import WebSocketClient, ReconnectPolicy
from src.ingestion.market_discovery import MarketDiscovery
from src.ingestion.oracle_feed import OracleFeed
from src.execution.order_manager import OrderManager
from src.execution.rate_limiter import RateLimiter
from src.execution.clob_client import CLOBClient
from src.risk.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from src.strategy.value_betting import EnsembleStrategy
from src.strategy.event_betting import EventBettingStrategy
from src.strategy.btc_15m import BTC15mStrategy
from src.portfolio.tracker import Portfolio
from src.ingestion.event_market_discovery import EventMarketDiscovery


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
        """Submit order to exchange via CLOB client.

        IMPORTANT: Return the exchange response; OrderManager owns state transitions.
        """
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

    # FIX #4: Initialize Portfolio for fill-based tracking
    bankroll = config.trading.bankroll  # Now configurable from YAML

    # Circuit breaker MUST use same starting capital as Portfolio!
    cb_config = CircuitBreakerConfig(
        daily_loss_soft_pct=config.risk.daily_loss_soft_limit_pct,
        daily_loss_hard_pct=config.risk.daily_loss_hard_limit_pct,
        max_drawdown_hard_pct=config.risk.max_drawdown_pct,
        max_drawdown_soft_pct=config.risk.max_drawdown_pct * 0.8,
        volatility_hard=config.risk.volatility_pause_threshold,
    )
    circuit_breaker = CircuitBreaker(starting_capital=bankroll, config=cb_config)

    portfolio = Portfolio(starting_capital=bankroll)
    
    # PERISTENCE: Load state from disk
    portfolio.load_state()
    
    # SYNC: Fetch real balance if possible
    # Even in dry_run, if we have keys (authed read-only), we get real balance
    try:
        real_balance = await clob_client.get_account_balance()
        if real_balance > 0:
            logger.info(u"Synced real balance from Polymarket", balance=f"${real_balance:.2f}")
            portfolio.current_capital = real_balance
            portfolio.peak_capital = max(portfolio.peak_capital, real_balance)
            # Update circuit breaker too
            circuit_breaker.update_drawdown(real_balance)
            # Update Kelly sizer too? (It's in strategy init, passed via portfolio reference usually or updated dynamically)
    except Exception as e:
        logger.warning("Failed to sync balance", error=str(e))
    
    # Sync to global state for API
    from src.api.state import get_state
    get_state().portfolio = portfolio

    # Create strategy
    strategy = EnsembleStrategy(
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
        logger.info("Starting crypto strategy")
        await strategy.run()
    except asyncio.CancelledError:
        logger.info("Crypto strategy task cancelled")
    except Exception as e:
        logger.exception("Crypto strategy crashed", error=str(e))
        raise # Re-raise to trigger restart logic if needed
    finally:
        # Cleanup
        await market_discovery.close()
        await oracle.close()
        
        # Log final metrics
        metrics = strategy.get_metrics()
        logger.info(
            "Crypto strategy completed",
            signals=metrics["signals_generated"],
            passed_gate=metrics["signals_passed_gate"],
            orders=metrics["orders_placed"],
        )


async def run_event_strategy(config: AppConfig) -> None:
    """Initialize and run the event betting strategy."""
    logger = get_logger(__name__)
    
    if not config.event_trading.enabled:
        logger.info("Event trading disabled")
        return
    
    logger.info(
        "Initializing event strategy",
        categories=config.event_trading.categories,
    )
    
    # Initialize components
    event_market_discovery = EventMarketDiscovery(base_url=config.api.gamma_base_url)
    rate_limiter = RateLimiter()

    # CLOB client for order execution
    clob_client = CLOBClient(
        dry_run=config.dry_run,
        rate_limiter=rate_limiter,
    )
    await clob_client.initialize()

    # Create submit callback
    async def submit_order_callback(order):
        """Submit order to exchange via CLOB client.

        IMPORTANT: Return the exchange response; OrderManager owns state transitions.
        """
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

    # Risk management
    # Risk management
    bankroll = config.trading.bankroll * 0.5  # Allocate 50% to event markets
    
    cb_config = CircuitBreakerConfig(
        daily_loss_soft_pct=config.risk.daily_loss_soft_limit_pct,
        daily_loss_hard_pct=config.risk.daily_loss_hard_limit_pct,
        max_drawdown_hard_pct=config.risk.max_drawdown_pct,
        max_drawdown_soft_pct=config.risk.max_drawdown_pct * 0.8,
        volatility_hard=config.risk.volatility_pause_threshold,
    )
    circuit_breaker = CircuitBreaker(starting_capital=bankroll, config=cb_config)

    from src.portfolio.tracker import Portfolio
    portfolio = Portfolio(starting_capital=bankroll)
    portfolio.load_state()

    # Create strategy
    strategy = EventBettingStrategy(
        config=config,
        market_discovery=event_market_discovery,
        order_manager=order_manager,
        rate_limiter=rate_limiter,
        circuit_breaker=circuit_breaker,
        clob_client=clob_client,
        portfolio=portfolio,
        bankroll=bankroll,
    )
    
    # Set up shutdown handler
    def handle_shutdown():
        logger.info("Shutdown requested for event strategy")
        strategy.stop()
    
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, handle_shutdown)
        except NotImplementedError:
            pass
    
    try:
        logger.info("Starting event betting strategy")
        await strategy.run()
    except asyncio.CancelledError:
        logger.info("Event strategy task cancelled")
    except Exception as e:
        logger.exception("Event strategy crashed", error=str(e))
        raise
    finally:
        # Cleanup
        await event_market_discovery.close()
        
        # Log final metrics
        metrics = strategy.get_metrics()
        logger.info(
            "Event strategy completed",
            markets_analyzed=metrics["markets_analyzed"],
            signals=metrics["signals_generated"],
            orders=metrics["orders_placed"],
            brier_score=metrics.get("brier_score"),
        )


async def run_btc15m_strategy(config: AppConfig) -> None:
    """Initialize and run the BTC 15m strategy."""
    logger = get_logger(__name__)
    
    # Check if enabled in config
    if not getattr(config, 'btc_15m', {}).get('enabled', False):
         logger.info("BTC 15m strategy disabled")
         return

    logger.info("Initializing BTC 15m strategy")
    
    # Initialize OracleFeed for safe price data
    oracle = OracleFeed()
    
    rate_limiter = RateLimiter()
    clob_client = CLOBClient(
        dry_run=config.dry_run,
        rate_limiter=rate_limiter,
    )
    await clob_client.initialize()
    
    strategy = BTC15mStrategy(
        config=config,
        clob_client=clob_client,
        oracle=oracle,
    )
    
    try:
        logger.info("Starting BTC 15m strategy")
        await strategy.run()
    except asyncio.CancelledError:
        logger.info("BTC 15m strategy task cancelled")
    except Exception as e:
        logger.exception("BTC 15m strategy crashed", error=str(e))
    finally:
        await oracle.close()


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
        "Value Betting": True,  # Core
        "Extreme Probability": config.trading.extreme_prob_enabled,
        "Arbitrage": config.trading.arbitrage_enabled,
        "Favorite Fallback": config.trading.favorite_bet_enabled,
        "Market Making": config.market_making.enabled,
        "VPIN Filter": config.vpin.enabled,
        "OBI Signal": config.obi.enabled,
        "Event Markets": config.event_trading.enabled,
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
    async def on_kill_switch(event: KillSwitchEvent):
        """Handle kill switch trigger."""
        logger.critical(
            "Kill switch activated - halting all strategies",
            reason=event.reason,
            type=event.switch_type.value,
        )
        # Cancel all strategy tasks
        for task in [crypto_task, event_task]:
            if not task.done():
                task.cancel()
    
    kill_switch.register_callback(on_kill_switch)
    await kill_switch.start_monitoring()
    
    try:
        # Run both strategies concurrently
        crypto_task = asyncio.create_task(run_crypto_strategy(config))
        event_task = asyncio.create_task(run_event_strategy(config))
        btc_15m_task = asyncio.create_task(run_btc15m_strategy(config))
        
        # Monitor for kill switch in parallel
        async def kill_switch_monitor():
            while True:
                if await kill_switch.check():
                    logger.critical("Kill switch detected - stopping strategies")
                    crypto_task.cancel()
                    event_task.cancel()
                    btc_15m_task.cancel()
                    break
                await asyncio.sleep(1.0)
        
        monitor_task = asyncio.create_task(kill_switch_monitor())
        
        # Use asyncio.gather to keep running until ALL tasks are done (or error)
        # return_exceptions=True prevents one crash from killing the other
        await asyncio.gather(
            crypto_task, 
            event_task, 
            btc_15m_task,
            monitor_task, 
            return_exceptions=True
        )
        
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
