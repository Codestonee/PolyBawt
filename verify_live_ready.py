"""
Pre-live trading verification script.

Checks all required components before enabling live trading:
1. Configuration validity
2. Required secrets
3. API connectivity
4. Oracle feeds
5. Risk parameters
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("verify_live")


class VerificationResult:
    def __init__(self):
        self.passed = []
        self.warnings = []
        self.failed = []

    def add_pass(self, check: str, detail: str = ""):
        self.passed.append((check, detail))
        logger.info(f"[OK] PASS: {check}" + (f" - {detail}" if detail else ""))

    def add_warning(self, check: str, detail: str):
        self.warnings.append((check, detail))
        logger.warning(f"[!] WARN: {check} - {detail}")

    def add_fail(self, check: str, detail: str):
        self.failed.append((check, detail))
        logger.error(f"[X] FAIL: {check} - {detail}")

    def summary(self) -> bool:
        print("\n" + "=" * 60)
        print("VERIFICATION SUMMARY")
        print("=" * 60)
        print(f"  Passed:   {len(self.passed)}")
        print(f"  Warnings: {len(self.warnings)}")
        print(f"  Failed:   {len(self.failed)}")
        print("=" * 60)

        if self.failed:
            print("\nFAILED CHECKS:")
            for check, detail in self.failed:
                print(f"  - {check}: {detail}")

        if self.warnings:
            print("\nWARNINGS:")
            for check, detail in self.warnings:
                print(f"  - {check}: {detail}")

        if not self.failed:
            print("\n[OK] ALL CRITICAL CHECKS PASSED - Ready for live trading")
            return True
        else:
            print("\n[X] VERIFICATION FAILED - Fix issues before live trading")
            return False


async def verify_config(result: VerificationResult):
    """Verify configuration file."""
    from src.infrastructure.config import load_config

    try:
        config = load_config("config/live.yaml")
        result.add_pass("Config load", f"live.yaml loaded successfully")

        # Check environment
        if config.environment == "live":
            result.add_pass("Environment", "Set to 'live'")
        else:
            result.add_warning("Environment", f"Set to '{config.environment}', expected 'live'")

        # Check dry_run is disabled
        if not config.dry_run:
            result.add_pass("Dry run", "Disabled (live trading enabled)")
        else:
            result.add_fail("Dry run", "Still enabled - set dry_run: false for live trading")

        # Verify trading parameters
        if config.trading.kelly_fraction <= 0.5:
            result.add_pass("Kelly fraction", f"{config.trading.kelly_fraction} (half-Kelly or less)")
        else:
            result.add_warning("Kelly fraction", f"{config.trading.kelly_fraction} is aggressive")

        if config.trading.min_edge_threshold >= 0.03:
            result.add_pass("Min edge", f"{config.trading.min_edge_threshold*100:.1f}% (covers fees)")
        else:
            result.add_warning("Min edge", f"{config.trading.min_edge_threshold*100:.1f}% may not cover fees")

        if config.trading.max_position_pct <= 0.05:
            result.add_pass("Max position", f"{config.trading.max_position_pct*100:.1f}% per trade")
        else:
            result.add_warning("Max position", f"{config.trading.max_position_pct*100:.1f}% is large")

        # Verify risk parameters
        if config.risk.max_drawdown_pct <= 0.15:
            result.add_pass("Max drawdown", f"{config.risk.max_drawdown_pct*100:.1f}% kill switch")
        else:
            result.add_warning("Max drawdown", f"{config.risk.max_drawdown_pct*100:.1f}% is risky")

        if config.risk.daily_loss_hard_limit_pct <= 0.10:
            result.add_pass("Daily loss limit", f"{config.risk.daily_loss_hard_limit_pct*100:.1f}%")
        else:
            result.add_warning("Daily loss limit", f"{config.risk.daily_loss_hard_limit_pct*100:.1f}% is high")

        return config

    except Exception as e:
        result.add_fail("Config load", str(e))
        return None


async def verify_secrets(result: VerificationResult):
    """Verify required secrets are set."""
    from src.infrastructure.config import load_secrets

    secrets = load_secrets()

    # Required for live trading
    required = [
        ("polymarket_private_key", "POLYMARKET_PRIVATE_KEY"),
        ("polymarket_api_key", "POLYMARKET_API_KEY"),
        ("polymarket_api_secret", "POLYMARKET_API_SECRET"),
        ("polymarket_passphrase", "POLYMARKET_PASSPHRASE"),
        ("polymarket_funder_address", "POLYMARKET_FUNDER_ADDRESS"),
    ]

    for attr, env_var in required:
        value = getattr(secrets, attr, "")
        if value:
            # Mask the value for display
            masked = value[:4] + "..." + value[-4:] if len(value) > 8 else "***"
            result.add_pass(f"Secret: {env_var}", f"Set ({masked})")
        else:
            result.add_fail(f"Secret: {env_var}", "Not set - required for live trading")

    # Optional but recommended
    chainlink_id = getattr(secrets, "chainlink_client_id", "")
    chainlink_secret = getattr(secrets, "chainlink_client_secret", "")

    if chainlink_id and chainlink_secret:
        result.add_pass("Chainlink credentials", "Set (sniper protection enabled)")
    else:
        result.add_warning(
            "Chainlink credentials",
            "Not set - sniper protection will use Binance as proxy"
        )

    return secrets


async def verify_api_connectivity(result: VerificationResult, config):
    """Verify API connectivity."""
    import aiohttp

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
        # Test Polymarket CLOB API
        try:
            url = f"{config.api.clob_base_url}/time"
            async with session.get(url) as resp:
                if resp.status == 200:
                    result.add_pass("Polymarket CLOB API", "Connected")
                else:
                    result.add_fail("Polymarket CLOB API", f"Status {resp.status}")
        except Exception as e:
            result.add_fail("Polymarket CLOB API", str(e))

        # Test Gamma API
        try:
            url = f"{config.api.gamma_base_url}/markets"
            async with session.get(url) as resp:
                if resp.status == 200:
                    result.add_pass("Polymarket Gamma API", "Connected")
                else:
                    result.add_fail("Polymarket Gamma API", f"Status {resp.status}")
        except Exception as e:
            result.add_fail("Polymarket Gamma API", str(e))

        # Test Binance API
        try:
            url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    price = float(data["price"])
                    result.add_pass("Binance Oracle", f"BTC=${price:,.2f}")
                else:
                    result.add_fail("Binance Oracle", f"Status {resp.status}")
        except Exception as e:
            result.add_fail("Binance Oracle", str(e))


async def verify_components(result: VerificationResult, config):
    """Verify bot components initialize correctly."""
    from src.execution.order_manager import OrderManager
    from src.ingestion.market_discovery import MarketDiscovery

    # Order manager
    try:
        om = OrderManager(dry_run=True, persistence_file="test_orders.json")
        result.add_pass("OrderManager", "Initialized with persistence")
        # Cleanup test file
        Path("test_orders.json").unlink(missing_ok=True)
    except Exception as e:
        result.add_fail("OrderManager", str(e))

    # Market discovery
    try:
        md = MarketDiscovery()
        result.add_pass("MarketDiscovery", "Initialized")
    except Exception as e:
        result.add_fail("MarketDiscovery", str(e))


async def verify_disk_space(result: VerificationResult):
    """Verify sufficient disk space for logs and state."""
    import shutil

    try:
        usage = shutil.disk_usage(".")
        free_gb = usage.free / (1024**3)
        if free_gb > 1:
            result.add_pass("Disk space", f"{free_gb:.1f} GB free")
        else:
            result.add_warning("Disk space", f"Only {free_gb:.2f} GB free")
    except Exception as e:
        result.add_warning("Disk space", f"Could not check: {e}")


async def main():
    print("=" * 60)
    print("POLYMARKET BOT - LIVE TRADING VERIFICATION")
    print("=" * 60)
    print()

    result = VerificationResult()

    # 1. Config
    print("\n[1/5] Verifying configuration...")
    config = await verify_config(result)
    if not config:
        print("\nCannot continue without valid config")
        sys.exit(1)

    # 2. Secrets
    print("\n[2/5] Verifying secrets...")
    await verify_secrets(result)

    # 3. API connectivity
    print("\n[3/5] Verifying API connectivity...")
    await verify_api_connectivity(result, config)

    # 4. Components
    print("\n[4/5] Verifying components...")
    await verify_components(result, config)

    # 5. System resources
    print("\n[5/5] Verifying system resources...")
    await verify_disk_space(result)

    # Summary
    success = result.summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
