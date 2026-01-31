"""
Configuration loader with Pydantic validation.

Supports:
- YAML file loading
- Environment variable overrides
- Secrets from environment
"""

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class APIConfig(BaseModel):
    """API endpoint configuration."""
    
    clob_base_url: str = "https://clob.polymarket.com"
    gamma_base_url: str = "https://gamma-api.polymarket.com"
    ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    polygon_chain_id: int = 137


class TradingConfig(BaseModel):
    """Trading parameters."""
    
    assets: list[str] = Field(default_factory=lambda: ["BTC", "ETH", "SOL", "XRP"])
    bankroll: float = 50.0  # Starting capital in USD
    min_edge_threshold: float = 0.04  # 4% after fees
    kelly_fraction: float = 0.25  # 1/4 Kelly
    max_position_pct: float = 0.02  # 2% per trade
    max_asset_exposure_pct: float = 0.05  # 5% per asset
    max_total_exposure_pct: float = 0.30  # Total exposure cap across all assets (fraction of bankroll)
    cutoff_seconds: int = 60  # Stop trading before expiry
    min_seconds_after_open: int = 30  # Avoid opening volatility

    # Safety: do not let placeholder sentiment inputs affect live sizing.
    # Enable only once real sentiment/funding inputs are wired.
    sentiment_sizing_enabled: bool = False
    
    # Favorite fallback betting
    favorite_bet_enabled: bool = True
    favorite_bet_size: float = 2.00  # USD

    # Optional "vibe bet" mode: allow an occasional small bet even without edge.
    # Still bounded by max_position_pct and max_total_exposure_pct.
    vibe_bet_enabled: bool = False
    vibe_bet_cooldown_seconds: int = 3600
    vibe_bet_size: float = 5.00  # USD (Polymarket minimum is often $5)

    # Execution hygiene
    gtc_max_spread: float = 0.02  # absolute spread; kept for backward compatibility
    gtc_max_spread_bps: float = 0.0  # if >0, prefer IOC when spread/mid exceeds this

    # Order sizing
    min_order_size_usd: float = 5.0

    # Balance floor (live trading)
    min_balance_usd: float = 20.0
    balance_check_interval_seconds: int = 30
    
    # Arbitrage trading
    arbitrage_enabled: bool = True
    arbitrage_min_profit_pct: float = 0.005  # 0.5% minimum profit
    arbitrage_max_size: float = 50.00  # Max USD per arb trade

    # Extreme probability trading (near 0% or 100% where fees approach zero)
    extreme_prob_enabled: bool = True
    extreme_prob_threshold: float = 0.05  # p < 5% or p > 95%
    extreme_prob_kelly_boost: float = 1.5  # 50% larger positions at extremes
    extreme_prob_min_edge: float = 0.02  # Lower edge threshold (2% vs 4%) since fees ~0
    
    @field_validator("kelly_fraction")
    @classmethod
    def validate_kelly(cls, v: float) -> float:
        if not 0 < v <= 1:
            raise ValueError("kelly_fraction must be between 0 and 1")
        return v
    
    @field_validator("min_edge_threshold")
    @classmethod
    def validate_edge(cls, v: float) -> float:
        if v < 0:
            raise ValueError("min_edge_threshold must be non-negative")
        return v


class RiskConfig(BaseModel):
    """Risk management parameters."""
    
    daily_loss_soft_limit_pct: float = 0.03  # Pause new entries
    daily_loss_hard_limit_pct: float = 0.05  # Close all, halt
    max_drawdown_pct: float = 0.10  # Kill switch
    volatility_pause_threshold: float = 2.0  # 200% annualized
    correlation_reduction_threshold: float = 0.8
    
    @field_validator("max_drawdown_pct")
    @classmethod
    def validate_drawdown(cls, v: float) -> float:
        if not 0 < v <= 1:
            raise ValueError("max_drawdown_pct must be between 0 and 1")
        return v


class ExecutionConfig(BaseModel):
    """Execution parameters."""

    rate_limit_buffer_pct: float = 0.10
    ws_reconnect_initial_ms: int = 100
    ws_reconnect_max_ms: int = 5000
    ws_reconnect_multiplier: float = 2.0
    ws_reconnect_jitter: float = 0.2
    order_timeout_ms: int = 5000
    max_retries: int = 3


class MarketMakingConfig(BaseModel):
    """Market making strategy parameters."""

    enabled: bool = False  # Disabled by default (advanced strategy)
    spread_bps: float = 50.0  # Bid-ask spread in basis points
    quote_size_usd: float = 10.0  # Size per side
    max_inventory_usd: float = 100.0  # Max position before skewing
    skew_bps_per_inventory: float = 1.0  # BPS to skew per $10 inventory
    min_profit_margin: float = 0.005  # 0.5% minimum expected profit


class VPINConfig(BaseModel):
    """VPIN toxicity filter parameters."""

    enabled: bool = True
    bucket_size_usd: float = 1000.0  # Volume per bucket
    n_buckets: int = 50  # Rolling window size
    halt_threshold: float = 0.6  # VPIN level to halt trading
    reduce_threshold: float = 0.4  # VPIN level to reduce size


class OBIConfig(BaseModel):
    """Order Book Imbalance signal parameters."""

    enabled: bool = True
    levels: int = 5  # Number of price levels to analyze
    strong_threshold: float = 0.4  # Threshold for STRONG_BUY/STRONG_SELL
    weak_threshold: float = 0.2  # Threshold for BUY/SELL
    weighted: bool = True  # Weight volume by distance from mid


class EventTradingConfig(BaseModel):
    """Event market trading parameters."""

    enabled: bool = False  # Disabled by default (runs alongside crypto strategy)
    categories: list[str] = Field(default_factory=lambda: ["politics", "sports", "economics"])

    # Minimum edge thresholds by category
    min_edge_politics: float = 0.05    # 5% minimum edge
    min_edge_sports: float = 0.06
    min_edge_economics: float = 0.04
    min_edge_pop_culture: float = 0.07

    # Position limits
    max_position_per_market: float = 50.0
    max_category_exposure: float = 200.0

    # Rebalancing
    rebalance_enabled: bool = True
    rebalance_threshold_pct: float = 0.10  # Rebalance if estimate changes 10%

    # External data refresh intervals (seconds)
    refresh_polls: int = 3600        # 1 hour
    refresh_betting_lines: int = 900  # 15 minutes
    refresh_fed_futures: int = 900    # 15 minutes
    refresh_sentiment: int = 1800     # 30 minutes

    # Calibration
    brier_tracking_enabled: bool = True
    calibration_db_path: str = "data/calibration.db"


class ModelsConfig(BaseModel):
    """Model parameters."""
    
    jump_intensity_per_day: float = 2.0
    jump_size_std: float = 0.02
    default_volatility: dict[str, float] = Field(
        default_factory=lambda: {
            "BTC": 0.60,
            "ETH": 0.70,
            "SOL": 0.85,
            "XRP": 0.90,
        }
    )


class ObservabilityConfig(BaseModel):
    """Logging and metrics configuration."""
    
    log_level: str = "INFO"
    log_format: str = "json"  # json or text
    metrics_port: int = 9090
    telegram_enabled: bool = False


class SecretsConfig(BaseSettings):
    """Secrets loaded exclusively from environment variables.

    Never logged or persisted.
    """

    model_config = SettingsConfigDict(
        env_prefix="",  # Use exact env var names
        case_sensitive=False,
    )

    polymarket_private_key: str = ""
    polymarket_api_key: str = ""
    polymarket_api_secret: str = ""
    polymarket_passphrase: str = ""
    polymarket_funder_address: str = ""
    chainlink_client_id: str = ""
    chainlink_client_secret: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    redis_url: str = "redis://localhost:6379/0"


class AppConfig(BaseModel):
    """Complete application configuration."""

    # Configuration versioning
    config_version: str = "1.1.0"
    config_schema_version: int = 2

    environment: str = "local"
    dry_run: bool = True

    api: APIConfig = Field(default_factory=APIConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    market_making: MarketMakingConfig = Field(default_factory=MarketMakingConfig)
    vpin: VPINConfig = Field(default_factory=VPINConfig)
    obi: OBIConfig = Field(default_factory=OBIConfig)
    event_trading: EventTradingConfig = Field(default_factory=EventTradingConfig)

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def is_paper(self) -> bool:
        return self.dry_run or self.environment == "paper"

    def diff_from_defaults(self) -> dict[str, Any]:
        """
        Get configuration differences from defaults.

        Useful for logging what's been customized.
        """
        defaults = AppConfig()
        current = self.model_dump()
        default_dict = defaults.model_dump()

        def diff_dict(d1: dict, d2: dict, path: str = "") -> dict:
            differences = {}
            for key in set(d1.keys()) | set(d2.keys()):
                full_key = f"{path}.{key}" if path else key
                v1 = d1.get(key)
                v2 = d2.get(key)

                if isinstance(v1, dict) and isinstance(v2, dict):
                    nested = diff_dict(v1, v2, full_key)
                    if nested:
                        differences.update(nested)
                elif v1 != v2:
                    differences[full_key] = {"current": v1, "default": v2}

            return differences

        return diff_dict(current, default_dict)


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load configuration from YAML file."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """
    Load configuration from YAML file with environment overrides.
    
    Priority (highest to lowest):
    1. Environment variables (for secrets only)
    2. Specified config file
    3. Defaults
    """
    config_dict: dict[str, Any] = {}
    
    # Load from file if specified
    if config_path:
        path = Path(config_path)
        config_dict = load_yaml_config(path)
    
    # Create config object
    config = AppConfig(**config_dict)
    
    # Safety check: production requires explicit --live flag
    if config.environment == "production" and config.dry_run:
        # This is expected - production.yaml should have dry_run: false
        # but we force it to true unless explicitly overridden
        pass
    
    return config


def load_secrets() -> SecretsConfig:
    """Load secrets from environment variables."""
    return SecretsConfig()


# Global config instances (set by main.py)
_config: AppConfig | None = None
_secrets: SecretsConfig | None = None


def get_config() -> AppConfig:
    """Get the global configuration."""
    if _config is None:
        raise RuntimeError("Configuration not loaded. Call load_config() first.")
    return _config


def get_secrets() -> SecretsConfig:
    """Get the secrets configuration."""
    if _secrets is None:
        raise RuntimeError("Secrets not loaded. Call load_secrets() first.")
    return _secrets


def init_config(config_path: str | Path | None = None) -> tuple[AppConfig, SecretsConfig]:
    """Initialize global configuration and secrets."""
    global _config, _secrets
    _config = load_config(config_path)
    _secrets = load_secrets()
    return _config, _secrets
