"""Tests for configuration loading."""

import pytest
from pathlib import Path

from src.infrastructure.config import (
    AppConfig,
    TradingConfig,
    RiskConfig,
    load_yaml_config,
    deep_merge,
)


class TestTradingConfig:
    """Tests for TradingConfig validation."""
    
    def test_default_assets(self):
        """Default assets should be BTC, ETH, SOL, XRP."""
        config = TradingConfig()
        assert config.assets == ["BTC", "ETH", "SOL", "XRP"]
    
    def test_kelly_fraction_validation(self):
        """Kelly fraction must be between 0 and 1."""
        with pytest.raises(ValueError):
            TradingConfig(kelly_fraction=1.5)
        
        with pytest.raises(ValueError):
            TradingConfig(kelly_fraction=0)
        
        # Valid values
        config = TradingConfig(kelly_fraction=0.25)
        assert config.kelly_fraction == 0.25
    
    def test_edge_threshold_validation(self):
        """Edge threshold must be non-negative."""
        with pytest.raises(ValueError):
            TradingConfig(min_edge_threshold=-0.01)
        
        # Zero is valid
        config = TradingConfig(min_edge_threshold=0)
        assert config.min_edge_threshold == 0

    def test_positive_safety_fields_validation(self):
        with pytest.raises(ValueError):
            TradingConfig(max_position_per_token_usd=0)
        with pytest.raises(ValueError):
            TradingConfig(order_book_ttl_seconds=0)
        with pytest.raises(ValueError):
            TradingConfig(market_concurrency=0)
        with pytest.raises(ValueError):
            TradingConfig(inter_market_jitter_ms_min=-1)


class TestRiskConfig:
    """Tests for RiskConfig validation."""
    
    def test_drawdown_validation(self):
        """Max drawdown must be between 0 and 1."""
        with pytest.raises(ValueError):
            RiskConfig(max_drawdown_pct=1.5)
        
        with pytest.raises(ValueError):
            RiskConfig(max_drawdown_pct=0)
        
        # Valid value
        config = RiskConfig(max_drawdown_pct=0.10)
        assert config.max_drawdown_pct == 0.10


class TestAppConfig:
    """Tests for AppConfig."""
    
    def test_default_dry_run(self):
        """Default should be dry run mode."""
        config = AppConfig()
        assert config.dry_run is True
    
    def test_is_paper_when_dry_run(self):
        """is_paper should be True when dry_run is True."""
        config = AppConfig(dry_run=True)
        assert config.is_paper is True
    
    def test_is_production(self):
        """is_production should be True only for production env."""
        config = AppConfig(environment="production")
        assert config.is_production is True
        
        config = AppConfig(environment="paper")
        assert config.is_production is False


class TestDeepMerge:
    """Tests for deep_merge utility."""
    
    def test_simple_merge(self):
        """Simple key override."""
        base = {"a": 1, "b": 2}
        override = {"b": 3}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 3}
    
    def test_nested_merge(self):
        """Nested dict merge."""
        base = {"a": {"x": 1, "y": 2}}
        override = {"a": {"y": 3}}
        result = deep_merge(base, override)
        assert result == {"a": {"x": 1, "y": 3}}
    
    def test_new_key(self):
        """New key addition."""
        base = {"a": 1}
        override = {"b": 2}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 2}
