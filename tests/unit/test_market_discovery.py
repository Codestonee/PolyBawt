"""Tests for market resolution helpers."""

import pytest

from src.ingestion.market_discovery import MarketDiscovery


class TestMarketResolution:
    def test_extract_resolution_from_winning_outcome(self):
        md = MarketDiscovery()
        raw = {"resolved": True, "winningOutcome": "Yes"}
        assert md._extract_yes_resolution(raw) is True

    def test_extract_resolution_from_terminal_outcome_prices(self):
        md = MarketDiscovery()
        raw = {
            "resolved": True,
            "outcomes": ["Yes", "No"],
            "outcomePrices": ["0", "1"],
        }
        assert md._extract_yes_resolution(raw) is False

    @pytest.mark.asyncio
    async def test_get_market_resolution_by_token(self):
        md = MarketDiscovery()

        async def fake_get_markets(*args, **kwargs):
            return [
                {
                    "conditionId": "c1",
                    "clobTokenIds": ["t_yes", "t_no"],
                    "resolved": True,
                    "winningOutcome": "No",
                }
            ]

        md.get_markets = fake_get_markets  # type: ignore[assignment]
        outcome = await md.get_market_resolution(token_id="t_yes")
        assert outcome is False
