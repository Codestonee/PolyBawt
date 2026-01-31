"""Gamma API parsing models.

Goal: stop scattering fragile parsing logic (stringified JSON, optional fields,
weird date formats) across multiple discovery modules.

These models accept the raw Gamma JSON dict and normalize it into predictable
Python types.

We keep the public-facing dataclasses (`Market`, `EventMarket`) for now to avoid
rewriting strategy code; these Pydantic models are an internal parsing layer.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


def _loads_if_json_str(v: Any) -> Any:
    if isinstance(v, str):
        s = v.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                return json.loads(s)
            except Exception:
                return v
    return v


def _parse_iso_dt(v: Any) -> datetime | None:
    if not v:
        return None
    if isinstance(v, datetime):
        return v
    if not isinstance(v, str):
        return None
    s = v.strip()
    # Gamma uses trailing Z a lot
    if s.endswith("Z"):
        s = s.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


class GammaMarketRaw(BaseModel):
    """Subset of Gamma market fields we actually use."""

    condition_id: str = Field(default="", alias="conditionId")
    question: str = ""
    slug: str = ""

    clob_token_ids: list[str] = Field(default_factory=list, alias="clobTokenIds")
    outcome_prices: list[float] = Field(default_factory=list, alias="outcomePrices")

    end_date: datetime | None = Field(default=None, alias="endDate")
    created_at: datetime | None = Field(default=None, alias="createdAt")

    active: bool = True

    # Keep original raw dict around for debugging
    raw: dict[str, Any] = Field(default_factory=dict)

    @field_validator("clob_token_ids", mode="before")
    @classmethod
    def _parse_clob_ids(cls, v: Any) -> list[str]:
        v = _loads_if_json_str(v)
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x) for x in v]
        return []

    @field_validator("outcome_prices", mode="before")
    @classmethod
    def _parse_outcome_prices(cls, v: Any) -> list[float]:
        v = _loads_if_json_str(v)
        if v is None:
            return []
        if isinstance(v, list):
            out: list[float] = []
            for x in v:
                try:
                    out.append(float(x))
                except Exception:
                    continue
            return out
        return []

    @field_validator("end_date", mode="before")
    @classmethod
    def _parse_end_date(cls, v: Any) -> datetime | None:
        # Some endpoints merge event data and may use alternate keys
        return _parse_iso_dt(v)

    @field_validator("created_at", mode="before")
    @classmethod
    def _parse_created_at(cls, v: Any) -> datetime | None:
        return _parse_iso_dt(v)

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "GammaMarketRaw":
        # Capture raw for debugging.
        # Also support alternate end-date key used in some merged objects.
        enriched = dict(raw)
        if "endDate" not in enriched and "end_date_iso" in enriched:
            enriched["endDate"] = enriched.get("end_date_iso")
        if "createdAt" not in enriched and "created_at" in enriched:
            enriched["createdAt"] = enriched.get("created_at")

        m = cls.model_validate(enriched)
        m.raw = raw
        return m


class GammaEventMarketRaw(BaseModel):
    """Subset for event markets used in EventMarketDiscovery."""

    condition_id: str = Field(default="", alias="conditionId")
    question: str = ""
    outcome_prices: list[float] = Field(default_factory=list, alias="outcomePrices")
    volume_24h: float = Field(default=0.0, alias="volume24hr")
    liquidity: float = 0.0
    end_date: datetime | None = Field(default=None, alias="endDate")

    tags: list[Any] = Field(default_factory=list)
    description: str = ""

    raw: dict[str, Any] = Field(default_factory=dict)

    @field_validator("outcome_prices", mode="before")
    @classmethod
    def _parse_outcome_prices(cls, v: Any) -> list[float]:
        v = _loads_if_json_str(v)
        if v is None:
            return []
        if isinstance(v, list):
            out: list[float] = []
            for x in v:
                try:
                    out.append(float(x))
                except Exception:
                    continue
            return out
        return []

    @field_validator("end_date", mode="before")
    @classmethod
    def _parse_end_date(cls, v: Any) -> datetime | None:
        return _parse_iso_dt(v)

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "GammaEventMarketRaw":
        m = cls.model_validate(raw)
        m.raw = raw
        return m
