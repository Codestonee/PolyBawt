"""Helpers for parsing Polymarket balance/allowance responses."""

from __future__ import annotations

from typing import Any


def parse_usdc_balance(resp: Any) -> float:
    """Best-effort parse of py-clob-client get_balance_allowance response.

    The response format has changed across versions; we accept several shapes.

    Returns:
        Balance in USDC (float).
    """
    if resp is None:
        return 0.0

    # Common: {"balance": "12.34", "allowance": "..."}
    if isinstance(resp, dict) and "balance" in resp:
        return _to_usdc(resp.get("balance"))

    # Sometimes: {"balances": [{"assetType":"COLLATERAL","balance":"..."}, ...]}
    if isinstance(resp, dict) and isinstance(resp.get("balances"), list):
        for item in resp["balances"]:
            if not isinstance(item, dict):
                continue
            at = str(item.get("assetType") or item.get("asset_type") or "").lower()
            if "collateral" in at or at == "0":
                return _to_usdc(item.get("balance"))

    # Fallback: try to coerce whole thing
    return _to_usdc(resp)


def _to_usdc(val: Any) -> float:
    if val is None:
        return 0.0
    try:
        # int-like (maybe base units)
        if isinstance(val, int):
            # Heuristic: if huge, assume 6 decimals base units
            return float(val) / 1_000_000 if val > 1_000_000 else float(val)
        if isinstance(val, float):
            return float(val)
        s = str(val).strip()
        if s == "":
            return 0.0
        # numeric string
        f = float(s)
        # Heuristic: large value => base units
        return f / 1_000_000 if f > 1_000_000 else f
    except Exception:
        return 0.0
