"""
Lightweight NDJSON recorder for strategy research datasets.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from threading import Lock
from typing import Any

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class ResearchRecorder:
    """Append-only NDJSON recorder for market snapshots, signals and fills."""

    def __init__(self, base_dir: str = "data/research") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def _append(self, filename: str, payload: dict[str, Any]) -> None:
        record = {"ts": time.time(), **payload}
        path = self.base_dir / filename
        line = json.dumps(record, separators=(",", ":"), default=str)
        try:
            with self._lock:
                with path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception as e:
            logger.warning("Research recorder write failed", file=str(path), error=str(e))

    def record_market_snapshot(self, payload: dict[str, Any]) -> None:
        self._append("market_snapshots.ndjson", payload)

    def record_signal(self, payload: dict[str, Any]) -> None:
        self._append("signals.ndjson", payload)

    def record_fill(self, payload: dict[str, Any]) -> None:
        self._append("fills.ndjson", payload)
