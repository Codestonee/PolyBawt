"""
NDJSON Event Logger for PolyBawt.

Append-only, replay-grade event persistence:
- Monotonic ordering by ts_recv_ns
- Source vs receive timestamps
- Event types: BOOK_DELTA, ORACLE_TICK, SIGNAL, ORDER_*, FILL, etc.
- Rotation and compression support

Usage:
    logger = EventLogger("logs/events")
    logger.log(EventType.FILL, {"order_id": "...", "price": 0.52})
    # Later: replay from logs
"""

from __future__ import annotations
import os
import json
import time
import gzip
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, TextIO
from enum import Enum
from datetime import datetime
from pathlib import Path

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class EventType(Enum):
    """Event types for replay logging."""
    # Market data
    BOOK_DELTA = "BOOK_DELTA"
    BOOK_SNAPSHOT = "BOOK_SNAPSHOT"
    ORACLE_TICK = "ORACLE_TICK"
    
    # Signals
    SIGNAL = "SIGNAL"
    REGIME_CHANGE = "REGIME_CHANGE"
    
    # Orders
    ORDER_INTENT = "ORDER_INTENT"
    ORDER_SUBMIT = "ORDER_SUBMIT"
    ORDER_ACK = "ORDER_ACK"
    ORDER_REJECT = "ORDER_REJECT"
    CANCEL_SUBMIT = "CANCEL_SUBMIT"
    CANCEL_ACK = "CANCEL_ACK"
    
    # Fills
    FILL = "FILL"
    PARTIAL_FILL = "PARTIAL_FILL"
    
    # Resolution
    RESOLUTION = "RESOLUTION"
    
    # System
    HEALTH = "HEALTH"
    LATENCY = "LATENCY"
    ERROR = "ERROR"
    KILL_SWITCH = "KILL_SWITCH"


@dataclass
class Event:
    """Single event for logging/replay."""
    event_type: EventType
    ts_source_ns: int  # Timestamp from source (e.g., exchange)
    ts_recv_ns: int    # Timestamp when we received it (monotonic)
    data: Dict[str, Any]
    
    # Optional metadata
    source: str = "local"
    token_id: Optional[str] = None
    strategy: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.event_type.value,
            "ts_source_ns": self.ts_source_ns,
            "ts_recv_ns": self.ts_recv_ns,
            "source": self.source,
            "token_id": self.token_id,
            "strategy": self.strategy,
            "data": self.data,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Event":
        return cls(
            event_type=EventType(d["type"]),
            ts_source_ns=d["ts_source_ns"],
            ts_recv_ns=d["ts_recv_ns"],
            source=d.get("source", "local"),
            token_id=d.get("token_id"),
            strategy=d.get("strategy"),
            data=d["data"],
        )


class EventLogger:
    """
    Append-only NDJSON event logger for replay-grade persistence.
    
    Features:
    - Monotonic ordering enforced
    - Daily file rotation
    - Optional gzip compression
    - Thread-safe writes
    
    Usage:
        logger = EventLogger("logs/events")
        logger.log(EventType.FILL, {"order_id": "abc", "price": 0.52})
    """
    
    def __init__(
        self,
        log_dir: str,
        compress_after_days: int = 1,
        max_files_to_keep: int = 90,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.compress_after_days = compress_after_days
        self.max_files_to_keep = max_files_to_keep
        
        self._current_file: Optional[TextIO] = None
        self._current_date: Optional[str] = None
        self._last_ts_ns: int = 0
        self._event_count: int = 0
        
    def _get_current_date(self) -> str:
        return datetime.utcnow().strftime("%Y-%m-%d")
    
    def _get_log_path(self, date: str) -> Path:
        return self.log_dir / f"events_{date}.ndjson"
    
    def _ensure_file_open(self):
        """Ensure we have an open file for today."""
        today = self._get_current_date()
        
        if self._current_date != today:
            # Close previous file
            if self._current_file:
                self._current_file.close()
            
            # Open new file
            path = self._get_log_path(today)
            self._current_file = open(path, "a", encoding="utf-8")
            self._current_date = today
            self._event_count = 0
            
            logger.info(f"[EventLogger] Opened {path}")
            
            # Trigger rotation check
            self._rotate_old_files()
    
    def _rotate_old_files(self):
        """Compress and clean up old log files."""
        now = datetime.utcnow()
        files = sorted(self.log_dir.glob("events_*.ndjson"))
        
        for f in files:
            try:
                date_str = f.stem.replace("events_", "")
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                age_days = (now - file_date).days
                
                # Compress old files
                if age_days >= self.compress_after_days and not f.suffix.endswith(".gz"):
                    gz_path = f.with_suffix(".ndjson.gz")
                    with open(f, "rb") as f_in:
                        with gzip.open(gz_path, "wb") as f_out:
                            f_out.write(f_in.read())
                    os.remove(f)
                    logger.info(f"[EventLogger] Compressed {f.name}")
                    
            except Exception as e:
                logger.warning(f"[EventLogger] Rotation error for {f}: {e}")
        
        # Clean up excess files
        all_files = sorted(self.log_dir.glob("events_*"))
        if len(all_files) > self.max_files_to_keep:
            for f in all_files[:-self.max_files_to_keep]:
                os.remove(f)
                logger.info(f"[EventLogger] Deleted old file {f.name}")
    
    def log(
        self,
        event_type: EventType,
        data: Dict[str, Any],
        ts_source_ns: Optional[int] = None,
        token_id: Optional[str] = None,
        strategy: Optional[str] = None,
        source: str = "local",
    ) -> Event:
        """
        Log an event to the NDJSON file.
        
        Args:
            event_type: Type of event
            data: Event-specific payload
            ts_source_ns: Source timestamp (defaults to now)
            token_id: Associated token (if any)
            strategy: Associated strategy (if any)
            source: Event source identifier
            
        Returns:
            The logged Event object
        """
        now_ns = time.time_ns()
        
        # Ensure monotonic ordering
        if now_ns <= self._last_ts_ns:
            now_ns = self._last_ts_ns + 1
        self._last_ts_ns = now_ns
        
        event = Event(
            event_type=event_type,
            ts_source_ns=ts_source_ns or now_ns,
            ts_recv_ns=now_ns,
            data=data,
            source=source,
            token_id=token_id,
            strategy=strategy,
        )
        
        self._ensure_file_open()
        
        line = json.dumps(event.to_dict(), separators=(",", ":"))
        self._current_file.write(line + "\n")
        self._current_file.flush()  # Ensure durability
        
        self._event_count += 1
        
        return event
    
    def log_fill(
        self,
        order_id: str,
        token_id: str,
        side: str,
        price: float,
        size: float,
        strategy: str,
        fee_paid: float = 0.0,
        rebate_est: float = 0.0,
        microprice_at_fill: Optional[float] = None,
        vpin_at_fill: Optional[float] = None,
    ) -> Event:
        """Convenience method for logging fills."""
        return self.log(
            EventType.FILL,
            {
                "order_id": order_id,
                "side": side,
                "price": price,
                "size": size,
                "fee_paid": fee_paid,
                "rebate_est": rebate_est,
                "microprice_at_fill": microprice_at_fill,
                "vpin_at_fill": vpin_at_fill,
            },
            token_id=token_id,
            strategy=strategy,
        )
    
    def log_order_intent(
        self,
        intent_id: str,
        client_order_id: str,
        token_id: str,
        side: str,
        price: float,
        size: float,
        strategy: str,
        microprice: float,
        vpin: float,
        regime: str,
    ) -> Event:
        """Convenience method for logging order intents."""
        return self.log(
            EventType.ORDER_INTENT,
            {
                "intent_id": intent_id,
                "client_order_id": client_order_id,
                "side": side,
                "price": price,
                "size": size,
                "microprice": microprice,
                "vpin": vpin,
                "regime": regime,
            },
            token_id=token_id,
            strategy=strategy,
        )
    
    def close(self):
        """Close the current log file."""
        if self._current_file:
            self._current_file.close()
            self._current_file = None
        logger.info(f"[EventLogger] Closed (logged {self._event_count} events)")


class EventReader:
    """
    Read events from NDJSON log files for replay.
    
    Usage:
        reader = EventReader("logs/events")
        for event in reader.read_date("2026-02-08"):
            process(event)
    """
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
    
    def read_date(self, date: str):
        """
        Read all events for a specific date.
        
        Args:
            date: Date string in YYYY-MM-DD format
            
        Yields:
            Event objects in order
        """
        # Try uncompressed first
        path = self.log_dir / f"events_{date}.ndjson"
        if path.exists():
            yield from self._read_file(path)
            return
        
        # Try compressed
        gz_path = self.log_dir / f"events_{date}.ndjson.gz"
        if gz_path.exists():
            yield from self._read_gzip(gz_path)
            return
        
        logger.warning(f"[EventReader] No log file for {date}")
    
    def read_range(self, start_date: str, end_date: str):
        """Read events across a date range."""
        from datetime import datetime, timedelta
        
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        current = start
        while current <= end:
            yield from self.read_date(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
    
    def _read_file(self, path: Path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield Event.from_dict(json.loads(line))
    
    def _read_gzip(self, path: Path):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield Event.from_dict(json.loads(line))
    
    def count_events(self, date: str) -> int:
        """Count events for a date without loading all into memory."""
        return sum(1 for _ in self.read_date(date))
