"""
Idempotency Registry for PolyBawt.

Prevents duplicate order submissions through:
- Client Order ID (COID) format: {run}:{strat}:{token}:{ts}:{nonce}
- In-memory duplicate detection
- Periodic disk persistence
- Intent-level deduplication

Usage:
    registry = IdempotencyRegistry()
    coid = registry.generate_coid("SpreadMaker", "token_123")
    if registry.can_submit(coid):
        registry.mark_submitted(coid)
"""

from __future__ import annotations
import os
import json
import time
import uuid
import hashlib
from dataclasses import dataclass, field
from typing import Dict, Set, Optional
from pathlib import Path
from threading import Lock

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


# Run ID generated at startup
_RUN_ID: str = hashlib.md5(f"{time.time()}{uuid.uuid4()}".encode()).hexdigest()[:8]


@dataclass
class IntentRecord:
    """Record of an order intent."""
    intent_id: str
    client_order_id: str
    strategy: str
    token_id: str
    side: str
    price: float
    size: float
    created_ns: int
    submitted_ns: Optional[int] = None
    acked_ns: Optional[int] = None
    status: str = "pending"  # pending, submitted, acked, rejected, expired


class IdempotencyRegistry:
    """
    Idempotency registry for order deduplication.
    
    Ensures no duplicate orders are submitted by tracking:
    - Generated client order IDs
    - Intent submission status
    - Recent order history
    
    Features:
    - Thread-safe operations
    - Periodic disk persistence
    - Configurable TTL for old entries
    """
    
    def __init__(
        self,
        run_id: Optional[str] = None,
        persist_path: Optional[str] = None,
        ttl_seconds: int = 3600,  # Keep entries for 1 hour
        flush_interval_seconds: int = 60,
    ):
        self.run_id = run_id or _RUN_ID
        self.persist_path = Path(persist_path) if persist_path else None
        self.ttl_seconds = ttl_seconds
        self.flush_interval = flush_interval_seconds
        
        self._submitted: Set[str] = set()
        self._intents: Dict[str, IntentRecord] = {}
        self._nonce_counter: int = 0
        self._lock = Lock()
        self._last_flush = time.time()
        
        # Load persisted state if available
        if self.persist_path and self.persist_path.exists():
            self._load_state()
        
        logger.info(f"[Idempotency] Initialized with run_id={self.run_id}")
    
    def generate_coid(
        self,
        strategy: str,
        token_id: str,
        intent_ts_ns: Optional[int] = None,
    ) -> str:
        """
        Generate a unique Client Order ID.
        
        Format: COID:{run_id}:{strategy}:{token_short}:{ts}:{nonce}
        
        Args:
            strategy: Strategy name
            token_id: Token identifier
            intent_ts_ns: Optional timestamp (defaults to now)
            
        Returns:
            Unique client order ID string
        """
        with self._lock:
            self._nonce_counter += 1
            nonce = self._nonce_counter
        
        ts = intent_ts_ns or time.time_ns()
        token_short = token_id[:8] if len(token_id) > 8 else token_id
        
        return f"COID:{self.run_id}:{strategy}:{token_short}:{ts}:{nonce}"
    
    def generate_intent_id(self) -> str:
        """Generate a unique intent ID."""
        return f"INT:{self.run_id}:{time.time_ns()}:{uuid.uuid4().hex[:8]}"
    
    def can_submit(self, client_order_id: str) -> bool:
        """Check if an order ID can be submitted (not a duplicate)."""
        with self._lock:
            return client_order_id not in self._submitted
    
    def mark_submitted(self, client_order_id: str) -> bool:
        """
        Mark an order as submitted.
        
        Returns:
            True if marked, False if was already submitted (duplicate)
        """
        with self._lock:
            if client_order_id in self._submitted:
                logger.warning(f"[Idempotency] Duplicate submission blocked: {client_order_id}")
                return False
            self._submitted.add(client_order_id)
        
        self._maybe_flush()
        return True
    
    def register_intent(
        self,
        strategy: str,
        token_id: str,
        side: str,
        price: float,
        size: float,
    ) -> IntentRecord:
        """
        Register an order intent and generate IDs.
        
        Returns:
            IntentRecord with generated intent_id and client_order_id
        """
        intent_id = self.generate_intent_id()
        client_order_id = self.generate_coid(strategy, token_id)
        
        record = IntentRecord(
            intent_id=intent_id,
            client_order_id=client_order_id,
            strategy=strategy,
            token_id=token_id,
            side=side,
            price=price,
            size=size,
            created_ns=time.time_ns(),
        )
        
        with self._lock:
            self._intents[intent_id] = record
        
        return record
    
    def mark_intent_submitted(self, intent_id: str) -> bool:
        """Mark an intent as submitted."""
        with self._lock:
            if intent_id not in self._intents:
                return False
            record = self._intents[intent_id]
            if record.status != "pending":
                return False
            
            record.submitted_ns = time.time_ns()
            record.status = "submitted"
            self._submitted.add(record.client_order_id)
        
        return True
    
    def mark_intent_acked(self, intent_id: str) -> bool:
        """Mark an intent as acknowledged by exchange."""
        with self._lock:
            if intent_id not in self._intents:
                return False
            record = self._intents[intent_id]
            record.acked_ns = time.time_ns()
            record.status = "acked"
        return True
    
    def mark_intent_rejected(self, intent_id: str) -> bool:
        """Mark an intent as rejected."""
        with self._lock:
            if intent_id not in self._intents:
                return False
            self._intents[intent_id].status = "rejected"
        return True
    
    def get_intent(self, intent_id: str) -> Optional[IntentRecord]:
        """Get an intent record by ID."""
        return self._intents.get(intent_id)
    
    def get_pending_intents(self) -> list[IntentRecord]:
        """Get all pending (not yet submitted) intents."""
        with self._lock:
            return [r for r in self._intents.values() if r.status == "pending"]
    
    def get_unacked_intents(self, timeout_seconds: float = 5.0) -> list[IntentRecord]:
        """Get submitted but not acked intents older than timeout."""
        cutoff_ns = time.time_ns() - int(timeout_seconds * 1e9)
        with self._lock:
            return [
                r for r in self._intents.values()
                if r.status == "submitted" and r.submitted_ns and r.submitted_ns < cutoff_ns
            ]
    
    def cleanup_old_entries(self):
        """Remove entries older than TTL."""
        cutoff_ns = time.time_ns() - int(self.ttl_seconds * 1e9)
        
        with self._lock:
            old_intents = [
                k for k, v in self._intents.items()
                if v.created_ns < cutoff_ns
            ]
            for k in old_intents:
                del self._intents[k]
        
        if old_intents:
            logger.debug(f"[Idempotency] Cleaned up {len(old_intents)} old intents")
    
    def _maybe_flush(self):
        """Flush to disk if interval has passed."""
        if not self.persist_path:
            return
        
        now = time.time()
        if now - self._last_flush > self.flush_interval:
            self._flush_state()
            self._last_flush = now
    
    def _flush_state(self):
        """Persist state to disk."""
        if not self.persist_path:
            return
        
        with self._lock:
            state = {
                "run_id": self.run_id,
                "submitted": list(self._submitted),
                "nonce_counter": self._nonce_counter,
            }
        
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.persist_path, "w") as f:
            json.dump(state, f)
        
        logger.debug(f"[Idempotency] Flushed state to {self.persist_path}")
    
    def _load_state(self):
        """Load state from disk."""
        try:
            with open(self.persist_path, "r") as f:
                state = json.load(f)
            
            # Only load if same run
            if state.get("run_id") == self.run_id:
                self._submitted = set(state.get("submitted", []))
                self._nonce_counter = state.get("nonce_counter", 0)
                logger.info(f"[Idempotency] Loaded {len(self._submitted)} submitted IDs")
        except Exception as e:
            logger.warning(f"[Idempotency] Failed to load state: {e}")
    
    def stats(self) -> Dict[str, int]:
        """Get registry statistics."""
        with self._lock:
            return {
                "submitted_count": len(self._submitted),
                "intent_count": len(self._intents),
                "pending_count": len([r for r in self._intents.values() if r.status == "pending"]),
                "nonce_counter": self._nonce_counter,
            }
