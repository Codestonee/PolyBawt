"""Database for storing predictions and outcomes."""

import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import aiosqlite
    HAS_AIOSQLITE = True
except ImportError:
    HAS_AIOSQLITE = False

from src.ingestion.event_market_discovery import MarketCategory
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PredictionRecord:
    """A recorded prediction with outcome."""
    
    prediction_id: str
    market_id: str
    market_question: str
    category: str
    
    # Prediction details
    predicted_probability: float
    model_name: str
    features: dict[str, Any]
    
    # Market state at prediction
    market_price: float
    time_to_resolution_hours: float | None
    
    # Timestamps
    predicted_at: datetime
    resolved_at: datetime | None = None
    
    # Outcome
    actual_outcome: bool | None = None
    brier_score: float | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prediction_id": self.prediction_id,
            "market_id": self.market_id,
            "market_question": self.market_question,
            "category": self.category,
            "predicted_probability": self.predicted_probability,
            "model_name": self.model_name,
            "features": json.dumps(self.features),
            "market_price": self.market_price,
            "time_to_resolution_hours": self.time_to_resolution_hours,
            "predicted_at": self.predicted_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "actual_outcome": self.actual_outcome,
            "brier_score": self.brier_score,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PredictionRecord":
        """Create from dictionary."""
        return cls(
            prediction_id=data["prediction_id"],
            market_id=data["market_id"],
            market_question=data["market_question"],
            category=data["category"],
            predicted_probability=data["predicted_probability"],
            model_name=data["model_name"],
            features=json.loads(data["features"]),
            market_price=data["market_price"],
            time_to_resolution_hours=data.get("time_to_resolution_hours"),
            predicted_at=datetime.fromisoformat(data["predicted_at"]),
            resolved_at=datetime.fromisoformat(data["resolved_at"]) if data.get("resolved_at") else None,
            actual_outcome=data.get("actual_outcome"),
            brier_score=data.get("brier_score"),
        )


class CalibrationDatabase:
    """
    SQLite database for storing predictions and outcomes.
    
    Enables:
    - Long-term Brier score tracking
    - Model performance analysis
    - Walk-forward validation
    - Feature importance analysis
    
    Schema:
    - predictions: All predictions made
    - outcomes: Resolved market outcomes
    - model_performance: Aggregated stats by model
    
    Usage:
        db = CalibrationDatabase("data/calibration.db")
        
        # Record prediction
        db.record_prediction(record)
        
        # Record outcome
        db.record_outcome(market_id, outcome=True)
        
        # Query results
        results = db.get_predictions(category="politics", resolved_only=True)
    """
    
    def __init__(self, db_path: str = "data/calibration.db"):
        """
        Initialize calibration database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id TEXT PRIMARY KEY,
                    market_id TEXT NOT NULL,
                    market_question TEXT,
                    category TEXT,
                    predicted_probability REAL NOT NULL,
                    model_name TEXT,
                    features TEXT,  -- JSON
                    market_price REAL,
                    time_to_resolution_hours REAL,
                    predicted_at TIMESTAMP NOT NULL,
                    resolved_at TIMESTAMP,
                    actual_outcome INTEGER,  -- 0 or 1
                    brier_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_market_id ON predictions(market_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_category ON predictions(category)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_predicted_at ON predictions(predicted_at)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_resolved ON predictions(resolved_at)
            """)
            
            conn.commit()
        
        logger.debug("Calibration database initialized", path=str(self.db_path))
    
    def record_prediction(self, record: PredictionRecord) -> bool:
        """
        Record a prediction in the database (sync version).
        
        Args:
            record: Prediction record
            
        Returns:
            True if successful
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO predictions (
                        prediction_id, market_id, market_question, category,
                        predicted_probability, model_name, features,
                        market_price, time_to_resolution_hours,
                        predicted_at, resolved_at, actual_outcome, brier_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.prediction_id,
                    record.market_id,
                    record.market_question,
                    record.category,
                    record.predicted_probability,
                    record.model_name,
                    json.dumps(record.features),
                    record.market_price,
                    record.time_to_resolution_hours,
                    record.predicted_at.isoformat(),
                    record.resolved_at.isoformat() if record.resolved_at else None,
                    record.actual_outcome,
                    record.brier_score,
                ))
                conn.commit()
            
            return True
            
        except Exception as e:
            logger.error("Failed to record prediction", error=str(e))
            return False
    
    def record_outcome(
        self,
        market_id: str,
        outcome: bool,
        resolved_at: datetime | None = None
    ) -> bool:
        """
        Record the outcome for a market.
        
        Args:
            market_id: Market identifier
            outcome: True if YES, False if NO
            resolved_at: When market resolved
            
        Returns:
            True if successful
        """
        try:
            resolved_time = resolved_at or datetime.now(timezone.utc)
            
            with sqlite3.connect(self.db_path) as conn:
                # Find predictions for this market
                cursor = conn.execute(
                    "SELECT prediction_id, predicted_probability FROM predictions WHERE market_id = ?",
                    (market_id,)
                )
                predictions = cursor.fetchall()
                
                for pred_id, pred_prob in predictions:
                    # Calculate Brier score
                    outcome_val = 1.0 if outcome else 0.0
                    brier = (pred_prob - outcome_val) ** 2
                    
                    # Update record
                    conn.execute("""
                        UPDATE predictions
                        SET actual_outcome = ?, resolved_at = ?, brier_score = ?
                        WHERE prediction_id = ?
                    """, (
                        outcome,
                        resolved_time.isoformat(),
                        brier,
                        pred_id,
                    ))
                
                conn.commit()
            
            logger.debug(
                "Outcome recorded",
                market_id=market_id,
                outcome=outcome,
                predictions_updated=len(predictions),
            )
            
            return True
            
        except Exception as e:
            logger.error("Failed to record outcome", error=str(e))
            return False
    
    # Async versions using aiosqlite (non-blocking)
    async def arecord_prediction(self, record: PredictionRecord) -> bool:
        """
        Record a prediction asynchronously (non-blocking).
        
        Args:
            record: Prediction record
            
        Returns:
            True if successful
        """
        if not HAS_AIOSQLITE:
            # Fallback to sync version in thread
            import asyncio
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.record_prediction, record)
        
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute("""
                    INSERT OR REPLACE INTO predictions (
                        prediction_id, market_id, market_question, category,
                        predicted_probability, model_name, features,
                        market_price, time_to_resolution_hours,
                        predicted_at, resolved_at, actual_outcome, brier_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.prediction_id,
                    record.market_id,
                    record.market_question,
                    record.category,
                    record.predicted_probability,
                    record.model_name,
                    json.dumps(record.features),
                    record.market_price,
                    record.time_to_resolution_hours,
                    record.predicted_at.isoformat(),
                    record.resolved_at.isoformat() if record.resolved_at else None,
                    record.actual_outcome,
                    record.brier_score,
                ))
                await conn.commit()
            return True
        except Exception as e:
            logger.error("Failed to record prediction async", error=str(e))
            return False
    
    async def arecord_outcome(
        self,
        market_id: str,
        outcome: bool,
        resolved_at: datetime | None = None
    ) -> bool:
        """
        Record the outcome asynchronously (non-blocking).
        
        Args:
            market_id: Market identifier
            outcome: True if YES, False if NO
            resolved_at: When market resolved
            
        Returns:
            True if successful
        """
        if not HAS_AIOSQLITE:
            import asyncio
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.record_outcome, market_id, outcome, resolved_at)
        
        try:
            resolved_time = resolved_at or datetime.now(timezone.utc)
            
            async with aiosqlite.connect(self.db_path) as conn:
                # Find predictions for this market
                async with conn.execute(
                    "SELECT prediction_id, predicted_probability FROM predictions WHERE market_id = ?",
                    (market_id,)
                ) as cursor:
                    predictions = await cursor.fetchall()
                
                for pred_id, pred_prob in predictions:
                    # Calculate Brier score
                    outcome_val = 1.0 if outcome else 0.0
                    brier = (pred_prob - outcome_val) ** 2
                    
                    await conn.execute("""
                        UPDATE predictions
                        SET actual_outcome = ?, resolved_at = ?, brier_score = ?
                        WHERE prediction_id = ?
                    """, (
                        outcome,
                        resolved_time.isoformat(),
                        brier,
                        pred_id,
                    ))
                
                await conn.commit()
            return True
        except Exception as e:
            logger.error("Failed to record outcome async", error=str(e))
            return False
    
    def get_predictions(
        self,
        category: str | None = None,
        model_name: str | None = None,
        resolved_only: bool = False,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 1000
    ) -> list[PredictionRecord]:
        """
        Query predictions from database.
        
        Args:
            category: Filter by category
            model_name: Filter by model
            resolved_only: Only return resolved predictions
            start_date: Filter by prediction date
            end_date: Filter by prediction date
            limit: Maximum results
            
        Returns:
            List of prediction records
        """
        query = "SELECT * FROM predictions WHERE 1=1"
        params: list[Any] = []
        
        if category:
            query += " AND category = ?"
            params.append(category)
        
        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)
        
        if resolved_only:
            query += " AND resolved_at IS NOT NULL"
        
        if start_date:
            query += " AND predicted_at >= ?"
            params.append(start_date.isoformat())
        
        if end_date:
            query += " AND predicted_at <= ?"
            params.append(end_date.isoformat())
        
        query += " ORDER BY predicted_at DESC LIMIT ?"
        params.append(limit)
        
        records: list[PredictionRecord] = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)
                
                for row in cursor.fetchall():
                    record = PredictionRecord.from_dict(dict(row))
                    records.append(record)
                    
        except Exception as e:
            logger.error("Failed to query predictions", error=str(e))
        
        return records
    
    def get_brier_stats(
        self,
        category: str | None = None,
        model_name: str | None = None,
        days: int = 30
    ) -> dict[str, Any]:
        """
        Get Brier score statistics.
        
        Args:
            category: Filter by category
            model_name: Filter by model
            days: Lookback period
            
        Returns:
            Statistics dictionary
        """
        query = """
            SELECT 
                COUNT(*) as count,
                AVG(brier_score) as avg_brier,
                MIN(brier_score) as min_brier,
                MAX(brier_score) as max_brier,
                AVG(predicted_probability) as avg_prediction,
                AVG(actual_outcome) as base_rate
            FROM predictions
            WHERE resolved_at IS NOT NULL
            AND predicted_at >= date('now', '-{} days')
        """.format(days)
        
        params: list[Any] = []
        
        if category:
            query += " AND category = ?"
            params.append(category)
        
        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(query, params)
                row = cursor.fetchone()
                
                if row and row[0] > 0:
                    return {
                        "count": row[0],
                        "avg_brier": row[1],
                        "min_brier": row[2],
                        "max_brier": row[3],
                        "avg_prediction": row[4],
                        "base_rate": row[5],
                        "skill_score": (0.25 - row[1]) / 0.25 if row[1] else 0,
                    }
                    
        except Exception as e:
            logger.error("Failed to get Brier stats", error=str(e))
        
        return {"count": 0}
    
    def get_pending_resolutions(self) -> list[dict[str, Any]]:
        """
        Get markets that have predictions but no outcomes.
        
        Returns:
            List of pending markets
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT DISTINCT market_id, market_question, category, predicted_at
                    FROM predictions
                    WHERE resolved_at IS NULL
                    ORDER BY predicted_at DESC
                """)
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error("Failed to get pending resolutions", error=str(e))
            return []
    
    def export_to_csv(self, filepath: str) -> bool:
        """
        Export all predictions to CSV.
        
        Args:
            filepath: Output CSV file path
            
        Returns:
            True if successful
        """
        import csv
        
        try:
            records = self.get_predictions(limit=100000)
            
            if not records:
                logger.warning("No records to export")
                return False
            
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=records[0].to_dict().keys())
                writer.writeheader()
                for record in records:
                    writer.writerow(record.to_dict())
            
            logger.info("Exported predictions to CSV", filepath=filepath, count=len(records))
            return True
            
        except Exception as e:
            logger.error("Failed to export CSV", error=str(e))
            return False


# Pre-instantiated database (in-memory for testing, file for production)
calibration_db = CalibrationDatabase()
