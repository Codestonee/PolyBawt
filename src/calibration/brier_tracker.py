"""Brier score tracking for model calibration."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.ingestion.event_market_discovery import MarketCategory
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BrierResult:
    """Brier score calculation result."""
    
    # Brier score components
    brier_score: float  # Mean squared error
    reliability: float  # How well calibrated
    resolution: float   # Ability to discriminate
    uncertainty: float  # Base rate uncertainty
    
    # Interpretation
    skill_score: float  # vs random (0.25 baseline)
    
    # Category breakdown
    by_category: dict[str, float] = field(default_factory=dict)
    
    # Metadata
    num_predictions: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def is_well_calibrated(self) -> bool:
        """Is the model well calibrated?"""
        return self.brier_score < 0.22  # Better than random guessing
    
    @property
    def calibration_grade(self) -> str:
        """Letter grade for calibration."""
        if self.brier_score < 0.15:
            return "A"
        elif self.brier_score < 0.20:
            return "B"
        elif self.brier_score < 0.22:
            return "C"
        elif self.brier_score < 0.25:
            return "D"
        else:
            return "F"


@dataclass
class PredictionOutcome:
    """A prediction and its eventual outcome."""
    prediction_id: str
    market_id: str
    category: MarketCategory
    predicted_probability: float
    actual_outcome: bool  # True = YES happened
    timestamp: datetime
    resolved_at: datetime
    
    @property
    def brier_contribution(self) -> float:
        """Contribution to Brier score."""
        outcome_value = 1.0 if self.actual_outcome else 0.0
        return (self.predicted_probability - outcome_value) ** 2


class BrierTracker:
    """
    Tracks Brier scores for model calibration.
    
    Brier Score = (1/N) Σ(p_i - o_i)²
    
    where p_i is the predicted probability and o_i is the actual outcome (0 or 1).
    
    Interpretation:
    - 0.00 = Perfect prediction
    - 0.25 = Random guessing (for binary outcomes at p=0.5)
    - 1.00 = Always wrong
    
    Target: Brier < 0.22 (better than random guessing)
    
    Usage:
        tracker = BrierTracker()
        
        # Record prediction
        tracker.record_prediction(
            market_id="abc123",
            category=MarketCategory.POLITICS,
            probability=0.65,
            timestamp=datetime.now()
        )
        
        # Later, when resolved
        tracker.record_outcome("abc123", outcome=True)
        
        # Check calibration
        result = tracker.calculate_brier()
        print(f"Brier Score: {result.brier_score:.4f}")
    """
    
    # Brier score thresholds
    EXCELLENT_THRESHOLD = 0.15
    GOOD_THRESHOLD = 0.20
    ACCEPTABLE_THRESHOLD = 0.22  # Better than random
    
    def __init__(self, window_size: int | None = None):
        """
        Initialize Brier tracker.
        
        Args:
            window_size: If set, only keep N most recent predictions
        """
        self.predictions: dict[str, PredictionOutcome] = {}
        self.pending_predictions: dict[str, dict[str, Any]] = {}
        self.window_size = window_size
        
        # Statistics by category
        self.category_stats: dict[str, dict[str, Any]] = {}
    
    def record_prediction(
        self,
        market_id: str,
        category: MarketCategory,
        probability: float,
        timestamp: datetime | None = None,
        prediction_id: str | None = None,
        metadata: dict[str, Any] | None = None
    ) -> str:
        """
        Record a prediction for later Brier calculation.
        
        Args:
            market_id: Market identifier
            category: Market category
            probability: Predicted probability (0-1)
            timestamp: When prediction was made
            prediction_id: Optional unique ID
            metadata: Additional metadata
            
        Returns:
            prediction_id
        """
        pred_id = prediction_id or f"{market_id}_{datetime.now(timezone.utc).isoformat()}"
        
        self.pending_predictions[pred_id] = {
            "prediction_id": pred_id,
            "market_id": market_id,
            "category": category,
            "predicted_probability": max(0.01, min(0.99, probability)),
            "timestamp": timestamp or datetime.now(timezone.utc),
            "metadata": metadata or {},
        }
        
        logger.debug(
            "Prediction recorded for Brier tracking",
            prediction_id=pred_id,
            market_id=market_id,
            probability=probability,
        )
        
        return pred_id
    
    def record_outcome(
        self,
        market_id: str,
        outcome: bool,
        resolved_at: datetime | None = None
    ) -> bool:
        """
        Record the outcome of a market.
        
        Args:
            market_id: Market identifier
            outcome: True if YES occurred, False if NO
            resolved_at: When market resolved
            
        Returns:
            True if matching prediction was found and recorded
        """
        # Find matching pending prediction
        matching_pred = None
        for pred_id, pred in self.pending_predictions.items():
            if pred["market_id"] == market_id:
                matching_pred = pred
                break
        
        if matching_pred is None:
            logger.warning(
                "No matching prediction found for outcome",
                market_id=market_id,
            )
            return False
        
        # Create outcome record
        outcome_record = PredictionOutcome(
            prediction_id=matching_pred["prediction_id"],
            market_id=market_id,
            category=matching_pred["category"],
            predicted_probability=matching_pred["predicted_probability"],
            actual_outcome=outcome,
            timestamp=matching_pred["timestamp"],
            resolved_at=resolved_at or datetime.now(timezone.utc),
        )
        
        # Store
        self.predictions[matching_pred["prediction_id"]] = outcome_record
        
        # Remove from pending
        del self.pending_predictions[matching_pred["prediction_id"]]
        
        # Manage window size
        if self.window_size and len(self.predictions) > self.window_size:
            oldest = min(self.predictions.items(), key=lambda x: x[1].timestamp)
            del self.predictions[oldest[0]]
        
        # Update category stats
        self._update_category_stats(outcome_record)
        
        logger.debug(
            "Outcome recorded",
            market_id=market_id,
            outcome=outcome,
            brier_contribution=outcome_record.brier_contribution,
        )
        
        return True
    
    def _update_category_stats(self, outcome: PredictionOutcome) -> None:
        """Update category statistics."""
        cat = outcome.category.value
        
        if cat not in self.category_stats:
            self.category_stats[cat] = {
                "count": 0,
                "sum_brier": 0.0,
                "correct": 0,
            }
        
        stats = self.category_stats[cat]
        stats["count"] += 1
        stats["sum_brier"] += outcome.brier_contribution
        
        # Track if prediction was "correct" (picked right side)
        predicted_yes = outcome.predicted_probability > 0.5
        if predicted_yes == outcome.actual_outcome:
            stats["correct"] += 1
    
    def calculate_brier(
        self,
        category: MarketCategory | None = None
    ) -> BrierResult:
        """
        Calculate Brier score.
        
        Args:
            category: If set, only calculate for this category
            
        Returns:
            BrierResult with score and components
        """
        # Filter by category if specified
        outcomes = list(self.predictions.values())
        if category:
            outcomes = [o for o in outcomes if o.category == category]
        
        if not outcomes:
            return BrierResult(
                brier_score=0.25,  # Random guessing baseline
                reliability=0.0,
                resolution=0.0,
                uncertainty=0.25,
                skill_score=0.0,
                num_predictions=0,
            )
        
        # Calculate Brier score
        total_brier = sum(o.brier_contribution for o in outcomes)
        brier_score = total_brier / len(outcomes)
        
        # Calculate components (simplified)
        # Reliability: how well calibrated
        reliability = self._calculate_reliability(outcomes)
        
        # Resolution: ability to discriminate
        resolution = self._calculate_resolution(outcomes)
        
        # Uncertainty: base rate
        base_rate = sum(o.actual_outcome for o in outcomes) / len(outcomes)
        uncertainty = base_rate * (1 - base_rate)
        
        # Skill score vs random (0.25)
        skill_score = (0.25 - brier_score) / 0.25
        
        # By category
        by_category = {}
        for cat in MarketCategory:
            cat_outcomes = [o for o in outcomes if o.category == cat]
            if cat_outcomes:
                cat_brier = sum(o.brier_contribution for o in cat_outcomes) / len(cat_outcomes)
                by_category[cat.value] = cat_brier
        
        return BrierResult(
            brier_score=brier_score,
            reliability=reliability,
            resolution=resolution,
            uncertainty=uncertainty,
            skill_score=skill_score,
            by_category=by_category,
            num_predictions=len(outcomes),
        )
    
    def _calculate_reliability(self, outcomes: list[PredictionOutcome]) -> float:
        """
        Calculate reliability component.
        
        Measures how well calibrated the predictions are.
        """
        # Bin predictions and compare to actual frequencies
        bins = {i: [] for i in range(10)}  # 0.0-0.1, 0.1-0.2, etc.
        
        for outcome in outcomes:
            bin_idx = min(9, int(outcome.predicted_probability * 10))
            bins[bin_idx].append(outcome.actual_outcome)
        
        reliability = 0.0
        total = 0
        
        for bin_idx, bin_outcomes in bins.items():
            if not bin_outcomes:
                continue
            
            bin_prob = (bin_idx + 0.5) / 10  # Center of bin
            actual_freq = sum(bin_outcomes) / len(bin_outcomes)
            
            reliability += len(bin_outcomes) * (bin_prob - actual_freq) ** 2
            total += len(bin_outcomes)
        
        return reliability / total if total > 0 else 0.0
    
    def _calculate_resolution(self, outcomes: list[PredictionOutcome]) -> float:
        """
        Calculate resolution component.
        
        Measures ability to discriminate between outcomes.
        """
        base_rate = sum(o.actual_outcome for o in outcomes) / len(outcomes)
        
        bins = {i: [] for i in range(10)}
        
        for outcome in outcomes:
            bin_idx = min(9, int(outcome.predicted_probability * 10))
            bins[bin_idx].append(outcome.actual_outcome)
        
        resolution = 0.0
        total = 0
        
        for bin_idx, bin_outcomes in bins.items():
            if not bin_outcomes:
                continue
            
            actual_freq = sum(bin_outcomes) / len(bin_outcomes)
            resolution += len(bin_outcomes) * (actual_freq - base_rate) ** 2
            total += len(bin_outcomes)
        
        return resolution / total if total > 0 else 0.0
    
    def get_calibration_report(self) -> dict[str, Any]:
        """Get comprehensive calibration report."""
        overall = self.calculate_brier()
        
        # Per-category results
        category_results = {}
        for cat in MarketCategory:
            cat_result = self.calculate_brier(category=cat)
            if cat_result.num_predictions > 0:
                category_results[cat.value] = {
                    "brier_score": cat_result.brier_score,
                    "num_predictions": cat_result.num_predictions,
                    "grade": cat_result.calibration_grade,
                }
        
        return {
            "overall_brier": overall.brier_score,
            "overall_grade": overall.calibration_grade,
            "skill_score": overall.skill_score,
            "num_predictions": overall.num_predictions,
            "num_pending": len(self.pending_predictions),
            "by_category": category_results,
            "thresholds": {
                "excellent": self.EXCELLENT_THRESHOLD,
                "good": self.GOOD_THRESHOLD,
                "acceptable": self.ACCEPTABLE_THRESHOLD,
            },
        }
    
    def reset(self) -> None:
        """Reset all tracking data."""
        self.predictions.clear()
        self.pending_predictions.clear()
        self.category_stats.clear()
        
        logger.info("Brier tracker reset")


# Pre-instantiated tracker
brier_tracker = BrierTracker()
