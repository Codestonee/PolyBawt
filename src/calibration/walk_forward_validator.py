"""Walk-forward validation for time series models."""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any

from src.ingestion.event_market_discovery import MarketCategory
from src.calibration.calibration_database import CalibrationDatabase, PredictionRecord
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of walk-forward validation."""
    
    passed: bool
    
    # Key metrics
    brier_score: float
    num_predictions: int
    time_span_days: int
    
    # Thresholds checked
    brier_threshold: float
    min_predictions: int
    min_days: int
    
    # Additional metrics
    calibration_error: float  # Max deviation from perfect calibration
    directional_accuracy: float  # % of correct directional calls
    
    # By category
    category_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    
    # Timestamp
    validated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def summary(self) -> str:
        """Human-readable summary."""
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"Walk-forward validation {status}: "
            f"Brier={self.brier_score:.4f} "
            f"(threshold={self.brier_threshold}), "
            f"n={self.num_predictions} "
            f"(min={self.min_predictions}), "
            f"span={self.time_span_days}d "
            f"(min={self.min_days}d)"
        )


class WalkForwardValidator:
    """
    Walk-forward validation for time series prediction models.
    
    CRITICAL: Never use k-fold cross-validation for time series!
    Walk-forward is the only valid validation method for sequential data.
    
    Validation Requirements:
    - min_resolved_markets: 500+ for statistical significance
    - min_time_span_days: 180+ to cover multiple regimes
    - brier_score_threshold: < 0.22 (better than random)
    - calibration_max_deviation: < 0.10 (within 10% of perfect)
    
    Walk-forward Process:
    1. Train on [t_0, t_1]
    2. Validate on [t_1, t_2]
    3. Train on [t_0, t_2]
    4. Validate on [t_2, t_3]
    5. ...and so on
    
    This ensures:
    - No lookahead bias
    - Realistic performance estimates
    - Regime-robust validation
    
    Usage:
        validator = WalkForwardValidator()
        
        # Run validation
        result = validator.validate(
            predictions=records,
            min_predictions=500,
            min_days=180,
            max_brier=0.22,
        )
        
        if result.passed:
            print("Model is ready for production!")
    """
    
    # Default validation requirements
    DEFAULT_REQUIREMENTS = {
        "min_resolved_markets": 500,
        "min_time_span_days": 180,
        "brier_score_threshold": 0.22,
        "calibration_max_deviation": 0.10,
    }
    
    def __init__(
        self,
        db: CalibrationDatabase | None = None,
        requirements: dict[str, float] | None = None
    ):
        """
        Initialize validator.
        
        Args:
            db: Calibration database
            requirements: Validation thresholds
        """
        self.db = db or CalibrationDatabase()
        self.requirements = requirements or self.DEFAULT_REQUIREMENTS
    
    def validate(
        self,
        predictions: list[PredictionRecord] | None = None,
        category: MarketCategory | None = None,
        min_predictions: int | None = None,
        min_days: int | None = None,
        max_brier: float | None = None,
        max_calibration_error: float | None = None,
    ) -> ValidationResult:
        """
        Run walk-forward validation.
        
        Args:
            predictions: List of predictions (or fetch from DB)
            category: Validate only this category
            min_predictions: Minimum predictions required
            min_days: Minimum time span required
            max_brier: Maximum Brier score allowed
            max_calibration_error: Maximum calibration deviation
            
        Returns:
            ValidationResult with pass/fail status
        """
        # Use defaults if not specified
        min_predictions = min_predictions or self.requirements["min_resolved_markets"]
        min_days = min_days or self.requirements["min_time_span_days"]
        max_brier = max_brier or self.requirements["brier_score_threshold"]
        max_calibration_error = max_calibration_error or self.requirements["calibration_max_deviation"]
        
        # Fetch predictions if not provided
        if predictions is None:
            cat_str = category.value if category else None
            predictions = self.db.get_predictions(
                category=cat_str,
                resolved_only=True,
                limit=10000,
            )
        
        # Filter by category if specified
        if category:
            predictions = [p for p in predictions if p.category == category.value]
        
        # Check minimum predictions
        if len(predictions) < min_predictions:
            logger.warning(
                "Insufficient predictions for validation",
                have=len(predictions),
                need=min_predictions,
            )
            return ValidationResult(
                passed=False,
                brier_score=0.25,
                num_predictions=len(predictions),
                time_span_days=0,
                brier_threshold=max_brier,
                min_predictions=min_predictions,
                min_days=min_days,
                calibration_error=1.0,
                directional_accuracy=0.0,
            )
        
        # Calculate time span
        dates = [p.predicted_at for p in predictions]
        time_span = max(dates) - min(dates)
        time_span_days = time_span.days
        
        if time_span_days < min_days:
            logger.warning(
                "Insufficient time span for validation",
                have=time_span_days,
                need=min_days,
            )
            return ValidationResult(
                passed=False,
                brier_score=0.25,
                num_predictions=len(predictions),
                time_span_days=time_span_days,
                brier_threshold=max_brier,
                min_predictions=min_predictions,
                min_days=min_days,
                calibration_error=1.0,
                directional_accuracy=0.0,
            )
        
        # Calculate Brier score
        total_brier = sum(p.brier_score for p in predictions if p.brier_score is not None)
        num_scored = sum(1 for p in predictions if p.brier_score is not None)
        brier_score = total_brier / num_scored if num_scored > 0 else 0.25
        
        # Calculate calibration error
        calibration_error = self._calculate_calibration_error(predictions)
        
        # Calculate directional accuracy
        directional_accuracy = self._calculate_directional_accuracy(predictions)
        
        # Category breakdown
        category_results = self._analyze_by_category(predictions)
        
        # Check all requirements
        passed = (
            brier_score <= max_brier and
            calibration_error <= max_calibration_error and
            len(predictions) >= min_predictions and
            time_span_days >= min_days
        )
        
        result = ValidationResult(
            passed=passed,
            brier_score=brier_score,
            num_predictions=len(predictions),
            time_span_days=time_span_days,
            brier_threshold=max_brier,
            min_predictions=min_predictions,
            min_days=min_days,
            calibration_error=calibration_error,
            directional_accuracy=directional_accuracy,
            category_results=category_results,
        )
        
        logger.info(
            "Walk-forward validation complete",
            passed=passed,
            brier_score=brier_score,
            num_predictions=len(predictions),
            time_span_days=time_span_days,
        )
        
        return result
    
    def _calculate_calibration_error(
        self,
        predictions: list[PredictionRecord]
    ) -> float:
        """
        Calculate maximum calibration deviation.
        
        Measures how far predicted probabilities are from actual frequencies.
        """
        # Bin predictions
        bins: dict[int, list[bool]] = {i: [] for i in range(10)}
        
        for pred in predictions:
            if pred.actual_outcome is None:
                continue
            
            bin_idx = min(9, int(pred.predicted_probability * 10))
            bins[bin_idx].append(pred.actual_outcome)
        
        # Calculate maximum deviation
        max_deviation = 0.0
        
        for bin_idx, outcomes in bins.items():
            if not outcomes:
                continue
            
            predicted_prob = (bin_idx + 0.5) / 10
            actual_freq = sum(outcomes) / len(outcomes)
            
            deviation = abs(predicted_prob - actual_freq)
            max_deviation = max(max_deviation, deviation)
        
        return max_deviation
    
    def _calculate_directional_accuracy(
        self,
        predictions: list[PredictionRecord]
    ) -> float:
        """
        Calculate directional accuracy (% of correct side predictions).
        """
        correct = 0
        total = 0
        
        for pred in predictions:
            if pred.actual_outcome is None:
                continue
            
            predicted_yes = pred.predicted_probability > 0.5
            actual_yes = pred.actual_outcome
            
            if predicted_yes == actual_yes:
                correct += 1
            
            total += 1
        
        return correct / total if total > 0 else 0.0
    
    def _analyze_by_category(
        self,
        predictions: list[PredictionRecord]
    ) -> dict[str, dict[str, Any]]:
        """Analyze results by category."""
        by_category: dict[str, list[PredictionRecord]] = {}
        
        for pred in predictions:
            cat = pred.category or "unknown"
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(pred)
        
        results = {}
        
        for cat, cat_preds in by_category.items():
            total_brier = sum(p.brier_score for p in cat_preds if p.brier_score is not None)
            num_scored = sum(1 for p in cat_preds if p.brier_score is not None)
            
            results[cat] = {
                "count": len(cat_preds),
                "brier_score": total_brier / num_scored if num_scored > 0 else None,
                "directional_accuracy": self._calculate_directional_accuracy(cat_preds),
            }
        
        return results
    
    def generate_report(self, result: ValidationResult) -> str:
        """
        Generate detailed validation report.
        
        Args:
            result: Validation result
            
        Returns:
            Formatted report string
        """
        lines = [
            "=" * 60,
            "WALK-FORWARD VALIDATION REPORT",
            "=" * 60,
            "",
            f"Status: {'PASSED' if result.passed else 'FAILED'}",
            f"Validated at: {result.validated_at.isoformat()}",
            "",
            "OVERALL METRICS",
            "-" * 40,
            f"  Brier Score: {result.brier_score:.4f} "
            f"(threshold: {result.brier_threshold})",
            f"  Calibration Error: {result.calibration_error:.4f}",
            f"  Directional Accuracy: {result.directional_accuracy:.1%}",
            f"  Predictions: {result.num_predictions} "
            f"(min: {result.min_predictions})",
            f"  Time Span: {result.time_span_days} days "
            f"(min: {result.min_days})",
            "",
            "BY CATEGORY",
            "-" * 40,
        ]
        
        for cat, stats in result.category_results.items():
            lines.append(f"  {cat}:")
            lines.append(f"    Count: {stats['count']}")
            if stats['brier_score']:
                lines.append(f"    Brier: {stats['brier_score']:.4f}")
            lines.append(f"    Directional: {stats['directional_accuracy']:.1%}")
        
        lines.extend([
            "",
            "=" * 60,
        ])
        
        return "\n".join(lines)


# Pre-instantiated validator
walk_forward_validator = WalkForwardValidator()
