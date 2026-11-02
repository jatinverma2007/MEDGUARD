"""Evaluation package: metrics and calibration."""

from medguard.evaluation.calibration import (
    IsotonicScaler,
    PlattScaler,
    TemperatureScaler,
    compare_calibrators,
    logits_from_probabilities,
)
from medguard.evaluation.metrics import (
    CalibrationMetrics,
    ClassificationMetrics,
    bootstrap_ci,
    calibration_metrics,
    classification_metrics,
    expected_calibration_error,
    leakage_impact,
    selective_risk_coverage,
)

__all__ = [
    "CalibrationMetrics",
    "ClassificationMetrics",
    "IsotonicScaler",
    "PlattScaler",
    "TemperatureScaler",
    "bootstrap_ci",
    "calibration_metrics",
    "classification_metrics",
    "compare_calibrators",
    "expected_calibration_error",
    "leakage_impact",
    "logits_from_probabilities",
    "selective_risk_coverage",
]
