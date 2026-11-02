"""Tests for calibration and selective prediction metrics."""

from __future__ import annotations

import numpy as np
import pytest

from medguard.evaluation.calibration import (
    TemperatureScaler,
    logits_from_probabilities,
)
from medguard.evaluation.metrics import (
    bootstrap_ci,
    calibration_metrics,
    classification_metrics,
    expected_calibration_error,
    selective_risk_coverage,
)


def test_perfectly_calibrated_probabilities_have_low_ece() -> None:
    # Probabilities that match the observed frequency exactly.
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_prob = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    ece, _, _, _, _ = expected_calibration_error(y_true, y_prob)
    assert ece < 0.05


def test_overconfident_model_has_high_ece() -> None:
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.99, 0.99, 0.99, 0.99])
    ece, _, _, _, _ = expected_calibration_error(y_true, y_prob)
    assert ece > 0.4


def test_temperature_scaling_increases_confidence_when_too_soft() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 500)
    # Under-confident logits: correct ranking, but probabilities too close to 0.5.
    logits = logits_from_probabilities(np.clip(y_true * 0.6 + 0.2, 0.01, 0.99))

    scaler = TemperatureScaler().fit(logits, y_true)
    before = calibration_metrics(y_true, 1 / (1 + np.exp(-logits)))
    after = calibration_metrics(y_true, scaler.predict_proba(logits))
    assert after.ece <= before.ece + 1e-6


def test_temperature_scaling_preserves_ranking() -> None:
    rng = np.random.default_rng(1)
    logits = rng.normal(size=200)
    scaler = TemperatureScaler(temperature=2.5)
    probs = scaler.predict_proba(logits)
    assert np.array_equal(np.argsort(probs), np.argsort(logits))


def test_calibration_metrics_fields() -> None:
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.8, 0.2])
    metrics = calibration_metrics(y_true, y_prob)
    assert metrics.ece >= 0
    assert metrics.brier_score >= 0
    assert metrics.nll >= 0
    assert len(metrics.bin_count) == metrics.n_bins


def test_selective_risk_coverage_monotonic_coverage() -> None:
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_prob = np.array([0.95, 0.90, 0.55, 0.45, 0.30, 0.10])
    curve = selective_risk_coverage(y_true, y_prob)
    coverages = [row["coverage"] for row in curve]
    assert coverages == sorted(coverages, reverse=True)
    assert curve[0]["coverage"] == pytest.approx(1.0)
    assert curve[-1]["n_kept"] <= curve[0]["n_kept"]


def test_selective_prediction_improves_accuracy_at_high_threshold() -> None:
    # The model is right when confident and wrong when not.
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.95, 0.94, 0.20, 0.10])
    curve = selective_risk_coverage(y_true, y_prob)
    full = curve[0]
    strict = curve[-1]
    assert strict["coverage"] <= full["coverage"]
    if strict["n_kept"] > 0:
        assert strict["accuracy"] >= full["accuracy"]


def test_classification_metrics() -> None:
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 0])
    metrics = classification_metrics(y_true, y_pred)
    assert metrics.accuracy == pytest.approx(0.75)
    assert metrics.support == 4
    assert 0 <= metrics.f1 <= 1


def test_bootstrap_ci_contains_point_estimate() -> None:
    rng = np.random.default_rng(3)
    y_true = rng.integers(0, 2, 200)
    y_pred = y_true.copy()
    lower, upper = bootstrap_ci(
        y_true, y_pred, lambda a, b: float((a == b).mean()), n_bootstraps=200
    )
    assert lower <= 1.0 <= upper
