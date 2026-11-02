"""Probability calibration.

Temperature scaling is the right default: it is a single parameter, it
cannot distort the ranking of predictions (so AUC is unchanged), and it is
almost always enough to fix an over-confident neural classifier.

Platt scaling and isotonic regression are included for comparison - isotonic
is more flexible but can overfit a small validation set, which is exactly
the situation in medical AI.

The rule for this project: **fit the calibrator on validation, evaluate on
test**. Fitting on test is the single most common calibration mistake.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TemperatureScaler:
    """Scale logits by a single temperature parameter.

    ``T > 1`` softens probabilities (less confident), ``T < 1`` sharpens
    them. Works on the log-odds of the positive class for binary problems.
    """

    temperature: float = 1.0

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> TemperatureScaler:
        """Fit ``T`` by minimising NLL on the validation set.

        Args:
            logits: Raw model logits (not probabilities).
            y_true: Binary labels.

        Returns:
            ``self``, with ``temperature`` updated.
        """
        from scipy.optimize import minimize_scalar

        logits = np.asarray(logits, dtype=float).reshape(-1)
        y_true = np.asarray(y_true, dtype=float).reshape(-1)

        if len(logits) == 0:
            return self
        if len(np.unique(y_true)) < 2:
            logger.warning("Calibration needs both classes; leaving T=1.0")
            return self

        def nll(log_t: float) -> float:
            t = float(np.exp(log_t))
            scaled = logits / max(t, 1e-6)
            # Numerically stable binary cross-entropy on logits.
            loss = np.maximum(scaled, 0) - scaled * y_true + np.log1p(np.exp(-np.abs(scaled)))
            return float(loss.mean())

        result = minimize_scalar(nll, bounds=(-4.0, 4.0), method="bounded")
        self.temperature = float(np.exp(result.x))
        logger.info("Fitted temperature: %.4f", self.temperature)
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """Convert logits to calibrated probabilities."""
        logits = np.asarray(logits, dtype=float).reshape(-1)
        scaled = logits / max(self.temperature, 1e-6)
        return 1.0 / (1.0 + np.exp(-scaled))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"temperature": self.temperature}, indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> TemperatureScaler:
        data = json.loads(Path(path).read_text())
        return cls(temperature=float(data["temperature"]))


@dataclass
class PlattScaler:
    """Sigmoid (Platt) calibration: ``sigmoid(a * logit + b)``."""

    a: float = 1.0
    b: float = 0.0

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> PlattScaler:
        from sklearn.linear_model import LogisticRegression

        logits = np.asarray(logits, dtype=float).reshape(-1, 1)
        y_true = np.asarray(y_true).reshape(-1)
        if len(np.unique(y_true)) < 2:
            logger.warning("Platt calibration needs both classes; leaving defaults")
            return self
        model = LogisticRegression(solver="lbfgs")
        model.fit(logits, y_true)
        self.a = float(model.coef_[0][0])
        self.b = float(model.intercept_[0])
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=float).reshape(-1)
        z = self.a * logits + self.b
        return 1.0 / (1.0 + np.exp(-z))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"a": self.a, "b": self.b}, indent=2))
        return path


@dataclass
class IsotonicScaler:
    """Isotonic regression calibration (monotonic, non-parametric)."""

    model: object | None = None

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> IsotonicScaler:
        from sklearn.isotonic import IsotonicRegression

        logits = np.asarray(logits, dtype=float).reshape(-1)
        y_true = np.asarray(y_true, dtype=float).reshape(-1)
        if len(np.unique(y_true)) < 2:
            logger.warning("Isotonic calibration needs both classes; leaving unfitted")
            return self
        self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.model.fit(logits, y_true)
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        if self.model is None:
            return 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=float).reshape(-1)))
        logits = np.asarray(logits, dtype=float).reshape(-1)
        return np.clip(self.model.predict(logits), 0.0, 1.0)

    def save(self, path: str | Path) -> Path:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        return path


def logits_from_probabilities(probabilities: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Invert a sigmoid to recover logits from probabilities."""
    p = np.clip(np.asarray(probabilities, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def compare_calibrators(
    logits: np.ndarray,
    y_true: np.ndarray,
    *,
    n_bins: int = 15,
) -> dict[str, dict[str, float]]:
    """Compare temperature / Platt / isotonic on the same validation data.

    Returns a dict of ``method -> {"ece", "nll", "brier"}`` so the best
    calibrator can be chosen on evidence rather than habit.
    """
    from medguard.evaluation.metrics import calibration_metrics

    y_true = np.asarray(y_true, dtype=float)
    results: dict[str, dict[str, float]] = {}

    scalers = {
        "uncalibrated": None,
        "temperature": TemperatureScaler().fit(logits, y_true),
        "platt": PlattScaler().fit(logits, y_true),
        "isotonic": IsotonicScaler().fit(logits, y_true),
    }

    for name, scaler in scalers.items():
        probs = (
            1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=float).reshape(-1)))
            if scaler is None
            else scaler.predict_proba(logits)
        )
        metrics = calibration_metrics(y_true, probs, n_bins=n_bins)
        results[name] = metrics.to_dict()

    return results
