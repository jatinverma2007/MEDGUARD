"""Evaluation metrics: classification, calibration and selective prediction.

These are the metrics that turn "I trained a model" into "I measured
something". Two of them are the differentiators most student projects miss:

**Calibration** (ECE, reliability diagram)
    A model that outputs 0.9 confidence should be right ~90% of the time.
    Deep nets usually are not, and in a medical setting an over-confident
    "no interaction" answer is the dangerous failure mode.

**Selective prediction** (risk-coverage curve)
    If the model can abstain, how much accuracy do we get on the subset it
    *does* answer? This is the honest way to report a safety-critical
    system's performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ClassificationMetrics:
    """Standard multi-class / binary classification metrics."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    macro_f1: float
    roc_auc: float | None = None
    average_precision: float | None = None
    support: int = 0

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "macro_f1": round(self.macro_f1, 4),
            "roc_auc": round(self.roc_auc, 4) if self.roc_auc is not None else None,
            "average_precision": (
                round(self.average_precision, 4) if self.average_precision is not None else None
            ),
            "support": self.support,
        }


@dataclass
class CalibrationMetrics:
    """Calibration quality of a probability head."""

    ece: float
    mce: float
    brier_score: float
    nll: float
    n_bins: int
    bin_edges: list[float] = field(default_factory=list)
    bin_confidence: list[float] = field(default_factory=list)
    bin_accuracy: list[float] = field(default_factory=list)
    bin_count: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "ece": round(self.ece, 4),
            "mce": round(self.mce, 4),
            "brier_score": round(self.brier_score, 4),
            "nll": round(self.nll, 4),
            "n_bins": self.n_bins,
        }


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    average: str = "binary",
    y_score: np.ndarray | None = None,
    pos_label: int | str = 1,
) -> ClassificationMetrics:
    """Compute classification metrics with sensible handling of edge cases."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if average == "binary":
        precision = float(precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
        recall = float(recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
    else:
        precision = float(precision_score(y_true, y_pred, average=average, zero_division=0))
        recall = float(recall_score(y_true, y_pred, average=average, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, average=average, zero_division=0))

    macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    roc_auc: float | None = None
    ap: float | None = None
    if y_score is not None:
        y_score = np.asarray(y_score)
        try:
            if len(np.unique(y_true)) == 2:
                roc_auc = float(roc_auc_score(y_true, y_score))
            else:
                roc_auc = float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro"))
        except ValueError:
            roc_auc = None
        try:
            ap = float(average_precision_score(y_true, y_score, average="macro"))
        except ValueError:
            ap = None

    return ClassificationMetrics(
        accuracy=float((y_true == y_pred).mean()) if len(y_true) else 0.0,
        precision=precision,
        recall=recall,
        f1=f1,
        macro_f1=macro,
        roc_auc=roc_auc,
        average_precision=ap,
        support=int(len(y_true)),
    )


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> tuple[float, float, list[float], list[float], list[int]]:
    """Expected Calibration Error and Maximum Calibration Error.

    Returns:
        ``(ece, mce, bin_edges, bin_counts, bin_accuracies)``
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    if len(y_true) == 0:
        return 0.0, 0.0, [], [], []

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_edges: list[float] = []
    bin_counts: list[int] = []
    bin_accuracies: list[float] = []

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_prob > lo) & (y_prob <= hi) if i > 0 else (y_prob >= lo) & (y_prob <= hi)
        count = int(mask.sum())
        bin_edges.append(float(hi))
        bin_counts.append(count)
        if count == 0:
            bin_accuracies.append(0.0)
            continue
        acc = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        bin_accuracies.append(acc)
        gap = abs(acc - conf)
        ece += (count / len(y_true)) * gap
        mce = max(mce, gap)

    return float(ece), float(mce), bin_edges, bin_counts, bin_accuracies


def calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> CalibrationMetrics:
    """Full calibration report for a probabilistic classifier."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-12, 1 - 1e-12)

    if len(y_true) == 0:
        return CalibrationMetrics(0.0, 0.0, 0.0, 0.0, n_bins)

    ece, mce, edges, counts, accuracies = expected_calibration_error(y_true, y_prob, n_bins)
    brier = float(np.mean((y_prob - y_true) ** 2))
    nll = float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))

    confidences: list[float] = []
    for i in range(len(counts)):
        lo = edges[i] - (1.0 / n_bins)
        hi = edges[i]
        mask = (y_prob > lo) & (y_prob <= hi) if i > 0 else (y_prob >= 0) & (y_prob <= hi)
        confidences.append(float(y_prob[mask].mean()) if counts[i] else 0.0)

    return CalibrationMetrics(
        ece=ece,
        mce=mce,
        brier_score=brier,
        nll=nll,
        n_bins=n_bins,
        bin_edges=edges,
        bin_confidence=confidences,
        bin_accuracy=accuracies,
        bin_count=counts,
    )


def selective_risk_coverage(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> list[dict[str, float]]:
    """Risk-coverage curve for a selective (abstaining) predictor.

    At each confidence threshold we keep predictions whose confidence is at
    least the threshold, and report the error rate ("risk") on that retained
    subset and the fraction retained ("coverage").

    A good safety-critical model shows risk falling as coverage falls.

    Returns:
        A list of ``{"threshold", "coverage", "risk", "accuracy", "n_kept"}``.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    if len(y_true) == 0:
        return []

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 21)

    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        keep = y_prob >= threshold
        n_kept = int(keep.sum())
        if n_kept == 0:
            rows.append(
                {
                    "threshold": float(threshold),
                    "coverage": 0.0,
                    "risk": 0.0,
                    "accuracy": 0.0,
                    "n_kept": 0,
                }
            )
            continue
        correct = float(y_true[keep].mean())
        rows.append(
            {
                "threshold": float(threshold),
                "coverage": n_kept / len(y_true),
                "risk": 1.0 - correct,
                "accuracy": correct,
                "n_kept": n_kept,
            }
        )
    return rows


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_bootstraps: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for any metric function."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) < 2:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_bootstraps):
        idx = rng.integers(0, len(y_true), len(y_true))
        try:
            values.append(float(metric_fn(y_true[idx], y_pred[idx])))
        except Exception:  # pragma: no cover - degenerate resample
            continue

    if not values:
        return (float("nan"), float("nan"))

    alpha = (1 - confidence) / 2
    lower = float(np.percentile(values, 100 * alpha))
    upper = float(np.percentile(values, 100 * (1 - alpha)))
    return lower, upper


def leakage_impact(
    y_true_leaky: np.ndarray,
    y_pred_leaky: np.ndarray,
    y_true_clean: np.ndarray,
    y_pred_clean: np.ndarray,
) -> dict[str, float]:
    """Compare metrics from a leaky split vs. a leak-free split.

    The gap is the headline number in the evaluation report: it shows the
    model memorised pairs rather than learning interaction structure.
    """
    leaky = classification_metrics(y_true_leaky, y_pred_leaky)
    clean = classification_metrics(y_true_clean, y_pred_clean)
    return {
        "leaky_f1": leaky.f1,
        "clean_f1": clean.f1,
        "f1_drop": leaky.f1 - clean.f1,
        "f1_drop_pct": (100 * (leaky.f1 - clean.f1) / leaky.f1 if leaky.f1 > 0 else 0.0),
        "leaky_accuracy": leaky.accuracy,
        "clean_accuracy": clean.accuracy,
    }
