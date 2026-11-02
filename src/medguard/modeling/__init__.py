"""Modeling package: baselines and the transformer classifier."""

from medguard.modeling.baseline import (
    CooccurrenceBaseline,
    DrugPairTfidfBaseline,
    build_baseline_df,
    pair_text,
)
from medguard.modeling.classifier import (
    DEFAULT_ENCODER,
    CalibratedInteractionClassifier,
)

__all__ = [
    "CalibratedInteractionClassifier",
    "CooccurrenceBaseline",
    "DEFAULT_ENCODER",
    "DrugPairTfidfBaseline",
    "build_baseline_df",
    "pair_text",
]
