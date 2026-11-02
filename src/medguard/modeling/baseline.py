"""Baseline models.

A baseline is not optional - it is the thing that proves your neural model
learned something. These baselines run on CPU in under a minute on the full
interaction table, which means you can iterate on them for free while the
GPU is busy with the real model.

``DrugPairTfidfBaseline``
    TF-IDF over the concatenated drug names + character n-grams, feeding a
    logistic regression. Surprising strength on DDI tasks because drug
    names carry pharmacokinetic information (CYP enzyme families, suffix
    families like ``-pril``, ``-sartan``, ``-statin``).

``CooccurrenceBaseline``
    "Does this pair of drugs appear together in the corpus?" A pure
    frequency baseline that must be beaten by any real model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from medguard.data.splits import canonical_pair

logger = logging.getLogger(__name__)


def pair_text(drug_a: str, drug_b: str) -> str:
    """Symmetric text representation of a drug pair.

    Both orderings are included so the representation is order-invariant
    without having to canonicalise at feature time.
    """
    a, b = sorted([str(drug_a).lower(), str(drug_b).lower()])
    return f"{a} {b} {b} {a}"


@dataclass
class CooccurrenceBaseline:
    """Frequency baseline: is this pair seen in the training data?"""

    pair_counts: dict[tuple[str, str], int] = field(default_factory=dict)
    total_pairs: int = 0
    positive_rate: float = 0.0

    def fit(self, df: pd.DataFrame, label_col: str = "label") -> CooccurrenceBaseline:
        counts: dict[tuple[str, str], int] = {}
        for a, b in zip(df["drug_a"], df["drug_b"], strict=False):
            key = canonical_pair(a, b)
            counts[key] = counts.get(key, 0) + 1
        self.pair_counts = counts
        self.total_pairs = len(df)
        self.positive_rate = float(df[label_col].mean()) if label_col in df.columns else 0.0
        logger.info("Cooccurrence baseline fitted on %d rows", self.total_pairs)
        return self

    def predict_proba(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Probability that a pair is an interaction, based on train frequency."""
        if not self.pair_counts:
            return np.full(len(pairs), self.positive_rate)

        scores = []
        for a, b in pairs:
            count = self.pair_counts.get(canonical_pair(a, b), 0)
            # Saturating function: seen once -> some signal, seen many -> more.
            scores.append(count / (count + 5.0) if count else self.positive_rate)
        return np.clip(np.asarray(scores), 0.0, 1.0)

    def predict(self, pairs: list[tuple[str, str]], threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(pairs) >= threshold).astype(int)


@dataclass
class DrugPairTfidfBaseline:
    """TF-IDF (word + char n-grams) over drug pairs + logistic regression."""

    pipeline: Pipeline | None = None
    classes_: np.ndarray | None = None

    def _build(self) -> Pipeline:
        features = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        lowercase=True,
                        min_df=1,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(2, 4),
                        lowercase=True,
                        min_df=2,
                        sublinear_tf=True,
                    ),
                ),
            ]
        )
        return Pipeline(
            [
                ("features", features),
                ("clf", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")),
            ]
        )

    def fit(self, df: pd.DataFrame, label_col: str = "label") -> DrugPairTfidfBaseline:
        texts = [pair_text(a, b) for a, b in zip(df["drug_a"], df["drug_b"], strict=False)]
        labels = df[label_col].to_numpy()
        self.pipeline = self._build()
        self.pipeline.fit(texts, labels)
        self.classes_ = self.pipeline.named_steps["clf"].classes_
        logger.info("TF-IDF baseline fitted on %d rows", len(df))
        return self

    def predict_proba(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Call fit() before predict_proba()")
        texts = [pair_text(a, b) for a, b in pairs]
        return self.pipeline.predict_proba(texts)

    def predict(self, pairs: list[tuple[str, str]], threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(pairs)
        if probs.shape[1] == 1:
            return (probs[:, 0] >= threshold).astype(int)
        positive_idx = int(np.where(self.classes_ == 1)[0][0]) if self.classes_ is not None else 1
        return (probs[:, positive_idx] >= threshold).astype(int)

    def positive_scores(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Raw logits for the positive class (needed by temperature scaling)."""
        if self.pipeline is None:
            raise RuntimeError("Call fit() before positive_scores()")
        texts = [pair_text(a, b) for a, b in pairs]
        decision = self.pipeline.decision_function(texts)
        if decision.ndim == 1:
            return decision.astype(float)
        positive_idx = int(np.where(self.classes_ == 1)[0][0]) if self.classes_ is not None else 1
        return decision[:, positive_idx].astype(float)


def build_baseline_df(
    kb_table: pd.DataFrame,
    negative_pairs: list[tuple[str, str]],
) -> pd.DataFrame:
    """Build a labelled dataset: KB pairs are positive, sampled pairs negative.

    Negative sampling must avoid creating pairs that are in the KB, and the
    negatives should be *hard* - random drug pairs are trivially easy.
    """
    from medguard.data.splits import canonical_pair

    positives = kb_table.copy()
    positives["label"] = 1
    positives = positives.rename(columns={"drug_a": "drug_a", "drug_b": "drug_b"})

    kb_pairs = {
        canonical_pair(a, b) for a, b in zip(kb_table["drug_a"], kb_table["drug_b"], strict=False)
    }
    seen: set[tuple[str, str]] = set()
    neg_rows: list[dict[str, object]] = []
    for a, b in negative_pairs:
        key = canonical_pair(a, b)
        if key in kb_pairs or key in seen:
            continue
        seen.add(key)
        neg_rows.append({"drug_a": a, "drug_b": b, "label": 0})
        if len(neg_rows) >= len(positives):
            break

    negatives = pd.DataFrame(neg_rows)
    combined = pd.concat([positives, negatives], ignore_index=True)
    return combined[["drug_a", "drug_b", "label"]]
