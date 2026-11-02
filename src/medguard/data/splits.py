"""Leak-free splitting for drug-drug interaction datasets.

This module is the methodological heart of MedGuard. Most published DDI
results are inflated because the *same drug pair* (or even the *same drug*)
appears in both train and test sets. A model that has memorised
"warfarin + aspirin -> bleeding" scores well on a random split but fails on
a genuinely unseen combination.

Three strategies are provided:

``random``
    Naive baseline. Kept only so the leakage effect can be *measured* and
    reported - never used for headline numbers.

``pairwise``
    No unordered drug pair appears in more than one split. Models still see
    the same drugs during training, so this measures generalisation to new
    *combinations*.

``cold_start``
    No drug appears in more than one split. This is the strictest setting and
    measures generalisation to entirely unseen drugs - the honest estimate of
    performance on a drug MedGuard has never seen before.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SplitStrategy(str, Enum):
    RANDOM = "random"
    PAIRWISE = "pairwise"
    COLD_START = "cold_start"


#: Column names that are checked, in order, when looking for drug identifiers.
DEFAULT_DRUG_COLUMNS: tuple[str, ...] = ("drug_a", "drug_b")


@dataclass
class SplitReport:
    """Diagnostics for a single split.

    ``leaked_pairs`` / ``leaked_drugs`` are the numbers you put in your
    README to demonstrate you understand why naive splits inflate results.
    """

    strategy: SplitStrategy
    train: int
    val: int
    test: int
    leaked_pairs: int = 0
    leaked_drugs: int = 0
    pair_overlap: set[tuple[str, str]] = field(default_factory=set)
    drug_overlap: set[str] = field(default_factory=set)
    class_distribution: dict[str, dict[str, int]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.train + self.val + self.test

    @property
    def is_degenerate(self) -> bool:
        """True when a split could not be populated at all."""
        return self.val == 0 or self.test == 0

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy.value,
            "n_train": self.train,
            "n_val": self.val,
            "n_test": self.test,
            "leaked_pairs": self.leaked_pairs,
            "leaked_drugs": self.leaked_drugs,
            "class_distribution": self.class_distribution,
            "warnings": self.warnings,
        }

    def summary(self) -> str:
        return (
            f"[{self.strategy.value}] train={self.train} val={self.val} test={self.test} "
            f"| leaked_pairs={self.leaked_pairs} leaked_drugs={self.leaked_drugs}"
        )


def canonical_pair(drug_a: str, drug_b: str) -> tuple[str, str]:
    """Order-independent key for an unordered drug pair."""
    a, b = str(drug_a).strip().lower(), str(drug_b).strip().lower()
    return (a, b) if a <= b else (b, a)


def find_drug_columns(
    df: pd.DataFrame, candidates: Sequence[str] = DEFAULT_DRUG_COLUMNS
) -> tuple[str, str]:
    """Locate the two drug identifier columns in ``df``.

    Raises:
        ValueError: if fewer than two usable columns are found.
    """
    found = [c for c in candidates if c in df.columns]
    if len(found) < 2:
        raise ValueError(
            f"Need two drug columns from {candidates}, found {found} in {list(df.columns)}"
        )
    return found[0], found[1]


def _pair_keys(df: pd.DataFrame, col_a: str, col_b: str) -> list[tuple[str, str]]:
    return [canonical_pair(a, b) for a, b in zip(df[col_a], df[col_b], strict=False)]


def _all_drugs(df: pd.DataFrame, col_a: str, col_b: str) -> set[str]:
    drugs: set[str] = set()
    for col in (col_a, col_b):
        drugs.update(str(v).strip().lower() for v in df[col] if str(v).strip())
    return drugs


def _assign_groups_to_splits(
    groups: list[tuple[str, str]],
    weights: dict[tuple[str, str], int],
    ratios: tuple[float, float, float],
    rng: np.random.Generator,
) -> dict[tuple[str, str], int]:
    """Greedily place groups into splits to hit the target ratios.

    Groups are shuffled then placed largest-first into whichever split is
    furthest below its target share. This keeps large groups together
    (required for leak-freedom) while keeping the splits balanced.
    """
    order = rng.permutation(len(groups))
    # Sort by size descending (stable within equal sizes) so big groups are
    # placed first while the random permutation still breaks ties.
    ordered = sorted(order, key=lambda i: -weights[groups[i]])

    targets = {0: ratios[0], 1: ratios[1], 2: ratios[2]}
    totals = {0: 0.0, 1: 0.0, 2: 0.0}
    total_weight = sum(weights.values()) or 1.0
    assignment: dict[tuple[str, str], int] = {}

    for idx in ordered:
        group = groups[idx]
        size = weights[group]
        deficits = {
            split: (targets[split] * total_weight - totals[split]) / max(size, 1)
            for split in targets
        }
        chosen = max(deficits, key=lambda s: (deficits[s], -s))
        assignment[group] = chosen
        totals[chosen] += size

    return assignment


def make_splits(
    df: pd.DataFrame,
    strategy: SplitStrategy = SplitStrategy.PAIRWISE,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    drug_columns: Sequence[str] = DEFAULT_DRUG_COLUMNS,
    label_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitReport]:
    """Create train/validation/test splits with no leakage.

    Args:
        df: Interaction dataframe with at least two drug columns.
        strategy: Which leak-free guarantee to enforce.
        ratios: Fractions for (train, val, test). Must sum to 1.
        seed: Random seed for reproducibility.
        drug_columns: Candidate column names holding drug identifiers.
        label_column: Optional label column, used to report class balance.

    Returns:
        ``(train, val, test, report)``
    """
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")
    if df.empty:
        raise ValueError("Cannot split an empty dataframe")

    col_a, col_b = find_drug_columns(df, drug_columns)
    rng = np.random.default_rng(seed)

    work = df.copy()
    work["__pair_key"] = _pair_keys(work, col_a, col_b)

    if strategy is SplitStrategy.RANDOM:
        idx = rng.permutation(len(work))
        n_train = int(len(work) * ratios[0])
        n_val = int(len(work) * ratios[1])
        assignment = np.zeros(len(work), dtype=int)
        assignment[idx[n_train : n_train + n_val]] = 1
        assignment[idx[n_train + n_val :]] = 2
        train = work[assignment == 0]
        val = work[assignment == 1]
        test = work[assignment == 2]
        report = SplitReport(strategy, len(train), len(val), len(test))
        report.leaked_pairs = len(
            set(train["__pair_key"]) & (set(val["__pair_key"]) | set(test["__pair_key"]))
        )
        report.leaked_drugs = len(
            _all_drugs(train, col_a, col_b)
            & (_all_drugs(val, col_a, col_b) | _all_drugs(test, col_a, col_b))
        )
    elif strategy is SplitStrategy.PAIRWISE:
        counts = work["__pair_key"].value_counts().to_dict()
        assignment_map = _assign_groups_to_splits(
            list(counts.keys()),
            counts,
            ratios,
            rng,  # type: ignore[arg-type]
        )
        split_ids = work["__pair_key"].map(assignment_map)
        train = work[split_ids == 0]
        val = work[split_ids == 1]
        test = work[split_ids == 2]
        report = _build_report(strategy, train, val, test, col_a, col_b)
    elif strategy is SplitStrategy.COLD_START:
        # Group drugs into connected components so a pair whose two drugs are
        # linked never straddles a split boundary.
        component_ids = _connected_components(work, col_a, col_b)
        counts = pd.Series(component_ids).value_counts().to_dict()
        comp_keys = list(counts.keys())
        assignment_map = _assign_groups_to_splits(
            comp_keys,
            counts,
            ratios,
            rng,  # type: ignore[arg-type]
        )
        split_ids = pd.Series([assignment_map[comp] for comp in component_ids])
        train = work[split_ids == 0]
        val = work[split_ids == 1]
        test = work[split_ids == 2]
        report = _build_report(strategy, train, val, test, col_a, col_b)
    else:  # pragma: no cover - exhaustive
        raise ValueError(f"Unknown strategy: {strategy}")

    if label_column and label_column in work.columns:
        report.class_distribution = {
            "train": train[label_column].value_counts().to_dict(),
            "val": val[label_column].value_counts().to_dict(),
            "test": test[label_column].value_counts().to_dict(),
        }

    logger.info(report.summary())
    _warn_on_degenerate_split(report, strategy)
    for split_name, frame in (("train", train), ("val", val), ("test", test)):
        if "__pair_key" in frame.columns:
            frame.drop(columns=["__pair_key"], inplace=True)
        logger.debug("split %s: %d rows", split_name, len(frame))

    return train, val, test, report


def _warn_on_degenerate_split(report: SplitReport, strategy: SplitStrategy) -> None:
    """Flag splits that could not be populated.

    A dense drug-interaction corpus (TWOSIDES, for example) forms an almost
    complete graph, so the whole corpus is one connected component and
    ``cold_start`` cannot split it without leaking drugs across the
    boundary. This must be loud, not silent - otherwise you accidentally
    report an F1 computed on a test set that is empty or leaked.
    """
    if not report.is_degenerate:
        return

    message = (
        f"Strategy '{strategy.value}' produced an empty split "
        f"(train={report.train}, val={report.val}, test={report.test}). "
        "The drug-interaction graph is too densely connected for this strategy: "
        "almost every drug is linked to almost every other drug, so the corpus "
        "forms a single connected component and cannot be divided without "
        "leaking drugs. Use 'pairwise' instead, or restrict the corpus to a "
        "sparser subset (e.g. a single therapeutic class) before attempting "
        "'cold_start'."
    )
    report.warnings.append(message)
    logger.warning(message)


def _connected_components(df: pd.DataFrame, col_a: str, col_b: str) -> list[str]:
    """Component id for every row, over the drug co-occurrence graph.

    Two drugs that co-occur in any row are linked; components capture the
    full interaction graph so cold-start splits contain complete subgraphs.

    Returns:
        A list of component ids aligned with the rows of ``df``.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for a, b in zip(df[col_a], df[col_b], strict=False):
        union(str(a).strip().lower(), str(b).strip().lower())

    roots = {drug: find(drug) for drug in parent}
    root_to_component: dict[str, str] = {}
    components: list[str] = []

    for a, _ in zip(df[col_a], df[col_b], strict=False):
        root = roots[str(a).strip().lower()]
        component = root_to_component.setdefault(root, f"comp_{len(root_to_component)}")
        components.append(component)

    return components


def _build_report(
    strategy: SplitStrategy,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    col_a: str,
    col_b: str,
) -> SplitReport:
    train_pairs = set(_pair_keys(train, col_a, col_b))
    val_pairs = set(_pair_keys(val, col_a, col_b))
    test_pairs = set(_pair_keys(test, col_a, col_b))

    overlap = (train_pairs & val_pairs) | (train_pairs & test_pairs) | (val_pairs & test_pairs)

    train_drugs = _all_drugs(train, col_a, col_b)
    val_drugs = _all_drugs(val, col_a, col_b)
    test_drugs = _all_drugs(test, col_a, col_b)
    drug_overlap = (train_drugs & val_drugs) | (train_drugs & test_drugs) | (val_drugs & test_drugs)

    return SplitReport(
        strategy=strategy,
        train=len(train),
        val=len(val),
        test=len(test),
        leaked_pairs=len(overlap),
        leaked_drugs=len(drug_overlap),
        pair_overlap=overlap,
        drug_overlap=drug_overlap,
    )


def assert_no_leakage(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    drug_columns: Sequence[str] = DEFAULT_DRUG_COLUMNS,
    allow_drug_overlap: bool = True,
) -> None:
    """Raise if the given splits violate the requested guarantees.

    Args:
        allow_drug_overlap: If False (cold-start), drugs must not be shared.

    Raises:
        AssertionError: on detected leakage.
    """
    col_a, col_b = find_drug_columns(train, drug_columns)
    train_pairs = set(_pair_keys(train, col_a, col_b))
    for name, frame in (("val", val), ("test", test)):
        other_pairs = set(_pair_keys(frame, col_a, col_b))
        shared = train_pairs & other_pairs
        assert not shared, (
            f"Pair leakage detected between train and {name}: {len(shared)} shared pairs "
            f"(e.g. {list(shared)[:5]})"
        )

    if not allow_drug_overlap:
        train_drugs = _all_drugs(train, col_a, col_b)
        for name, frame in (("val", val), ("test", test)):
            shared = train_drugs & _all_drugs(frame, col_a, col_b)
            assert not shared, (
                f"Cold-start violation between train and {name}: {len(shared)} shared drugs "
                f"(e.g. {list(shared)[:5]})"
            )


def iter_pairs(drugs: Iterable[str]) -> list[tuple[str, str]]:
    """All unordered pairs from a list of drugs (the polypharmacy surface)."""
    unique = sorted({str(d).strip().lower() for d in drugs if str(d).strip()})
    return [(unique[i], unique[j]) for i in range(len(unique)) for j in range(i + 1, len(unique))]
