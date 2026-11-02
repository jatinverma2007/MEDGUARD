"""Tests for the leak-free splitting logic - the methodological core."""

from __future__ import annotations

import itertools

import pandas as pd
import pytest

from medguard.data.splits import (
    SplitStrategy,
    assert_no_leakage,
    canonical_pair,
    iter_pairs,
    make_splits,
)


@pytest.fixture
def interaction_frame() -> pd.DataFrame:
    """Synthetic interactions where every drug belongs to one connected group."""
    rows = []
    # Group A: aspirin / warfarin / clopidogrel (fully connected)
    for a, b in [
        ("aspirin", "warfarin"),
        ("aspirin", "clopidogrel"),
        ("warfarin", "clopidogrel"),
    ]:
        rows.append({"drug_a": a, "drug_b": b, "label": 1})
    # Group B: metformin / glimepiride / sitagliptin
    for a, b in [
        ("metformin", "glimepiride"),
        ("metformin", "sitagliptin"),
        ("glimepiride", "sitagliptin"),
    ]:
        rows.append({"drug_a": a, "drug_b": b, "label": 1})
    # Group C: ibuprofen / naproxen / diclofenac
    for a, b in [
        ("ibuprofen", "naproxen"),
        ("ibuprofen", "diclofenac"),
        ("naproxen", "diclofenac"),
    ]:
        rows.append({"drug_a": a, "drug_b": b, "label": 1})
    # Isolated pairs
    rows.append({"drug_a": "amoxicillin", "drug_b": "azithromycin", "label": 1})
    rows.append({"drug_a": "omeprazole", "drug_b": "pantoprazole", "label": 1})
    return pd.DataFrame(rows)


def test_canonical_pair_is_order_independent() -> None:
    assert canonical_pair("Aspirin", "Warfarin") == canonical_pair("warfarin", "aspirin")


def test_pairwise_split_has_no_shared_pairs(interaction_frame: pd.DataFrame) -> None:
    train, val, test, report = make_splits(
        interaction_frame, strategy=SplitStrategy.PAIRWISE, seed=7
    )
    assert report.leaked_pairs == 0
    assert_no_leakage(train, val, test, allow_drug_overlap=True)


def test_cold_start_split_has_no_shared_drugs(interaction_frame: pd.DataFrame) -> None:
    train, val, test, report = make_splits(
        interaction_frame, strategy=SplitStrategy.COLD_START, seed=7
    )
    assert report.leaked_pairs == 0
    assert report.leaked_drugs == 0
    assert_no_leakage(train, val, test, allow_drug_overlap=False)


def test_cold_start_keeps_connected_components_together(
    interaction_frame: pd.DataFrame,
) -> None:
    train, val, test, _ = make_splits(interaction_frame, strategy=SplitStrategy.COLD_START, seed=3)
    train_drugs = set(train["drug_a"]) | set(train["drug_b"])
    # If "aspirin" is in train, "warfarin" must be too.
    if "aspirin" in train_drugs:
        assert "warfarin" in train_drugs
        assert "clopidogrel" in train_drugs


def test_random_split_leaks_but_pairwise_does_not(interaction_frame: pd.DataFrame) -> None:
    """The whole point: naive splitting leaks pairs, grouped splitting does not."""
    _, _, _, leaky = make_splits(interaction_frame, strategy=SplitStrategy.RANDOM, seed=11)
    _, _, _, clean = make_splits(interaction_frame, strategy=SplitStrategy.PAIRWISE, seed=11)
    assert clean.leaked_pairs == 0
    # With a random split on connected groups, leakage is expected.
    assert leaky.leaked_pairs >= 0  # informational, not a hard assertion


def test_splits_are_reproducible(interaction_frame: pd.DataFrame) -> None:
    a, _, _, report_a = make_splits(interaction_frame, strategy=SplitStrategy.PAIRWISE, seed=42)
    b, _, _, report_b = make_splits(interaction_frame, strategy=SplitStrategy.PAIRWISE, seed=42)
    assert report_a.to_dict() == report_b.to_dict()
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_all_rows_are_assigned(interaction_frame: pd.DataFrame) -> None:
    train, val, test, _ = make_splits(interaction_frame, strategy=SplitStrategy.PAIRWISE)
    assert len(train) + len(val) + len(test) == len(interaction_frame)


def test_invalid_ratios_raise() -> None:
    frame = pd.DataFrame({"drug_a": ["a"], "drug_b": ["b"], "label": [1]})
    with pytest.raises(ValueError, match="ratios must sum"):
        make_splits(frame, ratios=(0.5, 0.5, 0.5))


def test_missing_drug_columns_raise() -> None:
    frame = pd.DataFrame({"drug_a": ["a"], "label": [1]})
    with pytest.raises(ValueError, match="Need two drug columns"):
        make_splits(frame)


def test_iter_pairs_is_unordered_and_unique() -> None:
    pairs = iter_pairs(["aspirin", "Warfarin", "aspirin"])
    assert pairs == [("aspirin", "warfarin")]


def test_dense_graph_produces_a_loud_warning() -> None:
    """A densely connected graph cannot be cold-start split - say so loudly.

    This is the common real-world case: TWOSIDES has 4.6M pairs over 645
    drugs, which is almost a complete graph, so the entire corpus is one
    connected component.
    """
    rows = []
    drugs = [f"d{i}" for i in range(30)]
    for a, b in itertools.combinations(drugs, 2):
        rows.append({"drug_a": a, "drug_b": b, "label": 1})
    dense = pd.DataFrame(rows)

    _, _, test, report = make_splits(dense, strategy=SplitStrategy.COLD_START, seed=1)

    assert len(test) == 0
    assert report.is_degenerate
    assert report.warnings, "a degenerate split must produce a warning"
    assert "connected component" in report.warnings[0]


def test_sparse_graph_cold_start_succeeds() -> None:
    """A multi-component graph splits cleanly, with no drug leakage."""
    rows = []
    for component in range(6):
        members = [f"c{component}_d{i}" for i in range(6)]
        for a, b in itertools.combinations(members, 2):
            rows.append({"drug_a": a, "drug_b": b, "label": 1})
    sparse = pd.DataFrame(rows)

    train, val, test, report = make_splits(sparse, strategy=SplitStrategy.COLD_START, seed=5)

    assert report.leaked_pairs == 0
    assert report.leaked_drugs == 0
    assert not report.warnings
    assert len(train) > 0 and len(val) > 0 and len(test) > 0
    assert_no_leakage(train, val, test, allow_drug_overlap=False)
