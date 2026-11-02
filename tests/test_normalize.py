"""Tests for drug normalisation and alias resolution."""

from __future__ import annotations

import pytest

from medguard.normalize.drugs import (
    normalize,
    parse_dose,
    parse_frequency,
    resolve_alias,
    strip_salt,
)


@pytest.mark.parametrize(
    ("surface", "expected"),
    [
        ("Metformin HCl 500mg", "metformin"),
        ("Tab Dolo 650mg", "paracetamol"),
        ("DOLO 650", "paracetamol"),
        ("Tab. Azithral 500 mg", "azithromycin"),
        ("Gluconil 5mg", "glipizide"),
        ("  Ecosprin AV  ", "acetylsalicylic acid"),
        ("Pan 40", "pantoprazole"),
        ("Metformin", "metformin"),
    ],
)
def test_normalize(surface: str, expected: str) -> None:
    assert normalize(surface) == expected


def test_normalize_empty() -> None:
    assert normalize("") == ""
    assert normalize("   ") == ""


def test_strip_salt() -> None:
    assert strip_salt("metformin hydrochloride") == "metformin"
    assert strip_salt("amoxicillin trihydrate") == "amoxicillin"
    assert strip_salt("metformin") == "metformin"


def test_resolve_alias_unknown_returns_cleaned() -> None:
    assert resolve_alias("someunobknowndrug") == "someunobknowndrug"


@pytest.mark.parametrize(
    ("surface", "expected"),
    [
        ("Metformin 500mg", "500mg"),
        ("Tab Azithral 500 mg", "500 mg"),
        ("Dolo 650", "650"),
    ],
)
def test_parse_dose(surface: str, expected: str) -> None:
    assert parse_dose(surface) == expected


def test_parse_dose_missing() -> None:
    assert parse_dose("Metformin") is None


def test_parse_frequency() -> None:
    assert parse_frequency("Metformin 500mg BD") == "bd"
    assert parse_frequency("Metformin") is None
