"""Loaders that turn heterogeneous public datasets into one canonical table.

Canonical interaction table columns
-----------------------------------
``drug_a``, ``drug_b``
    Lower-cased, order-independent drug identifiers.
``event_type``
    Free-text interaction mechanism, e.g. ``bleeding``, ``hypoglycemia``.
``severity``
    One of ``none`` / ``minor`` / ``moderate`` / ``major`` /
    ``contraindicated`` / ``unknown``.
``description``
    Human-readable description of the clinical consequence.
``source``
    Which corpus the row came from.
``source_id``
    Stable identifier so a finding can always be traced back to its evidence.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from medguard.data.registry import DatasetSpec, get_spec
from medguard.normalize.drugs import canonical_key
from medguard.schemas import Severity

logger = logging.getLogger(__name__)

CANONICAL_COLUMNS = [
    "drug_a",
    "drug_b",
    "event_type",
    "severity",
    "description",
    "source",
    "source_id",
]

#: Textual severity markers mapped onto the MedGuard severity scale.
_SEVERITY_PATTERNS: tuple[tuple[str, Severity], ...] = (
    (r"\bcontraindicat", Severity.CONTRAINDICATED),
    (r"\bavoid\b", Severity.CONTRAINDICATED),
    (r"\bmajor\b", Severity.MAJOR),
    (r"\bsevere\b", Severity.MAJOR),
    (r"\bserious\b", Severity.MAJOR),
    (r"\bmoderate\b", Severity.MODERATE),
    (r"\bminor\b", Severity.MINOR),
    (r"\blow\b", Severity.MINOR),
    (r"\bnone\b", Severity.NONE),
)

#: Tokens stripped from drug names during normalisation.
_DOSE_PATTERN = re.compile(r"\b\d+(\.\d+)?\s*(mg|mcg|g|ml|iu|units?|%)\b.*$", re.IGNORECASE)
_BRACKET_PATTERN = re.compile(r"\(([^)]*)\)")


def clean_drug_name(name: str) -> str:
    """Normalise a raw drug string to a lookup key.

    Delegates to :func:`medguard.normalize.drugs.canonical_key` so the
    knowledge base and the extraction pipeline agree on drug identity.
    Using two different normalisers is a silent-recall bug: "Ecosprin AV"
    would resolve to "acetylsalicylic acid" while the KB still held
    "aspirin", and the interaction would never be found.

    Examples:
        >>> clean_drug_name("Metformin HCl 500mg")
        'metformin'
        >>> clean_drug_name("  AMOXICILLIN (500mg) ")
        'amoxicillin'
    """
    return canonical_key(name)


def map_severity(text: str) -> Severity:
    """Map a free-text severity description onto the :class:`Severity` scale."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return Severity.UNKNOWN
    lowered = str(text).strip().lower()
    if not lowered:
        return Severity.UNKNOWN
    for pattern, severity in _SEVERITY_PATTERNS:
        if re.search(pattern, lowered):
            return severity
    return Severity.UNKNOWN


def normalise_pair(drug_a: str, drug_b: str) -> tuple[str, str]:
    """Canonical (order-independent) pair of cleaned drug names."""
    a, b = clean_drug_name(drug_a), clean_drug_name(drug_b)
    if not a or not b or a == b:
        return a, b
    return (a, b) if a <= b else (b, a)


def build_interaction_table(
    rows: list[dict[str, object]],
    source: str,
    *,
    drug_cols: tuple[str, str] = ("drug_a", "drug_b"),
    severity_col: str | None = None,
    description_col: str | None = None,
    event_col: str | None = None,
    id_col: str | None = None,
) -> pd.DataFrame:
    """Build the canonical interaction table from loosely-shaped rows."""
    records: list[dict[str, object]] = []
    for i, row in enumerate(rows):
        raw_a = row.get(drug_cols[0], "")
        raw_b = row.get(drug_cols[1], "")
        a, b = normalise_pair(str(raw_a), str(raw_b))
        if not a or not b or a == b:
            continue
        severity = (
            map_severity(str(row.get(severity_col, ""))) if severity_col else Severity.UNKNOWN
        )
        records.append(
            {
                "drug_a": a,
                "drug_b": b,
                "event_type": str(row.get(event_col, "")).strip() if event_col else "",
                "severity": severity.value,
                "description": str(row.get(description_col, "")).strip() if description_col else "",
                "source": source,
                "source_id": str(row.get(id_col, f"{source}:{i}")) if id_col else f"{source}:{i}",
            }
        )
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    return df[CANONICAL_COLUMNS]


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------


def load_tdc_ddi(
    name: str = "TWOSIDES",
    *,
    split: str | None = None,
) -> pd.DataFrame:
    """Load a TDC DDI corpus.

    Args:
        name: ``"TWOSIDES"`` (CC BY 4.0, open) or ``"DrugBank"`` (licensed).
        split: ``None`` for the full frame, else ``"train"`` / ``"valid"`` / ``"test"``.

    Raises:
        ImportError: if ``tdc`` is not installed.
    """
    try:
        from tdc.multi_pred import DDI
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("TDC is not installed. Run: pip install 'medguard[data]'") from exc

    data = DDI(name=name)
    frame = data.get_data() if split is None else data.get_split()[split]
    logger.info("Loaded TDC DDI %s: %d rows, columns=%s", name, len(frame), list(frame.columns))
    return frame


def load_ddinter_hf(
    *,
    split: str = "train",
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """Load the severity-labelled DDInter subset from Hugging Face."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "The `datasets` package is required. Run: pip install 'medguard[ml]'"
        ) from exc

    ds = load_dataset(
        "Tassy24/K-Paths-inductive-reasoning-ddinter", split=split, cache_dir=cache_dir
    )
    frame = ds.to_pandas()
    logger.info("Loaded DDInter (HF): %d rows, columns=%s", len(frame), list(frame.columns))
    return frame


def load_ade_corpus_v2(
    *,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """Load ADE-Corpus-V2 as a flat dataframe of sentences + relations."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "The `datasets` package is required. Run: pip install 'medguard[ml]'"
        ) from exc

    ds = load_dataset("ade-benchmark-corpus/ade_corpus_v2", cache_dir=cache_dir)
    parts = []
    for split in ds:
        frame = ds[split].to_pandas()
        frame["__split"] = split
        parts.append(frame)
    frame = pd.concat(parts, ignore_index=True)
    logger.info("Loaded ADE-Corpus-V2: %d rows", len(frame))
    return frame


def load_dataset_by_id(dataset_id: str, **kwargs) -> pd.DataFrame:
    """Dispatch to the loader for a registry id."""
    spec: DatasetSpec = get_spec(dataset_id)
    if dataset_id in {"tdc_drugbank", "tdc_twosides"}:
        name = "DrugBank" if dataset_id == "tdc_drugbank" else "TWOSIDES"
        return load_tdc_ddi(name=name, **kwargs)
    if dataset_id == "ddinter_hf":
        return load_ddinter_hf(**kwargs)
    if dataset_id == "ade_corpus_v2":
        return load_ade_corpus_v2(**kwargs)
    if dataset_id == "meddialog":
        from datasets import load_dataset

        return load_dataset("bigbio/meddialog", **kwargs).to_pandas()
    raise ValueError(f"No loader implemented for {spec.id}")


def save_interaction_table(df: pd.DataFrame, path: Path) -> Path:
    """Persist the canonical interaction table as parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Saved %d interactions -> %s", len(df), path)
    return path
