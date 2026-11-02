#!/usr/bin/env python3
"""Download the public datasets MedGuard uses.

Usage::

    python scripts/download_data.py --all
    python scripts/download_data.py --dataset tdc_twosides
    python scripts/download_data.py --list

Everything lands in ``data/raw/`` and is processed into a single canonical
table at ``data/processed/interactions.parquet``.

Licence note: the DrugBank corpus needs an academic licence which TDC serves
after you accept their terms. ``TWOSIDES`` is CC BY 4.0 and needs nothing,
which is why it is the default.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medguard.config import get_settings  # noqa: E402
from medguard.data.ddi import (  # noqa: E402
    CANONICAL_COLUMNS,
    build_interaction_table,
    load_ade_corpus_v2,
    load_ddinter_hf,
    load_tdc_ddi,
    save_interaction_table,
)
from medguard.data.registry import DATASETS, describe_all  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("download_data")


def download_twosides(out_dir: Path) -> pd.DataFrame:
    """TDC TWOSIDES: 4.6M polypharmacy pairs, CC BY 4.0."""
    logger.info("Downloading TDC TWOSIDES (CC BY 4.0) ...")
    frame = load_tdc_ddi(name="TWOSIDES")
    # TWOSIDES is multi-label side effects; treat "any side effect" as the
    # interaction label and keep the side-effect list as the mechanism.
    if "Side Effect Name" in frame.columns:
        grouped = (
            frame.groupby(["Drug 1", "Drug 2"])["Side Effect Name"]
            .apply(lambda s: "; ".join(sorted(set(s))))
            .reset_index()
            .rename(
                columns={
                    "Drug 1": "drug_a",
                    "Drug 2": "drug_b",
                    "Side Effect Name": "event_type",
                }
            )
        )
        grouped["description"] = ""
        grouped["severity"] = "unknown"
        grouped["source"] = "tdc_twosides"
        grouped["source_id"] = [f"twosides:{i}" for i in range(len(grouped))]
        return grouped[CANONICAL_COLUMNS]

    frame = frame.rename(columns={"Drug 1": "drug_a", "Drug 2": "drug_b"})
    frame["event_type"] = ""
    frame["description"] = ""
    frame["severity"] = "unknown"
    frame["source"] = "tdc_twosides"
    return frame[CANONICAL_COLUMNS]


def download_drugbank(out_dir: Path) -> pd.DataFrame:
    """TDC DrugBank: 191k typed pairs. Requires an academic licence."""
    logger.info("Downloading TDC DrugBank (needs academic licence) ...")
    frame = load_tdc_ddi(name="DrugBank")
    frame = frame.rename(
        columns={
            "Drug 1": "drug_a",
            "Drug 2": "drug_b",
            "Interaction Type": "event_type",
            "Y": "label",
        }
    )
    frame["description"] = ""
    frame["severity"] = "unknown"
    frame["source"] = "tdc_drugbank"
    frame["source_id"] = [f"drugbank:{i}" for i in range(len(frame))]
    return frame[CANONICAL_COLUMNS]


def download_ddinter(out_dir: Path) -> pd.DataFrame:
    """DDInter subset with severity labels (Apache-2.0)."""
    logger.info("Downloading DDInter subset from Hugging Face ...")
    frame = load_ddinter_hf()
    logger.info("DDInter columns: %s", list(frame.columns))

    # The HF subset stores the two drugs under these names.
    drug_cols = [c for c in ("drug1", "drug2", "drug_a", "drug_b", "Drug 1", "Drug 2")]
    found = [c for c in drug_cols if c in frame.columns]
    if len(found) < 2:
        logger.warning("Unexpected DDInter schema; skipping conversion.")
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    severity_col = next(
        (c for c in ("severity", "Severity", "interaction_severity") if c in frame.columns), None
    )
    event_col = next(
        (c for c in ("interaction_type", "event_type", "reason") if c in frame.columns), None
    )
    desc_col = next(
        (c for c in ("description", "drug1_description", "summary") if c in frame.columns), None
    )

    rows = frame.rename(columns={found[0]: "drug_a", found[1]: "drug_b"}).to_dict("records")
    return build_interaction_table(
        rows,
        source="ddinter",
        severity_col=severity_col,
        event_col=event_col,
        description_col=desc_col,
    )


def download_ade(out_dir: Path) -> pd.DataFrame:
    """ADE-Corpus-V2: adverse drug event relations."""
    logger.info("Downloading ADE-Corpus-V2 ...")
    frame = load_ade_corpus_v2()
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_dir / "ade_corpus_v2.parquet", index=False)
    logger.info("Saved raw ADE-Corpus-V2 -> %s", out_dir / "ade_corpus_v2.parquet")
    return pd.DataFrame(columns=CANONICAL_COLUMNS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Download every open dataset.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), help="Download one dataset.")
    parser.add_argument("--list", action="store_true", help="List the registry and exit.")
    args = parser.parse_args()

    if args.list:
        print(describe_all())
        return 0

    settings = get_settings()
    settings.ensure_dirs()
    raw_dir = settings.raw_data_dir
    processed_dir = settings.processed_data_dir

    targets: list[str] = []
    if args.all:
        # DrugBank needs a licence, so it is opt-in only.
        targets = ["tdc_twosides", "ddinter_hf", "ade_corpus_v2"]
    elif args.dataset:
        targets = [args.dataset]
    else:
        parser.print_help()
        return 1

    frames: list[pd.DataFrame] = []
    for target in targets:
        try:
            if target == "tdc_twosides":
                frames.append(download_twosides(raw_dir))
            elif target == "tdc_drugbank":
                frames.append(download_drugbank(raw_dir))
            elif target == "ddinter_hf":
                frames.append(download_ddinter(raw_dir))
            elif target == "ade_corpus_v2":
                frames.append(download_ade(raw_dir))
            else:
                logger.info("Skipping %s (no loader in this script).", target)
        except Exception as exc:
            logger.error("Failed to download %s: %s", target, exc)
            logger.info("Continuing with the remaining datasets.")

    frames = [f for f in frames if not f.empty]
    if not frames:
        logger.error("No datasets were downloaded.")
        return 1

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["drug_a", "drug_b", "event_type"])
    combined = combined[CANONICAL_COLUMNS]

    out_path = processed_dir / "interactions.parquet"
    save_interaction_table(combined, out_path)
    logger.info(
        "Done: %d interaction rows, %d unique drugs -> %s",
        len(combined),
        len(set(combined["drug_a"]) | set(combined["drug_b"])),
        out_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
