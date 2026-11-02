#!/usr/bin/env python3
"""Build leak-free train/val/test splits and report the leakage comparison.

Usage::

    python scripts/make_splits.py --source drugbank --strategy grouped_by_drug
    python scripts/make_splits.py --strategy cold_start
    python scripts/make_splits.py --compare

The ``--compare`` mode is the one that produces a figure for your README: it
runs the same model on a naive random split and on a leak-free split, and
reports the gap.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medguard.config import get_settings  # noqa: E402
from medguard.data.splits import SplitStrategy, make_splits  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("make_splits")

STRATEGY_ALIASES = {
    "grouped_by_drug": SplitStrategy.PAIRWISE,
    "grouped": SplitStrategy.PAIRWISE,
    "pairwise": SplitStrategy.PAIRWISE,
    "cold_start": SplitStrategy.COLD_START,
    "cold-start": SplitStrategy.COLD_START,
    "random": SplitStrategy.RANDOM,
}


def resolve_strategy(name: str) -> SplitStrategy:
    try:
        return STRATEGY_ALIASES[name.lower()]
    except KeyError:
        raise SystemExit(
            f"Unknown strategy '{name}'. Use one of: {sorted(STRATEGY_ALIASES)}"
        ) from None


def compare_splits(table: Path, seed: int = 42) -> dict:
    """Train the same baseline on random vs. leak-free splits and compare.

    Splits that cannot be populated (for example ``cold_start`` on a densely
    connected interaction graph) are skipped with an explanation rather than
    crashing - the point of this script is to produce a report, and a report
    needs to say what was and was not measurable.
    """
    import numpy as np

    from medguard.evaluation.metrics import classification_metrics, leakage_impact
    from medguard.modeling.baseline import DrugPairTfidfBaseline, build_baseline_df

    frame = pd.read_parquet(table)
    results: dict[str, object] = {}

    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    skipped: dict[str, str] = {}

    for name in ("random", "pairwise", "cold_start"):
        strategy = resolve_strategy(name)
        train, val, test, report = make_splits(frame, strategy=strategy, seed=seed)

        if len(test) == 0 or len(train) == 0:
            skipped[name] = (
                f"{name}: split could not be populated "
                f"(train={len(train)}, val={len(val)}, test={len(test)}). "
                + (" ".join(report.warnings) if report.warnings else "")
            ).strip()
            logger.warning("Skipping %s - %s", name, skipped[name])
            continue

        all_drugs = sorted({str(d) for d in frame["drug_a"]} | {str(d) for d in frame["drug_b"]})
        negatives = [
            (all_drugs[i], all_drugs[j])
            for i in range(len(all_drugs))
            for j in range(i + 1, len(all_drugs))
        ]
        train_df = build_baseline_df(train, negatives)
        model = DrugPairTfidfBaseline().fit(train_df)

        test_pairs = list(zip(test["drug_a"], test["drug_b"], strict=False))
        labels = np.ones(len(test_pairs))
        preds = model.predict(test_pairs)

        metrics = classification_metrics(labels, preds)
        predictions[name] = (labels, preds)

        results[name] = {
            "split": report.to_dict(),
            "metrics": metrics.to_dict(),
        }
        logger.info("%s -> F1 %.4f", name, metrics.f1)

    if skipped:
        results["skipped"] = skipped

    if "random" in predictions and "pairwise" in predictions:
        results["leakage_impact_pairwise"] = leakage_impact(
            *predictions["random"], *predictions["pairwise"]
        )
    if "random" in predictions and "cold_start" in predictions:
        results["leakage_impact_cold_start"] = leakage_impact(
            *predictions["random"], *predictions["cold_start"]
        )

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", help="Dataset id (informational).")
    parser.add_argument(
        "--strategy",
        default="pairwise",
        help="random | pairwise | cold_start (aliases: grouped_by_drug)",
    )
    parser.add_argument("--table", type=Path, help="Canonical interaction table parquet.")
    parser.add_argument("--out-dir", type=Path, help="Where to write the splits.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compare", action="store_true", help="Run the leakage comparison.")
    args = parser.parse_args()

    settings = get_settings()
    settings.ensure_dirs()

    table = args.table or (settings.processed_data_dir / "interactions.parquet")
    if not table.exists():
        logger.error(
            "No interaction table at %s. Run `python scripts/download_data.py --all` first.",
            table,
        )
        return 1

    out_dir = args.out_dir or settings.processed_data_dir

    if args.compare:
        results = compare_splits(table, seed=args.seed)
        out_path = out_dir / "leakage_comparison.json"
        out_path.write_text(json.dumps(results, indent=2, default=str))
        print(json.dumps(results, indent=2, default=str))
        logger.info("Wrote %s", out_path)
        return 0

    frame = pd.read_parquet(table)
    train, val, test, report = make_splits(
        frame, strategy=resolve_strategy(args.strategy), seed=args.seed
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out_dir / "train.parquet", index=False)
    val.to_parquet(out_dir / "val.parquet", index=False)
    test.to_parquet(out_dir / "test.parquet", index=False)
    (out_dir / "split_report.json").write_text(json.dumps(report.to_dict(), indent=2))

    print(json.dumps(report.to_dict(), indent=2))
    logger.info("Saved splits -> %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
