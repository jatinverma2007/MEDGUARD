"""Data package: loaders, registry and leak-free splitting."""

from medguard.data.ddi import (
    CANONICAL_COLUMNS,
    build_interaction_table,
    clean_drug_name,
    load_ade_corpus_v2,
    load_dataset_by_id,
    load_ddinter_hf,
    load_tdc_ddi,
    map_severity,
    normalise_pair,
    save_interaction_table,
)
from medguard.data.registry import DATASETS, DatasetSpec, describe_all, get_spec
from medguard.data.splits import (
    SplitReport,
    SplitStrategy,
    assert_no_leakage,
    canonical_pair,
    iter_pairs,
    make_splits,
)

__all__ = [
    "CANONICAL_COLUMNS",
    "DATASETS",
    "DatasetSpec",
    "SplitReport",
    "SplitStrategy",
    "assert_no_leakage",
    "build_interaction_table",
    "canonical_pair",
    "clean_drug_name",
    "describe_all",
    "get_spec",
    "iter_pairs",
    "load_ade_corpus_v2",
    "load_dataset_by_id",
    "load_ddinter_hf",
    "load_tdc_ddi",
    "make_splits",
    "map_severity",
    "normalise_pair",
    "save_interaction_table",
]
