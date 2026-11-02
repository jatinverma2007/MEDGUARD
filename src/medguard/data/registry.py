"""Registry of the public datasets MedGuard uses.

Every entry records the *actual* verified availability so the student never
wastes a day chasing a dead URL. Licence notes are included because shipping
an unlicensed DrugBank dump in a public repo is a real problem.

Verified sources
----------------
``tdc_drugbank``
    TDC ``DDI(name="DrugBank")`` - 191,808 interaction pairs, 1,706 drugs,
    86 relation types. DrugBank content requires an academic licence;
    TDC serves it after you accept their terms. Install with
    ``pip install "medguard[data]"``.

``tdc_twosides``
    TDC ``DDI(name="TWOSIDES")`` - 4,649,441 pairs, 645 drugs, multi-label
    side effects. **CC BY 4.0** - no licence friction. This is the default
    corpus for training because it is fully open.

``ddinter_hf``
    ``Tassy24/K-Paths-inductive-reasoning-ddinter`` - a DDInter-derived
    subset with severity labels (Major / Moderate / Minor) and drug
    descriptions. Apache-2.0, ~10k-100k rows.

``ade_corpus_v2``
    ``ade-benchmark-corpus/ade_corpus_v2`` - adverse drug event relations
    (DRUG-AE.rel) plus negative sentences.

``meddialog``
    ``bigbio/meddialog`` / ``lighteval/med_dialog`` - doctor-patient
    dialogues, used for explanation style and faithfulness references.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)

DatasetId = Literal[
    "tdc_drugbank",
    "tdc_twosides",
    "ddinter_hf",
    "ade_corpus_v2",
    "meddialog",
]


@dataclass(frozen=True)
class DatasetSpec:
    """Metadata for one dataset."""

    id: str
    title: str
    url: str
    licence: str
    n_rows: str
    notes: str = ""
    requires_install: tuple[str, ...] = ()
    is_default: bool = False
    tags: tuple[str, ...] = field(default_factory=tuple)


DATASETS: dict[str, DatasetSpec] = {
    "tdc_twosides": DatasetSpec(
        id="tdc_twosides",
        title="TWOSIDES Polypharmacy Side Effects (TDC)",
        url="https://tdcommons.ai/multi_pred_tasks/ddi/",
        licence="CC BY 4.0",
        n_rows="4,649,441 pairs / 645 drugs",
        notes="Fully open. Default training corpus.",
        requires_install=("tdc",),
        is_default=True,
        tags=("ddi", "multilabel", "open"),
    ),
    "tdc_drugbank": DatasetSpec(
        id="tdc_drugbank",
        title="DrugBank Multi-Typed DDI (TDC)",
        url="https://tdcommons.ai/multi_pred_tasks/ddi/",
        licence="Academic licence required (DrugBank)",
        n_rows="191,808 pairs / 1,706 drugs",
        notes="86 relation types. Needs academic DrugBank licence via TDC.",
        requires_install=("tdc",),
        tags=("ddi", "typed", "licensed"),
    ),
    "ddinter_hf": DatasetSpec(
        id="ddinter_hf",
        title="DDInter (severity-labelled subset)",
        url="https://huggingface.co/datasets/Tassy24/K-Paths-inductive-reasoning-ddinter",
        licence="Apache-2.0",
        n_rows="10k-100k rows",
        notes="Severity grades: Major / Moderate / Minor.",
        requires_install=("datasets",),
        tags=("ddi", "severity", "open"),
    ),
    "ade_corpus_v2": DatasetSpec(
        id="ade_corpus_v2",
        title="ADE-Corpus-V2 (Adverse Drug Reactions)",
        url="https://huggingface.co/datasets/ade-benchmark-corpus/ade_corpus_v2",
        licence="See dataset card (public, unrestricted)",
        n_rows="~18,520 drug-AE relations",
        notes="Relation extraction + sentence classification.",
        requires_install=("datasets",),
        tags=("ade", "ner", "relation"),
    ),
    "meddialog": DatasetSpec(
        id="meddialog",
        title="MedDialog (doctor-patient dialogues)",
        url="https://huggingface.co/datasets/bigbio/meddialog",
        licence="Public, copyright retained by healthcaremagic.com / icliniq.com",
        n_rows="0.26M English dialogues",
        notes="Use for explanation style and faithfulness references.",
        requires_install=("datasets",),
        tags=("dialogue", "eval"),
    ),
}


def get_spec(dataset_id: str) -> DatasetSpec:
    try:
        return DATASETS[dataset_id]
    except KeyError:
        raise KeyError(f"Unknown dataset '{dataset_id}'. Known: {sorted(DATASETS)}") from None


def default_datasets() -> list[DatasetSpec]:
    """Datasets fetched by ``make download-data`` with ``--all``."""
    return [spec for spec in DATASETS.values() if spec.id in {"tdc_twosides", "ade_corpus_v2"}]


def describe_all() -> str:
    """Human-readable table of the registry (used by the CLI)."""
    lines = [f"{'ID':<18} {'ROWS':<28} {'LICENCE':<32} URL"]
    lines.append("-" * 120)
    for spec in DATASETS.values():
        lines.append(f"{spec.id:<18} {spec.n_rows:<28} {spec.licence:<32} {spec.url}")
    return "\n".join(lines)
