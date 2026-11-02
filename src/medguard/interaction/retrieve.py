"""Interaction retrieval against the knowledge base.

The knowledge base is the canonical interaction table (see
:mod:`medguard.data.ddi`) indexed by the unordered drug pair. Retrieval is
exact-match; the *model* layer is what predicts interactions for pairs that
are absent from the KB, which is the interesting ML problem.

Two retrieval modes:

``strict``
    Only exact KB hits. Maximum precision, poor coverage. This is the mode
    used for the "grounded" product path.

``augmented``
    Exact hits plus model-predicted candidates for unseen pairs. Higher
    coverage, lower precision - every such finding is explicitly tagged
    ``source="model"`` so the UI can distinguish it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from medguard.data.splits import canonical_pair
from medguard.normalize.drugs import canonical_key
from medguard.schemas import InteractionFinding, Severity

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeBase:
    """Indexed interaction table.

    Attributes:
        table: The canonical interaction dataframe.
        index: Mapping of ``(drug_a, drug_b)`` -> list of row dicts.
    """

    table: pd.DataFrame
    index: dict[tuple[str, str], list[dict[str, object]]] = field(default_factory=dict)

    @classmethod
    def from_frame(cls, table: pd.DataFrame) -> KnowledgeBase:
        required = {"drug_a", "drug_b", "severity"}
        missing = required - set(table.columns)
        if missing:
            raise ValueError(f"Interaction table missing columns: {sorted(missing)}")

        index: dict[tuple[str, str], list[dict[str, object]]] = {}
        for record in table.to_dict("records"):
            # Canonicalise on ingest so the KB agrees with the pipeline even
            # if the table was built outside `build_interaction_table`.
            a = canonical_key(str(record["drug_a"]))
            b = canonical_key(str(record["drug_b"]))
            if not a or not b or a == b:
                continue
            key = canonical_pair(a, b)
            index.setdefault(key, []).append(record)

        logger.info("Knowledge base: %d pairs, %d rows", len(index), len(table))
        return cls(table=table, index=index)

    def lookup(self, drug_a: str, drug_b: str) -> list[dict[str, object]]:
        """All knowledge-base rows for an unordered pair (possibly empty).

        The query is canonicalised first, so a brand name finds the same
        entry as its generic name ("Ecosprin" -> "acetylsalicylic acid").
        """
        a = canonical_key(str(drug_a))
        b = canonical_key(str(drug_b))
        if not a or not b:
            return []
        return self.index.get(canonical_pair(a, b), [])

    def has(self, drug_a: str, drug_b: str) -> bool:
        a = canonical_key(str(drug_a))
        b = canonical_key(str(drug_b))
        if not a or not b:
            return False
        return canonical_pair(a, b) in self.index

    def severity_for(self, drug_a: str, drug_b: str) -> Severity:
        """Highest severity recorded for a pair."""
        rows = self.lookup(drug_a, drug_b)
        if not rows:
            return Severity.NONE
        best = Severity.NONE
        for row in rows:
            sev = Severity(str(row.get("severity", "unknown")))
            if sev.rank > best.rank:
                best = sev
        return best

    @property
    def drug_count(self) -> int:
        drugs: set[str] = set()
        for key in self.index:
            drugs.update(key)
        return len(drugs)

    def coverage(self, drug_pairs: list[tuple[str, str]]) -> float:
        """Fraction of queried pairs present in the KB."""
        if not drug_pairs:
            return 0.0
        hits = sum(1 for a, b in drug_pairs if self.has(a, b))
        return hits / len(drug_pairs)


def rows_to_findings(rows: list[dict[str, object]]) -> list[InteractionFinding]:
    """Convert knowledge-base rows into findings.

    Drug names are canonicalised so a finding always reports the generic
    name, even when the source record used a brand name.
    """
    findings: list[InteractionFinding] = []
    for row in rows:
        severity = Severity(str(row.get("severity", "unknown")))
        findings.append(
            InteractionFinding(
                drug_a=canonical_key(str(row.get("drug_a", ""))),
                drug_b=canonical_key(str(row.get("drug_b", ""))),
                severity=severity,
                event_type=str(row.get("event_type", "")) or None,
                description=str(row.get("description", "")) or None,
                evidence=[str(row.get("source_id", ""))] if row.get("source_id") else [],
                source="kb",
            )
        )
    return findings


def merge_findings(*groups: list[InteractionFinding]) -> list[InteractionFinding]:
    """Merge finding lists, keeping the most severe entry per pair."""
    merged: dict[tuple[str, str], InteractionFinding] = {}
    for group in groups:
        for finding in group:
            key = canonical_pair(finding.drug_a, finding.drug_b)
            existing = merged.get(key)
            if existing is None or finding.severity.rank > existing.severity.rank:
                merged[key] = finding
            elif existing.source == "kb" and finding.source == "model":
                # Keep the KB evidence alongside the model finding.
                existing.evidence = list(dict.fromkeys([*existing.evidence, *finding.evidence]))
    return sorted(merged.values(), key=lambda f: -f.severity.rank)
