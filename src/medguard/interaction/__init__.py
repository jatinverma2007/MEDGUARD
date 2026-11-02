"""Interaction package: knowledge-base retrieval and severity grading."""

from medguard.interaction.retrieve import (
    KnowledgeBase,
    merge_findings,
    rows_to_findings,
)
from medguard.interaction.severity import (
    MANAGEMENT_BY_SEVERITY,
    RELATION_TYPE_SEVERITY,
    apply_confidence_cap,
    management_for,
    overall_severity,
    severity_from_relation,
)

__all__ = [
    "KnowledgeBase",
    "MANAGEMENT_BY_SEVERITY",
    "RELATION_TYPE_SEVERITY",
    "apply_confidence_cap",
    "management_for",
    "merge_findings",
    "overall_severity",
    "rows_to_findings",
    "severity_from_relation",
]
