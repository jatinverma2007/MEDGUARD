"""Severity grading for interaction findings.

Severity is *not* a single model's output - it is a deterministic mapping
from (a) the knowledge-base severity when a pair is known, (b) the
predicted relation type when it is not, and (c) an evidence-strength factor
that can downgrade a finding we are not confident about.

Keeping this deterministic (and separate from the model) means a clinician
can audit exactly why a finding was graded the way it was.
"""

from __future__ import annotations

from medguard.schemas import InteractionFinding, Severity

#: Relation type -> default severity. Used when a pair is not in the KB and
#: the model predicts the mechanism but not the grade.
RELATION_TYPE_SEVERITY: dict[str, Severity] = {
    # Potentially life-threatening
    "contraindicated": Severity.CONTRAINDICATED,
    "death": Severity.CONTRAINDICATED,
    "cardiac arrest": Severity.CONTRAINDICATED,
    "respiratory depression": Severity.MAJOR,
    "severe hypoglycemia": Severity.MAJOR,
    "hyperkalemia": Severity.MAJOR,
    "torsades de pointes": Severity.MAJOR,
    "hepatotoxicity": Severity.MAJOR,
    "neurotoxicity": Severity.MAJOR,
    "seizure": Severity.MAJOR,
    "gi bleeding": Severity.MAJOR,
    "bleeding": Severity.MAJOR,
    "hypertensive crisis": Severity.MAJOR,
    # Clinically significant
    "hypoglycemia": Severity.MODERATE,
    "hyperglycemia": Severity.MODERATE,
    "hypertension": Severity.MODERATE,
    "hypotension": Severity.MODERATE,
    "sedation": Severity.MODERATE,
    "dizziness": Severity.MODERATE,
    "qt prolongation": Severity.MODERATE,
    "renal impairment": Severity.MODERATE,
    "serotonin syndrome": Severity.MAJOR,
    "cushing syndrome": Severity.MODERATE,
    # Usually tolerable
    "drowsiness": Severity.MINOR,
    "nausea": Severity.MINOR,
    "headache": Severity.MINOR,
    "rash": Severity.MINOR,
    "constipation": Severity.MINOR,
}

#: Clinical management advice templates keyed by severity.
MANAGEMENT_BY_SEVERITY: dict[Severity, str] = {
    Severity.CONTRAINDICATED: (
        "Avoid this combination. Contact the prescriber to choose an alternative "
        "before the next dose."
    ),
    Severity.MAJOR: (
        "This combination can cause serious harm. The prescriber should review the "
        "dose, monitor the patient, or consider an alternative therapy."
    ),
    Severity.MODERATE: (
        "This combination may change the effect or side effects of one of the medicines. "
        "Monitoring or a dose adjustment is often needed."
    ),
    Severity.MINOR: (
        "This combination is usually well tolerated, but watch for the listed side "
        "effects and tell the prescriber if they occur."
    ),
    Severity.NONE: "No known interaction between these two medicines.",
    Severity.UNKNOWN: (
        "MedGuard could not determine the severity of this combination from its "
        "evidence base. Please confirm with the prescriber or pharmacist."
    ),
}


def severity_from_relation(event_type: str | None) -> Severity:
    """Map a relation/mechanism string onto a severity grade."""
    if not event_type:
        return Severity.UNKNOWN
    lowered = event_type.strip().lower()
    for key, severity in RELATION_TYPE_SEVERITY.items():
        if key in lowered:
            return severity
    return Severity.UNKNOWN


def apply_confidence_cap(
    severity: Severity, confidence: float | None, *, floor: float = 0.0
) -> Severity:
    """Downgrade a severity when the evidence backing it is weak.

    A "major" finding predicted with 0.30 confidence is not something to
    alarm a patient about. We keep the *type* of severity but move it down a
    notch, and let the abstention layer handle the rest.

    Args:
        severity: The severity proposed by the KB or the model.
        confidence: Calibrated confidence in ``[0, 1]`` (``None`` = trusted).
        floor: Minimum severity that will not be downgraded.

    Returns:
        The adjusted severity.
    """
    if confidence is None:
        return severity
    if confidence >= 0.75:
        return severity
    if confidence < 0.45 and severity.rank > floor:
        return Severity.from_rank(severity.rank - 1)
    return severity


def management_for(finding: InteractionFinding) -> str:
    """Clinical management advice for a finding."""
    if finding.management:
        return finding.management
    return MANAGEMENT_BY_SEVERITY.get(finding.severity, MANAGEMENT_BY_SEVERITY[Severity.UNKNOWN])


def overall_severity(findings: list[InteractionFinding]) -> Severity:
    """The highest severity across all findings."""
    if not findings:
        return Severity.NONE
    return max((f.severity for f in findings), key=lambda s: s.rank)
