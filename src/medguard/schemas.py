"""Domain schemas for MedGuard.

These are the contract between the pipeline stages and the API. Keeping them
strict (pydantic v2) means a malformed model output fails loudly at the
boundary instead of silently poisoning downstream reasoning.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Severity(str, Enum):
    """Clinical severity of a drug-drug interaction.

    Ordering matters: ``rank`` is used to compare findings and to decide the
    overall risk level of a prescription.
    """

    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CONTRAINDICATED = "contraindicated"
    UNKNOWN = "unknown"

    @property
    def rank(self) -> int:
        return _SEVERITY_RANK[self]

    @classmethod
    def from_rank(cls, rank: int) -> Severity:
        inverse = {v: k for k, v in _SEVERITY_RANK.items()}
        return inverse.get(rank, cls.UNKNOWN)


_SEVERITY_RANK: dict[Severity, int] = {
    Severity.NONE: 0,
    Severity.MINOR: 1,
    Severity.MODERATE: 2,
    Severity.MAJOR: 3,
    Severity.CONTRAINDICATED: 4,
    Severity.UNKNOWN: -1,
}


class Confidence(BaseModel):
    """Calibrated confidence for a single prediction."""

    raw: float = Field(ge=0.0, le=1.0, description="Uncalibrated model probability")
    calibrated: float = Field(ge=0.0, le=1.0, description="Post-calibration probability")
    temperature: float = Field(default=1.0, description="Temperature applied during calibration")

    def __lt__(self, other: Confidence) -> bool:  # pragma: no cover - convenience
        return self.calibrated < other.calibrated


class DrugMention(BaseModel):
    """A single drug detected in a prescription."""

    surface: str = Field(description="Raw text as written, e.g. 'Metformin HCl 500mg'")
    normalized: str | None = Field(default=None, description="Canonical drug key")
    canonical_name: str | None = Field(default=None, description="Human-readable canonical name")
    rxnorm_id: str | None = Field(default=None)
    atc_code: str | None = Field(default=None)
    dose: str | None = Field(default=None, description="Dose and unit, if present")
    confidence: Confidence | None = None
    source: Literal["ner", "heuristic", "ocr", "manual"] = "ner"
    span: tuple[int, int] | None = None

    @field_validator("surface")
    @classmethod
    def _strip_surface(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("surface must not be empty")
        return value


class InteractionFinding(BaseModel):
    """A detected interaction between two drugs."""

    drug_a: str
    drug_b: str
    severity: Severity = Severity.UNKNOWN
    event_type: str | None = Field(default=None, description="e.g. 'bleeding', 'hypoglycemia'")
    description: str | None = None
    management: str | None = Field(default=None, description="Clinical management advice")
    evidence: list[str] = Field(default_factory=list, description="Source dataset ids / citations")
    confidence: Confidence | None = None
    source: Literal["kb", "model", "both"] = "kb"

    @property
    def pair(self) -> tuple[str, str]:
        return (self.drug_a, self.drug_b)


class Explanation(BaseModel):
    """Patient/clinician-facing explanation."""

    text: str
    language: Literal["en", "hi"] = "en"
    grounded: bool = Field(default=True, description="Every claim traceable to evidence")
    citations: list[str] = Field(default_factory=list)


class Abstention(BaseModel):
    """Records *why* the system refused to give a definitive answer."""

    reason: str
    threshold: float | None = None
    confidence: float | None = None
    details: dict[str, str | float | int] = Field(default_factory=dict)


class AnalysisResult(BaseModel):
    """Complete output of the MedGuard pipeline for one prescription."""

    drugs: list[DrugMention] = Field(default_factory=list)
    findings: list[InteractionFinding] = Field(default_factory=list)
    explanations: list[Explanation] = Field(default_factory=list)
    abstention: Abstention | None = None
    overall_severity: Severity = Severity.NONE
    raw_text: str | None = Field(default=None, exclude=True, description="PHI - never serialise")
    processing_ms: float | None = None
    pipeline_version: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def max_severity_rank(self) -> int:
        return max((f.severity.rank for f in self.findings), default=0)

    @property
    def is_abstained(self) -> bool:
        return self.abstention is not None
