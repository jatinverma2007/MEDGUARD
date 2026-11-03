"""The MedGuard pipeline.

    image / text
        -> OCR
        -> drug NER
        -> normalisation
        -> pairwise interaction lookup (KB + optional model)
        -> severity grading
        -> calibration / abstention
        -> grounded bilingual explanations

The pipeline is designed so that **every stage can be disabled**. That is
what lets the whole thing run on a laptop CPU in under a second for
testing, while the same code runs with a 3B VLM and a DeBERTa classifier on
a free Colab T4.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from medguard.config import Settings, get_settings
from medguard.data.splits import canonical_pair, iter_pairs
from medguard.explain.templates import explain_finding, overall_summary
from medguard.extract.ner import DrugExtractor, build_extractor
from medguard.extract.ocr import GroqVisionOcr, OcrBackend, PassthroughOcr
from medguard.interaction.retrieve import KnowledgeBase, merge_findings, rows_to_findings
from medguard.interaction.severity import (
    apply_confidence_cap,
    management_for,
    overall_severity,
)
from medguard.normalize.drugs import normalize, parse_dose
from medguard.schemas import (
    Abstention,
    AnalysisResult,
    Confidence,
    DrugMention,
    InteractionFinding,
    Severity,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Runtime knobs for a single pipeline instance."""

    use_model_predictions: bool = False
    use_llm_explanations: bool = False
    language: str = "en"
    abstain_threshold: float = 0.55
    safe_assertion_threshold: float = 0.80


@dataclass
class MedGuardPipeline:
    """End-to-end prescription safety analysis."""

    settings: Settings = field(default_factory=get_settings)
    kb: KnowledgeBase | None = None
    extractor: DrugExtractor | None = None
    ocr: OcrBackend | None = None
    classifier: object | None = None
    config: PipelineConfig = field(default_factory=PipelineConfig)

    def __post_init__(self) -> None:
        if self.extractor is None:
            self.extractor = build_extractor(
                use_heavy_models=self.settings.use_heavy_models,
                model_id=self.settings.ner_model,
            )
        if self.ocr is None:
            if self.settings.llm_api_key:
                self.ocr = GroqVisionOcr(
                    api_key=self.settings.llm_api_key,
                    base_url=self.settings.llm_base_url,
                )
            else:
                self.ocr = PassthroughOcr()

    # ------------------------------------------------------------- loading
    @classmethod
    def from_interactions(
        cls,
        interactions_path: str | Path,
        **kwargs,
    ) -> MedGuardPipeline:
        """Build a pipeline from a saved canonical interaction table."""
        import pandas as pd

        path = Path(interactions_path)
        if not path.exists():
            raise FileNotFoundError(
                f"No interaction table at {path}. Run `make data` and `python scripts/make_splits.py` first."
            )
        table = pd.read_parquet(path)
        return cls(kb=KnowledgeBase.from_frame(table), **kwargs)

    def load_knowledge_base(self) -> bool:
        """Load the knowledge base from the configured path.

        Prefers the full downloaded corpus and falls back to the small
        committed sample, so the project runs immediately after cloning.

        Returns:
            True if a knowledge base was loaded.
        """
        import pandas as pd

        path = self.settings.interactions_path
        if not path.exists():
            logger.warning("No interaction table at %s", path)
            return False
        self.kb = KnowledgeBase.from_frame(pd.read_parquet(path))
        logger.info("Loaded knowledge base: %d drugs", self.kb.drug_count)
        return True

    def load_classifier(self, model_dir: str | Path) -> None:
        """Load a trained + calibrated interaction classifier."""
        from medguard.modeling.classifier import CalibratedInteractionClassifier

        classifier = CalibratedInteractionClassifier(
            encoder_name=self.settings.relation_model,
            abstain_threshold=self.config.abstain_threshold,
        )
        classifier.load(model_dir)
        self.classifier = classifier

    # ------------------------------------------------------------ extraction
    def extract_drugs(self, text: str) -> list[DrugMention]:
        """Detect and normalise drug mentions in raw prescription text."""
        assert self.extractor is not None
        mentions: list[DrugMention] = []
        seen: set[str] = set()

        for span in self.extractor.extract(text):
            normalized = normalize(span.surface)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            mentions.append(
                DrugMention(
                    surface=span.surface.strip(),
                    normalized=normalized,
                    canonical_name=normalized.title(),
                    dose=parse_dose(span.surface),
                    confidence=Confidence(raw=span.score, calibrated=span.score),
                    source="ner",
                    span=(span.start, span.end),
                )
            )
        return mentions

    # ---------------------------------------------------------- interactions
    def find_interactions(
        self,
        drugs: list[DrugMention],
    ) -> tuple[list[InteractionFinding], list[tuple[str, str]]]:
        """Look up every drug pair against the KB and (optionally) the model.

        Returns:
            ``(findings, abstained_pairs)``
        """
        keys = [d.normalized for d in drugs if d.normalized]
        pairs = iter_pairs(keys)

        kb_findings: list[InteractionFinding] = []
        model_findings: list[InteractionFinding] = []
        abstained: list[tuple[str, str]] = []

        if self.kb is not None:
            for a, b in pairs:
                rows = self.kb.lookup(a, b)
                if rows:
                    kb_findings.extend(rows_to_findings(rows))

        if self.config.use_model_predictions and self.classifier is not None:
            missing = [p for p in pairs if not (self.kb and self.kb.has(*p))]
            if missing:
                model_findings, abstained = self.classifier.predict_with_abstain(missing)
                for finding in model_findings:
                    finding.severity = apply_confidence_cap(
                        finding.severity,
                        finding.confidence.calibrated if finding.confidence else None,
                    )

        findings = merge_findings(kb_findings, model_findings)
        for finding in findings:
            finding.management = management_for(finding)
        return findings, abstained

    # --------------------------------------------------------------- analysis
    def analyze_text(
        self,
        text: str,
        *,
        language: str | None = None,
    ) -> AnalysisResult:
        """Analyse already-extracted prescription text."""
        started = time.perf_counter()
        lang = language or self.config.language

        drugs = self.extract_drugs(text)
        findings, abstained = self.find_interactions(drugs)

        explanations = [explain_finding(f, language=lang) for f in findings]
        explanations.insert(0, overall_summary(findings, len(drugs), language=lang))

        result = AnalysisResult(
            drugs=drugs,
            findings=findings,
            explanations=explanations,
            overall_severity=overall_severity(findings),
            raw_text=text if self.settings.log_phi else None,
            processing_ms=round((time.perf_counter() - started) * 1000, 2),
            pipeline_version=self.settings.app_version,
        )

        # Abstention: too few drugs, an unreadable prescription, or a low
        # confidence "no interaction" claim.
        result.abstention = self._maybe_abstain(drugs, findings, abstained)
        return result

    def analyze_image(
        self,
        image_path: str | Path,
        *,
        language: str | None = None,
    ) -> AnalysisResult:
        """Analyse a prescription image."""
        assert self.ocr is not None
        ocr_result = self.ocr.read(image_path)
        result = self.analyze_text(ocr_result.text, language=language)
        if ocr_result.confidence is not None and ocr_result.confidence < 0.5:
            result.abstention = Abstention(
                reason="OCR confidence is low; the prescription image may not be legible.",
                confidence=ocr_result.confidence,
            )
        return result

    def _maybe_abstain(
        self,
        drugs: list[DrugMention],
        findings: list[InteractionFinding],
        abstained: list[tuple[str, str]],
    ) -> Abstention | None:
        if len(drugs) == 0:
            return Abstention(
                reason="No medicines could be identified in the prescription.",
                threshold=self.config.abstain_threshold,
            )

        if abstained:
            pairs = ", ".join(f"{a} + {b}" for a, b in abstained[:5])
            return Abstention(
                reason=(
                    f"Model confidence was below the safety threshold for {len(abstained)} "
                    f"pair(s): {pairs}. Please confirm these combinations with a clinician."
                ),
                threshold=self.config.abstain_threshold,
                details={"n_abstained": len(abstained)},
            )

        # Asserting "no interaction" also needs confidence.
        if not findings:
            weakest = min(
                (d.confidence.calibrated for d in drugs if d.confidence),
                default=1.0,
            )
            if weakest < self.config.safe_assertion_threshold:
                return Abstention(
                    reason=(
                        "Medicine identification confidence is too low to assert that there "
                        "is no interaction. Please verify the prescription manually."
                    ),
                    threshold=self.config.safe_assertion_threshold,
                    confidence=weakest,
                )

        return None

    # ------------------------------------------------------------- utilities
    def knowledge_base_coverage(self, drugs: list[DrugMention]) -> float:
        """Fraction of drug pairs covered by the knowledge base."""
        if self.kb is None:
            return 0.0
        keys = [d.normalized for d in drugs if d.normalized]
        return self.kb.coverage(iter_pairs(keys))

    def lookup_pair(self, drug_a: str, drug_b: str) -> list[InteractionFinding]:
        """Direct lookup for a single pair (useful for testing and demos)."""
        if self.kb is None:
            return []
        return rows_to_findings(
            self.kb.lookup(canonical_pair(drug_a)[0], canonical_pair(drug_a)[1])
        )


def severity_of(findings: list[InteractionFinding]) -> Severity:
    """Convenience wrapper around :func:`overall_severity`."""
    return overall_severity(findings)
