"""End-to-end pipeline tests."""

from __future__ import annotations

from medguard.extract.ner import LexiconDrugNer
from medguard.interaction.retrieve import KnowledgeBase
from medguard.interaction.severity import (
    apply_confidence_cap,
    severity_from_relation,
)
from medguard.pipeline import MedGuardPipeline, PipelineConfig
from medguard.schemas import Severity


def test_lexicon_ner_finds_known_drugs() -> None:
    extractor = LexiconDrugNer()
    text = "Tab Dolo 650mg for fever, Tab Azithral 500mg for infection"
    surfaces = {s.surface.lower() for s in extractor.extract(text)}
    assert any("dolo" in s for s in surfaces)
    assert any("azithral" in s for s in surfaces)


def test_pipeline_detects_interaction(pipeline: MedGuardPipeline) -> None:
    text = "Tab Aspirin 75mg and Tab Warfarin 5mg once daily"
    result = pipeline.analyze_text(text)

    assert len(result.drugs) >= 2
    assert result.findings, "expected an interaction finding for aspirin + warfarin"
    finding = result.findings[0]
    assert {finding.drug_a, finding.drug_b} == {"acetylsalicylic acid", "warfarin"}
    assert finding.severity == Severity.MAJOR
    assert finding.source == "kb"


def test_pipeline_produces_explanations(pipeline: MedGuardPipeline) -> None:
    result = pipeline.analyze_text("Aspirin 75mg, Warfarin 5mg")
    assert result.explanations
    assert "decision-support" in result.explanations[0].text.lower()


def test_pipeline_hindi_explanation(pipeline: MedGuardPipeline) -> None:
    result = pipeline.analyze_text("Aspirin 75mg, Warfarin 5mg", language="hi")
    assert result.explanations
    assert "डॉक्टर" in result.explanations[0].text


def test_pipeline_abstains_when_no_drugs_found(pipeline: MedGuardPipeline) -> None:
    result = pipeline.analyze_text("Hello, how are you today?")
    assert result.is_abstained
    assert result.abstention is not None
    assert "No medicines" in result.abstention.reason


def test_pipeline_severity_escalation(pipeline: MedGuardPipeline) -> None:
    result = pipeline.analyze_text("Aspirin 75mg, Warfarin 5mg, Metformin 500mg")
    assert result.overall_severity == Severity.MAJOR


def test_severity_from_relation_mapping() -> None:
    assert severity_from_relation("gi bleeding") == Severity.MAJOR
    assert severity_from_relation("drowsiness") == Severity.MINOR
    assert severity_from_relation(None) == Severity.UNKNOWN
    assert severity_from_relation("something obscure") == Severity.UNKNOWN


def test_confidence_cap_downgrades_weak_predictions() -> None:
    assert apply_confidence_cap(Severity.MAJOR, 0.9) == Severity.MAJOR
    assert apply_confidence_cap(Severity.MAJOR, 0.3) == Severity.MODERATE
    assert apply_confidence_cap(Severity.MAJOR, None) == Severity.MAJOR


def test_pipeline_coverage(pipeline: MedGuardPipeline) -> None:
    result = pipeline.analyze_text("Aspirin 75mg, Warfarin 5mg")
    assert pipeline.knowledge_base_coverage(result.drugs) == 1.0


def test_pipeline_without_kb_still_runs() -> None:
    pipeline = MedGuardPipeline(config=PipelineConfig())
    result = pipeline.analyze_text("Aspirin 75mg, Warfarin 5mg")
    assert result.drugs
    assert result.findings == []


def test_kb_and_pipeline_agree_on_drug_identity() -> None:
    """Regression test: the KB and the extractor must canonicalise identically.

    A brand name on the prescription must resolve to the same key as the
    generic name stored in the knowledge base, otherwise the interaction is
    silently missed.
    """
    from medguard.data.ddi import build_interaction_table

    table = build_interaction_table(
        [{"drug_a": "Aspirin", "drug_b": "Warfarin"}],
        source="test",
        severity_col="severity",
    )
    kb = KnowledgeBase.from_frame(table)
    pipeline = MedGuardPipeline(kb=kb, config=PipelineConfig())

    result = pipeline.analyze_text("Tab Ecosprin AV 75mg and Tab Warfarin 5mg")
    assert result.findings, "brand name 'Ecosprin AV' must match the KB entry 'Aspirin'"
    assert {result.findings[0].drug_a, result.findings[0].drug_b} == {
        "acetylsalicylic acid",
        "warfarin",
    }


def test_kb_lookup_canonicalises_the_query(knowledge_base: KnowledgeBase) -> None:
    """A brand name must find the same KB entry as its generic name."""
    assert knowledge_base.has("aspirin", "warfarin")
    assert knowledge_base.has("Ecosprin AV", "Warfarin Sodium")
    assert knowledge_base.has("Warfarin", "Ecosprin")

    rows = knowledge_base.lookup("Ecosprin", "warfarin")
    assert rows, "brand-name lookup must resolve"
    assert rows[0]["severity"] == "major"

    findings = knowledge_base.lookup("Ecosprin", "warfarin")
    assert findings


def test_canonical_key_is_idempotent() -> None:
    """canonical_key(canonical_key(x)) must equal canonical_key(x)."""
    from medguard.normalize.drugs import canonical_key

    for name in [
        "Ecosprin AV",
        "Aspirin",
        "Warfarin Sodium",
        "Metformin HCl 500mg",
        "Tab Dolo 650mg",
        "Gluconil 5mg",
        "Pan 40",
    ]:
        once = canonical_key(name)
        assert canonical_key(once) == once, f"not idempotent for {name!r}"
