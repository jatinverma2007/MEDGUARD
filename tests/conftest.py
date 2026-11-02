"""Shared pytest fixtures."""

from __future__ import annotations

import pandas as pd
import pytest

from medguard.interaction.retrieve import KnowledgeBase
from medguard.pipeline import MedGuardPipeline, PipelineConfig


@pytest.fixture
def sample_interactions() -> pd.DataFrame:
    """A tiny interaction table with known relationships.

    Built through ``build_interaction_table`` so the keys are canonicalised
    exactly the way the real loaders do it - the KB and the extraction
    pipeline must agree on drug identity.
    """
    from medguard.data.ddi import build_interaction_table

    return build_interaction_table(
        [
            {
                "drug_a": "Aspirin",
                "drug_b": "Warfarin",
                "event_type": "bleeding",
                "severity": "Major",
                "description": "Both drugs inhibit platelet function and anticoagulation.",
            },
            {
                "drug_a": "Metformin",
                "drug_b": "Glimepiride",
                "event_type": "hypoglycemia",
                "severity": "Moderate",
                "description": "Additive lowering of blood glucose.",
            },
            {
                "drug_a": "Ciprofloxacin",
                "drug_b": "Theophylline",
                "event_type": "seizure",
                "severity": "Major",
                "description": "CYP1A2 inhibition raises theophylline levels.",
            },
            {
                "drug_a": "Lisinopril",
                "drug_b": "Potassium Chloride",
                "event_type": "hyperkalemia",
                "severity": "Moderate",
                "description": "Additive potassium retention.",
            },
        ],
        source="unit-test",
        severity_col="severity",
        event_col="event_type",
        description_col="description",
    )


@pytest.fixture
def knowledge_base(sample_interactions: pd.DataFrame) -> KnowledgeBase:
    return KnowledgeBase.from_frame(sample_interactions)


@pytest.fixture
def pipeline(knowledge_base: KnowledgeBase) -> MedGuardPipeline:
    return MedGuardPipeline(
        kb=knowledge_base,
        config=PipelineConfig(language="en"),
    )
