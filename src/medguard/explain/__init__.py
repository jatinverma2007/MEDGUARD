"""Explain package: grounded bilingual explanations."""

from medguard.explain.llm import (
    DEFAULT_NLI_MODEL,
    GroundedLlmExplainer,
    NliVerifier,
    VerificationResult,
    split_sentences,
    verify_and_filter,
)
from medguard.explain.templates import (
    ADVICE_EN,
    ADVICE_HI,
    DISCLAIMER_EN,
    DISCLAIMER_HI,
    MECHANISM_EN,
    MECHANISM_HI,
    SEVERITY_LABEL_EN,
    SEVERITY_LABEL_HI,
    explain_all,
    explain_finding,
    overall_summary,
)

__all__ = [
    "ADVICE_EN",
    "ADVICE_HI",
    "DEFAULT_NLI_MODEL",
    "DISCLAIMER_EN",
    "DISCLAIMER_HI",
    "GroundedLlmExplainer",
    "MECHANISM_EN",
    "MECHANISM_HI",
    "NliVerifier",
    "SEVERITY_LABEL_EN",
    "SEVERITY_LABEL_HI",
    "VerificationResult",
    "explain_all",
    "explain_finding",
    "overall_summary",
    "split_sentences",
    "verify_and_filter",
]
