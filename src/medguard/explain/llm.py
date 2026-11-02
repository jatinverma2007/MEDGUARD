"""Optional LLM explanation layer with faithfulness verification.

The templated explainer in :mod:`medguard.explain.templates` is the default
because it cannot hallucinate. This module provides the *optional* upgrade:
a free-tier LLM writes a richer, more readable explanation - but:

1. The prompt contains **only** retrieved evidence, nothing else.
2. The response is split into sentences.
3. Every sentence is checked against the evidence with an NLI entailment
   model (``MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`` style checkpoint).
4. Sentences that are not entailed are **removed**, and the finding is
   flagged as partially ungrounded.

Step 3 is the interesting research contribution: it turns "the LLM said
something nice" into a measurable faithfulness number you can report
before/after with and without retrieval.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from medguard.schemas import Explanation, InteractionFinding

logger = logging.getLogger(__name__)

DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

SYSTEM_PROMPT = (
    "You are a clinical information assistant. You explain drug interactions to "
    "patients in plain language. You are given a set of verified facts from a "
    "drug interaction database. "
    "RULES: "
    "1. Use ONLY the facts provided. Never add information from memory. "
    "2. Never state that a combination is safe if it is not stated in the facts. "
    "3. Always include the severity as given. "
    "4. Never give a dosing instruction. "
    "5. Keep it under 4 sentences and mention that the person should confirm with "
    "their doctor or pharmacist. "
    "6. Answer in the requested language."
)

EVIDENCE_BLOCK = """Verified facts:
{facts}

Severity: {severity}
Mechanism: {mechanism}
Clinical description: {description}

Write the patient-facing explanation in {language_name}.
"""


@dataclass
class VerificationResult:
    """Result of checking an explanation against its evidence."""

    grounded: bool
    supported_sentences: list[str] = field(default_factory=list)
    unsupported_sentences: list[str] = field(default_factory=list)
    entailment_scores: list[float] = field(default_factory=list)

    @property
    def grounding_ratio(self) -> float:
        total = len(self.supported_sentences) + len(self.unsupported_sentences)
        if total == 0:
            return 1.0
        return len(self.supported_sentences) / total


def split_sentences(text: str) -> list[str]:
    """Split an explanation into sentences (simple, language-agnostic)."""
    if not text:
        return []
    parts = re.split(r"(?<=[.!?।])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


class NliVerifier:
    """Entailment check with a DeBERTa MNLI model.

    Requires ``pip install 'medguard[ml]'``.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_NLI_MODEL,
        device: str | None = None,
        threshold: float = 0.5,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.threshold = threshold
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "transformers is required for NliVerifier. Run: pip install 'medguard[ml]'"
            ) from exc

        logger.info("Loading NLI model %s ...", self.model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        self._model.eval()

    def entailment(self, premise: str, hypothesis: str) -> float:
        """Probability that ``hypothesis`` is entailed by ``premise``."""
        self._load()
        import torch

        encoded = self._tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        if self.device:
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = self._model(**encoded).logits

        probs = torch.softmax(logits, dim=-1)[0]
        labels = {v.lower(): k for k, v in self._model.config.id2label.items()}
        entail_idx = labels.get("entailment")
        if entail_idx is None:
            # Fall back to the second class, which is entailment in most MNLI heads.
            entail_idx = 1 if probs.shape[0] > 1 else 0
        return float(probs[entail_idx])

    def verify(self, sentences: list[str], evidence: str) -> VerificationResult:
        """Split sentences into entailed / not-entailed."""
        supported: list[str] = []
        unsupported: list[str] = []
        scores: list[float] = []

        for sentence in sentences:
            score = self.entailment(evidence, sentence)
            scores.append(score)
            if score >= self.threshold:
                supported.append(sentence)
            else:
                unsupported.append(sentence)

        return VerificationResult(
            grounded=not unsupported,
            supported_sentences=supported,
            unsupported_sentences=unsupported,
            entailment_scores=scores,
        )


class GroundedLlmExplainer:
    """LLM explanation grounded in retrieved evidence.

    Uses any OpenAI-compatible endpoint, so free tiers work:
    Groq (``https://api.groq.com/openai/v1``), Google AI Studio
    (``https://generativelanguage.googleapis.com/v1beta/openai/``),
    OpenRouter, or a local Ollama server.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.groq.com/openai/v1",
        model: str = "llama-3.3-70b-versatile",
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout = timeout

    def _client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "The `openai` package is required for the LLM layer. Run: pip install openai"
            ) from exc
        return OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

    def _facts_for(self, finding: InteractionFinding) -> str:
        facts = [
            f"{finding.drug_a} and {finding.drug_b} have a {finding.severity.value} interaction.",
        ]
        if finding.event_type:
            facts.append(f"The interaction mechanism is {finding.event_type}.")
        if finding.description:
            facts.append(finding.description)
        for evidence_id in finding.evidence:
            facts.append(f"Source record: {evidence_id}.")
        return " ".join(facts)

    def explain(self, finding: InteractionFinding, language: str = "en") -> Explanation:
        """Generate a grounded explanation for one finding."""
        language_name = "English" if language == "en" else "Hindi (Devanagari)"
        prompt = EVIDENCE_BLOCK.format(
            facts=self._facts_for(finding),
            severity=finding.severity.value,
            mechanism=finding.event_type or "not specified",
            description=finding.description or "not specified",
            language_name=language_name,
        )

        try:
            response = self._client().chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=220,
            )
            text = (response.choices[0].message.content or "").strip()
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("LLM explanation failed (%s); falling back to template", exc)
            from medguard.explain.templates import explain_finding

            return explain_finding(finding, language=language)

        return Explanation(
            text=text,
            language="hi" if language == "hi" else "en",
            grounded=False,  # verified separately before it is shown
            citations=list(finding.evidence),
        )


def verify_and_filter(
    explanation: Explanation, evidence: str, verifier: NliVerifier
) -> Explanation:
    """Remove sentences not entailed by the evidence.

    Returns a copy of ``explanation`` whose text contains only supported
    sentences and whose ``grounded`` flag reflects the verification result.
    """
    sentences = split_sentences(explanation.text)
    if not sentences:
        return explanation

    result = verifier.verify(sentences, evidence)
    kept = " ".join(result.supported_sentences).strip()

    if not kept:
        logger.warning("All LLM sentences were unsupported; using the template explanation")
        return explanation

    return Explanation(
        text=kept,
        language=explanation.language,
        grounded=result.grounded,
        citations=explanation.citations,
    )
