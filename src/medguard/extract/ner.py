"""Drug mention extraction.

Two extractors:

``BioBertNer``
    Token classification with a biomedical encoder. Verified checkpoints:
    ``michiyasunaga/BioLinkBERT-base`` (Apache-2.0) and
    ``microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`` (MIT).
    Both are trained on PubMed and transfer well to clinical text.

``LexiconDrugNer``
    Dictionary matching against the alias table plus a curated list of common
    generic drugs. No GPU, no download, fully deterministic.

The lexicon extractor is not a throwaway - it is the **baseline** your
fine-tuned model has to beat, and it is fast enough to run on every request
as a sanity net around the neural extractor.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from medguard.normalize.drugs import ALIASES, parse_dose, strip_salt

logger = logging.getLogger(__name__)

#: Common generic drugs in Indian outpatient prescriptions. Expand this with
#: the drugs you see in your own collected prescriptions.
COMMON_GENERICS: tuple[str, ...] = (
    "metformin",
    "glimepiride",
    "glipizide",
    "insulin",
    "sitagliptin",
    "teneligliptin",
    "atorvastatin",
    "rosuvastatin",
    "simvastatin",
    "fenofibrate",
    "ezetimibe",
    "amlodipine",
    "telmisartan",
    "losartan",
    "valsartan",
    "ramipril",
    "lisinopril",
    "perindopril",
    "enalapril",
    "metoprolol",
    "atenolol",
    "bisoprolol",
    "carvedilol",
    "aspirin",
    "clopidogrel",
    "ticagrelor",
    "warfarin",
    "rivaroxaban",
    "apixaban",
    "dabigatran",
    "acenocoumarol",
    "atorvastatin",
    "digoxin",
    "levothyroxine",
    "amitriptyline",
    "sertraline",
    "escitalopram",
    "fluoxetine",
    "diazepam",
    "lorazepam",
    "alprazolam",
    "clonazepam",
    "gabapentin",
    "pregabalin",
    "levetiracetam",
    "carbamazepine",
    "phenytoin",
    "valproate",
    "topiramate",
    "paracetamol",
    "ibuprofen",
    "diclofenac",
    "naproxen",
    "aceclofenac",
    "aspirin",
    "meloxicam",
    "prednisolone",
    "dexamethasone",
    "methotrexate",
    "pantoprazole",
    "rabeprazole",
    "omeprazole",
    "esomeprazole",
    "ranitidine",
    "domperidone",
    "metoclopramide",
    "ondansetron",
    "loperamide",
    "loratadine",
    "cetirizine",
    "levocetirizine",
    "fexofenadine",
    "antiemetic",
    "montelukast",
    "salbutamol",
    "budesonide",
    "formoterol",
    "prednisone",
    "salbutamol",
    "ipratropium",
    "tiotropium",
    "amoxicillin",
    "amoxicillin clavulanate",
    "azithromycin",
    "ciprofloxacin",
    "levofloxacin",
    "cefixime",
    "cefuroxime",
    "ceftriaxone",
    "doxycycline",
    "clindamycin",
    "metronidazole",
    "nitrofurantoin",
    "fusidic acid",
    "albendazole",
    "ivermectin",
    "antihistamine",
    "montelukast",
    "hydroxychloroquine",
    "methotrexate",
    "azathioprine",
    "allopurinol",
    "colchicine",
    "febuxostat",
    "levocetirizine",
    "cetirizine",
    "iron",
    "ferrous sulfate",
    "folic acid",
    "vitamin b12",
    "cholecalciferol",
    "calcium carbonate",
    "calcium gluconate",
    "zinc",
    "multivitamin",
)

#: Tokens that are never drug names even though they look like them.
_STOP_WORDS: frozenset[str] = frozenset(
    {
        "tablet",
        "tablets",
        "tab",
        "capsule",
        "capsules",
        "cap",
        "syrup",
        "syp",
        "injection",
        "inj",
        "sachet",
        "cream",
        "ointment",
        "gel",
        "drops",
        "lotion",
        "patch",
        "suspension",
        "powder",
        "spray",
        "paint",
        "morning",
        "evening",
        "night",
        "daily",
        "after",
        "before",
        "food",
        "doctor",
        "patient",
        "mr",
        "mrs",
        "miss",
        "age",
        "years",
        "weight",
        "bp",
        "sugar",
        "prescription",
        "rx",
        "diagnosis",
        "advice",
        "mcg",
        "mg",
        "ml",
        "iu",
        "units",
        "dose",
        "doses",
        "date",
        "sign",
    }
)


@dataclass
class DrugSpan:
    """A raw drug mention with its character offsets in the source text."""

    surface: str
    start: int
    end: int
    score: float = 1.0

    @property
    def text(self) -> str:
        return self.surface


class DrugExtractor(ABC):
    """Interface for drug mention extractors."""

    name: str = "base"

    @abstractmethod
    def extract(self, text: str) -> list[DrugSpan]:
        """Return drug mentions found in ``text``."""

    def extract_with_dose(self, text: str) -> list[tuple[DrugSpan, str | None]]:
        """Return drug mentions paired with any parsed dose."""
        return [(span, parse_dose(span.surface)) for span in self.extract(text)]


class LexiconDrugNer(DrugExtractor):
    """Dictionary / longest-match extractor.

    Matches the longest alias first so ``amoxicillin clavulanate`` wins over
    ``amoxicillin``.
    """

    name = "lexicon"

    def __init__(self, extra_drugs: tuple[str, ...] = ()) -> None:
        self.lexicon: dict[str, str] = {}
        for surface, canonical in ALIASES.items():
            self.lexicon[surface] = canonical
        for generic in (*COMMON_GENERICS, *extra_drugs):
            self.lexicon[generic] = strip_salt(generic)
        # Longest first so multiword names win.
        self._ordered = sorted(self.lexicon.keys(), key=len, reverse=True)

    def extract(self, text: str) -> list[DrugSpan]:
        if not text:
            return []

        lowered = text.lower()
        spans: list[DrugSpan] = []
        claimed: list[tuple[int, int]] = []

        def overlaps(start: int, end: int) -> bool:
            return any(start < c_end and end > c_start for c_start, c_end in claimed)

        for alias in self._ordered:
            if alias in _STOP_WORDS:
                continue
            pattern = re.compile(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])")
            for match in pattern.finditer(lowered):
                start, end = match.span()
                if overlaps(start, end):
                    continue
                # A match followed by a dose is still a drug mention.
                trailing = text[end : end + 24]
                dose_match = re.match(r"\s*(\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|iu))", trailing, re.I)
                if dose_match:
                    end = end + dose_match.end()
                claimed.append((start, end))
                spans.append(DrugSpan(surface=text[start:end], start=start, end=end, score=1.0))

        spans.sort(key=lambda s: s.start)
        return spans


class BioBertNer(DrugExtractor):
    """Biomedical token-classification NER.

    Requires ``pip install 'medguard[ml]'``.
    """

    name = "biobert"

    def __init__(
        self,
        model_id: str = "michiyasunaga/BioLinkBERT-base",
        device: str | None = None,
        max_length: int = 512,
        confidence_threshold: float = 0.5,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.max_length = max_length
        self.confidence_threshold = confidence_threshold
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import (
                AutoModelForTokenClassification,
                AutoTokenizer,
            )
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "transformers is required for BioBertNer. Run: pip install 'medguard[ml]'"
            ) from exc

        logger.info("Loading NER model %s ...", self.model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForTokenClassification.from_pretrained(self.model_id)
        if self.device:
            self._model.to(self.device)
        self._model.eval()

    def extract(self, text: str) -> list[DrugSpan]:
        self._load()
        import torch

        if not text.strip():
            return []

        encoded = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        encoded = {k: v.to(self._model.device) for k, v in encoded.items()}

        with torch.no_grad():
            logits = self._model(**encoded).logits

        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        label_ids = probs.argmax(axis=-1)
        scores = probs.max(axis=-1)

        id2label = self._model.config.id2label
        spans: list[DrugSpan] = []
        current: tuple[int, int, float] | None = None

        for idx, (start, end) in enumerate(offsets):
            if start == end:  # special token
                continue
            label = str(id2label.get(int(label_ids[idx]), "O"))
            score = float(scores[idx])

            if "B" in label and label != "O":
                if current:
                    spans.append(self._finish(current, text))
                current = (start, end, score)
            elif "I" in label and label != "O" and current:
                current = (current[0], end, max(current[2], score))
            else:
                if current:
                    spans.append(self._finish(current, text))
                current = None

        if current:
            spans.append(self._finish(current, text))

        return [
            s
            for s in spans
            if s.score >= self.confidence_threshold and s.surface.strip().lower() not in _STOP_WORDS
        ]

    @staticmethod
    def _finish(current: tuple[int, int, float], text: str) -> DrugSpan:
        start, end, score = current
        return DrugSpan(surface=text[start:end], start=start, end=end, score=score)


class UnionExtractor(DrugExtractor):
    """Union of two extractors with span de-duplication.

    Neural mentions win over lexicon mentions on overlap, because the neural
    model handles context the lexicon cannot (e.g. "aspirin" inside
    "aspirin-induced asthma").
    """

    name = "union"

    def __init__(self, primary: DrugExtractor, secondary: DrugExtractor) -> None:
        self.primary = primary
        self.secondary = secondary

    def extract(self, text: str) -> list[DrugSpan]:
        primary = self.primary.extract(text)
        secondary = self.secondary.extract(text)
        kept: list[DrugSpan] = list(primary)

        for span in secondary:
            if any(span.start < k.end and span.end > k.start for k in kept):
                continue
            kept.append(span)

        kept.sort(key=lambda s: s.start)
        return kept


def build_extractor(
    use_heavy_models: bool = False,
    model_id: str = "michiyasunaga/BioLinkBERT-base",
    device: str | None = None,
) -> DrugExtractor:
    """Factory: neural extractor when available, lexicon otherwise."""
    lexicon = LexiconDrugNer()
    if not use_heavy_models:
        return lexicon
    try:
        neural = BioBertNer(model_id=model_id, device=device)
        return UnionExtractor(neural, lexicon)
    except Exception as exc:  # pragma: no cover - fallback path
        logger.warning("Falling back to lexicon NER: %s", exc)
        return lexicon
