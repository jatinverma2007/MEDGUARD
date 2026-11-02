"""Extraction package: OCR and drug NER."""

from medguard.extract.ner import (
    COMMON_GENERICS,
    BioBertNer,
    DrugExtractor,
    DrugSpan,
    LexiconDrugNer,
    UnionExtractor,
    build_extractor,
)
from medguard.extract.ocr import (
    OcrBackend,
    OcrResult,
    PassthroughOcr,
    QwenVLOcr,
    TesseractOcr,
    strip_dosage_instructions,
)

__all__ = [
    "BioBertNer",
    "COMMON_GENERICS",
    "DrugExtractor",
    "DrugSpan",
    "LexiconDrugNer",
    "OcrBackend",
    "OcrResult",
    "PassthroughOcr",
    "QwenVLOcr",
    "TesseractOcr",
    "UnionExtractor",
    "build_extractor",
    "strip_dosage_instructions",
]
