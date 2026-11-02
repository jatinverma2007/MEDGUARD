"""MedGuard - prescription safety AI.

Pipeline stages
---------------
1. ``extract.ocr``      : prescription image -> raw text (VLM + Tesseract fallback)
2. ``extract.ner``      : raw text -> drug mentions (biomedical NER)
3. ``normalize.drugs``  : drug mentions -> canonical drug keys
4. ``interaction.*``    : canonical keys -> interaction findings + severity
5. ``evaluation.*``     : calibration, ECE, selective prediction
6. ``explain.*``        : grounded bilingual explanations

MedGuard is a *decision-support* tool. It is not a diagnostic device and must
not be used to make autonomous clinical decisions.
"""

__version__ = "0.1.0"

from medguard.config import Settings, get_settings
from medguard.schemas import (
    AnalysisResult,
    DrugMention,
    InteractionFinding,
    Severity,
)

__all__ = [
    "AnalysisResult",
    "DrugMention",
    "InteractionFinding",
    "Severity",
    "Settings",
    "get_settings",
    "__version__",
]
