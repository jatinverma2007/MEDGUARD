"""Prescription OCR.

Two backends:

``QwenVLOcr``
    ``Qwen/Qwen2.5-VL-3B-Instruct``. The model card reports **DocVQA 93.9**
    and **InfoVQA 77.5**, which makes it the strongest small open VLM for
    reading prescriptions, forms and handwritten tables. Fits a Colab T4
    in 4-bit or bf16.

``TesseractOcr``
    Classic OCR via ``pytesseract``. No GPU, no download, and surprisingly
    good on clean printed prescriptions - it is the right baseline, and
    beating a Tesseract baseline is a legitimate result.

Both return a :class:`OcrResult` with the raw text plus the per-token
confidence when the backend can provide it.
"""

from __future__ import annotations

import importlib.util
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

#: Prompt tuned for prescriptions: asks for layout-preserving plain text.
PRESCRIPTION_PROMPT = (
    "Transcribe this prescription image verbatim. Preserve the line order and the "
    "drug names exactly as written. Output only the transcribed text, with no "
    "commentary, no markdown and no added interpretation. If a line is illegible, "
    "write [ILLEGIBLE]."
)


@dataclass
class OcrResult:
    """Output of an OCR backend."""

    text: str
    confidence: float | None = None
    backend: str = "unknown"
    lines: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.lines and self.text:
            self.lines = [ln for ln in self.text.splitlines() if ln.strip()]


class OcrBackend(ABC):
    """Interface implemented by every OCR backend."""

    name: str = "base"

    @abstractmethod
    def read(self, image_path: str | Path) -> OcrResult:
        """Transcribe an image file."""

    def read_text(self, text: str) -> OcrResult:
        """Pass-through for already-extracted text (used by the API/tests)."""
        return OcrResult(text=text, confidence=1.0, backend=f"{self.name}:passthrough")


class TesseractOcr(OcrBackend):
    """CPU OCR using Tesseract. Requires the ``tesseract`` binary."""

    name = "tesseract"

    def __init__(self, lang: str = "eng+hin", psm: int = 6) -> None:
        self.lang = lang
        self.psm = psm

    def read(self, image_path: str | Path) -> OcrResult:
        try:
            import pytesseract
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "pytesseract + Pillow are required for TesseractOcr. "
                "Run: pip install 'medguard[ocr]'"
            ) from exc

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        with Image.open(path) as img:
            text = pytesseract.image_to_string(img, lang=self.lang, config=f"--psm {self.psm}")

        words = self._mean_word_confidence(path)
        return OcrResult(text=text.strip(), confidence=words, backend=self.name)

    @staticmethod
    def _mean_word_confidence(image_path: Path) -> float | None:
        try:
            import pytesseract
            from PIL import Image

            with Image.open(image_path) as img:
                data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            confidences = [
                float(c)
                for c in data.get("conf", [])
                if str(c).lstrip("-").isdigit() and float(c) >= 0
            ]
            return sum(confidences) / len(confidences) / 100.0 if confidences else None
        except Exception:  # pragma: no cover - best effort
            return None


class QwenVLOcr(OcrBackend):
    """Vision-language OCR using Qwen2.5-VL-3B-Instruct.

    Requires ``pip install 'medguard[ml]'`` and a GPU with >= 12GB VRAM
    (or 4-bit quantisation for 8GB).
    """

    name = "qwen2.5-vl"

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: str | None = None,
        max_new_tokens: int = 512,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1280 * 28 * 28,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "transformers + torch are required for QwenVLOcr. Run: pip install 'medguard[ml]'"
            ) from exc

        if importlib.util.find_spec("torch") is None:  # pragma: no cover
            raise ImportError("torch is required for QwenVLOcr. Run: pip install 'medguard[ml]'")

        logger.info("Loading %s ...", self.model_id)
        self._processor = AutoProcessor.from_pretrained(
            self.model_id,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map=self.device or "auto",
        )

    def read(self, image_path: str | Path) -> OcrResult:
        self._load()
        from PIL import Image

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        with Image.open(path) as img:
            image = img.convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PRESCRIPTION_PROMPT},
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self._processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        generated = self._model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
        )
        trimmed = [
            out[len(inp) :] for inp, out in zip(inputs["input_ids"], generated, strict=False)
        ]
        transcription = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return OcrResult(text=transcription.strip(), backend=self.name)


class PassthroughOcr(OcrBackend):
    """No-op backend used when the caller already has text.

    Keeps the pipeline testable without any model downloads.
    """

    name = "passthrough"

    def read(self, image_path: str | Path) -> OcrResult:  # pragma: no cover - trivial
        raise ValueError("PassthroughOcr cannot read images; pass text directly instead.")


def strip_dosage_instructions(text: str) -> str:
    """Remove dosing instructions that are not part of the drug name.

    Kept separate from :func:`medguard.normalize.drugs.normalize` because
    the OCR output needs cleaning *before* NER, not after.
    """
    cleaned = re.sub(r"\b(OD|BD|TDS|QID|HS|STAT|PRN|SOS|ROA|PO|IV|IM)\b", " ", text)
    cleaned = re.sub(r"\b\d+\s*(days?|weeks?|months?)\b.*$", " ", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip()
