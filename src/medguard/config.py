"""Application configuration.

Everything is overridable via environment variables prefixed ``MEDGUARD_``
or via a local ``.env`` file. See ``.env.example``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Language = Literal["en", "hi"]

#: Canonical Hugging Face model identifiers.
#:
#: Verified on Hugging Face:
#:   * ``Qwen/Qwen2.5-VL-3B-Instruct``                 - OCR / layout understanding
#:   * ``michiyasunaga/BioLinkBERT-base``               - biomedical NER (Apache-2.0)
#:   * ``microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`` - biomedical NER (MIT)
#:   * ``microsoft/deberta-v3-base``                     - relation / severity classifier
DEFAULT_OCR_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_NER_MODEL = "michiyasunaga/BioLinkBERT-base"
DEFAULT_RELATION_MODEL = "microsoft/deberta-v3-base"


class Settings(BaseSettings):
    """Runtime settings for the MedGuard pipeline."""

    model_config = SettingsConfigDict(
        env_prefix="MEDGUARD_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Paths ---
    project_root: Path = Field(default=Path(__file__).resolve().parents[2])
    data_dir: Path = Field(default=Path("data"))
    artifacts_dir: Path = Field(default=Path("artifacts"))

    # --- Models ---
    ocr_model: str = DEFAULT_OCR_MODEL
    ner_model: str = DEFAULT_NER_MODEL
    relation_model: str = DEFAULT_RELATION_MODEL

    #: When False (default) the pipeline runs fully on CPU with heuristic
    #: extractors, which keeps CI and local development fast.
    use_heavy_models: bool = False

    # --- LLM explanation layer (optional, free tiers work) ---
    llm_api_key: str | None = None
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "llama-3.3-70b-versatile"

    # --- Decision thresholds ---
    #: Below this calibrated probability the system abstains and defers to a
    #: clinician instead of asserting "safe" / "no interaction".
    abstain_threshold: float = Field(default=0.55, ge=0.0, le=1.0)

    #: Minimum confidence to assert that a pair is safe (absence of finding).
    safe_assertion_threshold: float = Field(default=0.80, ge=0.0, le=1.0)

    # --- Behaviour ---
    default_language: Language = "en"
    #: Logging raw prescription text is PHI. Keep False.
    log_phi: bool = False

    # --- API ---
    app_title: str = "MedGuard API"
    app_version: str = "0.1.0"
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    @property
    def raw_data_dir(self) -> Path:
        return self.project_root / self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.project_root / self.data_dir / "processed"

    @property
    def artifacts_path(self) -> Path:
        return self.project_root / self.artifacts_dir

    @property
    def interactions_path(self) -> Path:
        """Path to the canonical interaction table.

        Prefers the full downloaded corpus at
        ``data/processed/interactions.parquet`` and falls back to the small
        committed sample at ``data/processed/demo_interactions.parquet`` so
        the project runs out of the box after cloning.
        """
        primary = self.processed_data_dir / "interactions.parquet"
        if primary.exists():
            return primary
        return self.processed_data_dir / "demo_interactions.parquet"

    def ensure_dirs(self) -> None:
        """Create the directories MedGuard writes into."""
        for path in (self.raw_data_dir, self.processed_data_dir, self.artifacts_path):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor (use as a FastAPI dependency)."""
    return Settings()
