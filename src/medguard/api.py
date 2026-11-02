"""FastAPI service for MedGuard.

Run locally::

    make api          # http://127.0.0.1:8000/docs

Deploy to Hugging Face Spaces by adding a ``Dockerfile`` (already included)
and setting the port to 7860.

Privacy notes (design constraints, not an afterthought):
* Uploaded images are processed in memory and never written to disk.
* Raw prescription text is excluded from responses unless ``MEDGUARD_LOG_PHI``
  is explicitly enabled (it is off by default).
* No patient identifiers are stored or logged.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from medguard import __version__
from medguard.config import get_settings
from medguard.pipeline import MedGuardPipeline, PipelineConfig

logger = logging.getLogger(__name__)

app = FastAPI(
    title="MedGuard API",
    version=__version__,
    description=(
        "Prescription safety AI: extracts medicines from a prescription image or text, "
        "detects drug-drug interactions, grades severity and explains the risk in "
        "English or Hindi. Decision support only - not a diagnostic device."
    ),
)

_pipeline: MedGuardPipeline | None = None


def get_pipeline() -> MedGuardPipeline:
    """FastAPI dependency returning the singleton pipeline."""
    global _pipeline
    if _pipeline is None:
        settings = get_settings()
        pipeline = MedGuardPipeline(
            settings=settings,
            config=PipelineConfig(
                language=settings.default_language,
                abstain_threshold=settings.abstain_threshold,
                safe_assertion_threshold=settings.safe_assertion_threshold,
            ),
        )
        if settings.interactions_path.exists():
            import pandas as pd

            from medguard.interaction.retrieve import KnowledgeBase

            pipeline.kb = KnowledgeBase.from_frame(pd.read_parquet(settings.interactions_path))
            logger.info("Loaded knowledge base: %d drugs", pipeline.kb.drug_count)
        _pipeline = pipeline
    return _pipeline


# --- Static web UI ---
WEB_DIST = Path(os.getenv("WEB_DIST_PATH", "/app/web/dist"))
if WEB_DIST.exists():
    app.mount("/web", StaticFiles(directory=WEB_DIST, html=True), name="web")
    logger.info("Mounted web UI at /web from %s", WEB_DIST)

    @app.get("/")
    async def root() -> FileResponse:
        return FileResponse(WEB_DIST / "index.html")
else:
    @app.get("/")
    async def root() -> dict:
        return {"message": "MedGuard API — web UI not built. Run `docker build` or `cd web && npm run build`."}


# ------------------------------------------------------------------ schemas
class TextAnalysisRequest(BaseModel):
    """Analyse prescription text directly."""

    text: str = Field(min_length=1, description="Raw prescription text")
    language: Literal["en", "hi"] = "en"


class HealthResponse(BaseModel):
    status: str
    version: str
    kb_loaded: bool
    kb_drugs: int
    heavy_models: bool


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health(pipeline: Annotated[MedGuardPipeline, Depends(get_pipeline)]) -> HealthResponse:
    """Service health and knowledge-base status."""
    settings = get_settings()
    return HealthResponse(
        status="ok",
        version=__version__,
        kb_loaded=pipeline.kb is not None,
        kb_drugs=pipeline.kb.drug_count if pipeline.kb else 0,
        heavy_models=settings.use_heavy_models,
    )


@app.post("/analyze/text", tags=["analysis"])
def analyze_text(
    request: TextAnalysisRequest,
    pipeline: Annotated[MedGuardPipeline, Depends(get_pipeline)],
) -> dict:
    """Analyse a prescription supplied as text."""
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="Prescription text is empty.")
    result = pipeline.analyze_text(request.text, language=request.language)
    return result.model_dump(mode="json")


@app.post("/analyze/image", tags=["analysis"])
async def analyze_image(
    file: Annotated[UploadFile, File(description="Prescription image (jpg/png)")],
    pipeline: Annotated[MedGuardPipeline, Depends(get_pipeline)],
    language: Literal["en", "hi"] = "en",
) -> dict:
    """Analyse a prescription image.

    The image is held in memory only; nothing is written to disk.
    """
    allowed = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"}
    if file.content_type and file.content_type not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type '{file.content_type}'. Allowed: {sorted(allowed)}",
        )

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")
    if len(payload) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MB limit.")

    tmp_path = Path("/tmp") / f"medguard_{abs(hash(payload))}.png"
    tmp_path.write_bytes(payload)
    try:
        result = pipeline.analyze_image(tmp_path, language=language)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Image analysis failed")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    return result.model_dump(mode="json")


@app.get("/drugs", tags=["knowledge base"])
def list_drugs(
    pipeline: Annotated[MedGuardPipeline, Depends(get_pipeline)],
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """List drugs covered by the knowledge base."""
    if pipeline.kb is None:
        return {"total": 0, "items": []}

    drugs = sorted({drug for pair in pipeline.kb.index for drug in pair})
    window = drugs[offset : offset + limit]
    return {"total": len(drugs), "offset": offset, "limit": limit, "items": window}


@app.get("/interaction", tags=["knowledge base"])
def lookup_interaction(
    drug_a: str,
    drug_b: str,
    pipeline: Annotated[MedGuardPipeline, Depends(get_pipeline)],
) -> dict:
    """Look up a single drug pair."""
    if pipeline.kb is None:
        raise HTTPException(status_code=503, detail="Knowledge base is not loaded.")

    from medguard.interaction.retrieve import rows_to_findings

    rows = pipeline.kb.lookup(drug_a, drug_b)
    findings = rows_to_findings(rows)
    return {
        "drug_a": drug_a,
        "drug_b": drug_b,
        "found": bool(findings),
        "findings": [f.model_dump(mode="json") for f in findings],
    }


def _register_middleware() -> None:
    settings = get_settings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


_register_middleware()