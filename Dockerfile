FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-hin \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install ".[ocr]"

COPY scripts ./scripts
COPY tests ./tests

RUN useradd --create-home --uid 1000 medguard && chown -R medguard:medguard /app
USER medguard

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:7860/health || exit 1

CMD ["python", "-m", "uvicorn", "medguard.api:app", "--host", "0.0.0.0", "--port", "7860"]


# --- Web UI build stage ---
FROM node:20-alpine AS web-builder

WORKDIR /web
COPY web/package.json web/package-lock.json* ./
RUN npm ci

COPY web/ ./
RUN npm run build


# --- Final image with API + static web UI ---
FROM base AS final

USER root
COPY --from=web-builder /web/dist /app/web/dist
RUN chown -R medguard:medguard /app/web/dist

# FastAPI will serve the static files at /web and the API at /api
ENV WEB_DIST_PATH=/app/web/dist

USER medguard
