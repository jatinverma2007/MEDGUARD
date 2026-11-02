.PHONY: help install install-dev data splits test lint fmt typecheck api eval compare demo all clean

PY ?= python3

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime deps
	$(PY) -m pip install -e .

install-dev:  ## Install runtime + dev + ML deps
	$(PY) -m pip install -e ".[all]"

data:  ## Download public datasets into data/raw
	$(PY) scripts/download_data.py --all

splits:  ## Build leak-free train/val/test splits
	$(PY) scripts/make_splits.py --source drugbank --strategy grouped_by_drug

test:  ## Run the test suite
	$(PY) -m pytest

lint:  ## Lint with ruff
	$(PY) -m ruff check src tests scripts

fmt:  ## Auto-format with ruff
	$(PY) -m ruff format src tests scripts
	$(PY) -m ruff check --fix src tests scripts

typecheck:  ## Static type check with mypy
	$(PY) -m mypy

api:  ## Run the local API on :8000
	$(PY) -m uvicorn medguard.api:app --reload --port 8000

eval:  ## Reproduce the headline metrics table
	$(PY) -m medguard.cli eval --out artifacts/metrics.json

compare:  ## Quantify the leakage gap between random and leak-free splits
	$(PY) scripts/make_splits.py --compare

demo:  ## Run the pipeline on a sample prescription
	$(PY) -m medguard.cli analyze "Tab Ecosprin AV 75mg, Tab Warfarin 5mg, Tab Digoxin 0.25mg"

all: fmt lint typecheck test  ## Format, lint, typecheck, test
