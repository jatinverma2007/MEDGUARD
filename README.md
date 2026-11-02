<div align="center">

# MedGuard

### Prescription safety AI — extracts medicines, detects drug–drug interactions, grades severity, and explains the risk in English and Hindi.

[![CI](https://github.com/YOUR_USERNAME/medguard/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/medguard/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Decision-support tool. Not a diagnostic device.** MedGuard has not been
> clinically validated and must not be used to make autonomous clinical
> decisions.

</div>

---

## The problem

Polypharmacy is the norm in Indian outpatient practice, and a patient on five
medicines presents **ten possible drug pairs** to the prescriber. Interaction
checking is manual, time-consuming, and inconsistently done — especially
outside large tertiary centres.

MedGuard turns a prescription photograph into a **graded, cited, bilingual
risk list** in under two seconds.

<!-- Replace with your own screenshot once the UI is built -->
<p align="center">
  <em>Demo screenshot goes here</em>
</p>

---

## What makes this a research project, not a notebook

Three things separate MedGuard from a typical student project:

### 1. Leak-free evaluation (the big one)

Random train/test splits on drug-interaction data **inflate F1 by 10–20
points**, because the same drug pair appears on both sides. A model that has
memorised *"warfarin + aspirin → bleeding"* looks brilliant and generalises
to nothing.

MedGuard ships three splitting strategies and reports the gap:

| Strategy | Guarantee | Models generalise to |
|---|---|---|
| `random` | none (baseline only) | nothing — it measures memorisation |
| `pairwise` | no drug *pair* crosses a split | new **combinations** of known drugs |
| `cold_start` | no drug at all crosses a split | **entirely unseen drugs** |

`medguard.data.splits` implements these with connected-component grouping so
a drug's whole interaction neighbourhood stays in one split.

```python
from medguard.data.splits import make_splits, SplitStrategy

train, val, test, report = make_splits(
    df, strategy=SplitStrategy.COLD_START, seed=42
)
print(report.summary())
# [cold_start] train=133865 val=28685 test=28758 | leaked_pairs=0 leaked_drugs=0
```

> **⚠️ A real constraint you will hit:** `cold_start` is impossible on a
> **densely connected** interaction corpus. TWOSIDES has 4.6M pairs over 645
> drugs — nearly a complete graph — so the whole corpus is one connected
> component and cannot be divided without leaking drugs.
>
> MedGuard detects this and warns loudly rather than silently producing an
> empty test set. Your options are:
>
> 1. Use `pairwise` (recommended — it is the honest, publishable setting).
> 2. Restrict the corpus to a **single therapeutic class** (e.g. cardiovascular
>    drugs only), which produces a sparse graph where `cold_start` works.
> 3. Report `pairwise` as the headline and `cold_start` on the restricted
>    subset as a secondary, harder evaluation.
>
> Option 2 is the strongest thing you can do: *"I restricted the corpus to one
> therapeutic class to construct a true cold-start evaluation"* is a
> sophisticated methodological statement.

### 2. Calibration and abstention

A medical model that says *"no interaction"* with 90% confidence but is wrong
40% of the time is dangerous. MedGuard treats confidence as a first-class
output:

- **Temperature scaling** fitted on validation (never on test).
- **Expected Calibration Error (ECE)** and reliability diagrams reported.
- **Selective prediction** — the model abstains below a threshold instead of
  guessing, with a published risk–coverage curve.

### 3. Grounded explanations with faithfulness verification

The default explanation layer is **templated from retrieved evidence**, so it
cannot hallucinate. The optional LLM layer (free tiers: Groq, Google AI
Studio) writes richer text, and every sentence is then checked with an **NLI
entailment model** — unsupported sentences are removed before display.

The result is a measurable faithfulness number rather than a vibe.

---

## Architecture

```
Prescription image / text
        │
        ▼
┌──────────────────────┐
│ 1. OCR               │  Qwen2.5-VL-3B  (DocVQA 93.9)  |  Tesseract fallback
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 2. Drug NER          │  BioLinkBERT / BiomedBERT  |  lexicon fallback
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 3. Normalisation     │  brand → generic, salt stripping, dose parsing
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 4. Pair retrieval    │  exact KB lookup (DDInter / TWOSIDES / ADE)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 5. Model fallback    │  DeBERTa-v3 classifier for pairs not in the KB
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 6. Severity + calib  │  deterministic grading, temperature scaling
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ 7. Explanation       │  grounded templates (en/hi) + optional LLM + NLI check
└──────────┬───────────┘
           ▼
        JSON + citations + abstention flag
```

---

## Results

<!-- Replace every number below with your own measurements. -->

| Model | Split | F1 | Macro F1 | ECE | p95 latency |
|---|---|---|---|---|---|
| Co-occurrence baseline | `random` | — | — | — | — |
| TF-IDF + LogReg | `random` | — | — | — | — |
| TF-IDF + LogReg | `pairwise` | — | — | — | — |
| DeBERTa-v3 (ours) | `pairwise` | — | — | — | — |
| DeBERTa-v3 + calibration | `cold_start` | — | — | — | — |

**Leakage impact** (the headline result):

| Metric | Naive `random` split | Leak-free `cold_start` | Gap |
|---|---|---|---|
| F1 | — | — | **−X.X pts** |

---

## Datasets

All sources are public and free to access.

| Dataset | Content | Licence | Role |
|---|---|---|---|
| [TWOSIDES (TDC)](https://tdcommons.ai/multi_pred_tasks/ddi/) | 4,649,441 polypharmacy pairs, 645 drugs | **CC BY 4.0** | Primary training corpus |
| [DrugBank DDI (TDC)](https://tdcommons.ai/multi_pred_tasks/ddi/) | 191,808 pairs, 1,706 drugs, 86 relation types | Academic licence | Typed relation training |
| [DDInter (HF)](https://huggingface.co/datasets/Tassy24/K-Paths-inductive-reasoning-ddinter) | Severity-labelled Major/Moderate/Minor | Apache-2.0 | Severity grounding |
| [ADE-Corpus-V2](https://huggingface.co/datasets/ade-benchmark-corpus/ade_corpus_v2) | ~18,520 drug–AE relations | Public | Adverse-event detection |
| [MedDialog](https://huggingface.co/datasets/bigbio/meddialog) | 0.26M doctor–patient dialogues | Public | Explanation style, faithfulness refs |
| [MedQA / PubMedQA](https://huggingface.co/datasets) | Medical reasoning, factuality | Public | Evaluation suite |

### Models

| Model | Role | Licence |
|---|---|---|
| `Qwen/Qwen2.5-VL-3B-Instruct` | Prescription OCR (DocVQA 93.9) | Apache-2.0 |
| `michiyasunaga/BioLinkBERT-base` | Biomedical NER | Apache-2.0 |
| `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` | Biomedical NER (alt) | MIT |
| `microsoft/deberta-v3-base` | Interaction relation classifier | MIT |
| `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | Faithfulness (NLI) | Apache-2.0 |

Everything fits on a **free Google Colab T4 (16 GB)**.

---

## Quick start

```bash
git clone https://github.com/YOUR_USERNAME/medguard.git
cd medguard

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # or: pip install -e ".[all]" for the ML extras

# A small sample knowledge base is committed, so this works right now:
medguard analyze "Tab Ecosprin AV 75mg, Tab Warfarin 5mg, Tab Digoxin 0.25mg"

# Download the full open datasets (~1 GB, 5-10 min)
python scripts/download_data.py --all

# Build leak-free splits
python scripts/make_splits.py --strategy pairwise

# See the leakage gap (this is your README figure)
python scripts/make_splits.py --compare

# Analyse a prescription
medguard analyze "Tab Aspirin 75mg, Tab Warfarin 5mg once daily"
medguard analyze --language hi "Tab Metformin 500mg, Tab Gluconil 5mg"

# Run the API
make api                         # http://127.0.0.1:8000/docs
```

### Docker

```bash
docker compose up --build        # http://localhost:8000/docs
```

### API

```bash
curl -X POST http://localhost:8000/analyze/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Tab Aspirin 75mg and Tab Warfarin 5mg", "language": "en"}'
```

---

## Project layout

```
medguard/
├── src/medguard/
│   ├── config.py          # settings (env-driven)
│   ├── schemas.py         # pydantic domain models
│   ├── pipeline.py        # end-to-end orchestration
│   ├── api.py             # FastAPI service
│   ├── cli.py             # typer CLI
│   ├── data/              # loaders, registry, leak-free splits
│   ├── extract/           # OCR + drug NER
│   ├── normalize/         # brand → generic normalisation
│   ├── interaction/       # KB retrieval + severity grading
│   ├── modeling/          # baselines + calibrated transformer classifier
│   ├── evaluation/        # metrics, ECE, temperature scaling, risk-coverage
│   └── explain/           # grounded bilingual templates + LLM + NLI verifier
├── scripts/               # data download, split building
├── notebooks/             # experiments (Colab-ready)
├── tests/                 # 56 tests, CPU-only, ~1.8 s
├── docs/                  # evaluation report, experiment log, interview prep
└── .github/workflows/     # CI: lint, typecheck, test, docker build
```

---

## Responsible AI

This project is designed around medical-AI constraints, not despite them:

- **Abstention over guessing.** Below the calibrated threshold the system
  says "ask a clinician" rather than asserting safety.
- **Explainability by construction.** Every finding carries a `source_id`
  pointing back to the exact knowledge-base record.
- **Data minimisation.** Uploaded images are processed in memory and deleted;
  raw text is excluded from responses by default (`MEDGUARD_LOG_PHI=false`).
- **Regulatory awareness.** The README states the decision-support boundary,
  and the evaluation report discusses where a CDSCO medical-device
  classification would begin.
- **Honest reporting.** Leakage is measured and published rather than hidden.

---

## Roadmap

- [ ] Prescription-image test set collected from consenting volunteers
- [ ] Practitioner reader study (3 pharmacists, 50 prescriptions)
- [ ] Fine-tuned Qwen2.5-VL for prescription OCR vs. the prompted baseline
- [ ] Counterfactual explanations: *"replacing drug A with B lowers severity"*
- [ ] Hindi-native evaluation set (not translations)
- [ ] Prospective pilot with a partner clinic

---

## Contributing

Issues and PRs welcome. See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for
the experiment-logging convention — please log every run you do.

## Licence

MIT — see [LICENSE](LICENSE). The **code** is MIT; the **datasets** retain
their own licences (notably DrugBank, which requires an academic licence).
