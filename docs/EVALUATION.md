# Evaluation report — template

> Fill this in as you go. Aim for 5–8 pages, ACM/IEEE two-column format if
> your college expects a report. Keep every number reproducible with a single
> command.

**How to cite this in an interview:** *"I wrote a 7-page evaluation report
with three split strategies, a calibration study and an error taxonomy."*

---

## 1. Problem statement and scope

- **Who has the problem?** Patients on 3+ medicines, their families,
  community pharmacists, and primary-care prescribers.
- **What is the current workaround?** Manual cross-checking against a
  formulary or a mobile app; inconsistent, time-consuming, error-prone.
- **What exactly does MedGuard do?** Reads a prescription, identifies the
  medicines, checks every pair against an interaction knowledge base,
  grades severity, and explains the risk in the patient's language.
- **Explicitly out of scope:** dose adjustment recommendations, diagnosis,
  replacement therapy selection, contraindication checking against patient
  conditions (no patient records are available).
- **Success criteria (set before you built it):**
  - NER F1 ≥ 90 on a held-out prescription set.
  - Interaction detection F1 ≥ 0.80 on the leak-free `pairwise` split.
  - ECE ≤ 0.05 after calibration.
  - p95 end-to-end latency ≤ 3 s on CPU.

---

## 2. Data

### 2.1 Sources

| Dataset | Licence | Rows | Used for |
|---|---|---|---|
| TWOSIDES (TDC) | CC BY 4.0 | 4,649,441 pairs | Training corpus |
| DDInter (HF subset) | Apache-2.0 | — | Severity grounding |
| ADE-Corpus-V2 | Public | 18,520 relations | ADE detection |
| Own prescription images | Consent obtained | — | OCR + NER evaluation |

### 2.2 Splits

| Strategy | Train | Val | Test | Leaked pairs | Leaked drugs |
|---|---|---|---|---|---|
| `random` | — | — | — | — | — |
| `pairwise` | — | — | — | 0 | — |
| `cold_start` | — | — | — | 0 | 0 |

### 2.3 The leakage finding (put this early — it is your headline)

> "A naive random split reports F1 = X. Under a leak-free `cold_start`
> split, the same model scores F1 = Y. The X − Y gap is the portion of the
> headline number that came from memorising drug pairs rather than learning
> interaction structure."

<!-- Paste the output of: python scripts/make_splits.py --compare -->

---

## 3. Models

| Stage | Model | Params | Fits on Colab T4 |
|---|---|---|---|
| OCR | Qwen2.5-VL-3B-Instruct | 3B | ✅ (bf16) |
| NER | BioLinkBERT-base | 110M | ✅ |
| Relation | DeBERTa-v3-base | 86M + head | ✅ |
| Calibration | Temperature scaling | 1 param | CPU |
| Explanation | Templates / LLM + NLI | — | ✅ |

**Baselines (mandatory):**

1. Majority class.
2. Co-occurrence frequency.
3. TF-IDF + logistic regression.
4. Random-split version of your own model (to quantify leakage).

---

## 4. Results

### 4.1 Main table

| Model | Split | Accuracy | Precision | Recall | F1 | Macro F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|
| Majority class | `pairwise` | — | — | — | — | — | — |
| Co-occurrence | `pairwise` | — | — | — | — | — | — |
| TF-IDF + LogReg | `pairwise` | — | — | — | — | — | — |
| DeBERTa-v3 | `pairwise` | — | — | — | — | — | — |
| DeBERTa-v3 | `cold_start` | — | — | — | — | — | — |

### 4.2 Calibration

| Model | ECE before | ECE after | Brier | Method | T |
|---|---|---|---|---|---|
| DeBERTa-v3 | — | — | — | Temperature | — |

Include:
- A reliability diagram (matplotlib).
- A comparison of temperature vs. Platt vs. isotonic
  (`medguard.evaluation.calibration.compare_calibrators`).

### 4.3 Selective prediction

Include the risk–coverage curve and state the operating point:

> "At a 0.85 confidence threshold MedGuard answers 62% of prescriptions and
> its error rate on those falls from 14.1% to 3.8%. The remaining 38% are
> escalated to a clinician."

### 4.4 Cross-dataset generalisation

| Train | Test | F1 | Drop |
|---|---|---|---|
| TWOSIDES | DDInter | — | — |
| DDI-2010 | TWOSIDES | — | — |

### 4.5 Ablations

| Removed component | F1 | Δ | ECE |
|---|---|---|---|
| Full model | — | — | — |
| − calibration | — | — | — |
| − normalisation (brand→generic) | — | — | — |
| − lexicon fallback | — | — | — |
| — | — | — | — |

---

## 5. Error analysis

Bucket every error in the test set. This section is what separates a report
from a report.

| Error type | Count | % | Example | Fix |
|---|---|---|---|---|
| OCR misread drug name | — | — | "Ecosprin" → "Ecosprn" | better OCR prompt / preprocessing |
| Drug not in alias table | — | — | "Telma" missed | expand alias table |
| Wrong normalisation | — | — | "Pan 40" → pantoprazole ✓ | — |
| KB coverage gap | — | — | pair absent from corpus | add corpus / model fallback |
| Genuinely ambiguous interaction | — | — | context-dependent severity | needs patient context |
| Severity over-grading | — | — | minor flagged as major | recalibrate severity mapping |

**Discussion:** which errors are *reducible* by better modelling, and which
require information MedGuard cannot have (e.g. renal function, current
medication list)? Be explicit — this shows research maturity.

---

## 6. Robustness

- Typos: "Warfrin", "Aspirin 75m", "Metfromin".
- Unusual formats: vertical handwriting, multi-column layouts, staple marks.
- Alias swaps: brand ↔ generic for the same drug.
- Out-of-distribution drugs not in the KB.

---

## 7. Human evaluation (the part that makes it real)

If you interviewed practitioners, report it here.

| Participant | Role | Years | Task | Agreement with MedGuard | Notes |
|---|---|---|---|---|---|
| — | — | — | — | — | — |

Include direct anonymised quotes. "One pharmacist said severity labelling,
not detection, was their real bottleneck — so I rebuilt the severity module
around that." That sentence is worth more than any metric.

---

## 8. Limitations

State at least five, honestly.

1. Public interaction corpora are biased toward US/European formularies and
   under-represent many Indian generics.
2. No patient context: renal function, hepatic function, allergies, pregnancy,
   and current medication list are all unavailable, so severity is a
   population-level estimate.
3. Prescription images in the wild are messier than any public benchmark.
4. The explanation templates are not yet validated for patient
   understanding.
5. No prospective clinical validation has been performed.
6. DDInter severity labels are coarse; the mapping to fine-grained clinical
   severity is heuristic.

---

## 9. Future work

1. Condition-aware severity (needs a patient record → consent + DPDP Act
   compliance).
2. Hindi-native evaluation set built with practitioners, not translated.
3. Counterfactual severity explanations.
4. Prospective pilot with a partner pharmacy.
5. Regulatory pathway assessment if a dosing recommendation is ever added.

---

## Appendix A — Reproduction

```bash
make install-dev
python scripts/download_data.py --all
python scripts/make_splits.py --strategy cold_start
python scripts/make_splits.py --compare
medguard eval --table data/processed/interactions.parquet --out artifacts/metrics.json
pytest --cov=medguard
```

Exact package versions are pinned in `pyproject.toml`; the exact commit hash
that produced the numbers above is:

```
commit: <paste git rev-parse HEAD>
date:   <paste date>
```
