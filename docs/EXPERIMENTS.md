# Experiment log

Every experiment gets a row. This is the single highest-leverage habit for an
AI portfolio — it proves you are systematic, and it takes five minutes per run.

Use `make eval` to reproduce any row.

---

## Template

```markdown
### EXP-XXX — <short title>

- **Date:** YYYY-MM-DD
- **Branch/commit:** `<git rev-parse --short HEAD>`
- **Goal:** What question does this answer?
- **Hypothesis:** What do I expect, and why?
- **Setup:** Model, data split, seed, hyperparameters, compute (GPU hours)
- **Result:** The numbers (paste actual output, not a summary)
- **Conclusion:** Was the hypothesis confirmed?
- **Next:** What to try next
```

---

## Log

### EXP-001 — Baseline: majority class on TWOSIDES

- **Date:** _(fill in)_
- **Goal:** Establish the floor. Any model below this is worthless.
- **Hypothesis:** Majority-class F1 will be low (~0.0–0.3) because the
  positive class dominates.
- **Setup:** `random` split, seed 42.
- **Result:** _(paste)_
- **Conclusion:** _(fill in)_

### EXP-002 — Baseline: TF-IDF + logistic regression

- **Goal:** Does drug-name morphology alone predict interactions?
- **Hypothesis:** Yes, surprisingly well — name families (`-pril`, `-sartan`,
  `-statin`) encode pharmacokinetic classes.
- **Setup:** `pairwise` split, seed 42, `sublinear_tf=True`, balanced class
  weights.
- **Result:** _(paste)_
- **Conclusion:** _(fill in)_

### EXP-003 — **The leakage experiment** (do this early)

- **Goal:** Quantify how much a random split inflates results.
- **Hypothesis:** F1 drops by 10–20 points from `random` to `cold_start`.
- **Setup:** Identical model, identical seed, three split strategies.
- **Result:**

  | Split | F1 | Macro F1 |
  |---|---|---|
  | `random` | | |
  | `pairwise` | | |
  | `cold_start` | | |

- **Conclusion:** _(fill in)_
- **Why it matters:** This is the first slide of your presentation and the
  first thing you say in the interview.

> **Expect a small gap on synthetic data.** If you test with made-up drug
> names (`drug_001`, `drug_002`), the leakage gap will be near zero — because
> a model cannot memorise a pair it has never seen twice. The gap only
> appears with **real drug names**, where the same drugs recur across many
> pairs and the model can learn "warfarin pairs are dangerous" rather than
> "warfarin + aspirin is dangerous". If your first run shows no gap, check
> that you are using the real corpus before concluding there is no leakage.
>
> This is also why `pairwise` is the right headline setting, and why
> `cold_start` needs a sparse subset (one therapeutic class) to be
> measurable at all.

### EXP-004 — Fine-tune DeBERTa-v3 on the leak-free split

- **Goal:** Beat the TF-IDF baseline honestly.
- **Hypothesis:** +5–15 F1 over TF-IDF on `pairwise`.
- **Setup:** `microsoft/deberta-v3-base`, lr 2e-5, 3 epochs, batch 32, bf16,
  Colab T4, seed 42.
- **Result:** _(paste)_
- **Conclusion:** _(fill in)_

### EXP-005 — Calibration: temperature scaling

- **Goal:** Make the probabilities trustworthy.
- **Hypothesis:** ECE drops below 0.05 without changing AUC (temperature
  scaling is monotone, so ranking is preserved).
- **Setup:** Fit on validation logits, evaluate on test.
- **Result:** ECE before → after; T = _(value)_
- **Conclusion:** _(fill in)_
- **Sanity check:** confirm AUC is unchanged. If it changed, you have a bug.

### EXP-006 — Selective prediction / risk–coverage

- **Goal:** Find the operating point where the model is safe to answer.
- **Hypothesis:** At threshold 0.85, error rate on answered cases is < 5%.
- **Result:** _(paste the curve)_

### EXP-007 — Ablation: remove normalisation

- **Goal:** Measure how much brand→generic resolution matters.
- **Hypothesis:** F1 drops sharply because brand names miss the KB.
- **Result:** _(paste)_

### EXP-008 — Cross-dataset generalisation

- **Goal:** Test on a corpus the model never saw.
- **Setup:** Train on TWOSIDES, test on DDInter (severity-labelled).
- **Result:** _(paste)_
- **Conclusion:** _(fill in)_

### EXP-009 — OCR: Qwen2.5-VL vs. Tesseract

- **Goal:** Quantify the OCR contribution.
- **Setup:** 50 self-collected prescription images; measure character error
  rate and downstream drug-name F1.
- **Result:** _(paste)_

### EXP-010 — Hallucination audit of the LLM explanation layer

- **Goal:** Show that the NLI filter removes unsupported claims.
- **Setup:** 200 findings, compare the raw LLM output against the
  NLI-verified output. Report the grounding ratio before and after.
- **Result:** grounding ratio raw → verified: _(values)_
- **Conclusion:** _(fill in)_

---

## Ablation summary table (paste into the README)

| Configuration | F1 | ECE | p95 latency | Notes |
|---|---|---|---|---|
| Majority class | | | | |
| TF-IDF + LogReg | | | | |
| DeBERTa-v3 | | | | |
| DeBERTa-v3 + calibration | | | | |
| + lexicon fallback | | | | |
| + abstention | | | | |
