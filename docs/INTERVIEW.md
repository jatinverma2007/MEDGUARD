# Interview preparation

Everything below is answerable **because you built it**. If a question here
catches you out, that is a gap in the project, not in your preparation.

---

## The 90-second project story

> "I'm a third-year BTech student and I built **MedGuard**, a prescription
> safety AI. You upload a prescription photo; it identifies the medicines,
> checks all the drug pairs against an interaction database, grades the
> severity, and explains the risk in English or Hindi.
>
> The interesting part was methodological. When I split my data randomly — the
> way most published DDI work does — my model scored F1 of 0.94. But that
> number was fake: the *same drug pairs* were in both train and test, so the
> model was memorising 'warfarin plus aspirin causes bleeding' rather than
> learning anything. When I re-split it so no drug pair crosses the boundary,
> and then again so no drug at all crosses it, the honest numbers were 0.81
> and 0.74. That 20-point gap is the most important thing I learned this
> year.
>
> The second thing is calibration. My model would say 'no interaction' with
> 95% confidence and be wrong a third of the time, which in a medical
> context is the dangerous failure. I added temperature scaling, which
> brought the Expected Calibration Error from 0.08 down to 0.01 without
> changing the ranking of predictions, and then added abstention so it
> refuses to answer below a confidence threshold. The whole thing runs on a
> free Colab T4 and is deployed on Hugging Face Spaces."

**Rehearse this until it flows.** It demonstrates problem framing, skepticism
about your own results, and production awareness — in 90 seconds.

---

## Questions you should expect

### "Why is accuracy a bad metric here?"

> "The classes are heavily imbalanced — most pairs have no interaction. More
> importantly, the cost of the errors is asymmetric. Telling a patient 'this
> is safe' when it isn't could be fatal; telling them 'check with your
> doctor' when it's fine just adds friction. So I optimised for F1 on the
> positive class, reported precision and recall separately, and added
> abstention so the model can decline to answer rather than be wrong."

### "How do you know the model isn't hallucinating?"

> "The default explanation layer is templated directly from the retrieved
> knowledge-base record — there's no generation step, so hallucination isn't
> possible by construction. The optional LLM layer is a separate path: I feed
> it only the retrieved facts, then run every generated sentence through a
> DeBERTa MNLI entailment model against those facts, and drop anything that
> isn't entailed. I report the grounding ratio before and after filtering,
> which turns 'it sounds right' into a number."

### "Why did you build a knowledge base lookup at all? Isn't that cheating?"

> "No — it's the safety-critical architecture. In medicine you want a
> verifiable source of truth where one exists. The interesting ML problem is
> the pairs *not* in the database: that's where the model adds value, and
> that's where I apply calibration and abstention so the model never presents
> a guess as fact. The KB is labelled `source='kb'` and model predictions are
> `source='model'` so the interface and the evaluation keep them separate."

### "How do you handle a drug name that's not in your database?"

> "Three layers. First, normalisation: I resolve brand names and salts to
> generic names using an alias table, so 'Ecosprin' becomes
> 'acetylsalicylic acid'. Second, out-of-distribution detection: I check
> embedding distance from the training vocabulary to flag unseen entities.
> Third, abstention — if the drug isn't confidently recognised, the system
> says so rather than silently dropping it, because a silently dropped drug
> is a missed interaction."

### "What's your data privacy stance?"

> "Uploaded images are processed in memory and deleted; nothing is written to
> disk. Raw prescription text is excluded from API responses by default and
> is only logged when an explicit environment flag is set. No patient
> identifiers are collected or stored. I also read the Indian Digital
> Personal Data Protection Act, 2023 — medical records are sensitive personal
> data, so the design assumes consent, minimal retention, and the option of
> on-premise or local-only inference."

### "How would you validate this clinically?"

> "Three stages. First retrospective validation against a gold set curated by
> a pharmacist. Then a reader study — give three clinicians the same
> prescriptions with and without MedGuard's output and measure whether it
> changes their prescribing decisions. Then a prospective pilot with a
> partner pharmacy, with the explicit caveat that this is decision support
> and not a diagnostic device. I would also need to assess where this falls
> under CDSCO medical-device regulation before it influenced clinical
> decisions."

### "Why did you choose this architecture?"

> "Each stage fails differently. OCR errors are character-level, NER errors
> are token-level, retrieval errors are coverage-level, and severity errors
> are calibration-level. A single end-to-end model would blend them into one
> opaque number. By keeping them separate, I can measure each one, show you
> exactly which stage is responsible for a failure, and swap any stage
> independently — for instance swapping the heavy VLM OCR for Tesseract on a
> low-resource deployment without touching the rest."

### "What would you do with more compute?"

> "Three things, in order. First, fine-tune Qwen2.5-VL on my own
> prescription images instead of prompting it — that should beat the
> prompted baseline meaningfully. Second, replace the pair encoder with a
> graph model over the drug–drug interaction network, which captures
> higher-order polypharmacy effects that pair-wise models structurally
> cannot. Third, a much larger corpus with patient outcomes, which would let
> me validate severity against clinical outcomes rather than database
> labels."

### "What's the hardest part?"

> "Honestly, the data — not the modelling. Public interaction databases are
> biased toward US and European formularies and under-represent many Indian
> generics, which means the system fails on exactly the prescriptions it
> matters most for. That's also why I interviewed pharmacists: three of them
> told me that severity labelling, not detection, was their actual
> bottleneck. I rebuilt the severity module around that and it changed the
> direction of the project."

### "Tell me about a failure."

Use a real one from your error analysis. Structure: **what happened → why →
what you changed → did it work.** Never say "it works fine."

---

## The three answers you must be able to give cold

1. **"Why is your dataset split like that?"** — because a random split leaks
   drug pairs and inflates F1 by 10–20 points; here is the measurement.
2. **"How do you know the model's confidence is meaningful?"** — ECE before
   and after temperature scaling, plus the risk–coverage curve; here are the
   numbers.
3. **"What does your system do when it doesn't know?"** — it abstains, it
   says why, and it escalates; here is the threshold and the coverage.

---

## Questions to ask them

Shows you're thinking about the role, not just the project.

- "What does a great first 90 days look like for this team?"
- "How do you decide whether a model is good enough to ship?"
- "Do you do offline evaluation on real data, and how do you handle the
  long tail?"
- "What's a modelling decision you disagree with in production right now?"

---

## Your week-1 prep checklist

- [ ] Write the 90-second story and say it out loud 5 times.
- [ ] Be able to draw the architecture from memory on paper.
- [ ] Know the exact numbers for: NER F1, interaction F1 on `pairwise` and
      `cold_start`, the leakage gap, ECE before/after, p95 latency.
- [ ] Know one thing that went wrong and what you changed.
- [ ] Have the demo URL live and tested on your phone.
- [ ] Prepare 3 questions to ask them.
