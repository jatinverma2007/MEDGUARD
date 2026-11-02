"""Transformer relation classifier for drug-drug interactions.

The model predicts, for a pair of drugs, whether an interaction exists and
which mechanism is involved. It is trained on the canonical interaction table
with the leak-free splits from :mod:`medguard.data.splits`.

Design decisions that matter for the interview:

1. **Drug names are inputs.** The pair is encoded as
   ``"[SEP] drug_a [SEP] drug_b [SEP]"`` using a biomedical encoder, so the
   model can learn pharmacokinetic structure from name morphology and
   co-occurrence context.

2. **The classifier is separated from the calibrator.** The model produces
   logits; :class:`medguard.evaluation.calibration.TemperatureScaler` is
   fitted afterwards on the validation split. Keeping them apart is what
   makes the calibration numbers trustworthy.

3. **Abstention is part of the model wrapper.**
   :class:`CalibratedInteractionClassifier` exposes a ``predict_with_abstain``
   method that returns a finding only when the calibrated probability clears
   the threshold.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from medguard.evaluation.calibration import TemperatureScaler
from medguard.schemas import Confidence, InteractionFinding, Severity

logger = logging.getLogger(__name__)

DEFAULT_ENCODER = "microsoft/deberta-v3-base"


@dataclass
class CalibratedInteractionClassifier:
    """Transformer classifier + calibrator + abstention threshold.

    The transformer is optional so the class can be exercised with a stub
    during testing and used with a real encoder in production.
    """

    encoder_name: str = DEFAULT_ENCODER
    threshold: float = 0.5
    abstain_threshold: float = 0.55
    max_length: int = 128
    batch_size: int = 32
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.1

    model: object | None = None
    tokenizer: object | None = None
    calibrator: TemperatureScaler = field(default_factory=TemperatureScaler)
    label_map: dict[str, int] = field(default_factory=dict)
    device: str | None = None

    # ---------------------------------------------------------------- input
    @staticmethod
    def encode_pair(drug_a: str, drug_b: str) -> str:
        """Order-invariant textual encoding of a drug pair."""
        a, b = sorted([str(drug_a).lower(), str(drug_b).lower()])
        return f"{a} [SEP] {b}"

    # ------------------------------------------------------------- training
    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        *,
        label_col: str = "label",
        output_dir: str | Path | None = None,
    ) -> CalibratedInteractionClassifier:
        """Fine-tune the encoder on the training split.

        Requires ``pip install 'medguard[ml]'``.
        """
        try:
            import torch
            from torch.utils.data import DataLoader, Dataset
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
                get_linear_schedule_with_warmup,
            )
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "transformers + torch are required for training. Run: pip install 'medguard[ml]'"
            ) from exc

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        labels = sorted(train_df[label_col].unique().tolist())
        self.label_map = {str(label): idx for idx, label in enumerate(labels)}

        self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.encoder_name,
            num_labels=len(labels),
            id2label={i: str(label) for label, i in self.label_map.items()},
            label2id=self.label_map,
        )
        self.model.to(self.device)

        class _PairDataset(Dataset):
            def __init__(self, frame: pd.DataFrame) -> None:
                self.frame = frame.reset_index(drop=True)

            def __len__(self) -> int:
                return len(self.frame)

            def __getitem__(self, idx: int) -> dict[str, object]:
                row = self.frame.iloc[idx]
                text = CalibratedInteractionClassifier.encode_pair(row["drug_a"], row["drug_b"])
                return {
                    "text": text,
                    "label": self.label_map[str(row[label_col])],
                }

        def collate(batch: list[dict[str, object]]) -> dict[str, object]:
            texts = [b["text"] for b in batch]
            labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded["labels"] = labels
            return encoded

        train_loader = DataLoader(
            _PairDataset(train_df),
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate,
        )
        val_loader = DataLoader(
            _PairDataset(val_df),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate,
        )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(train_loader) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * self.warmup_ratio),
            num_training_steps=total_steps,
        )

        best_f1 = -1.0
        best_state: dict[str, object] | None = None

        for epoch in range(self.num_epochs):
            self.model.train()
            running = 0.0
            for step, batch in enumerate(train_loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                loss = self.model(**batch).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                running += float(loss.item())
                if step % 50 == 0:
                    logger.info("epoch %d step %d loss %.4f", epoch, step, float(loss.item()))

            val_logits, val_labels = self._infer(val_loader)
            metrics = self._quick_metrics(val_logits, val_labels)
            logger.info("epoch %d | val F1 %.4f", epoch, metrics["f1"])
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_state = {
                    k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                }

        if best_state is not None:
            self.model.load_state_dict(best_state)
            logger.info("Restored best checkpoint (val F1 %.4f)", best_f1)

        # Fit the calibrator on the validation split (never on test).
        val_logits, val_labels = self._infer(val_loader)
        if len(self.label_map) == 2:
            self.calibrator = TemperatureScaler().fit(val_logits, val_labels)
            logger.info("Calibrator fitted: T=%.4f", self.calibrator.temperature)

        if output_dir:
            self.save(Path(output_dir))
        return self

    def _infer(self, loader) -> tuple[np.ndarray, np.ndarray]:
        import torch

        self.model.eval()
        all_logits: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                labels = batch.pop("labels").cpu().numpy()
                batch = {k: v.to(self.device) for k, v in batch.items()}
                logits = self.model(**batch).logits.cpu().numpy()
                all_logits.append(logits)
                all_labels.append(labels)
        return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)

    def _quick_metrics(self, logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        from sklearn.metrics import f1_score

        if len(self.label_map) == 2:
            probs = 1.0 / (1.0 + np.exp(-(logits[:, 1] - logits[:, 0])))
            preds = (probs >= 0.5).astype(int)
        else:
            preds = logits.argmax(axis=-1)
        return {"f1": float(f1_score(labels, preds, average="macro", zero_division=0))}

    # ----------------------------------------------------------- inference
    def predict_logits(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Raw logits for a list of drug pairs."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded. Call train() or load() first.")
        import torch

        self.model.eval()
        texts = [self.encode_pair(a, b) for a, b in pairs]
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                chunk = texts[start : start + self.batch_size]
                encoded = self.tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**encoded).logits.cpu().numpy()
                outputs.append(logits)
        return np.concatenate(outputs, axis=0)

    def predict_proba(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Calibrated probability of an interaction for each pair."""
        logits = self.predict_logits(pairs)
        if logits.shape[1] != 2:
            raise ValueError("Calibrated probabilities require a binary classifier")
        binary_logits = logits[:, 1] - logits[:, 0]
        return self.calibrator.predict_proba(binary_logits)

    def predict_with_abstain(
        self,
        pairs: list[tuple[str, str]],
        *,
        event_types: list[str | None] | None = None,
    ) -> tuple[list[InteractionFinding], list[tuple[str, str]]]:
        """Predict findings, abstaining on low-confidence pairs.

        Returns:
            ``(findings, abstained_pairs)``
        """
        from medguard.interaction.severity import severity_from_relation

        probs = self.predict_proba(pairs)
        findings: list[InteractionFinding] = []
        abstained: list[tuple[str, str]] = []

        for idx, (a, b) in enumerate(pairs):
            prob = float(probs[idx])
            if prob < self.abstain_threshold:
                abstained.append((a, b))
                continue
            event = event_types[idx] if event_types else None
            severity = severity_from_relation(event)
            findings.append(
                InteractionFinding(
                    drug_a=a,
                    drug_b=b,
                    severity=severity,
                    event_type=event,
                    confidence=Confidence(
                        raw=prob,
                        calibrated=prob,
                        temperature=self.calibrator.temperature,
                    ),
                    source="model",
                )
            )
        return findings, abstained

    # ------------------------------------------------------------ persistence
    def save(self, output_dir: str | Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.model is not None and self.tokenizer is not None:
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
        self.calibrator.save(output_dir / "calibrator.json")
        (output_dir / "label_map.json").write_text(json.dumps(self.label_map, indent=2))
        return output_dir

    def load(self, input_dir: str | Path) -> CalibratedInteractionClassifier:
        input_dir = Path(input_dir)
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("transformers is required. Run: pip install 'medguard[ml]'") from exc

        self.tokenizer = AutoTokenizer.from_pretrained(input_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(input_dir)
        self.model.eval()
        calib_path = input_dir / "calibrator.json"
        if calib_path.exists():
            self.calibrator = TemperatureScaler.load(calib_path)
        label_path = input_dir / "label_map.json"
        if label_path.exists():
            self.label_map = json.loads(label_path.read_text())
        return self


def severity_from_confidence(
    base: Severity, confidence: float, *, floor: Severity = Severity.NONE
) -> Severity:
    """Convenience re-export used by the pipeline."""
    from medguard.interaction.severity import apply_confidence_cap

    return apply_confidence_cap(base, confidence, floor=floor.rank)
