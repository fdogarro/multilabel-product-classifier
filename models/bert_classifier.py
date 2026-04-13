"""
BERT / RoBERTa Multi-Label Classifier
--------------------------------------
Architecture:
  - Pretrained encoder (bert-base-uncased or roberta-base)
  - Dropout(0.1) on [CLS] pooled output
  - Linear head: hidden_size → num_labels (raw logits, no sigmoid)

Loss: BCEWithLogitsLoss with optional pos_weight tensor for class imbalance.

Training:
  - AdamW optimizer
  - Linear warmup schedule (10% of total steps)
  - Gradient clipping (max_norm=1.0)
  - Best checkpoint saved by Micro-F1 on validation set
"""

import json
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from eval.metrics import compute_metrics, tune_thresholds

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------

class BertMultiLabelClassifier(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased", num_labels: int = 30):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # Use [CLS] pooled output
        pooled = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs.last_hidden_state[:, 0]
        logits = self.classifier(self.dropout(pooled))
        return logits  # raw logits — BCEWithLogitsLoss applies sigmoid internally

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str) -> "BertMultiLabelClassifier":
        checkpoint_dir = Path(checkpoint_dir)
        with open(checkpoint_dir / "config.json") as f:
            config = json.load(f)
        model = cls(model_name=config["model_name"], num_labels=config["num_labels"])
        model.load_state_dict(torch.load(checkpoint_dir / "model.pt", map_location="cpu"))
        return model


# -----------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------

class BertTrainer:
    def __init__(
        self,
        model: BertMultiLabelClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        checkpoint_dir: str = "checkpoints/bert",
        pos_weight: list[float] = None,
        device: str = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"Training on: {self.device}")

        # Loss
        pw = torch.tensor(pos_weight, dtype=torch.float32).to(self.device) if pos_weight else None
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get("learning_rate", 2e-5),
            weight_decay=config.get("weight_decay", 0.01),
        )

        # Scheduler — linear warmup over 10% of total steps
        num_epochs = config.get("num_epochs", 5)
        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(0.10 * total_steps)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        self.history = {"train_loss": [], "val_loss": [], "micro_f1": [], "macro_f1": []}
        self.best_micro_f1 = 0.0
        self.best_thresholds = None

    def train(self) -> dict:
        num_epochs = self.config.get("num_epochs", 5)

        for epoch in range(1, num_epochs + 1):
            train_loss = self._train_epoch(epoch, num_epochs)
            val_loss, metrics, thresholds = self._validate_epoch()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["micro_f1"].append(metrics["micro_f1"])
            self.history["macro_f1"].append(metrics["macro_f1"])

            logger.info(
                f"Epoch {epoch}/{num_epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"micro_f1={metrics['micro_f1']:.4f} | macro_f1={metrics['macro_f1']:.4f}"
            )

            # Save best checkpoint
            if metrics["micro_f1"] > self.best_micro_f1:
                self.best_micro_f1 = metrics["micro_f1"]
                self.best_thresholds = thresholds
                self._save_checkpoint(epoch, metrics)
                logger.info(f"  -> New best checkpoint (micro_f1={self.best_micro_f1:.4f})")

        return {"history": self.history, "best_micro_f1": self.best_micro_f1,
                "best_thresholds": self.best_thresholds}

    def _train_epoch(self, epoch: int, total_epochs: int) -> float:
        self.model.train()
        total_loss = 0.0
        max_grad_norm = self.config.get("max_grad_norm", 1.0)

        for step, batch in enumerate(self.train_loader, 1):
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask, token_type_ids)
            loss = self.criterion(logits, labels)
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

            if step % 100 == 0:
                avg = total_loss / step
                logger.info(f"  Epoch {epoch}/{total_epochs} | step {step}/{len(self.train_loader)} | avg_loss={avg:.4f}")

        return total_loss / len(self.train_loader)

    def _validate_epoch(self):
        self.model.eval()
        total_loss = 0.0
        all_probs, all_labels = [], []

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.device)
                labels = batch["labels"].to(self.device)

                logits = self.model(input_ids, attention_mask, token_type_ids)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(labels.cpu().numpy())

        import numpy as np
        y_prob = np.vstack(all_probs)
        y_true = np.vstack(all_labels)

        # Tune thresholds on this validation pass
        thresholds = tune_thresholds(y_true, y_prob)
        metrics = compute_metrics(y_true, y_prob, thresholds)

        return total_loss / len(self.val_loader), metrics, thresholds

    def _save_checkpoint(self, epoch: int, metrics: dict):
        torch.save(self.model.state_dict(), self.checkpoint_dir / "model.pt")
        config_out = {
            "model_name":  self.model.model_name,
            "num_labels":  self.model.num_labels,
            "epoch":       epoch,
            "micro_f1":    metrics["micro_f1"],
            "macro_f1":    metrics["macro_f1"],
            "config":      self.config,
        }
        with open(self.checkpoint_dir / "config.json", "w") as f:
            json.dump(config_out, f, indent=2)

        if self.best_thresholds is not None:
            with open(self.checkpoint_dir / "thresholds.json", "w") as f:
                json.dump(self.best_thresholds, f)


# -----------------------------------------------------------------------
# Inference helper
# -----------------------------------------------------------------------

def predict_batch(
    model: BertMultiLabelClassifier,
    tokenizer,
    texts: list[str],
    thresholds: list[float],
    device: str = None,
    max_length: int = 120,
) -> tuple[list[list[int]], list[list[float]]]:
    """
    Run inference on a list of pre-combined text strings.

    Returns:
        binary_preds: list of binary label vectors
        prob_scores:  list of per-label probability scores
    """
    import numpy as np

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    encoding = tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = model(
            encoding["input_ids"].to(device),
            encoding["attention_mask"].to(device),
            encoding.get("token_type_ids", {}).get("token_type_ids"),
        )
        probs = torch.sigmoid(logits).cpu().numpy()

    thresholds_arr = np.array(thresholds)
    binary = (probs >= thresholds_arr).astype(int)

    return binary.tolist(), probs.tolist()


def measure_latency(
    model: BertMultiLabelClassifier,
    tokenizer,
    n_samples: int = 100,
    max_length: int = 120,
    device: str = None,
) -> dict:
    """Measure mean and p95 inference latency in milliseconds."""
    import numpy as np

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    dummy_text = ["Wireless Bluetooth speaker waterproof outdoor audio device"] * 1
    encoding = tokenizer(dummy_text, max_length=max_length, padding="max_length",
                         truncation=True, return_tensors="pt")
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    latencies = []
    with torch.no_grad():
        for _ in range(n_samples):
            start = time.perf_counter()
            model(input_ids, attention_mask)
            if device == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)

    return {
        "mean_ms":   round(float(np.mean(latencies)), 2),
        "median_ms": round(float(np.median(latencies)), 2),
        "p95_ms":    round(float(np.percentile(latencies, 95)), 2),
        "p99_ms":    round(float(np.percentile(latencies, 99)), 2),
    }
