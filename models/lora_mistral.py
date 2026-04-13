"""
LoRA-Mistral-7B Multi-Label Classifier (QLoRA, 4-bit)
-------------------------------------------------------
Architecture:
  - Mistral-7B-Instruct-v0.2 loaded in 4-bit (NF4 quantization, double quant)
  - LoRA adapters on q_proj and v_proj: r=16, alpha=32, dropout=0.1
  - Linear classification head on the last non-padding token's hidden state
  - Raw logit output — BCEWithLogitsLoss applied during training

Why last token: Mistral is a decoder-only model with no [CLS] pooler.
The last real token aggregates the full left context, making it the
natural pooling point for classification.

Expected GPU memory: ~18–24 GB peak on A100 with batch_size=4, max_length=120.
"""

import json
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)

from eval.metrics import compute_metrics, tune_thresholds

logger = logging.getLogger(__name__)

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"


# -----------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------

class LoraMistralClassifier(nn.Module):
    """
    Wraps the QLoRA-adapted Mistral backbone with a linear classification head.
    The backbone weights are 4-bit frozen; only LoRA adapters and the head train.
    """

    def __init__(self, backbone: nn.Module, hidden_size: int, num_labels: int = 30):
        super().__init__()
        self.backbone = backbone
        self.num_labels = num_labels
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Last non-padding token per sample
        seq_lengths = attention_mask.sum(dim=1) - 1          # (batch,)
        last_hidden  = outputs.hidden_states[-1]              # (batch, seq, hidden)
        batch_size   = input_ids.size(0)
        pooled = last_hidden[torch.arange(batch_size, device=input_ids.device), seq_lengths]

        logits = self.classifier(self.dropout(pooled))        # (batch, num_labels)
        return logits

    def save(self, output_dir: str):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save LoRA adapters
        self.backbone.save_pretrained(str(output_dir))
        # Save classification head separately
        torch.save(self.classifier.state_dict(), output_dir / "classifier_head.pt")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str,
        base_model_name: str = BASE_MODEL,
        num_labels: int = 30,
    ) -> "LoraMistralClassifier":
        checkpoint_dir = Path(checkpoint_dir)
        bnb_config = _bnb_config()
        backbone = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        backbone = PeftModel.from_pretrained(backbone, str(checkpoint_dir))
        hidden_size = backbone.config.hidden_size
        model = cls(backbone, hidden_size, num_labels)
        model.classifier.load_state_dict(
            torch.load(checkpoint_dir / "classifier_head.pt", map_location="cpu")
        )
        return model


# -----------------------------------------------------------------------
# Build helpers
# -----------------------------------------------------------------------

def _bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def build_model_and_tokenizer(
    base_model_name: str = BASE_MODEL,
    num_labels: int = 30,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
) -> tuple["LoraMistralClassifier", AutoTokenizer]:
    """
    Load Mistral-7B in 4-bit, apply QLoRA, attach classification head.

    Returns:
        (LoraMistralClassifier, tokenizer)
    """
    logger.info(f"Loading {base_model_name} in 4-bit...")
    bnb_config = _bnb_config()

    backbone = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    backbone = prepare_model_for_kbit_training(backbone)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    backbone = get_peft_model(backbone, lora_config)
    backbone.print_trainable_parameters()

    hidden_size = backbone.config.hidden_size
    model = LoraMistralClassifier(backbone, hidden_size, num_labels)

    # Tokenizer — set pad token to EOS, right-pad for classification
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token     = tokenizer.eos_token
    tokenizer.padding_side  = "right"

    return model, tokenizer


# -----------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------

class LoraMistralTrainer:
    def __init__(
        self,
        model: LoraMistralClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        checkpoint_dir: str = "checkpoints/lora_mistral",
        pos_weight: list[float] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Mistral uses device_map="auto" — don't manually move the backbone
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Move only the classification head explicitly
        self.model.classifier.to(self.device)
        self.model.dropout.to(self.device)

        pw = torch.tensor(pos_weight, dtype=torch.float32).to(self.device) if pos_weight else None
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

        # Only optimize LoRA params + classification head (backbone weights are frozen/4-bit)
        trainable_params = (
            list(filter(lambda p: p.requires_grad, model.backbone.parameters()))
            + list(model.classifier.parameters())
        )
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.get("learning_rate", 2e-4),
            weight_decay=config.get("weight_decay", 0.01),
        )

        num_epochs  = config.get("num_epochs", 3)
        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(0.10 * total_steps)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        self.history = {"train_loss": [], "val_loss": [], "micro_f1": [], "macro_f1": []}
        self.best_micro_f1 = 0.0
        self.best_thresholds = None

    def train(self) -> dict:
        num_epochs = self.config.get("num_epochs", 3)

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

            if metrics["micro_f1"] > self.best_micro_f1:
                self.best_micro_f1 = metrics["micro_f1"]
                self.best_thresholds = thresholds
                self._save_checkpoint(epoch, metrics)
                logger.info(f"  -> New best checkpoint (micro_f1={self.best_micro_f1:.4f})")

        return {
            "history":          self.history,
            "best_micro_f1":    self.best_micro_f1,
            "best_thresholds":  self.best_thresholds,
        }

    def _train_epoch(self, epoch: int, total_epochs: int) -> float:
        self.model.train()
        total_loss = 0.0
        max_grad_norm = self.config.get("max_grad_norm", 1.0)

        for step, batch in enumerate(self.train_loader, 1):
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels         = batch["labels"].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            loss   = self.criterion(logits, labels)
            loss.backward()

            nn.utils.clip_grad_norm_(
                list(filter(lambda p: p.requires_grad, self.model.backbone.parameters()))
                + list(self.model.classifier.parameters()),
                max_grad_norm,
            )
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

            if step % 50 == 0:
                avg = total_loss / step
                mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                logger.info(
                    f"  Epoch {epoch}/{total_epochs} | step {step}/{len(self.train_loader)} "
                    f"| avg_loss={avg:.4f} | GPU mem={mem:.1f}GB"
                )

        return total_loss / len(self.train_loader)

    def _validate_epoch(self):
        self.model.eval()
        total_loss = 0.0
        all_probs, all_labels = [], []

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["labels"].to(self.device)

                logits = self.model(input_ids, attention_mask)
                loss   = self.criterion(logits, labels)
                total_loss += loss.item()

                probs = torch.sigmoid(logits).cpu().float().numpy()
                all_probs.append(probs)
                all_labels.append(labels.cpu().numpy())

        import numpy as np
        y_prob = np.vstack(all_probs)
        y_true = np.vstack(all_labels)

        thresholds = tune_thresholds(y_true, y_prob)
        metrics    = compute_metrics(y_true, y_prob, thresholds)

        return total_loss / len(self.val_loader), metrics, thresholds

    def _save_checkpoint(self, epoch: int, metrics: dict):
        self.model.save(str(self.checkpoint_dir))
        meta = {
            "base_model":  BASE_MODEL,
            "num_labels":  self.model.num_labels,
            "epoch":       epoch,
            "micro_f1":    metrics["micro_f1"],
            "macro_f1":    metrics["macro_f1"],
            "config":      self.config,
        }
        with open(self.checkpoint_dir / "training_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        if self.best_thresholds:
            with open(self.checkpoint_dir / "thresholds.json", "w") as f:
                json.dump(self.best_thresholds, f)


# -----------------------------------------------------------------------
# Inference helper
# -----------------------------------------------------------------------

def measure_latency(
    model: LoraMistralClassifier,
    tokenizer,
    n_samples: int = 50,
    max_length: int = 120,
) -> dict:
    """Measure single-sample inference latency in milliseconds."""
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()

    dummy = ["Wireless Bluetooth speaker waterproof outdoor audio device"]
    enc   = tokenizer(dummy, max_length=max_length, padding="max_length",
                      truncation=True, return_tensors="pt")
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    latencies = []
    with torch.no_grad():
        for _ in range(n_samples):
            if device == "cuda":
                torch.cuda.synchronize()
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
