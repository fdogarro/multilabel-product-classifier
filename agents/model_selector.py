"""
Model Selector Agent
--------------------
Confidence-calibrated routing policy.
  High confidence  → BERT
  Ambiguous        → LoRA-Mistral
  OOD / fallback   → LLM API

Thresholds are tuned on the validation set (F1-maximization).
Implemented in Phase 5.
"""

ROUTE_BERT = "bert"
ROUTE_LORA = "lora_mistral"
ROUTE_LLM = "llm_api"


class ModelSelector:
    def __init__(self, high_conf_threshold: float = 0.75, low_conf_threshold: float = 0.50):
        self.high_conf_threshold = high_conf_threshold
        self.low_conf_threshold = low_conf_threshold

    def run(self, state: dict) -> dict:
        raise NotImplementedError("Phase 5")
