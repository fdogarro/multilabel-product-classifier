"""
Evaluation Metrics
------------------
Hamming loss, Micro/Macro-F1, coverage error, ranking loss,
per-label F1, and inference latency.
Threshold tuning via F1-maximization on the validation set.
Implemented in Phase 2 (BERT baseline); extended in Phases 3–5.
"""


def compute_metrics(y_true, y_pred_proba, thresholds=None) -> dict:
    """
    Args:
        y_true:        (n_samples, n_labels) binary ground truth
        y_pred_proba:  (n_samples, n_labels) sigmoid scores
        thresholds:    per-label thresholds; defaults to 0.5 if None
    Returns:
        dict of all metrics
    """
    raise NotImplementedError("Phase 2")


def tune_thresholds(y_true, y_pred_proba) -> list[float]:
    """F1-maximization sweep per label over the validation set."""
    raise NotImplementedError("Phase 2")


def per_label_f1(y_true, y_pred_binary, label_names: list[str]) -> dict:
    raise NotImplementedError("Phase 2")
