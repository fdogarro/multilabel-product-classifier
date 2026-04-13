"""
Evaluation Metrics
------------------
Covers all metrics specified in the project proposal:
  - Hamming loss
  - Micro-F1 / Macro-F1
  - Coverage error
  - Ranking loss
  - Per-label F1
  - Inference latency (measured in bert_classifier.py)

Threshold tuning via per-label F1-maximization sweep on the validation set.
"""

import numpy as np
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    coverage_error,
    label_ranking_loss,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    thresholds: list[float] = None,
) -> dict:
    """
    Args:
        y_true:       (n_samples, n_labels) binary ground truth
        y_pred_proba: (n_samples, n_labels) sigmoid probability scores
        thresholds:   per-label thresholds; defaults to 0.5 for all labels

    Returns:
        dict containing all proposal metrics
    """
    n_labels = y_true.shape[1]

    if thresholds is None:
        thresholds = [0.5] * n_labels

    thresholds_arr = np.array(thresholds)
    y_pred_binary = (y_pred_proba >= thresholds_arr).astype(int)

    micro_f1 = f1_score(y_true, y_pred_binary, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred_binary, average="macro", zero_division=0)
    h_loss   = hamming_loss(y_true, y_pred_binary)

    # Coverage error and ranking loss require probability scores
    cov_error   = coverage_error(y_true, y_pred_proba)
    rank_loss   = label_ranking_loss(y_true, y_pred_proba)

    # Per-label F1
    per_label = f1_score(y_true, y_pred_binary, average=None, zero_division=0)

    return {
        "micro_f1":       round(float(micro_f1), 4),
        "macro_f1":       round(float(macro_f1), 4),
        "hamming_loss":   round(float(h_loss), 4),
        "coverage_error": round(float(cov_error), 4),
        "ranking_loss":   round(float(rank_loss), 4),
        "per_label_f1":   [round(float(v), 4) for v in per_label],
    }


def tune_thresholds(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    sweep_start: float = 0.10,
    sweep_end: float = 0.90,
    sweep_steps: int = 81,
) -> list[float]:
    """
    Per-label F1-maximization threshold sweep.

    For each label, finds the threshold in [sweep_start, sweep_end] that
    maximizes F1 on the provided (validation) set.

    Args:
        y_true:       (n_samples, n_labels) binary ground truth
        y_pred_proba: (n_samples, n_labels) probability scores
        sweep_start:  lowest threshold to test
        sweep_end:    highest threshold to test
        sweep_steps:  number of candidate values

    Returns:
        list of per-label optimal thresholds, length n_labels
    """
    n_labels = y_true.shape[1]
    candidates = np.linspace(sweep_start, sweep_end, sweep_steps)
    best_thresholds = []

    for label_idx in range(n_labels):
        best_t = 0.5
        best_f1 = -1.0

        col_true = y_true[:, label_idx]
        col_prob = y_pred_proba[:, label_idx]

        # Skip labels with no positive examples — default to 0.5
        if col_true.sum() == 0:
            best_thresholds.append(0.5)
            continue

        for t in candidates:
            col_pred = (col_prob >= t).astype(int)
            f1 = f1_score(col_true, col_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        best_thresholds.append(round(float(best_t), 4))

    return best_thresholds


def per_label_f1(
    y_true: np.ndarray,
    y_pred_binary: np.ndarray,
    label_names: list[str],
) -> dict:
    """
    Returns a dict mapping each label name to its F1 score.

    Args:
        y_true:        (n_samples, n_labels) binary ground truth
        y_pred_binary: (n_samples, n_labels) binary predictions
        label_names:   list of label tag strings, length n_labels
    """
    scores = f1_score(y_true, y_pred_binary, average=None, zero_division=0)
    return {name: round(float(score), 4) for name, score in zip(label_names, scores)}


def compare_models(results: dict[str, dict], label_names: list[str]) -> dict:
    """
    Aggregate metrics from multiple model runs into a comparison table.

    Args:
        results: { model_name: compute_metrics() output }
        label_names: list of label tag strings

    Returns:
        dict with summary table and per-label breakdown
    """
    summary = {}
    per_label_breakdown = {name: {} for name in label_names}

    for model_name, metrics in results.items():
        summary[model_name] = {
            "micro_f1":       metrics["micro_f1"],
            "macro_f1":       metrics["macro_f1"],
            "hamming_loss":   metrics["hamming_loss"],
            "coverage_error": metrics["coverage_error"],
            "ranking_loss":   metrics["ranking_loss"],
        }
        for label_name, f1 in zip(label_names, metrics["per_label_f1"]):
            per_label_breakdown[label_name][model_name] = f1

    return {"summary": summary, "per_label": per_label_breakdown}
