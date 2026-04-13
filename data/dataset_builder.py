"""
Dataset Builder
---------------
Takes raw DataFrames (Amazon train corpus + Flipkart holdout) and produces:
  - Stratified 80/10/10 train/val/test splits from the Amazon corpus
  - A completely isolated Flipkart cross-domain evaluation set
  - Class imbalance statistics and pos_weight vector for BCEWithLogitsLoss

Stratification is done per primary label (the highest-frequency active label
per sample) to ensure each label is represented proportionally in every split.

Output files (Parquet):
  data/processed/train.parquet
  data/processed/val.parquet
  data/processed/test.parquet
  data/processed/flipkart_eval.parquet
  data/processed/stats.json
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10
RANDOM_SEED = 42


class DatasetBuilder:
    def __init__(self, num_labels: int = 30):
        self.num_labels = num_labels

    def build(self, amazon_df: pd.DataFrame, flipkart_df: pd.DataFrame = None) -> dict:
        """
        Args:
            amazon_df:   Raw Amazon products with 'labels' column (list[int], length 30)
            flipkart_df: Raw Flipkart products — kept entirely separate, never seen during training

        Returns:
            dict with keys: train, val, test, flipkart_eval (DataFrames) + stats (dict)
        """
        logger.info(f"Amazon corpus: {len(amazon_df):,} products")

        amazon_df = self._validate(amazon_df)

        # Stratify by primary label
        amazon_df["_primary_label"] = amazon_df["labels"].apply(self._primary_label)

        train_df, val_df, test_df = self._stratified_split(amazon_df)
        train_df = train_df.drop(columns=["_primary_label"])
        val_df   = val_df.drop(columns=["_primary_label"])
        test_df  = test_df.drop(columns=["_primary_label"])

        stats = self._compute_stats(train_df, val_df, test_df, flipkart_df)

        # Persist
        train_df.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
        val_df.to_parquet(PROCESSED_DIR / "val.parquet", index=False)
        test_df.to_parquet(PROCESSED_DIR / "test.parquet", index=False)

        if flipkart_df is not None:
            flipkart_df = self._validate(flipkart_df)
            flipkart_df.to_parquet(PROCESSED_DIR / "flipkart_eval.parquet", index=False)
            logger.info(f"Flipkart holdout: {len(flipkart_df):,} products")

        with open(PROCESSED_DIR / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(
            f"Split sizes — train: {len(train_df):,} | val: {len(val_df):,} | test: {len(test_df):,}"
        )
        return {"train": train_df, "val": val_df, "test": test_df,
                "flipkart_eval": flipkart_df, "stats": stats}

    def compute_pos_weight(self, train_df: pd.DataFrame) -> list[float]:
        """
        BCEWithLogitsLoss pos_weight: (n_negative / n_positive) per label.
        Returns a list of length num_labels. Labels with zero positives get weight 1.0.
        """
        label_matrix = np.array(train_df["labels"].tolist())
        n = len(label_matrix)
        pos_counts = label_matrix.sum(axis=0)
        neg_counts = n - pos_counts
        weights = []
        for pos, neg in zip(pos_counts, neg_counts):
            weights.append(float(neg / pos) if pos > 0 else 1.0)
        return weights

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate(self, df: pd.DataFrame) -> pd.DataFrame:
        assert "labels" in df.columns, "DataFrame must have a 'labels' column"
        assert "title" in df.columns, "DataFrame must have a 'title' column"
        # Ensure labels are lists of ints with correct length
        df = df[df["labels"].apply(lambda x: isinstance(x, list) and len(x) == self.num_labels)]
        df = df[df["labels"].apply(lambda x: sum(x) > 0)]  # drop zero-label rows
        return df.reset_index(drop=True)

    @staticmethod
    def _primary_label(labels: list[int]) -> int:
        """Return index of first active label (used as stratification key)."""
        for i, v in enumerate(labels):
            if v == 1:
                return i
        return -1

    def _stratified_split(self, df: pd.DataFrame):
        """80 / 10 / 10 stratified by _primary_label."""
        # First cut: 80% train, 20% temp
        train, temp = train_test_split(
            df, test_size=(VAL_RATIO + TEST_RATIO),
            stratify=df["_primary_label"], random_state=RANDOM_SEED
        )
        # Second cut: 50/50 of the 20% → 10% val, 10% test
        val, test = train_test_split(
            temp, test_size=0.5,
            stratify=temp["_primary_label"], random_state=RANDOM_SEED
        )
        return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

    def _compute_stats(self, train_df, val_df, test_df, flipkart_df) -> dict:
        label_matrix = np.array(train_df["labels"].tolist())
        pos_counts = label_matrix.sum(axis=0).tolist()
        pos_weights = self.compute_pos_weight(train_df)

        avg_labels_per_product = float(label_matrix.sum(axis=1).mean())
        label_coverage = float((label_matrix.sum(axis=0) > 0).mean())

        return {
            "n_train": len(train_df),
            "n_val":   len(val_df),
            "n_test":  len(test_df),
            "n_flipkart": len(flipkart_df) if flipkart_df is not None else 0,
            "num_labels": self.num_labels,
            "avg_labels_per_product": round(avg_labels_per_product, 3),
            "label_coverage_pct": round(label_coverage * 100, 1),
            "per_label_pos_counts": [int(c) for c in pos_counts],
            "per_label_pos_weights": [round(w, 4) for w in pos_weights],
        }
