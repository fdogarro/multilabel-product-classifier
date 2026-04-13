"""
Product Dataset
---------------
PyTorch Dataset that wraps the processed Parquet splits produced by
DatasetBuilder. Returns tokenized inputs and a float32 label tensor
suitable for BCEWithLogitsLoss.

Usage:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    ds = ProductDataset("data/processed/train.parquet", tokenizer, max_length=120)
    loader = DataLoader(ds, batch_size=32, shuffle=True)
"""

import ast

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class ProductDataset(Dataset):
    def __init__(
        self,
        parquet_path: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 120,
        num_labels: int = 30,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_labels = num_labels

        df = pd.read_parquet(parquet_path)
        self.titles = df["title"].fillna("").tolist()
        self.descriptions = df["description"].fillna("").tolist()

        # labels may be stored as list or stringified list
        raw_labels = df["labels"].tolist()
        self.labels = [
            self._parse_labels(l) for l in raw_labels
        ]

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        title = self.titles[idx]
        description = self.descriptions[idx]
        combined = f"{title} [SEP] {description}"

        encoding = self.tokenizer(
            combined,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)

        item = {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         label_tensor,
        }

        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

        return item

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_labels(self, raw) -> list:
        if isinstance(raw, (list, np.ndarray)):
            return [int(v) for v in raw]
        if isinstance(raw, str):
            return [int(v) for v in ast.literal_eval(raw)]
        return [0] * self.num_labels
