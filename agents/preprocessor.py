"""
Preprocessor Agent
------------------
Cleans and normalizes product text for downstream model input.

Steps:
  1. Strip HTML tags
  2. Normalize whitespace and unicode
  3. Concatenate: "{title} [SEP] {description}"
  4. Enforce 20–120 token length (truncate or mark as too short)
  5. Deduplicate on (title, description) hash
"""

import hashlib
import re
import unicodedata
from typing import Optional


class Preprocessor:
    SEP_TOKEN = "[SEP]"
    MIN_TOKENS = 20
    MAX_TOKENS = 120

    def __init__(self, tokenizer, min_length: int = MIN_TOKENS, max_length: int = MAX_TOKENS):
        self.tokenizer = tokenizer
        self.min_length = min_length
        self.max_length = max_length
        self._seen_hashes: set = set()

    def run(self, product: dict) -> Optional[dict]:
        """
        Process a single product dict.

        Args:
            product: { "title": str, "description": str, ... }

        Returns:
            Enriched state dict, or None if the product is a duplicate
            or too short after cleaning.
        """
        title = self._clean_text(product.get("title", ""))
        description = self._clean_text(product.get("description", ""))

        # Deduplicate
        fingerprint = self._hash(title, description)
        if fingerprint in self._seen_hashes:
            return None
        self._seen_hashes.add(fingerprint)

        # Concatenate
        combined = f"{title} {self.SEP_TOKEN} {description}".strip()

        # Tokenize and enforce length bounds
        tokens = self.tokenizer(
            combined,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        if len(tokens["input_ids"]) < self.min_length:
            return None

        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "token_type_ids": tokens.get("token_type_ids"),
            "raw_title": title,
            "raw_description": description,
            "combined_text": combined,
            "fingerprint": fingerprint,
            # Pass-through fields for enrichment
            "vendor": product.get("vendor", ""),
            "product_type": product.get("product_type", ""),
        }

    def reset_dedup(self):
        """Clear the deduplication set — call between train/val/test splits."""
        self._seen_hashes.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_text(text: str) -> str:
        if not isinstance(text, str):
            text = str(text) if text else ""

        # Strip HTML tags
        text = re.sub(r"<[^>]+>", " ", text)

        # Decode HTML entities (&amp; &lt; etc.)
        text = re.sub(r"&amp;", "&", text)
        text = re.sub(r"&lt;", "<", text)
        text = re.sub(r"&gt;", ">", text)
        text = re.sub(r"&quot;", '"', text)
        text = re.sub(r"&#?[a-zA-Z0-9]+;", " ", text)

        # Normalize unicode to ASCII-compatible form
        text = unicodedata.normalize("NFKC", text)

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    @staticmethod
    def _hash(title: str, description: str) -> str:
        content = f"{title}|||{description}".encode("utf-8")
        return hashlib.md5(content).hexdigest()
