"""
Few-Shot LLM API Tagger
------------------------
Zero-training-cost baseline and OOD fallback for the live pipeline.
Supports GPT-4o-mini (OpenAI) and Claude Haiku (Anthropic).

Design decisions:
  - 5-shot prompt with diverse label combinations (single, multi, edge-case)
  - Structured JSON output: model returns a list of tag strings
  - All responses are MD5-cached to disk — safe to re-run evaluation
  - Unknown tags returned by the model are logged as hallucinations
  - Refusals and malformed JSON are logged as hard failures

Cost model (April 2026 approximate pricing):
  GPT-4o-mini:   $0.150 / 1M input tokens  |  $0.600 / 1M output tokens
  Claude Haiku:  $0.080 / 1M input tokens  |  $0.250 / 1M output tokens
"""

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Pricing (per 1M tokens, USD)
# -----------------------------------------------------------------------
PRICING = {
    "gpt-4o-mini": {
        "input_per_1m":  0.150,
        "output_per_1m": 0.600,
    },
    "claude-haiku-4-5-20251001": {
        "input_per_1m":  0.080,
        "output_per_1m": 0.250,
    },
}

# -----------------------------------------------------------------------
# Few-shot examples — 5 examples covering diverse label combinations
# -----------------------------------------------------------------------
FEW_SHOT_EXAMPLES = [
    {
        "title":       "Sony WH-1000XM5 Wireless Noise Cancelling Headphones",
        "description": "Industry-leading noise cancellation with 30-hour battery life. "
                       "Foldable design, multipoint connection, and built-in Alexa.",
        "tags":        ["electronics", "audio"],
    },
    {
        "title":       "Nike Air Zoom Pegasus 41 Running Shoes",
        "description": "Responsive cushioning for everyday training. "
                       "Breathable mesh upper, React foam midsole, rubber outsole.",
        "tags":        ["shoes", "sports"],
    },
    {
        "title":       "CeraVe Moisturizing Cream + Vitamin C 1000mg Supplement Bundle",
        "description": "Daily moisturizer with ceramides and hyaluronic acid. "
                       "Includes 90-count vitamin C immune support capsules.",
        "tags":        ["beauty", "health"],
    },
    {
        "title":       "DEWALT 20V MAX Cordless Drill Combo Kit",
        "description": "Includes drill/driver, circular saw, and reciprocating saw. "
                       "Compact design for tight spaces. LED work light built in.",
        "tags":        ["tools-hardware"],
    },
    {
        "title":       "The Pragmatic Programmer: 20th Anniversary Edition",
        "description": "Classic software development guide covering topics from "
                       "career development to architectural techniques.",
        "tags":        ["books", "office"],
    },
]


# -----------------------------------------------------------------------
# Main class
# -----------------------------------------------------------------------

class LLMApiTagger:
    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-haiku-4-5-20251001",
        num_labels: int = 30,
        taxonomy_path: str = "data/taxonomy.json",
        cache_dir: str = "data/raw/llm_cache",
        request_delay: float = 0.5,
    ):
        self.provider = provider.lower()
        self.model = model
        self.num_labels = num_labels
        self.request_delay = request_delay

        # Load taxonomy
        with open(taxonomy_path) as f:
            taxonomy = json.load(f)
        self.labels = [l["tag"] for l in taxonomy["labels"]]
        self._tag_to_id = {l["tag"]: l["id"] for l in taxonomy["labels"]}

        # Cache
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Lazily initialized client
        self._client = None

        # Telemetry
        self.total_input_tokens  = 0
        self.total_output_tokens = 0
        self.failure_log: list[dict] = []

    def predict(self, product: dict, few_shot_examples: list[dict] = None) -> dict:
        """
        Args:
            product: { "title": str, "description": str }
            few_shot_examples: override default 5-shot set

        Returns:
            {
              "labels":        list[int]  — binary vector length num_labels,
              "tags":          list[str]  — predicted tag strings,
              "raw_response":  str        — model's raw text output,
              "hallucinations": list[str] — tags returned but not in taxonomy,
              "failed":        bool       — True if response was unparseable,
              "input_tokens":  int,
              "output_tokens": int,
            }
        """
        examples = few_shot_examples or FEW_SHOT_EXAMPLES
        prompt   = self._build_prompt(product, examples)
        cache_key = self._cache_key(prompt)
        cached    = self._load_cache(cache_key)

        if cached:
            raw_response   = cached["raw_response"]
            input_tokens   = cached.get("input_tokens", 0)
            output_tokens  = cached.get("output_tokens", 0)
        else:
            raw_response, input_tokens, output_tokens = self._call_api(prompt)
            self._save_cache(cache_key, {
                "raw_response":  raw_response,
                "input_tokens":  input_tokens,
                "output_tokens": output_tokens,
            })
            time.sleep(self.request_delay)

        self.total_input_tokens  += input_tokens
        self.total_output_tokens += output_tokens

        tags, hallucinations, failed = self._parse_response(raw_response, product)
        label_vector = self._tags_to_vector(tags)

        return {
            "labels":          label_vector,
            "tags":            tags,
            "raw_response":    raw_response,
            "hallucinations":  hallucinations,
            "failed":          failed,
            "input_tokens":    input_tokens,
            "output_tokens":   output_tokens,
        }

    def cost_estimate(self, n_products: int) -> dict:
        """
        Estimate cost to tag n_products using current token averages.
        Falls back to prompt-length estimate if no calls made yet.
        """
        if self.total_input_tokens > 0:
            avg_in  = self.total_input_tokens  / max(1, len(self.failure_log) + 1)
            avg_out = self.total_output_tokens / max(1, len(self.failure_log) + 1)
        else:
            # Conservative estimate: ~600 input tokens (prompt + product), ~30 output tokens
            avg_in, avg_out = 600, 30

        pricing = PRICING.get(self.model, {"input_per_1m": 0.15, "output_per_1m": 0.60})
        cost = (
            (avg_in  * n_products / 1_000_000) * pricing["input_per_1m"]
            + (avg_out * n_products / 1_000_000) * pricing["output_per_1m"]
        )
        return {
            "model":           self.model,
            "n_products":      n_products,
            "avg_input_tokens": round(avg_in),
            "avg_output_tokens": round(avg_out),
            "estimated_cost_usd": round(cost, 4),
        }

    def failure_report(self) -> dict:
        total = len(self.failure_log)
        hallucination_events = [f for f in self.failure_log if f.get("hallucinations")]
        parse_failures       = [f for f in self.failure_log if f.get("failed")]
        return {
            "total_failures":     total,
            "hallucination_count": len(hallucination_events),
            "parse_failure_count": len(parse_failures),
            "examples":           self.failure_log[:10],
        }

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self, product: dict, examples: list[dict]) -> str:
        taxonomy_str = ", ".join(self.labels)

        shots = []
        for ex in examples:
            shots.append(
                f"Title: {ex['title']}\n"
                f"Description: {ex['description']}\n"
                f"Tags: {json.dumps(ex['tags'])}"
            )
        shots_str = "\n\n".join(shots)

        return (
            f"You are a product taxonomy classifier for an ecommerce store.\n"
            f"Given a product title and description, identify which tags from the list below apply.\n"
            f"Return ONLY a JSON array of applicable tag strings — no explanation, no markdown.\n"
            f"If no tags apply, return an empty array [].\n\n"
            f"Available tags: [{taxonomy_str}]\n\n"
            f"--- Examples ---\n\n"
            f"{shots_str}\n\n"
            f"--- Product to classify ---\n\n"
            f"Title: {product.get('title', '')}\n"
            f"Description: {product.get('description', '')}\n"
            f"Tags:"
        )

    # ------------------------------------------------------------------
    # API calls
    # ------------------------------------------------------------------

    def _call_api(self, prompt: str) -> tuple[str, int, int]:
        """Returns (raw_text, input_tokens, output_tokens)."""
        if self.provider == "anthropic":
            return self._call_anthropic(prompt)
        elif self.provider == "openai":
            return self._call_openai(prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _call_anthropic(self, prompt: str) -> tuple[str, int, int]:
        import anthropic
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

        message = self._client.messages.create(
            model=self.model,
            max_tokens=128,
            messages=[{"role": "user", "content": prompt}],
        )
        raw    = message.content[0].text.strip()
        in_tok = message.usage.input_tokens
        out_tok = message.usage.output_tokens
        return raw, in_tok, out_tok

    def _call_openai(self, prompt: str) -> tuple[str, int, int]:
        import openai
        if self._client is None:
            self._client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=128,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw     = response.choices[0].message.content.strip()
        in_tok  = response.usage.prompt_tokens
        out_tok = response.usage.completion_tokens
        return raw, in_tok, out_tok

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(
        self, raw: str, product: dict
    ) -> tuple[list[str], list[str], bool]:
        """
        Returns (valid_tags, hallucinated_tags, failed).
        failed=True means the response couldn't be parsed at all.
        """
        # Extract JSON array — handle markdown code fences and trailing text
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if not match:
            self.failure_log.append({
                "product": product.get("title", ""),
                "raw":     raw,
                "failed":  True,
                "hallucinations": [],
            })
            return [], [], True

        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            self.failure_log.append({
                "product": product.get("title", ""),
                "raw":     raw,
                "failed":  True,
                "hallucinations": [],
            })
            return [], [], True

        if not isinstance(parsed, list):
            return [], [], True

        valid, hallucinations = [], []
        for tag in parsed:
            tag = str(tag).strip().lower().replace(" ", "-")
            if tag in self._tag_to_id:
                valid.append(tag)
            else:
                hallucinations.append(tag)

        if hallucinations:
            self.failure_log.append({
                "product":       product.get("title", ""),
                "raw":           raw,
                "failed":        False,
                "hallucinations": hallucinations,
            })

        return valid, hallucinations, False

    def _tags_to_vector(self, tags: list[str]) -> list[int]:
        vector = [0] * self.num_labels
        for tag in tags:
            idx = self._tag_to_id.get(tag)
            if idx is not None:
                vector[idx] = 1
        return vector

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_key(self, prompt: str) -> str:
        return hashlib.md5(f"{self.model}:{prompt}".encode()).hexdigest()

    def _load_cache(self, key: str) -> Optional[dict]:
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None

    def _save_cache(self, key: str, data: dict):
        with open(self.cache_dir / f"{key}.json", "w") as f:
            json.dump(data, f)
