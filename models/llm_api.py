"""
Few-Shot LLM API Client
------------------------
Zero-training-cost baseline and OOD fallback.
Supports GPT-4o-mini (OpenAI) and Claude Haiku (Anthropic).
Responses are cached to avoid redundant API calls during evaluation.
Implemented in Phase 4.
"""


class LLMApiTagger:
    def __init__(self, provider: str = "anthropic", model: str = "claude-haiku-4-5-20251001",
                 num_labels: int = 30, cache_responses: bool = True):
        self.provider = provider
        self.model = model
        self.num_labels = num_labels
        self.cache_responses = cache_responses
        self._cache: dict = {}

    def predict(self, product: dict, few_shot_examples: list) -> dict:
        raise NotImplementedError("Phase 4")

    def _build_prompt(self, product: dict, few_shot_examples: list) -> str:
        raise NotImplementedError("Phase 4")

    def _parse_response(self, response: str) -> list[int]:
        raise NotImplementedError("Phase 4")
