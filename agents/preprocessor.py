"""
Preprocessor Agent
------------------
Tokenizes, cleans, and normalizes product text.
Enforces 20–120 token length; concatenates title and description.
Implemented in Phase 1.
"""


class Preprocessor:
    def __init__(self, tokenizer, max_length: int = 120):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def run(self, product: dict) -> dict:
        raise NotImplementedError("Phase 1")
