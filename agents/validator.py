"""
Validator Agent
---------------
Checks minimum tag coverage (≥1 label) and per-label confidence floor.
Flags low-confidence outputs and routes rejects back through the Orchestrator.
Implemented in Phase 5.
"""


class ValidatorAgent:
    def __init__(self, min_tags: int = 1, confidence_floor: float = 0.50):
        self.min_tags = min_tags
        self.confidence_floor = confidence_floor

    def run(self, state: dict) -> dict:
        raise NotImplementedError("Phase 5")
