"""
Enrichment Agent
----------------
Injects vendor/product_type metadata into the token sequence.
RAG retrieval is an optional extension (stubbed below).
Implemented in Phase 5.
"""


class EnrichmentAgent:
    def __init__(self, use_rag: bool = False):
        self.use_rag = use_rag

    def run(self, state: dict) -> dict:
        raise NotImplementedError("Phase 5")

    def _rag_retrieve(self, query: str) -> str:
        """Optional RAG extension — not implemented in baseline."""
        raise NotImplementedError("RAG extension")
