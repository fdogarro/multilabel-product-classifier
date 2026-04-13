"""
Tagger Agent
------------
Dispatches to the model assigned by ModelSelector.
Returns a binary label vector and per-label confidence scores.
Implemented in Phase 5.
"""


class TaggerAgent:
    def __init__(self, bert_model=None, lora_model=None, llm_client=None):
        self.bert_model = bert_model
        self.lora_model = lora_model
        self.llm_client = llm_client

    def run(self, state: dict) -> dict:
        raise NotImplementedError("Phase 5")
