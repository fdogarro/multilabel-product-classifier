"""
Ablation Studies
----------------
Isolates each agent's contribution to overall pipeline performance.

Conditions:
  1. Full pipeline (baseline)
  2. Enrichment disabled — Preprocessor feeds directly to ModelSelector
  3. Validator bypassed — Tagger output goes directly to output
  4. ModelSelector replaced with fixed BERT routing
Implemented in Phase 5.
"""


def run_ablations(dataset, pipeline_config: dict) -> dict:
    raise NotImplementedError("Phase 5")
