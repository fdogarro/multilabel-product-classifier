# Multi-Label Product Classifier — Multi-Agent LLM Pipeline

**CSCI E-222 · Harvard Extension School · Spring 2026**

A confidence-calibrated multi-agent pipeline for automatic product taxonomy classification in e-commerce. Products are routed through BERT (high confidence), LoRA-Mistral-7B (ambiguous), or a few-shot LLM API call (out-of-distribution) based on real-time confidence scores.

---

## Quick Start (Google Colab — Recommended)

1. Open [`notebooks/shopify_tagger_full_v7.ipynb`](notebooks/shopify_tagger_full_v7.ipynb) in Google Colab
2. Set runtime to A100 GPU: `Runtime → Change runtime type → A100 GPU`
3. Run the **first setup cell** (clones repo and installs dependencies)
4. Add your Anthropic and OpenAI API keys when prompted in Phase 4
5. Run all cells top to bottom

> **Runtime restart recovery:** If the Colab session disconnects after training, run the Phase 5 Recovery cell to reload all state from disk without re-running Phases 1–3.

---

## Local Setup

```bash
git clone https://github.com/fdogarro/multilabel-product-classifier.git
cd multilabel-product-classifier
pip install -r requirements.txt
jupyter notebook notebooks/shopify_tagger_full_v7.ipynb
```

**Python:** 3.12 · **GPU:** NVIDIA A100 recommended for Phase 3 (LoRA training)

---

## Data

The Amazon Reviews 2023 dataset is downloaded automatically via HuggingFace Datasets using streaming mode — no manual download required.

**Source:** https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023

No data files need to be provided. The pipeline loads ~80,000 products across 21 Amazon categories at runtime.

---

## Pipeline Phases

| Phase | Description | Est. Runtime (A100) |
|-------|-------------|---------------------|
| 1 | Data loading, preprocessing, splits | ~25 min |
| 2 | BERT fine-tuning (5 epochs) | ~25 min |
| 3 | LoRA-Mistral-7B training (3 epochs) | ~4 hours |
| 4 | LLM API evaluation (Claude Haiku + GPT-4o-mini) | ~15 min |
| 5 | Orchestrator + Gradio demo | ~10 min |

---

## Checkpoints

Trained checkpoints (BERT and LoRA-Mistral) were saved to the submitter's Google Drive during training. To reproduce results from scratch, run Phases 1–3 in order before Phase 5. See the **Note on Checkpoints** cell in the notebook for full details.

---

## API Keys

Phase 4 requires API keys for Claude Haiku (Anthropic) and GPT-4o-mini (OpenAI). Add them when prompted in the notebook cells. Never commit API keys to the repository.

---

## Repository Structure

```
notebooks/
  shopify_tagger_full_v7.ipynb  ← main entry point
data/
  taxonomy.json                 ← 30-label product taxonomy
  amazon_loader.py
  dataset_builder.py
  taxonomy_mapper.py
agents/                         ← Preprocessor, ModelSelector, Orchestrator, Validator
models/                         ← BertMultiLabelClassifier, LoraMistralClassifier
report/
  report.md                     ← full project report
  images/                       ← embedded figures
requirements.txt
```

---

## Results

| Model | Micro-F1 | Macro-F1 | Latency |
|-------|----------|----------|---------|
| BERT (bert-base-uncased) | 0.856 | 0.652 | 15.3 ms |
| LoRA-Mistral-7B (v0.2) | 0.868 | 0.662 | 262.7 ms |

Macro-F1 across the 23 labels with training coverage: **0.850** (BERT) / **0.864** (LoRA-Mistral).

