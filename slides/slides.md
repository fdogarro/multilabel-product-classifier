---
theme: default
background: '#ffffff'
highlighter: shiki
lineNumbers: false
colorSchema: light
fonts:
  sans: 'Inter'
  mono: 'Fira Code'
title: 'Shopify Product Tagger — Multi-Agent LLM Pipeline'
info: CSCI E-222 · Spring 2026
css: |
  :root {
    --slidev-theme-background: #ffffff;
    --slidev-theme-color: #111111;
  }
  .slidev-layout {
    background: #ffffff !important;
    color: #111111 !important;
  }
  h1, h2, h3, h4, p, li, td, th {
    color: #111111 !important;
  }
  code {
    color: #1e1e1e !important;
  }
---

<img src="./harvard_shield.png" style="height:80px; margin-bottom:16px;" />

# Shopify Product Tagger
## A Confidence-Calibrated Multi-Agent LLM Pipeline

**CSCI E-222 · Spring 2026**

Harvard Extension School

Felicia D. O'Garro

---

# The Problem

Merchants listing products on Shopify must manually assign taxonomy tags.

- **Slow** — tagging hundreds of SKUs by hand takes hours
- **Inconsistent** — different people apply different tags to similar products
- **Inaccurate** — wrong tags hurt search ranking and discoverability

**Goal:** Build an automated pipeline that assigns the correct Shopify taxonomy tags to any product given its title and description.

---

# Why Multi-Agent?

No single model handles all cases well.

| Scenario | Challenge |
|----------|-----------|
| Common products | Need speed and low cost |
| Ambiguous products | Need deeper language understanding |
| Out-of-distribution products | Need general world knowledge |

**Solution:** Route each product to the most appropriate model based on confidence — fast when confident, powerful when not.

---

# System Architecture

<div style="font-family: monospace; font-size: 0.85em; line-height: 1.8; background: #f8f8f8; padding: 16px; border-radius: 8px;">
Product (title + description)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br/>
&nbsp;&nbsp;[1. Preprocessor] — tokenize, deduplicate, enforce length<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br/>
&nbsp;&nbsp;[2. BERT Inference] — always runs first; produces confidence scores<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br/>
&nbsp;&nbsp;[3. Model Selector] — routes based on max confidence<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\<br/>
&nbsp;BERT&nbsp;&nbsp;&nbsp;LoRA-Mistral&nbsp;&nbsp;LLM API<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br/>
&nbsp;&nbsp;[4. Tagger] — selected model produces final predictions<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br/>
&nbsp;&nbsp;[5. Validator Agent] — checks tag coverage and confidence floor<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br/>
&nbsp;&nbsp;Predicted Tags + Routing Metadata
</div>

---

# Five Agents

| Agent | Role |
|-------|------|
| **Preprocessor** | Tokenizes, deduplicates, enforces min/max length |
| **BERT Classifier** | Fast inference + confidence estimation (always runs) |
| **Model Selector** | Routes based on max confidence score |
| **LLM API Tagger** | Few-shot Claude Haiku / GPT-4o-mini fallback |
| **Validator** | Flags low-confidence or tag-sparse predictions |

BERT serves dual purpose: it is both the confidence estimator and the primary tagger. LoRA-Mistral and the LLM API are fallbacks that only activate when BERT is not confident enough.

---

# Routing Policy

```python
if max_confidence >= 0.75:
    route → BERT          # high confidence — fast, free

elif max_confidence >= 0.50:
    route → LoRA-Mistral  # ambiguous — deeper model

else:
    route → LLM API       # out-of-distribution — general knowledge
```

Cost and latency increase left to right. The pipeline is designed to keep the majority of traffic on the cheapest, fastest path.

---

# Methods: Why These Models?

**BERT (fast path)**
- 110M parameters, fine-tuned end-to-end on product text
- `BCEWithLogitsLoss` with per-label `pos_weight` for class imbalance
- Per-label threshold tuning on validation set (0.50–0.90)

**LoRA-Mistral-7B (ambiguous path)**
- QLoRA: base weights quantized to 4-bit, only low-rank adapters trained
- ~8M trainable parameters on top of frozen 7B base
- Feasible on a single A100 GPU — no multi-GPU setup required

**Few-Shot LLM API (out-of-distribution fallback)**
- 5-shot prompt with full taxonomy + example products
- Claude Haiku or GPT-4o-mini — general world knowledge
- Only called when both fine-tuned models are not confident

---

# Data

**Source:** McAuley Lab Amazon Reviews 2023 (HuggingFace Datasets)

| Split | Products |
|-------|----------|
| Train | 61,102 |
| Val   | 7,638 |
| Test  | 7,638 |
| **Total** | **76,378** |

- 21 Amazon categories → 30 Shopify taxonomy labels
- Loaded via `streaming=True` — no disk overhead
- Avg labels per product: **1.08**
- Label coverage: **24 / 30** labels represented

---

# Taxonomy — 30 Labels

```
electronics   computers     phones        audio
cameras       tv-video      clothing      shoes
jewelry       bags          beauty        health
sports        outdoors      home-kitchen  furniture
garden        tools-hardware automotive   toys
baby          books         music         video-games
office        pet-supplies  food-grocery  arts-crafts
industrial    software
```

Designed to map directly to Shopify's standard product taxonomy.

---

# Phase 2 — BERT Baseline Results

| Metric | Score |
|--------|-------|
| **Micro-F1** | **0.856** |
| **Macro-F1** | **0.652** |
| Hamming Loss | 0.011 |
| Coverage Error | 1.528 |

**Inference latency — single sample, A100**

| mean | median | p95 | p99 |
|------|--------|-----|-----|
| 15.3 ms | 15.3 ms | 15.7 ms | 15.9 ms |

---

# BERT — Training Curves

<img src="./bert_training_curves.png" style="max-height: 340px; display: block; margin: 0 auto;" />

Loss and validation F1 over 5 epochs. Micro-F1 peaks at epoch 5 — no overfitting observed.

---

# BERT — Per-Label F1

<img src="./per_label_f1.png" style="max-height: 360px; display: block; margin: 0 auto;" />

Six labels score 0.0 due to insufficient training representation. Covered labels perform strongly, with seven labels above F1 = 0.93.

---

# Phase 3 — LoRA-Mistral-7B Results

| Metric | BERT | LoRA-Mistral | Δ |
|--------|------|-------------|---|
| **Micro-F1** | 0.856 | **0.868** | +0.012 |
| **Macro-F1** | 0.652 | **0.662** | +0.010 |
| Hamming Loss | 0.011 | **0.010** | -0.001 |
| Coverage Error | 1.528 | **1.478** | -0.050 |

<img src="./model_comparison.png" style="max-height: 200px; display: block; margin: 8px auto 0;" />

---

# Latency Comparison

| Model | Mean | p95 | vs BERT |
|-------|------|-----|---------|
| BERT | 15.3 ms | 15.7 ms | 1× |
| LoRA-Mistral | 262.7 ms | 267.1 ms | **17× slower** |
| LLM API | ~800 ms + network | — | ~50× slower (est.) |

**This tradeoff directly motivates the routing layer.**

- BERT handles high-confidence products at ~15ms
- LoRA-Mistral activates only for ambiguous products
- LLM API reserved for out-of-distribution cases

---

# Phase 5 — Routing Results

<img src="./routing_decisions.png" style="max-height: 260px; display: block; margin: 0 auto;" />

Routing on 497 demo products — BERT achieved max confidence ≥ 0.75 on all in-distribution products. Validator flagged: **0 / 497**.

---

# Example Predictions

| Product | Predicted Tags | Model |
|---------|---------------|-------|
| Gotoh Modern Bridge for Tele | music | BERT |
| JOYIN 72 Pcs Animal Easter Eggs | toys | BERT |
| GREENARK Replacement Canon CLI-251 | office | BERT |
| Top Collection Enchanted Garden | home-kitchen, furniture | BERT |
| KZ ZS6 HiFi Quad Driver Earphones | music | BERT |
| Penny Rose Paper Dolls Bakery | toys, arts-crafts | BERT |
| Wearable4U H&K VP9 Airsoft | sports, outdoors | BERT |

---

# Interactive Gradio Demo

**Live product tagger with full routing visibility**

- Input: product title + description
- Output: predicted tags with confidence scores
- Routing Decision panel: which model ran, why, flagged status

Built with `gradio.Interface` + `share=True` for a public URL in Colab.

---

# Strengths, Limitations & Future Work

**Strengths**
- Strong performance on covered labels — seven labels above F1 = 0.93
- LoRA-Mistral outperforms BERT on all metrics with only 3 epochs
- Cost-efficient by design — most traffic stays on the free BERT path

**Limitations**
- 6 of 30 labels have zero training coverage → F1 = 0.0
- Routing thresholds manually tuned — could be learned
- LoRA and LLM API paths not exercised on in-distribution test data

**Future Work**
- Active learning to close label coverage gap
- Shopify OAuth integration — tag products directly via API
- Merchant feedback loop for continual improvement

---

# Conclusion

A confidence-calibrated multi-agent pipeline that:

1. **Preprocesses** product text and deduplicates efficiently
2. **Classifies** with BERT at 85.6% micro-F1 and 15ms latency
3. **Designed to route** ambiguous cases to LoRA-Mistral or LLM API
4. **Validates** outputs before returning to the merchant
5. **Scales** cost-efficiently — most traffic never hits a paid API

**GitHub:** `github.com/fdogarro/shopify-tagger`

---

# Video Presentation

**Watch the full 7–15 minute demo and walkthrough:**

`[INSERT VIDEO URL HERE]`

*(YouTube / Zoom / Panopto)*

**GitHub:** `github.com/fdogarro/shopify-tagger`
