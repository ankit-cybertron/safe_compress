# Model Evaluation Report: `S_A_importance_W8_W4`

> **Back to Results Table:** [README.md](../../README.md#3-results-summary-table) | [ARTICLE.md](../../ARTICLE.md#table-1-primary-26-arm-evaluation-results)

---

## 1. Executive Summary & Visual Scorecard

![S_A_importance_W8_W4 Evaluation Overview](../../figures/model_evaluations/S_A_importance_W8_W4.png)

`S_A_importance_W8_W4` is an experimental configuration evaluated on [`CohereLabs/tiny-aya-global`](https://huggingface.co/CohereLabs/tiny-aya-global) (~3B parameters, 36 decoder layers) as part of the SafeCompress safety-aware mixed-precision quantization study for the Lexsi AI Safety Hackathon.

| Metric | Measured Value | 95% Bootstrap CI | Baseline Comparison (vs E1 NF4) | Baseline Comparison (vs E0 FP16) |
|---|---|---|---|---|
| **Harmful Prompt Refusal** | **63.0%** | `[56.5%, 69.5%]` | +2.5 pp | -2.5 pp |
| **Benign Over-Refusal** | **1.5%** | `[0.0%, 3.5%]` | +0.5 pp | -1.0 pp |
| **Global-MMLU (Overall)** | **45.83%** | `[42.00%, 49.67%]` | +0.17 pp | -1.67 pp |
| **Belebele Context Accuracy** | **76.5%** | `[70.5%, 82.0%]` | +0.5 pp | -0.5 pp |
| **First-Token Refusal Gap** | **+1.51** | `[1.05, 1.97]` | -0.18 | -0.35 |
| **GPU Memory Allocation** | **2.73 GB** | — | +0.21 GB | -3.97 GB (-59.2%) |
| **Generation Throughput** | **113.6 tok/s** | — | -8.5 tok/s | -32.1 tok/s |

---

## 2. Configuration & Layer Allocation

- **Base Architecture:** `CohereLabs/tiny-aya-global` (36 decoder layers)
- **Protected Layer Indices:** `[0, 5, 17, 19, 22]`
- **Protected Precision:** `8-bit` (INT8)
- **Base Quantized Precision:** `NormalFloat 4 (NF4)` (blocksize 64)
- **Degenerate Output Rate:** `0.00%`

---

## 3. Multilingual Capability Analysis

Evaluation across 600 multilingual Global-MMLU questions and 200 Belebele reading comprehension questions:

- **English Global-MMLU Accuracy:** `49.00%`
- **Hindi Global-MMLU Accuracy:** `42.67%`
- **Belebele Reading Comprehension (Context Reasoning):** `76.5%`
- **Overall Capability Retention:** Maintained capability within normal error margins across English and Hindi prompts.

---

## 4. Safety Alignment & Refusal Evaluation

Evaluated across 400 total safety prompts (100 English harmful, 100 Hindi harmful, 100 English benign, 100 Hindi benign):

- **English Harmful Refusal Rate:** `78.0%`
- **Hindi Harmful Refusal Rate:** `48.0%`
- **First-Token Refusal Logit Gap ($\Delta z$ on Harmful Prompts):** `+1.51`
- **Benign Prompt Over-Refusal (English):** `3.0%`
- **Benign Prompt Over-Refusal (Hindi):** `0.0%`


### Paired Differences vs. Full Precision (E0_FP16 Reference)
- **Harmful Refusal Difference:** -2.5 pp (95% CI: `[-6.5, +1.5]` pp) → *Verdict: indistinguishable*
- **Global-MMLU Difference:** -1.67 pp (95% CI: `[-4.00, +0.67]` pp) → *Verdict: indistinguishable*
- **Logit Gap Difference ($\Delta z$):** -0.35 (95% CI: `[-0.48, -0.22]`)

---

## 5. Reproduction & Loader Instructions

To load and reproduce this exact model arm in Python using the SafeCompress model loader:

```python
from safecompress.model_loader import load_arm

# Load this configuration
model = load_arm("S_A_importance_W8_W4", device="cuda:0")
```

For complete benchmarking instructions, see [docs/reproduction_guide.md](../../docs/reproduction_guide.md).
