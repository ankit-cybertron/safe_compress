# Model Evaluation Report: `S_X_C4_W8_W4`

> **Back to Results Table:** [README.md](../../README.md#3-results-summary-table) | [ARTICLE.md](../../ARTICLE.md#table-1-primary-26-arm-evaluation-results)

---

## 1. Executive Summary & Visual Scorecard

![S_X_C4_W8_W4 Evaluation Overview](../../figures/model_evaluations/S_X_C4_W8_W4.png)

`S_X_C4_W8_W4` is an experimental configuration evaluated on [`CohereLabs/tiny-aya-global`](https://huggingface.co/CohereLabs/tiny-aya-global) (~3B parameters, 36 decoder layers) as part of the SafeCompress safety-aware mixed-precision quantization study for the Lexsi AI Safety Hackathon.

| Metric | Measured Value | 95% Bootstrap CI | Baseline Comparison (vs E1 NF4) | Baseline Comparison (vs E0 FP16) |
|---|---|---|---|---|
| **Harmful Prompt Refusal** | **63.0%** | `[56.0%, 69.5%]` | +2.5 pp | -2.5 pp |
| **Benign Over-Refusal** | **2.0%** | `[0.5%, 4.0%]` | +1.0 pp | -0.5 pp |
| **Global-MMLU (Overall)** | **47.17%** | `[43.17%, 51.00%]` | +1.50 pp | -0.33 pp |
| **Belebele Context Accuracy** | **76.5%** | `[70.5%, 82.0%]` | +0.5 pp | -0.5 pp |
| **First-Token Refusal Gap** | **+2.27** | `[1.79, 2.75]` | +0.58 | +0.41 |
| **GPU Memory Allocation** | **2.70 GB** | — | +0.18 GB | -4.00 GB (-59.7%) |
| **Generation Throughput** | **104.7 tok/s** | — | -17.4 tok/s | -40.9 tok/s |

---

## 2. Configuration & Layer Allocation

- **Base Architecture:** `CohereLabs/tiny-aya-global` (36 decoder layers)
- **Protected Layer Indices:** `[7, 16, 27, 35]`
- **Protected Precision:** `8-bit` (INT8)
- **Base Quantized Precision:** `NormalFloat 4 (NF4)` (blocksize 64)
- **Degenerate Output Rate:** `0.00%`

---

## 3. Multilingual Capability Analysis

Evaluation across 600 multilingual Global-MMLU questions and 200 Belebele reading comprehension questions:

- **English Global-MMLU Accuracy:** `50.00%`
- **Hindi Global-MMLU Accuracy:** `44.33%`
- **Belebele Reading Comprehension (Context Reasoning):** `76.5%`
- **Overall Capability Retention:** Maintained capability within normal error margins across English and Hindi prompts.

---

## 4. Safety Alignment & Refusal Evaluation

Evaluated across 400 total safety prompts (100 English harmful, 100 Hindi harmful, 100 English benign, 100 Hindi benign):

- **English Harmful Refusal Rate:** `79.0%`
- **Hindi Harmful Refusal Rate:** `47.0%`
- **First-Token Refusal Logit Gap ($\Delta z$ on Harmful Prompts):** `+2.27`
- **Benign Prompt Over-Refusal (English):** `4.0%`
- **Benign Prompt Over-Refusal (Hindi):** `0.0%`


### Paired Differences vs. Full Precision (E0_FP16 Reference)
- **Harmful Refusal Difference:** -2.5 pp (95% CI: `[-6.0, +1.0]` pp) → *Verdict: indistinguishable*
- **Global-MMLU Difference:** -0.33 pp (95% CI: `[-3.00, +2.33]` pp) → *Verdict: indistinguishable*
- **Logit Gap Difference ($\Delta z$):** +0.41 (95% CI: `[+0.30, +0.52]`)

---

## 5. Reproduction & Loader Instructions

To load and reproduce this exact model arm in Python using the SafeCompress model loader:

```python
from safecompress.model_loader import load_arm

# Load this configuration
model = load_arm("S_X_C4_W8_W4", device="cuda:0")
```

For complete benchmarking instructions, see [docs/reproduction_guide.md](../../docs/reproduction_guide.md).
