# Model Evaluation Report: `S_X_Cons5_W8_W4`

> **Back to Results Table:** [README.md](../../README.md#3-results-summary-table) | [ARTICLE.md](../../ARTICLE.md#table-1-primary-26-arm-evaluation-results)

---

## 1. Executive Summary & Visual Scorecard

![S_X_Cons5_W8_W4 Evaluation Overview](../../figures/model_evaluations/S_X_Cons5_W8_W4.png)

`S_X_Cons5_W8_W4` is an experimental configuration evaluated on [`CohereLabs/tiny-aya-global`](https://huggingface.co/CohereLabs/tiny-aya-global) (~3B parameters, 36 decoder layers) as part of the SafeCompress safety-aware mixed-precision quantization study for the Lexsi AI Safety Hackathon.

| Metric | Measured Value | 95% Bootstrap CI | Baseline Comparison (vs E1 NF4) | Baseline Comparison (vs E0 FP16) |
|---|---|---|---|---|
| **Harmful Prompt Refusal** | **65.5%** | `[59.0%, 72.0%]` | +5.0 pp | +0.0 pp |
| **Benign Over-Refusal** | **1.5%** | `[0.0%, 3.5%]` | +0.5 pp | -1.0 pp |
| **Global-MMLU (Overall)** | **46.33%** | `[42.33%, 50.17%]` | +0.67 pp | -1.17 pp |
| **Belebele Context Accuracy** | **75.0%** | `[69.0%, 81.0%]` | -1.0 pp | -2.0 pp |
| **First-Token Refusal Gap** | **+2.26** | `[1.80, 2.74]` | +0.57 | +0.40 |
| **GPU Memory Allocation** | **2.71 GB** | — | +0.19 GB | -3.98 GB (-59.5%) |
| **Generation Throughput** | **104.3 tok/s** | — | -17.9 tok/s | -41.4 tok/s |

### Hugging Face Artifact Link
- **Hugging Face Repository:** [`cybertronak/safecompress-tinyaya-recipes-loader`](https://huggingface.co/cybertronak/safecompress-tinyaya-recipes-loader)
- **Deployment Format:** Loader

---

## 2. Configuration & Layer Allocation

- **Base Architecture:** `CohereLabs/tiny-aya-global` (36 decoder layers)
- **Protected Layer Indices:** `[0, 7, 14, 16, 35]`
- **Protected Precision:** `8-bit` (INT8)
- **Base Quantized Precision:** `NormalFloat 4 (NF4)` (blocksize 64)
- **Degenerate Output Rate:** `0.25%`

---

## 3. Multilingual Capability Analysis

Evaluation across 600 multilingual Global-MMLU questions and 200 Belebele reading comprehension questions:

- **English Global-MMLU Accuracy:** `50.33%`
- **Hindi Global-MMLU Accuracy:** `42.33%`
- **Belebele Reading Comprehension (Context Reasoning):** `75.0%`
- **Overall Capability Retention:** Maintained capability within normal error margins across English and Hindi prompts.

---

## 4. Safety Alignment & Refusal Evaluation

Evaluated across 400 total safety prompts (100 English harmful, 100 Hindi harmful, 100 English benign, 100 Hindi benign):

- **English Harmful Refusal Rate:** `85.0%`
- **Hindi Harmful Refusal Rate:** `46.0%`
- **First-Token Refusal Logit Gap ($\Delta z$ on Harmful Prompts):** `+2.26`
- **Benign Prompt Over-Refusal (English):** `3.0%`
- **Benign Prompt Over-Refusal (Hindi):** `0.0%`


### Paired Differences vs. Full Precision (E0_FP16 Reference)
- **Harmful Refusal Difference:** +0.0 pp (95% CI: `[-4.0, +4.0]` pp) → *Verdict: indistinguishable*
- **Global-MMLU Difference:** -1.17 pp (95% CI: `[-3.67, +1.33]` pp) → *Verdict: indistinguishable*
- **Logit Gap Difference ($\Delta z$):** +0.40 (95% CI: `[+0.29, +0.52]`)

---

## 5. Reproduction & Loader Instructions

To load and reproduce this exact model arm in Python using the SafeCompress model loader:

```python
from safecompress.model_loader import load_arm

# Load this configuration
model = load_arm("S_X_Cons5_W8_W4", device="cuda:0")
```

For complete benchmarking instructions, see [docs/reproduction_guide.md](../../docs/reproduction_guide.md).
