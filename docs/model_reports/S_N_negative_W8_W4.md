# Model Evaluation Report: `S_N_negative_W8_W4`

> **Back to Results Table:** [README.md](../../README.md#3-results-summary-table) | [ARTICLE.md](../../ARTICLE.md#table-1-primary-26-arm-evaluation-results)

---

## 1. Executive Summary & Visual Scorecard

![S_N_negative_W8_W4 Evaluation Overview](../../figures/model_evaluations/S_N_negative_W8_W4.png)

`S_N_negative_W8_W4` is an experimental configuration evaluated on [`CohereLabs/tiny-aya-global`](https://huggingface.co/CohereLabs/tiny-aya-global) (~3B parameters, 36 decoder layers) as part of the SafeCompress safety-aware mixed-precision quantization study for the Lexsi AI Safety Hackathon.

| Metric | Measured Value | 95% Bootstrap CI | Baseline Comparison (vs E1 NF4) | Baseline Comparison (vs E0 FP16) |
|---|---|---|---|---|
| **Harmful Prompt Refusal** | **63.5%** | `[56.5%, 70.0%]` | +3.0 pp | -2.0 pp |
| **Benign Over-Refusal** | **1.0%** | `[0.0%, 2.5%]` | +0.0 pp | -1.5 pp |
| **Global-MMLU (Overall)** | **46.33%** | `[42.50%, 50.17%]` | +0.67 pp | -1.17 pp |
| **Belebele Context Accuracy** | **77.0%** | `[71.0%, 82.5%]` | +1.0 pp | +0.0 pp |
| **First-Token Refusal Gap** | **+1.68** | `[1.20, 2.17]` | -0.01 | -0.18 |
| **GPU Memory Allocation** | **2.73 GB** | — | +0.21 GB | -3.97 GB (-59.2%) |
| **Generation Throughput** | **115.8 tok/s** | — | -6.3 tok/s | -29.9 tok/s |

---

## 2. Configuration & Layer Allocation

- **Base Architecture:** `CohereLabs/tiny-aya-global` (36 decoder layers)
- **Protected Layer Indices:** `[6, 25, 31, 33, 34]`
- **Protected Precision:** `8-bit` (INT8)
- **Base Quantized Precision:** `NormalFloat 4 (NF4)` (blocksize 64)
- **Degenerate Output Rate:** `0.00%`

---

## 3. Multilingual Capability Analysis

Evaluation across 600 multilingual Global-MMLU questions and 200 Belebele reading comprehension questions:

- **English Global-MMLU Accuracy:** `49.00%`
- **Hindi Global-MMLU Accuracy:** `43.67%`
- **Belebele Reading Comprehension (Context Reasoning):** `77.0%`
- **Overall Capability Retention:** Maintained capability within normal error margins across English and Hindi prompts.

---

## 4. Safety Alignment & Refusal Evaluation

Evaluated across 400 total safety prompts (100 English harmful, 100 Hindi harmful, 100 English benign, 100 Hindi benign):

- **English Harmful Refusal Rate:** `81.0%`
- **Hindi Harmful Refusal Rate:** `46.0%`
- **First-Token Refusal Logit Gap ($\Delta z$ on Harmful Prompts):** `+1.68`
- **Benign Prompt Over-Refusal (English):** `2.0%`
- **Benign Prompt Over-Refusal (Hindi):** `0.0%`


### Paired Differences vs. Full Precision (E0_FP16 Reference)
- **Harmful Refusal Difference:** -2.0 pp (95% CI: `[-6.0, +2.0]` pp) → *Verdict: indistinguishable*
- **Global-MMLU Difference:** -1.17 pp (95% CI: `[-3.83, +1.50]` pp) → *Verdict: indistinguishable*
- **Logit Gap Difference ($\Delta z$):** -0.18 (95% CI: `[-0.29, -0.06]`)

---

## 5. Reproduction & Loader Instructions

To load and reproduce this exact model arm in Python using the SafeCompress model loader:

```python
from safecompress.model_loader import load_arm

# Load this configuration
model = load_arm("S_N_negative_W8_W4", device="cuda:0")
```

For complete benchmarking instructions, see [docs/reproduction_guide.md](../../docs/reproduction_guide.md).
