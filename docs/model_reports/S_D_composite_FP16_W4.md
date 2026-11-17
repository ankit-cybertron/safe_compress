# Model Evaluation Report: `S_D_composite_FP16_W4`

> **Back to Results Table:** [README.md](../../README.md#3-results-summary-table) | [ARTICLE.md](../../ARTICLE.md#table-1-primary-26-arm-evaluation-results)

---

## 1. Executive Summary & Visual Scorecard

![S_D_composite_FP16_W4 Evaluation Overview](../../figures/model_evaluations/S_D_composite_FP16_W4.png)

`S_D_composite_FP16_W4` is an experimental configuration evaluated on [`CohereLabs/tiny-aya-global`](https://huggingface.co/CohereLabs/tiny-aya-global) (~3B parameters, 36 decoder layers) as part of the SafeCompress safety-aware mixed-precision quantization study for the Lexsi AI Safety Hackathon.

| Metric | Measured Value | 95% Bootstrap CI | Baseline Comparison (vs E1 NF4) | Baseline Comparison (vs E0 FP16) |
|---|---|---|---|---|
| **Harmful Prompt Refusal** | **65.0%** | `[58.5%, 71.5%]` | +4.5 pp | -0.5 pp |
| **Benign Over-Refusal** | **1.5%** | `[0.0%, 3.5%]` | +0.5 pp | -1.0 pp |
| **Global-MMLU (Overall)** | **46.67%** | `[42.67%, 50.50%]` | +1.00 pp | -0.83 pp |
| **Belebele Context Accuracy** | **75.0%** | `[68.5%, 81.0%]` | -1.0 pp | -2.0 pp |
| **First-Token Refusal Gap** | **+1.89** | `[1.41, 2.37]` | +0.20 | +0.03 |
| **GPU Memory Allocation** | **3.12 GB** | — | +0.60 GB | -3.58 GB (-53.4%) |
| **Generation Throughput** | **131.5 tok/s** | — | +9.4 tok/s | -14.2 tok/s |

---

## 2. Configuration & Layer Allocation

- **Base Architecture:** `CohereLabs/tiny-aya-global` (36 decoder layers)
- **Protected Layer Indices:** `[0, 14, 16, 17, 19]`
- **Protected Precision:** `16-bit` (FP16)
- **Base Quantized Precision:** `NormalFloat 4 (NF4)` (blocksize 64)
- **Degenerate Output Rate:** `0.25%`

---

## 3. Multilingual Capability Analysis

Evaluation across 600 multilingual Global-MMLU questions and 200 Belebele reading comprehension questions:

- **English Global-MMLU Accuracy:** `50.00%`
- **Hindi Global-MMLU Accuracy:** `43.33%`
- **Belebele Reading Comprehension (Context Reasoning):** `75.0%`
- **Overall Capability Retention:** Maintained capability within normal error margins across English and Hindi prompts.

---

## 4. Safety Alignment & Refusal Evaluation

Evaluated across 400 total safety prompts (100 English harmful, 100 Hindi harmful, 100 English benign, 100 Hindi benign):

- **English Harmful Refusal Rate:** `80.0%`
- **Hindi Harmful Refusal Rate:** `50.0%`
- **First-Token Refusal Logit Gap ($\Delta z$ on Harmful Prompts):** `+1.89`
- **Benign Prompt Over-Refusal (English):** `3.0%`
- **Benign Prompt Over-Refusal (Hindi):** `0.0%`

### Independent AuditKIT Multilingual Safety Validation
This arm was independently scored using Lexsi's `lexsi/auditkit` toolkit across English and Hindi safety probes:
- **AuditKIT Harmful Refusal Rate:** **48.0%** (95% Bootstrap CI: `[41.0%, 55.0%]`)
- **AuditKIT Benign Over-Refusal:** **1.5%**
- **Unscored Probes:** Harmful: 0, Benign: 0


### Paired Differences vs. Full Precision (E0_FP16 Reference)
- **Harmful Refusal Difference:** -0.5 pp (95% CI: `[-5.0, +4.0]` pp) → *Verdict: indistinguishable*
- **Global-MMLU Difference:** -0.83 pp (95% CI: `[-3.17, +1.67]` pp) → *Verdict: indistinguishable*
- **Logit Gap Difference ($\Delta z$):** +0.03 (95% CI: `[-0.08, +0.15]`)

---

## 5. Reproduction & Loader Instructions

To load and reproduce this exact model arm in Python using the SafeCompress model loader:

```python
from safecompress.model_loader import load_arm

# Load this configuration
model = load_arm("S_D_composite_FP16_W4", device="cuda:0")
```

For complete benchmarking instructions, see [docs/reproduction_guide.md](../../docs/reproduction_guide.md).
