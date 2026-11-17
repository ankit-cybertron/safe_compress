# Model Evaluation Report: `S_CK_circuitkit_W8_W4`

> **Back to Results Table:** [README.md](../../README.md#3-results-summary-table) | [ARTICLE.md](../../ARTICLE.md#table-1-primary-26-arm-evaluation-results)

---

## 1. Executive Summary & Visual Scorecard

![S_CK_circuitkit_W8_W4 Evaluation Overview](../../figures/model_evaluations/S_CK_circuitkit_W8_W4.png)

`S_CK_circuitkit_W8_W4` is an experimental configuration evaluated on [`CohereLabs/tiny-aya-global`](https://huggingface.co/CohereLabs/tiny-aya-global) (~3B parameters, 36 decoder layers) as part of the SafeCompress safety-aware mixed-precision quantization study for the Lexsi AI Safety Hackathon.

| Metric | Measured Value | 95% Bootstrap CI | Baseline Comparison (vs E1 NF4) | Baseline Comparison (vs E0 FP16) |
|---|---|---|---|---|
| **Harmful Prompt Refusal** | **65.0%** | `[58.5%, 71.5%]` | +4.5 pp | -0.5 pp |
| **Benign Over-Refusal** | **1.0%** | `[0.0%, 2.5%]` | +0.0 pp | -1.5 pp |
| **Global-MMLU (Overall)** | **47.83%** | `[44.00%, 51.67%]` | +2.17 pp | +0.33 pp |
| **Belebele Context Accuracy** | **73.5%** | `[67.0%, 79.5%]` | -2.5 pp | -3.5 pp |
| **First-Token Refusal Gap** | **+2.10** | `[1.66, 2.56]` | +0.42 | +0.25 |
| **GPU Memory Allocation** | **2.74 GB** | — | +0.22 GB | -3.96 GB (-59.1%) |
| **Generation Throughput** | **104.5 tok/s** | — | -17.7 tok/s | -41.2 tok/s |

### Hugging Face Artifact Link
- **Hugging Face Repository:** [`cybertronak/safecompress-tinyaya-recipes-loader`](https://huggingface.co/cybertronak/safecompress-tinyaya-recipes-loader)
- **Deployment Format:** Loader

---

## 2. Configuration & Layer Allocation

- **Base Architecture:** `CohereLabs/tiny-aya-global` (36 decoder layers)
- **Protected Layer Indices:** `[0, 14, 16, 20, 35]`
- **Protected Precision:** `8-bit` (INT8)
- **Base Quantized Precision:** `NormalFloat 4 (NF4)` (blocksize 64)
- **Degenerate Output Rate:** `0.00%`

---

## 3. Multilingual Capability Analysis

Evaluation across 600 multilingual Global-MMLU questions and 200 Belebele reading comprehension questions:

- **English Global-MMLU Accuracy:** `52.00%`
- **Hindi Global-MMLU Accuracy:** `43.67%`
- **Belebele Reading Comprehension (Context Reasoning):** `73.5%`
- **Overall Capability Retention:** Maintained capability within normal error margins across English and Hindi prompts.

---

## 4. Safety Alignment & Refusal Evaluation

Evaluated across 400 total safety prompts (100 English harmful, 100 Hindi harmful, 100 English benign, 100 Hindi benign):

- **English Harmful Refusal Rate:** `83.0%`
- **Hindi Harmful Refusal Rate:** `47.0%`
- **First-Token Refusal Logit Gap ($\Delta z$ on Harmful Prompts):** `+2.10`
- **Benign Prompt Over-Refusal (English):** `2.0%`
- **Benign Prompt Over-Refusal (Hindi):** `0.0%`

### Independent AuditKIT Multilingual Safety Validation
This arm was independently scored using Lexsi's `lexsi/auditkit` toolkit across English and Hindi safety probes:
- **AuditKIT Harmful Refusal Rate:** **54.0%** (95% Bootstrap CI: `[47.0%, 61.0%]`)
- **AuditKIT Benign Over-Refusal:** **1.0%**
- **Unscored Probes:** Harmful: 0, Benign: 0


### Paired Differences vs. Full Precision (E0_FP16 Reference)
- **Harmful Refusal Difference:** -0.5 pp (95% CI: `[-4.5, +3.5]` pp) → *Verdict: indistinguishable*
- **Global-MMLU Difference:** +0.33 pp (95% CI: `[-2.00, +2.67]` pp) → *Verdict: indistinguishable (non-inferior within 3 pts)*
- **Logit Gap Difference ($\Delta z$):** +0.25 (95% CI: `[+0.13, +0.37]`)

---

## 5. Reproduction & Loader Instructions

To load and reproduce this exact model arm in Python using the SafeCompress model loader:

```python
from safecompress.model_loader import load_arm

# Load this configuration
model = load_arm("S_CK_circuitkit_W8_W4", device="cuda:0")
```

For complete benchmarking instructions, see [docs/reproduction_guide.md](../../docs/reproduction_guide.md).
