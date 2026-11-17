# Model Evaluation Report: `S_C_refusalDrop_FP16_W4`

> **Back to Results Table:** [README.md](../../README.md#3-results-summary-table) | [ARTICLE.md](../../ARTICLE.md#table-1-primary-26-arm-evaluation-results)

---

## 1. Executive Summary & Visual Scorecard

![S_C_refusalDrop_FP16_W4 Evaluation Overview](../../figures/model_evaluations/S_C_refusalDrop_FP16_W4.png)

`S_C_refusalDrop_FP16_W4` is an experimental configuration evaluated on [`CohereLabs/tiny-aya-global`](https://huggingface.co/CohereLabs/tiny-aya-global) (~3B parameters, 36 decoder layers) as part of the SafeCompress safety-aware mixed-precision quantization study for the Lexsi AI Safety Hackathon.

| Metric | Measured Value | 95% Bootstrap CI | Baseline Comparison (vs E1 NF4) | Baseline Comparison (vs E0 FP16) |
|---|---|---|---|---|
| **Harmful Prompt Refusal** | **65.5%** | `[59.0%, 72.0%]` | +5.0 pp | +0.0 pp |
| **Benign Over-Refusal** | **1.5%** | `[0.0%, 3.5%]` | +0.5 pp | -1.0 pp |
| **Global-MMLU (Overall)** | **46.33%** | `[42.33%, 50.17%]` | +0.67 pp | -1.17 pp |
| **Belebele Context Accuracy** | **76.0%** | `[70.0%, 81.5%]` | +0.0 pp | -1.0 pp |
| **First-Token Refusal Gap** | **+2.39** | `[1.92, 2.89]` | +0.71 | +0.54 |
| **GPU Memory Allocation** | **3.12 GB** | — | +0.60 GB | -3.58 GB (-53.4%) |
| **Generation Throughput** | **128.9 tok/s** | — | +6.8 tok/s | -16.8 tok/s |

### Hugging Face Artifact Link
- **Hugging Face Repository:** [`cybertronak/safecompress-tinyaya-refusaldrop-fp16prot-nf4`](https://huggingface.co/cybertronak/safecompress-tinyaya-refusaldrop-fp16prot-nf4)
- **Deployment Format:** Checkpoint

---

## 2. Configuration & Layer Allocation

- **Base Architecture:** `CohereLabs/tiny-aya-global` (36 decoder layers)
- **Protected Layer Indices:** `[7, 14, 16, 27, 35]`
- **Protected Precision:** `16-bit` (FP16)
- **Base Quantized Precision:** `NormalFloat 4 (NF4)` (blocksize 64)
- **Degenerate Output Rate:** `0.00%`

---

## 3. Multilingual Capability Analysis

Evaluation across 600 multilingual Global-MMLU questions and 200 Belebele reading comprehension questions:

- **English Global-MMLU Accuracy:** `50.67%`
- **Hindi Global-MMLU Accuracy:** `42.00%`
- **Belebele Reading Comprehension (Context Reasoning):** `76.0%`
- **Overall Capability Retention:** Maintained capability within normal error margins across English and Hindi prompts.

---

## 4. Safety Alignment & Refusal Evaluation

Evaluated across 400 total safety prompts (100 English harmful, 100 Hindi harmful, 100 English benign, 100 Hindi benign):

- **English Harmful Refusal Rate:** `84.0%`
- **Hindi Harmful Refusal Rate:** `47.0%`
- **First-Token Refusal Logit Gap ($\Delta z$ on Harmful Prompts):** `+2.39`
- **Benign Prompt Over-Refusal (English):** `3.0%`
- **Benign Prompt Over-Refusal (Hindi):** `0.0%`

### Independent AuditKIT Multilingual Safety Validation
This arm was independently scored using Lexsi's `lexsi/auditkit` toolkit across English and Hindi safety probes:
- **AuditKIT Harmful Refusal Rate:** **56.0%** (95% Bootstrap CI: `[49.0%, 62.5%]`)
- **AuditKIT Benign Over-Refusal:** **1.5%**
- **Unscored Probes:** Harmful: 0, Benign: 0


### Paired Differences vs. Full Precision (E0_FP16 Reference)
- **Harmful Refusal Difference:** +0.0 pp (95% CI: `[-3.5, +3.5]` pp) → *Verdict: indistinguishable*
- **Global-MMLU Difference:** -1.17 pp (95% CI: `[-3.83, +1.50]` pp) → *Verdict: indistinguishable*
- **Logit Gap Difference ($\Delta z$):** +0.54 (95% CI: `[+0.43, +0.64]`)

---

## 5. Reproduction & Loader Instructions

To load and reproduce this exact model arm in Python using the SafeCompress model loader:

```python
from safecompress.model_loader import load_arm

# Load this configuration
model = load_arm("S_C_refusalDrop_FP16_W4", device="cuda:0")
```

For complete benchmarking instructions, see [docs/reproduction_guide.md](../../docs/reproduction_guide.md).
