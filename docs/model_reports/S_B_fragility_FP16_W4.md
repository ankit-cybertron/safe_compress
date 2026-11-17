# Model Evaluation Report: `S_B_fragility_FP16_W4`

> **Back to Results Table:** [README.md](../../README.md#3-results-summary-table) | [ARTICLE.md](../../ARTICLE.md#table-1-primary-26-arm-evaluation-results)

---

## 1. Executive Summary & Visual Scorecard

![S_B_fragility_FP16_W4 Evaluation Overview](../../figures/model_evaluations/S_B_fragility_FP16_W4.png)

`S_B_fragility_FP16_W4` is an experimental configuration evaluated on [`CohereLabs/tiny-aya-global`](https://huggingface.co/CohereLabs/tiny-aya-global) (~3B parameters, 36 decoder layers) as part of the SafeCompress safety-aware mixed-precision quantization study for the Lexsi AI Safety Hackathon.

| Metric | Measured Value | 95% Bootstrap CI | Baseline Comparison (vs E1 NF4) | Baseline Comparison (vs E0 FP16) |
|---|---|---|---|---|
| **Harmful Prompt Refusal** | **64.5%** | `[58.0%, 71.0%]` | +4.0 pp | -1.0 pp |
| **Benign Over-Refusal** | **2.5%** | `[0.5%, 5.0%]` | +1.5 pp | +0.0 pp |
| **Global-MMLU (Overall)** | **46.83%** | `[42.83%, 50.67%]` | +1.17 pp | -0.67 pp |
| **Belebele Context Accuracy** | **75.5%** | `[69.5%, 81.0%]` | -0.5 pp | -1.5 pp |
| **First-Token Refusal Gap** | **+2.15** | `[1.69, 2.63]` | +0.46 | +0.29 |
| **GPU Memory Allocation** | **3.12 GB** | — | +0.60 GB | -3.58 GB (-53.4%) |
| **Generation Throughput** | **129.6 tok/s** | — | +7.5 tok/s | -16.1 tok/s |

---

## 2. Configuration & Layer Allocation

- **Base Architecture:** `CohereLabs/tiny-aya-global` (36 decoder layers)
- **Protected Layer Indices:** `[0, 7, 12, 16, 35]`
- **Protected Precision:** `16-bit` (FP16)
- **Base Quantized Precision:** `NormalFloat 4 (NF4)` (blocksize 64)
- **Degenerate Output Rate:** `0.00%`

---

## 3. Multilingual Capability Analysis

Evaluation across 600 multilingual Global-MMLU questions and 200 Belebele reading comprehension questions:

- **English Global-MMLU Accuracy:** `49.67%`
- **Hindi Global-MMLU Accuracy:** `44.00%`
- **Belebele Reading Comprehension (Context Reasoning):** `75.5%`
- **Overall Capability Retention:** Maintained capability within normal error margins across English and Hindi prompts.

---

## 4. Safety Alignment & Refusal Evaluation

Evaluated across 400 total safety prompts (100 English harmful, 100 Hindi harmful, 100 English benign, 100 Hindi benign):

- **English Harmful Refusal Rate:** `84.0%`
- **Hindi Harmful Refusal Rate:** `45.0%`
- **First-Token Refusal Logit Gap ($\Delta z$ on Harmful Prompts):** `+2.15`
- **Benign Prompt Over-Refusal (English):** `5.0%`
- **Benign Prompt Over-Refusal (Hindi):** `0.0%`

### Independent AuditKIT Multilingual Safety Validation
This arm was independently scored using Lexsi's `lexsi/auditkit` toolkit across English and Hindi safety probes:
- **AuditKIT Harmful Refusal Rate:** **54.0%** (95% Bootstrap CI: `[47.0%, 61.0%]`)
- **AuditKIT Benign Over-Refusal:** **2.5%**
- **Unscored Probes:** Harmful: 0, Benign: 0


### Paired Differences vs. Full Precision (E0_FP16 Reference)
- **Harmful Refusal Difference:** -1.0 pp (95% CI: `[-4.5, +2.5]` pp) → *Verdict: indistinguishable*
- **Global-MMLU Difference:** -0.67 pp (95% CI: `[-3.17, +1.83]` pp) → *Verdict: indistinguishable*
- **Logit Gap Difference ($\Delta z$):** +0.29 (95% CI: `[+0.19, +0.40]`)

---

## 5. Reproduction & Loader Instructions

To load and reproduce this exact model arm in Python using the SafeCompress model loader:

```python
from safecompress.model_loader import load_arm

# Load this configuration
model = load_arm("S_B_fragility_FP16_W4", device="cuda:0")
```

For complete benchmarking instructions, see [docs/reproduction_guide.md](../../docs/reproduction_guide.md).
