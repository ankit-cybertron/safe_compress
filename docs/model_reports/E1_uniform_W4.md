# Model Evaluation Report: `E1_uniform_W4`

> **Back to Results Table:** [README.md](../../README.md#3-results-summary-table) | [ARTICLE.md](../../ARTICLE.md#table-1-primary-26-arm-evaluation-results)

---

## 1. Executive Summary & Visual Scorecard

![E1_uniform_W4 Evaluation Overview](../../figures/model_evaluations/E1_uniform_W4.png)

`E1_uniform_W4` is an experimental configuration evaluated on [`CohereLabs/tiny-aya-global`](https://huggingface.co/CohereLabs/tiny-aya-global) (~3B parameters, 36 decoder layers) as part of the SafeCompress safety-aware mixed-precision quantization study for the Lexsi AI Safety Hackathon.

| Metric | Measured Value | 95% Bootstrap CI | Baseline Comparison (vs E1 NF4) | Baseline Comparison (vs E0 FP16) |
|---|---|---|---|---|
| **Harmful Prompt Refusal** | **60.5%** | `[53.5%, 67.0%]` | +0.0 pp | -5.0 pp |
| **Benign Over-Refusal** | **1.0%** | `[0.0%, 2.5%]` | +0.0 pp | -1.5 pp |
| **Global-MMLU (Overall)** | **45.67%** | `[41.83%, 49.67%]` | +0.00 pp | -1.83 pp |
| **Belebele Context Accuracy** | **76.0%** | `[70.0%, 81.5%]` | +0.0 pp | -1.0 pp |
| **First-Token Refusal Gap** | **+1.69** | `[1.22, 2.16]` | +0.00 | -0.17 |
| **GPU Memory Allocation** | **2.53 GB** | — | +0.00 GB | -4.17 GB (-62.3%) |
| **Generation Throughput** | **122.1 tok/s** | — | +0.0 tok/s | -23.6 tok/s |

### Hugging Face Artifact Link
- **Hugging Face Repository:** [`cybertronak/safecompress-tinyaya-uniform-nf4`](https://huggingface.co/cybertronak/safecompress-tinyaya-uniform-nf4)
- **Deployment Format:** Checkpoint

---

## 2. Configuration & Layer Allocation

- **Base Architecture:** `CohereLabs/tiny-aya-global` (36 decoder layers)
- **Protected Layer Indices:** `[]`
- **Protected Precision:** `16-bit` (None)
- **Base Quantized Precision:** `NormalFloat 4 (NF4)` (blocksize 64)
- **Degenerate Output Rate:** `0.25%`

---

## 3. Multilingual Capability Analysis

Evaluation across 600 multilingual Global-MMLU questions and 200 Belebele reading comprehension questions:

- **English Global-MMLU Accuracy:** `48.33%`
- **Hindi Global-MMLU Accuracy:** `43.00%`
- **Belebele Reading Comprehension (Context Reasoning):** `76.0%`
- **Overall Capability Retention:** Maintained capability within normal error margins across English and Hindi prompts.

---

## 4. Safety Alignment & Refusal Evaluation

Evaluated across 400 total safety prompts (100 English harmful, 100 Hindi harmful, 100 English benign, 100 Hindi benign):

- **English Harmful Refusal Rate:** `77.0%`
- **Hindi Harmful Refusal Rate:** `44.0%`
- **First-Token Refusal Logit Gap ($\Delta z$ on Harmful Prompts):** `+1.69`
- **Benign Prompt Over-Refusal (English):** `2.0%`
- **Benign Prompt Over-Refusal (Hindi):** `0.0%`

### Independent AuditKIT Multilingual Safety Validation
This arm was independently scored using Lexsi's `lexsi/auditkit` toolkit across English and Hindi safety probes:
- **AuditKIT Harmful Refusal Rate:** **44.0%** (95% Bootstrap CI: `[37.5%, 51.0%]`)
- **AuditKIT Benign Over-Refusal:** **1.0%**
- **Unscored Probes:** Harmful: 0, Benign: 0


### Paired Differences vs. Full Precision (E0_FP16 Reference)
- **Harmful Refusal Difference:** -5.0 pp (95% CI: `[-9.0, -1.5]` pp) → *Verdict: slightly lower (within margin)*
- **Global-MMLU Difference:** -1.83 pp (95% CI: `[-4.34, +0.83]` pp) → *Verdict: indistinguishable*
- **Logit Gap Difference ($\Delta z$):** -0.17 (95% CI: `[-0.30, -0.04]`)

---

## 5. Reproduction & Loader Instructions

To load and reproduce this exact model arm in Python using the SafeCompress model loader:

```python
from safecompress.model_loader import load_arm

# Load this configuration
model = load_arm("E1_uniform_W4", device="cuda:0")
```

For complete benchmarking instructions, see [docs/reproduction_guide.md](../../docs/reproduction_guide.md).
