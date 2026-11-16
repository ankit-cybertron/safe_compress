# SafeCompress: do "safety layers" survive quantization? A small, honest test on Tiny Aya

> [!NOTE]
> **HISTORICAL PILOT EXPERIMENT ARCHIVE (Phase 1 — 12 Arms):**  
> This article documents the preliminary exploratory pilot study investigating initial activation patching across 12 experimental arms.  
> **For the authoritative, complete 20-arm research study** (evaluating Refusal Drop Strategy C, NF4 KL Fragility, Gradient Attribution, AuditKIT independent verification, and Belebele context reasoning), refer to the root repository documentation:
> - **Primary Research Overview:** [README.md](../README.md)
> - **Full Published Research Paper:** [ARTICLE.md](../ARTICLE.md)
> - **Canonical 20-Arm Evaluation Data:** [results/tables/results_table.csv](../results/tables/results_table.csv)

*Ankit Kumar Tiwari. Lexsi AI Safety and Alignment Hackathon, Track 2b (safety-aware compression).*

## Question
Does keeping patching-identified layers at higher precision during quantization preserve refusal, over-refusal and capability (English and Hindi) in CohereLabs/tiny-aya-global better than parameter-matched random layers, least-effect layers and uniform quantization?

## Result in one paragraph
Keeping the patching-identified layers at higher precision did NOT help refusal: it was worse than random layers in at least one scheme. Other metrics: over-refusal: no distinguishable difference; capability: no distinguishable difference. Numbers below are percentages with 95% bootstrap intervals; every comparison is paired over the same prompts. Read the limitations before quoting any number.

## Setup
- Model: CohereLabs/tiny-aya-global (36 decoder layers, 3.35B unique parameters). Quantisation: bitsandbytes NF4 (block 64) and LLM.int8, applied per layer to all linear weights.
- Layer ranking: bidirectional activation patching (denoise and noise, whole-block outputs) on equal-length harmful/unharmful pairs, English and Hindi, from PolyGuardMix. The ranking was frozen before any quantized evaluation. Top-5 layers [0, 3, 5, 17, 22]; least-effect control [20, 28, 30, 31, 35]; random sets matched on the number of full-attention layers among the five.
- Arms: FP16; uniform W4; and for each of two schemes (five layers kept in FP16, the rest NF4 (4-bit); five layers kept in INT8, the rest NF4) the circuit set, the least-effect set and three random sets, all protecting 5 x 78.1M = 390.6M parameters.
- Safety prompts: HarmBench (standard, contextual), StrongREJECT, XSTest (safe and unsafe), IndicSafe (Hindi). Capability: 500 paired Global-MMLU questions in English and Hindi.
- Size is measured from the stored state dict: FP16 6.70 GB, uniform W4 2.53 GB, protected FP16+W4 3.11 GB, protected INT8+W4 2.72 GB.

## Results
| arm | bits/param | size GB | MMLU EN+HI % | refusal on harmful % | refusal on benign % |
|---|---|---|---|---|---|
| E0_FP16 | 16.00 | 6.70 | 46.5 (43.3-49.7) | 70.6 (67.8-73.3) | 4.5 (2.6-6.7) |
| E1_uniform_W4 | 6.03 | 2.53 | 44.8 (41.7-47.9) | 66.3 (63.5-69.2) | 2.6 (1.2-4.3) |
| E3_circuit_FP16_W4 | 7.42 | 3.11 | 46.3 (43.3-49.4) | 67.1 (64.2-69.9) | 3.1 (1.7-4.8) |
| E4_least_FP16_W4 | 7.42 | 3.11 | 45.1 (42.0-48.3) | 68.1 (65.4-71.0) | 2.6 (1.2-4.3) |
| E5_rand0_FP16_W4 | 7.42 | 3.11 | 46.8 (43.7-50.0) | 67.0 (64.1-69.7) | 2.4 (1.0-4.0) |
| E6_rand1_FP16_W4 | 7.42 | 3.11 | 45.6 (42.5-48.8) | 69.0 (66.2-71.7) | 3.1 (1.7-5.0) |
| E7_rand2_FP16_W4 | 7.42 | 3.11 | 45.6 (42.5-48.8) | 69.0 (66.1-71.7) | 2.4 (1.0-3.8) |
| E8_circuit_W8_W4 | 6.49 | 2.72 | 46.2 (43.1-49.3) | 67.1 (64.2-69.9) | 3.1 (1.7-4.8) |
| E9_least_W8_W4 | 6.49 | 2.72 | 45.3 (42.2-48.5) | 67.3 (64.4-70.2) | 2.1 (1.0-3.6) |
| E10_rand0_W8_W4 | 6.49 | 2.72 | 47.3 (44.2-50.5) | 67.3 (64.4-70.1) | 2.6 (1.2-4.3) |
| E11_rand1_W8_W4 | 6.49 | 2.72 | 44.7 (41.7-47.9) | 69.3 (66.4-72.0) | 3.1 (1.7-5.0) |
| E12_rand2_W8_W4 | 6.49 | 2.72 | 45.4 (42.3-48.6) | 68.3 (65.5-71.1) | 2.4 (1.0-3.8) |

Paired differences (percentage points):

**Refusal on harmful prompts**
- circuit minus mean of 3 random sets, FP16_W4: -1.2 points (95% CI -2.4 to -0.1), circuit worse
- circuit minus least-effect layers, FP16_W4: -1.0 points (95% CI -2.6 to +0.4), indistinguishable
- circuit minus mean of 3 random sets, W8_W4: -1.2 points (95% CI -2.5 to +0.1), indistinguishable
- circuit minus least-effect layers, W8_W4: -0.2 points (95% CI -1.6 to +1.2), indistinguishable
- circuit (FP16+W4) minus uniform W4: +0.8 points (95% CI -0.5 to +2.0), indistinguishable

**Refusal on benign prompts (over-refusal)**
- circuit minus mean of 3 random sets, FP16_W4: +0.5 points (95% CI -0.3 to +1.3), indistinguishable
- circuit minus least-effect layers, FP16_W4: +0.5 points (95% CI +0.0 to +1.2), indistinguishable
- circuit minus mean of 3 random sets, W8_W4: +0.4 points (95% CI -0.4 to +1.3), indistinguishable
- circuit minus least-effect layers, W8_W4: +1.0 points (95% CI +0.2 to +1.9), circuit worse
- circuit (FP16+W4) minus uniform W4: +0.5 points (95% CI -0.7 to +1.7), indistinguishable

**Global-MMLU accuracy**
- circuit minus mean of 3 random sets, FP16_W4: +0.3 points (95% CI -1.0 to +1.6), indistinguishable
- circuit minus least-effect layers, FP16_W4: +1.2 points (95% CI -0.4 to +2.8), indistinguishable
- circuit minus mean of 3 random sets, W8_W4: +0.4 points (95% CI -1.0 to +1.8), indistinguishable
- circuit minus least-effect layers, W8_W4: +0.9 points (95% CI -0.6 to +2.4), indistinguishable
- circuit (FP16+W4) minus uniform W4: +1.5 points (95% CI +0.1 to +2.9), circuit better

Figures: figures/fig1 (metrics vs bits), fig2 (calibration), fig3 (layer scores), fig4 (circuit vs random).

## What this does and does not show
The cost side is solid: protecting five layers costs a known number of bytes (table). The benefit side is only as strong as the intervals above. If an interval contains zero, we have no evidence of a difference.

## Limitations
- The layer ranking comes from our own activation-patching code (whole-sequence, forward passes, equal-length harmful/unharmful pairs from PolyGuardMix), not from CircuitKIT/EAP. Early layers (layer 0 scores 0.998) show large denoising effects with near-zero noising effects, which suggests the score partly measures prompt-content flow rather than a safety-specific circuit. The 5th and 6th layers score almost the same (0.305 vs 0.279). Treat the set as "patching-identified layers", not a verified refusal circuit.
- Only 3 random layer sets per scheme (the frozen manifest contains 10, only 3 were evaluated). Three controls cannot give a reliable null distribution.
- Refusal is detected with a rule-based phrase matcher on the first 250 characters of a 64-token greedy response, not a judge model. Our first matcher counted words such as "harmful" and "illegal" as refusals; the numbers here use a stricter matcher and we report the original too in final_experimental_matrix.csv.
- The first safety run truncated every prompt at 512 tokens (including a ~365-token default system preamble). Prompts above that length (186 prompts incl. HarmBench-copyright) were removed from the analysis. Length filter applied: True.
- Capability is 500 paired Global-MMLU questions per language; a single arm has a 95% interval of roughly +-4.4 points, so differences under about 3 points are not resolvable. Prompts used a short answer-letter format with a fixed letter-token set.
- One model (Tiny Aya Global), one quantiser family (bitsandbytes NF4 and LLM.int8), greedy decoding, English and Hindi only. No 3-bit arm and no GPTQ/AWQ/HQQ backend. No human review of outputs.
- Safety benchmark prompts and model outputs on them are not redistributed. Only ids, labels and numbers are in this repository.

## Reproduce
Code, frozen manifests and per-prompt labels: https://github.com/<your-user>/safecompress. Arms are rebuilt deterministically from the upstream FP16 model and layer_selection.json (see load_safecompress.py). Model repository: https://huggingface.co/<your-user>/safecompress-tiny-aya-global.
