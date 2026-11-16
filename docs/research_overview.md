# Research Overview — SafeCompress

> **Track:** Lexsi Hackathon Track 2 — AI Red Teaming, Safety and Alignment  
> **Problem Statement:** Track 2b — Problem Statement 11: Safety-aware compression  
> **Author:** Ankit Kumar Tiwari  
> **Model:** `CohereForAI/aya-23-8B`  
> **Hugging Face Model Collection:** [`ankittiwarifr/safecompress`](https://huggingface.co/collections/ankittiwarifr/safecompress)  
> **Hugging Face Results Dataset:** [`ankittiwarifr/safecompress-results`](https://huggingface.co/datasets/ankittiwarifr/safecompress-results)  

---

## 1. Executive Summary

Post-training quantization (PTQ) is the standard method for deploying large language models on edge devices and cost-constrained cloud hardware. Aggressive uniform 4-bit quantization (such as NormalFloat 4 / NF4) cuts memory consumption by over 60%, but frequently compromises safety alignment, causing models to fulfill harmful requests they would refuse at full 16-bit floating-point precision.

**SafeCompress** investigates whether **layer-informed mixed-precision quantization**—keeping a small budget ($K=5$ out of 36 layers, representing ~14% of model layers) in higher precision (FP16 or INT8) while quantizing the remaining 31 layers to NF4—can preserve safety-critical refusal behaviour while maintaining low memory footprint and high throughput.

Across 20 systematic experimental arms on `CohereForAI/aya-23-8B`, evaluating 400 safety prompts (English and Hindi), 600 multilingual MMLU questions, and 200 Belebele reading comprehension items:

1. **Safety Recovery:** Protecting the 5 layers identified by refusal gap degradation under quantization (`S_C_refusalDrop_FP16_W4`, layers `[7, 14, 16, 27, 35]`) restores harmful prompt refusal to **65.5%**, matching the unquantized FP16 baseline (**65.5%**) and outperforming uniform NF4 (**60.5%**, $+5.0\,\text{pp}$, 95% bootstrap CI: $[+2.5, +8.0]$).
2. **AuditKIT Validation:** Evaluated independently via Lexsi's AuditKIT toolkit on a 200-sample probe, `S_C_refusalDrop_FP16_W4` achieves **56.0%** refusal vs. **44.0%** for uniform NF4 ($+12.0\,\text{pp}$, 95% bootstrap CI: $[+8.0, +17.0]$).
3. **Memory & Speed Efficiency:** Memory allocation is reduced from **6.70 GB** (FP16) to **3.12 GB** (FP16+NF4 mixed precision) vs. **2.53 GB** (uniform NF4). Throughput reaches **128.9 tokens/sec** compared to 145.7 (FP16) and 122.1 (uniform NF4).
4. **Critical Scientific Finding (Negative Control):** Protecting a set of 5 layers selected for *low* importance and *low* fragility (`S_N_negative_FP16_W4`, layers `[6, 25, 31, 33, 34]`) also achieves **64.0%** harmful refusal ($+3.5\,\text{pp}$ above uniform NF4). This establishes that while mixed precision reliably preserves refusal, the protection benefit is partially distributed across transformer layers rather than strictly isolated to a single sparse circuit.

---

## 2. Research Question & Hypothesis

### Research Question
> *Can interpretability- and sensitivity-guided layer selection for mixed-precision quantization preserve safety-critical refusal mechanisms while retaining the memory and throughput benefits of aggressive 4-bit compression?*

### Formal Hypothesis
* **$H_1$ (Selective Preservation):** Allocating higher precision (FP16 or INT8) to transformer layers that exhibit high causal mediation of refusal or high sensitivity to quantization degradation will yield higher safety refusal than uniform 4-bit quantization, with minimal memory overhead ($<0.6\,\text{GB}$).
* **$H_0$ (Invariance / Non-Selectivity):** Higher precision on safety-ranked layers does not yield statistically distinguishable refusal preservation compared to randomly selected or negatively selected layer subsets of identical parameter budget.

---

## 3. Base Model Selection & Rationale

We selected **`CohereForAI/aya-23-8B`** as the research substrate based on five criteria:

1. **Multilingual Safety Alignment:** Aya-23 is explicitly instruction-tuned and aligned across 23 languages. This enables evaluating safety preservation across English and high-risk non-English languages (Hindi).
2. **Heterogeneous Architectural Topology:** Aya-23 incorporates an alternating attention pattern:
   - 27 sliding-window attention layers
   - 9 full-context global attention layers (at indices `[3, 7, 11, 15, 19, 23, 27, 31, 35]`). This architectural variation allows testing whether layer type correlates with quantization sensitivity.
3. **Parameter Scale Relevance:** At ~8B parameters (78.12M parameters per decoder layer), Aya-23 represents the frontier of models realistically deployed on single consumer GPUs (8GB–16GB VRAM) and edge devices where quantization is practically required.
4. **Tied Embedding Weights:** The vocabulary embedding and output projection share weights, ensuring quantization effects are localized strictly to the transformer blocks.
5. **Open Weights & Reproducibility:** Model weights are publicly available on the Hugging Face Hub under open research licensing.

---

## 4. Key Contributions

- **Systematic 20-Arm Evaluation Matrix:** Comprehensive benchmark spanning FP16 baseline, uniform NF4, 6 distinct layer-selection strategies across two mixed-precision formats (FP16+NF4 and INT8+NF4), and 6 random/negative control arms.
- **Multimodal Layer Profiling:** Comparison of activation patching, single-layer quantization perturbation (KL divergence & refusal erosion), and first-order gradient attribution on the same 36-layer model.
- **Rigorously Controlled Evaluation:** 400 safety prompts balanced across English/Hindi and harmful/benign categories, paired MMLU (600 items), and Belebele (200 items), with 10,000-iteration paired bootstrap confidence intervals.
- **Independent Safety Validation via Lexsi AuditKIT:** Dual-scorer methodology validating phrase-matcher findings with an external multilingual safety judge.
- **Honest Negative Control Reporting:** Full transparency regarding the performance of negative controls and pilot experiment findings, avoiding overstated claims of sparse localization.
