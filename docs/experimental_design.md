# Experimental Design — SafeCompress

The experimental grid consists of **20 systematically configured arms** evaluating safety, capability, memory footprint, and generation speed under identical hardware and prompt conditions.

---

## 1. Arm Taxonomy

The 20 arms are organized into 4 logical tiers:

```
20 Experimental Arms
├── 1. Baselines (2 arms)
│   ├── E0_FP16 (Unquantized reference, 16-bit across all 36 layers)
│   └── E1_uniform_W4 (Standard 4-bit baseline, NF4 across all 36 layers)
│
├── 2. Strategy Arms — FP16 Protected (6 arms: 5 layers FP16 + 31 layers NF4)
│   ├── S_A_importance_FP16_W4 (Layers [0, 5, 17, 19, 22])
│   ├── S_B_fragility_FP16_W4  (Layers [0, 7, 12, 16, 35])
│   ├── S_C_refusalDrop_FP16_W4 (Layers [7, 14, 16, 27, 35])
│   ├── S_D_composite_FP16_W4  (Layers [0, 14, 16, 17, 19])
│   ├── S_G_gradattr_FP16_W4   (Layers [17, 19, 22, 27, 29])
│   └── S_N_negative_FP16_W4   (Layers [6, 25, 31, 33, 34] — Negative Control)
│
├── 3. Strategy Arms — INT8 Protected (6 arms: 5 layers INT8 + 31 layers NF4)
│   ├── S_A_importance_W8_W4   (Layers [0, 5, 17, 19, 22])
│   ├── S_B_fragility_W8_W4    (Layers [0, 7, 12, 16, 35])
│   ├── S_C_refusalDrop_W8_W4  (Layers [7, 14, 16, 27, 35])
│   ├── S_D_composite_W8_W4    (Layers [0, 14, 16, 17, 19])
│   ├── S_G_gradattr_W8_W4     (Layers [17, 19, 22, 27, 29])
│   └── S_N_negative_W8_W4     (Layers [6, 25, 31, 33, 34] — Negative Control)
│
└── 4. Random Controls (6 arms: uniformly sampled 5-layer sets)
    ├── R0_FP16_W4 (Layers [7, 13, 15, 24, 32] at FP16)
    ├── R1_FP16_W4 (Layers [11, 15, 22, 27, 30] at FP16)
    ├── R2_FP16_W4 (Layers [13, 14, 27, 31, 34] at FP16)
    ├── R3_FP16_W4 (Layers [1, 5, 19, 27, 33] at FP16)
    ├── R0_W8_W4   (Layers [7, 13, 15, 24, 32] at INT8)
    └── R1_W8_W4   (Layers [11, 15, 22, 27, 30] at INT8)
```

---

## 2. Experimental Controls Rationale

To make valid causal inferences regarding whether layer selection matters, several control layers are required:

| Control Type | Arm(s) | Scientific Purpose |
|---|---|---|
| **Full Precision Reference** | `E0_FP16` | Upper bound on safety and reasoning capabilities. |
| **Uniform Quantization** | `E1_uniform_W4` | Lower bound representing naive post-training compression. |
| **Negative Control** | `S_N_negative_*` | Protects layers ranked *lowest* in importance and fragility. Tests whether *any* 5 layers in FP16 provide the observed benefit. |
| **Random Controls** | `R0_*` through `R3_*` | Protects pseudo-random subsets of 5 layers. Measures parameter volume preservation vs. targeted selection. |
| **Precision Ablation** | `*_W8_W4` vs `*_FP16_W4` | Assesses whether 8-bit integer precision suffices for protected layers or if 16-bit floating point is necessary. |

---

## 3. Evaluation Workload

All 20 arms are evaluated on the exact same prompt instances with greedy decoding (`temperature=0.0`):

1. **Safety Refusal Benchmark (400 items total):**
   - 100 English harmful prompts (HarmBench-standard, direct harmful requests)
   - 100 Hindi harmful prompts (PolyGuardMix / HarmBench translated & verified)
   - 100 English benign prompts (everyday assistance, coding, writing)
   - 100 Hindi benign prompts (everyday non-harmful queries)
2. **Multilingual MMLU (600 items total):**
   - 300 English 4-choice questions across STEM, humanities, and social sciences
   - 300 Hindi 4-choice questions
3. **Belebele Reading Comprehension (200 items total):**
   - 100 English context-passage questions
   - 100 Hindi context-passage questions
4. **Efficiency Metrics:**
   - Model memory allocation on GPU (`alloc_gb` via `torch.cuda.memory_allocated`)
   - Peak memory consumption (`peak_gb` via `torch.cuda.max_memory_allocated`)
   - Token generation throughput (`tok_s`, measured over generation tokens divided by decode time)
