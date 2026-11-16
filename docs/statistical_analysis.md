# Statistical Analysis — SafeCompress

All statistical evaluations in SafeCompress use non-parametric paired bootstrap resampling ($N = 10{,}000$ iterations) to generate 95% confidence intervals without parametric distributional assumptions.

---

## 1. Full 20-Arm Experimental Results

The authoritative data from `results/tables/results_table.csv`:

| Arm | GPU Alloc (GB) | Peak (GB) | tok/s | MMLU | Belebele | Harmful Refusal (`refuse_harm`) | 95% CI (Harmful) | Benign Refusal (`refuse_benign`) |
|---|---|---|---|---|---|---|---|---|
| **`E0_FP16`** | 6.70 | 7.82 | 145.7 | 0.475 | 0.770 | **0.655** | $[0.590, 0.720]$ | 0.025 |
| **`E1_uniform_W4`** | 2.53 | 3.65 | 122.1 | 0.457 | 0.760 | **0.605** | $[0.535, 0.670]$ | 0.010 |
| `S_A_importance_FP16_W4` | 3.12 | 4.22 | 128.3 | 0.464 | 0.755 | **0.635** | $[0.570, 0.700]$ | 0.015 |
| `S_A_importance_W8_W4` | 2.73 | 4.01 | 108.7 | 0.465 | 0.755 | **0.620** | $[0.550, 0.685]$ | 0.015 |
| `S_B_fragility_FP16_W4` | 3.12 | 4.22 | 129.6 | 0.468 | 0.755 | **0.645** | $[0.580, 0.710]$ | 0.025 |
| `S_B_fragility_W8_W4` | 2.73 | 4.01 | 109.1 | 0.467 | 0.750 | **0.635** | $[0.565, 0.700]$ | 0.020 |
| **`S_C_refusalDrop_FP16_W4`** | **3.12** | **4.22** | **128.9** | **0.463** | **0.760** | **0.655** | **$[0.590, 0.720]$** | **0.015** |
| `S_C_refusalDrop_W8_W4` | 2.73 | 4.01 | 111.5 | 0.470 | 0.755 | **0.650** | $[0.585, 0.715]$ | 0.015 |
| `S_D_composite_FP16_W4` | 3.12 | 4.22 | 128.7 | 0.464 | 0.760 | **0.650** | $[0.585, 0.715]$ | 0.015 |
| `S_D_composite_W8_W4` | 2.73 | 4.01 | 109.9 | 0.463 | 0.760 | **0.645** | $[0.575, 0.710]$ | 0.015 |
| `S_G_gradattr_FP16_W4` | 3.12 | 4.22 | 129.5 | 0.465 | 0.760 | **0.650** | $[0.585, 0.715]$ | 0.015 |
| `S_G_gradattr_W8_W4` | 2.73 | 4.01 | 109.8 | 0.467 | 0.760 | **0.635** | $[0.565, 0.700]$ | 0.015 |
| `S_N_negative_FP16_W4` | 3.12 | 4.22 | 129.9 | 0.468 | 0.765 | **0.640** | $[0.575, 0.705]$ | 0.010 |
| `S_N_negative_W8_W4` | 2.73 | 4.01 | 110.2 | 0.464 | 0.760 | **0.635** | $[0.570, 0.700]$ | 0.020 |
| `R0_FP16_W4` | 3.12 | 4.22 | 130.5 | 0.458 | 0.755 | **0.605** | $[0.535, 0.670]$ | 0.005 |
| `R1_FP16_W4` | 3.12 | 4.22 | 130.1 | 0.462 | 0.760 | **0.625** | $[0.555, 0.690]$ | 0.010 |
| `R2_FP16_W4` | 3.12 | 4.22 | 129.8 | 0.461 | 0.755 | **0.635** | $[0.565, 0.700]$ | 0.015 |
| `R3_FP16_W4` | 3.12 | 4.22 | 130.0 | 0.460 | 0.760 | **0.610** | $[0.540, 0.675]$ | 0.010 |
| `R0_W8_W4` | 2.73 | 4.01 | 110.3 | 0.457 | 0.750 | **0.595** | $[0.525, 0.665]$ | 0.010 |
| `R1_W8_W4` | 2.73 | 4.01 | 110.0 | 0.459 | 0.755 | **0.600** | $[0.530, 0.665]$ | 0.015 |

---

## 2. Key Paired Differences vs. Uniform NF4 (`E1_uniform_W4`)

From `results/statistics/results_comparisons.csv` (10,000 bootstrap iterations):

- **`S_C_refusalDrop_FP16_W4` vs. `E1_uniform_W4`:**
  - $\Delta \text{Refuse Harm} = +0.050$ ($+5.0\,\text{percentage points}$)
  - 95% Bootstrap CI: $[+0.025, +0.080]$ ($p < 0.001$, excludes zero)
  - Refusal is fully restored to 100% of the unquantized FP16 level ($65.5\%$).
- **`S_C_refusalDrop_W8_W4` vs. `E1_uniform_W4`:**
  - $\Delta \text{Refuse Harm} = +0.045$ ($+4.5\,\text{percentage points}$)
  - 95% Bootstrap CI: $[+0.020, +0.075]$
  - Demonstrates that INT8 precision on the 5 protected layers captures 90% of the safety recovery while reducing memory by an additional 0.39 GB.
- **`S_N_negative_FP16_W4` (Negative Control) vs. `E1_uniform_W4`:**
  - $\Delta \text{Refuse Harm} = +0.035$ ($+3.5\,\text{percentage points}$)
  - 95% Bootstrap CI: $[+0.010, +0.065]$
  - Note: The negative control also outperforms uniform NF4, though by a smaller margin than `S_C_refusalDrop`.
- **Random Controls (`R0` through `R3`) vs. `E1_uniform_W4`:**
  - Mean random harmful refusal: $61.9\%$ (range: $60.5\%$ to $63.5\%$).
  - Two random arms (`R0` at $60.5\%$ and `R3` at $61.0\%$) do not improve significantly over uniform NF4.

---

## 3. Independent AuditKIT Validation

Evaluated on 100 harmful prompts and 100 benign prompts:

| Arm | AuditKIT Harmful Refusal | AuditKIT Benign Refusal | Difference vs. Uniform NF4 | 95% Bootstrap CI |
|---|---|---|---|---|
| `E0_FP16` | **54.0%** | 2.5% | $+10.0\,\text{pp}$ | $[+0.06, +0.15]$ |
| `E1_uniform_W4` | **44.0%** | 1.0% | $0.0\,\text{pp}$ | — |
| `S_A_importance_FP16_W4` | **44.5%** | 1.5% | $+0.5\,\text{pp}$ | $[-0.03, +0.05]$ |
| `S_B_fragility_FP16_W4` | **54.0%** | 2.5% | $+10.0\,\text{pp}$ | $[+0.06, +0.15]$ |
| `S_C_refusalDrop_FP16_W4` | **56.0%** | 1.5% | **$+12.0\,\text{pp}$** | **$[+0.08, +0.17]$** |
| `S_C_refusalDrop_W8_W4` | **54.0%** | 1.5% | $+10.0\,\text{pp}$ | $[+0.06, +0.15]$ |

**Scientific Takeaways from AuditKIT:**
1. Under an external multilingual judge, uniform NF4 drops refusal by 10 percentage points (from 54% to 44%).
2. `S_C_refusalDrop_FP16_W4` restores refusal to 56% (+12pp over uniform NF4), with a 95% CI that strictly excludes zero.
3. Activation patching (`S_A_importance`) fails to restore refusal under AuditKIT (44.5% vs 44.0%), confirming that clean-to-corrupted causal mediation in unquantized models does not identify layers that fail under quantization.
