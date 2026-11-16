| Arm | Description | GPU Alloc (GB) | Peak (GB) | tok/s | MMLU | Belebele | Harmful Refusal | 95% Bootstrap CI | Benign Refusal (Over-refusal) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **`E0_FP16`** | Unquantized FP16 Reference | 6.70 | 7.82 | 145.7 | 0.475 | 0.770 | **0.655** | [0.590, 0.720] | 0.025 |
| **`E1_uniform_W4`** | Uniform NF4 4-bit Baseline | 2.53 | 3.64 | 122.1 | 0.457 | 0.760 | **0.605** | [0.535, 0.670] | 0.010 |
| `S_A_importance_FP16_W4` | Strategy A (Activation Patching) | 3.12 | 4.22 | 131.3 | 0.458 | 0.760 | **0.630** | [0.565, 0.695] | 0.015 |
| `S_A_importance_W8_W4` | Strategy A (INT8 Protected) | 2.73 | 3.83 | 113.6 | 0.458 | 0.765 | **0.630** | [0.565, 0.695] | 0.015 |
| `S_B_fragility_FP16_W4` | Strategy B (NF4 KL Sensitivity) | 3.12 | 4.22 | 129.6 | 0.468 | 0.755 | **0.645** | [0.580, 0.710] | 0.025 |
| `S_B_fragility_W8_W4` | Strategy B (INT8 Protected) | 2.73 | 4.01 | 111.5 | 0.467 | 0.760 | **0.645** | [0.580, 0.710] | 0.025 |
| **`S_C_refusalDrop_FP16_W4`** | **Strategy C (Refusal Drop)** | **3.12** | **4.22** | **128.9** | **0.463** | **0.760** | **0.655** | **[0.590, 0.720]** | **0.015** |
| `S_C_refusalDrop_W8_W4` | Strategy C (INT8 Protected) | 2.73 | 4.01 | 111.5 | 0.470 | 0.755 | **0.650** | [0.580, 0.715] | 0.015 |
| `S_D_composite_FP16_W4` | Strategy D (Composite Score) | 3.12 | 4.22 | 131.5 | 0.467 | 0.750 | **0.650** | [0.585, 0.715] | 0.015 |
| `S_D_composite_W8_W4` | Strategy D (INT8 Protected) | 2.73 | 3.83 | 114.4 | 0.463 | 0.750 | **0.650** | [0.585, 0.715] | 0.020 |
| `S_N_negative_FP16_W4` | Negative Control (Low Rank) | 3.12 | 4.22 | 129.9 | 0.468 | 0.765 | **0.640** | [0.575, 0.705] | 0.010 |
| `S_N_negative_W8_W4` | Negative Control (INT8 Protected) | 2.73 | 3.99 | 115.8 | 0.463 | 0.770 | **0.635** | [0.565, 0.700] | 0.010 |
| `R0_FP16_W4` | Random Control Set 0 | 3.12 | 4.22 | 130.5 | 0.458 | 0.755 | **0.605** | [0.540, 0.670] | 0.005 |
| `R0_W8_W4` | Random Control Set 0 (INT8) | 2.73 | 3.96 | 115.5 | 0.460 | 0.765 | **0.610** | [0.545, 0.675] | 0.005 |
| `R1_FP16_W4` | Random Control Set 1 | 3.12 | 4.22 | 128.3 | 0.460 | 0.755 | **0.615** | [0.545, 0.680] | 0.010 |
| `R1_W8_W4` | Random Control Set 1 (INT8) | 2.73 | 3.93 | 114.5 | 0.458 | 0.755 | **0.605** | [0.535, 0.670] | 0.010 |
| `R2_FP16_W4` | Random Control Set 2 | 3.12 | 4.22 | 129.8 | 0.458 | 0.745 | **0.615** | [0.550, 0.680] | 0.010 |
| `R3_FP16_W4` | Random Control Set 3 | 3.12 | 4.22 | 124.7 | 0.462 | 0.755 | **0.620** | [0.555, 0.685] | 0.015 |
| `S_G_gradattr_FP16_W4` | Strategy G (Gradient Attribution) | 3.10 | 4.23 | 113.6 | 0.462 | 0.750 | **0.615** | [0.550, 0.680] | 0.005 |
| `S_G_gradattr_W8_W4` | Strategy G (INT8 Protected) | 2.71 | 3.93 | 100.2 | 0.458 | 0.760 | **0.615** | [0.545, 0.680] | 0.005 |