# SafeCompress final evaluation (Tiny Aya Global, bitsandbytes NF4 / INT8, real quantization)

Layer sets come from per-layer patching importance and one-layer-at-a-time quantization fragility (see FINDINGS_layer_scoring.md). Random sets: [[7, 13, 15, 24, 32], [11, 15, 22, 27, 30], [13, 14, 27, 31, 34], [1, 5, 19, 27, 33]]. Evaluation: 600 Global-MMLU items, 200 Belebele items, 400 safety prompts (EN: HarmBench, StrongREJECT, XSTest; HI: IndicSafe). Refusal = phrase matcher on the first 300 characters of a 48-token reply plus the continuous first-token refusal gap.

| arm | protected layers | GB | tok/s | MMLU | Belebele | refuse harmful | gap harmful | refuse benign |
|---|---|---|---|---|---|---|---|---|
| E0_FP16 | [] | 6.70 | 146 | 0.475 | 0.770 | 0.655 | 1.86 | 0.025 |
| E1_uniform_W4 | [] | 2.53 | 122 | 0.457 | 0.760 | 0.605 | 1.69 | 0.010 |
| S_A_importance_FP16_W4 | [0, 5, 17, 19, 22] | 3.12 | 131 | 0.458 | 0.760 | 0.630 | 1.56 | 0.015 |
| S_A_importance_W8_W4 | [0, 5, 17, 19, 22] | 2.73 | 114 | 0.458 | 0.765 | 0.630 | 1.51 | 0.015 |
| S_B_fragility_FP16_W4 | [0, 7, 12, 16, 35] | 3.12 | 130 | 0.468 | 0.755 | 0.645 | 2.15 | 0.025 |
| S_B_fragility_W8_W4 | [0, 7, 12, 16, 35] | 2.73 | 111 | 0.467 | 0.760 | 0.645 | 2.15 | 0.025 |
| S_C_refusalDrop_FP16_W4 | [7, 14, 16, 27, 35] | 3.12 | 129 | 0.463 | 0.760 | 0.655 | 2.39 | 0.015 |
| S_C_refusalDrop_W8_W4 | [7, 14, 16, 27, 35] | 2.73 | 111 | 0.470 | 0.755 | 0.650 | 2.31 | 0.015 |
| S_D_composite_FP16_W4 | [0, 14, 16, 17, 19] | 3.12 | 131 | 0.467 | 0.750 | 0.650 | 1.89 | 0.015 |
| S_D_composite_W8_W4 | [0, 14, 16, 17, 19] | 2.73 | 114 | 0.463 | 0.750 | 0.650 | 1.90 | 0.020 |
| S_N_negative_FP16_W4 | [6, 25, 31, 33, 34] | 3.12 | 130 | 0.468 | 0.765 | 0.640 | 1.70 | 0.010 |
| S_N_negative_W8_W4 | [6, 25, 31, 33, 34] | 2.73 | 116 | 0.463 | 0.770 | 0.635 | 1.68 | 0.010 |
| R0_FP16_W4 | [7, 13, 15, 24, 32] | 3.12 | 131 | 0.458 | 0.755 | 0.605 | 1.86 | 0.005 |
| R0_W8_W4 | [7, 13, 15, 24, 32] | 2.73 | 116 | 0.460 | 0.765 | 0.610 | 1.75 | 0.005 |
| R1_FP16_W4 | [11, 15, 22, 27, 30] | 3.12 | 128 | 0.460 | 0.755 | 0.615 | 1.32 | 0.010 |
| R1_W8_W4 | [11, 15, 22, 27, 30] | 2.73 | 115 | 0.458 | 0.755 | 0.605 | 1.29 | 0.010 |
| R2_FP16_W4 | [13, 14, 27, 31, 34] | 3.12 | 130 | 0.458 | 0.745 | 0.615 | 1.72 | 0.010 |
| R3_FP16_W4 | [1, 5, 19, 27, 33] | 3.12 | 125 | 0.462 | 0.755 | 0.620 | 1.75 | 0.015 |
| S_G_gradattr_FP16_W4 | [17, 19, 22, 27, 29] | 3.10 | 114 | 0.462 | 0.750 | 0.615 | 1.67 | 0.005 |
| S_G_gradattr_W8_W4 | [17, 19, 22, 27, 29] | 2.71 | 100 | 0.458 | 0.760 | 0.615 | 1.67 | 0.005 |

Paired differences:

| metric | comparison | diff | 95% CI |
|---|---|---|---|
| mmlu | E1 uniform W4 - E0 FP16 | -0.018 | [-0.043, +0.008] |
| mmlu | A_importance [FP16_W4] - uniform W4 | +0.002 | [-0.015, +0.018] |
| mmlu | A_importance [FP16_W4] - mean(4 random) | -0.001 | [-0.017, +0.015] |
| mmlu | B_fragility [FP16_W4] - uniform W4 | +0.012 | [-0.005, +0.028] |
| mmlu | B_fragility [FP16_W4] - mean(4 random) | +0.009 | [-0.006, +0.023] |
| mmlu | C_refusalDrop [FP16_W4] - uniform W4 | +0.007 | [-0.008, +0.022] |
| mmlu | C_refusalDrop [FP16_W4] - mean(4 random) | +0.004 | [-0.010, +0.017] |
| mmlu | D_composite [FP16_W4] - uniform W4 | +0.010 | [-0.008, +0.028] |
| mmlu | D_composite [FP16_W4] - mean(4 random) | +0.007 | [-0.008, +0.022] |
| mmlu | N_negative [FP16_W4] - uniform W4 | +0.012 | [-0.003, +0.027] |
| mmlu | N_negative [FP16_W4] - mean(4 random) | +0.009 | [-0.005, +0.023] |
| mmlu | G_gradattr [FP16_W4] - uniform W4 | +0.005 | [-0.010, +0.020] |
| mmlu | G_gradattr [FP16_W4] - mean(4 random) | +0.002 | [-0.010, +0.015] |
| mmlu | A_importance [W8_W4] - uniform W4 | +0.002 | [-0.017, +0.020] |
| mmlu | A_importance [W8_W4] - mean(2 random) | -0.001 | [-0.020, +0.019] |
| mmlu | B_fragility [W8_W4] - uniform W4 | +0.010 | [-0.010, +0.030] |
| mmlu | B_fragility [W8_W4] - mean(2 random) | +0.007 | [-0.011, +0.025] |
| mmlu | C_refusalDrop [W8_W4] - uniform W4 | +0.013 | [-0.002, +0.028] |
| mmlu | C_refusalDrop [W8_W4] - mean(2 random) | +0.011 | [-0.005, +0.027] |
| mmlu | D_composite [W8_W4] - uniform W4 | +0.007 | [-0.013, +0.027] |
| mmlu | D_composite [W8_W4] - mean(2 random) | +0.004 | [-0.015, +0.023] |
| mmlu | N_negative [W8_W4] - uniform W4 | +0.007 | [-0.008, +0.022] |
| mmlu | N_negative [W8_W4] - mean(2 random) | +0.004 | [-0.013, +0.022] |
| mmlu | G_gradattr [W8_W4] - uniform W4 | +0.002 | [-0.013, +0.017] |
| mmlu | G_gradattr [W8_W4] - mean(2 random) | -0.001 | [-0.017, +0.016] |
| bele | E1 uniform W4 - E0 FP16 | -0.010 | [-0.045, +0.025] |
| bele | A_importance [FP16_W4] - uniform W4 | +0.000 | [-0.015, +0.015] |
| bele | A_importance [FP16_W4] - mean(4 random) | +0.007 | [-0.011, +0.025] |
| bele | B_fragility [FP16_W4] - uniform W4 | -0.005 | [-0.030, +0.015] |
| bele | B_fragility [FP16_W4] - mean(4 random) | +0.003 | [-0.021, +0.026] |
| bele | C_refusalDrop [FP16_W4] - uniform W4 | +0.000 | [-0.025, +0.025] |
| bele | C_refusalDrop [FP16_W4] - mean(4 random) | +0.007 | [-0.014, +0.029] |
| bele | D_composite [FP16_W4] - uniform W4 | -0.010 | [-0.030, +0.010] |
| bele | D_composite [FP16_W4] - mean(4 random) | -0.003 | [-0.024, +0.018] |
| bele | N_negative [FP16_W4] - uniform W4 | +0.005 | [+0.000, +0.015] |
| bele | N_negative [FP16_W4] - mean(4 random) | +0.013 | [-0.003, +0.029] |
| bele | G_gradattr [FP16_W4] - uniform W4 | -0.010 | [-0.030, +0.005] |
| bele | G_gradattr [FP16_W4] - mean(4 random) | -0.003 | [-0.020, +0.015] |
| bele | A_importance [W8_W4] - uniform W4 | +0.005 | [-0.010, +0.020] |
| bele | A_importance [W8_W4] - mean(2 random) | +0.005 | [-0.018, +0.028] |
| bele | B_fragility [W8_W4] - uniform W4 | +0.000 | [-0.025, +0.025] |
| bele | B_fragility [W8_W4] - mean(2 random) | +0.000 | [-0.028, +0.028] |
| bele | C_refusalDrop [W8_W4] - uniform W4 | -0.005 | [-0.030, +0.015] |
| bele | C_refusalDrop [W8_W4] - mean(2 random) | -0.005 | [-0.033, +0.022] |
| bele | D_composite [W8_W4] - uniform W4 | -0.010 | [-0.030, +0.010] |
| bele | D_composite [W8_W4] - mean(2 random) | -0.010 | [-0.037, +0.015] |
| bele | N_negative [W8_W4] - uniform W4 | +0.010 | [+0.000, +0.025] |
| bele | N_negative [W8_W4] - mean(2 random) | +0.010 | [-0.013, +0.033] |
| bele | G_gradattr [W8_W4] - uniform W4 | +0.000 | [-0.015, +0.015] |
| bele | G_gradattr [W8_W4] - mean(2 random) | +0.000 | [-0.022, +0.022] |
| refuse_harm | E1 uniform W4 - E0 FP16 | -0.050 | [-0.090, -0.015] |
| refuse_harm | A_importance [FP16_W4] - uniform W4 | +0.025 | [+0.000, +0.050] |
| refuse_harm | A_importance [FP16_W4] - mean(4 random) | +0.016 | [-0.007, +0.040] |
| refuse_harm | B_fragility [FP16_W4] - uniform W4 | +0.040 | [+0.010, +0.070] |
| refuse_harm | B_fragility [FP16_W4] - mean(4 random) | +0.031 | [+0.005, +0.060] |
| refuse_harm | C_refusalDrop [FP16_W4] - uniform W4 | +0.050 | [+0.025, +0.080] |
| refuse_harm | C_refusalDrop [FP16_W4] - mean(4 random) | +0.041 | [+0.014, +0.072] |
| refuse_harm | D_composite [FP16_W4] - uniform W4 | +0.045 | [+0.020, +0.075] |
| refuse_harm | D_composite [FP16_W4] - mean(4 random) | +0.036 | [+0.013, +0.062] |
| refuse_harm | N_negative [FP16_W4] - uniform W4 | +0.035 | [+0.010, +0.060] |
| refuse_harm | N_negative [FP16_W4] - mean(4 random) | +0.026 | [+0.006, +0.049] |
| refuse_harm | G_gradattr [FP16_W4] - uniform W4 | +0.010 | [-0.010, +0.030] |
| refuse_harm | G_gradattr [FP16_W4] - mean(4 random) | +0.001 | [-0.019, +0.021] |
| refuse_harm | A_importance [W8_W4] - uniform W4 | +0.025 | [+0.000, +0.050] |
| refuse_harm | A_importance [W8_W4] - mean(2 random) | +0.022 | [+0.003, +0.043] |
| refuse_harm | B_fragility [W8_W4] - uniform W4 | +0.040 | [+0.010, +0.070] |
| refuse_harm | B_fragility [W8_W4] - mean(2 random) | +0.037 | [+0.005, +0.072] |
| refuse_harm | C_refusalDrop [W8_W4] - uniform W4 | +0.045 | [+0.020, +0.075] |
| refuse_harm | C_refusalDrop [W8_W4] - mean(2 random) | +0.043 | [+0.018, +0.072] |
| refuse_harm | D_composite [W8_W4] - uniform W4 | +0.045 | [+0.020, +0.075] |
| refuse_harm | D_composite [W8_W4] - mean(2 random) | +0.043 | [+0.018, +0.070] |
| refuse_harm | N_negative [W8_W4] - uniform W4 | +0.030 | [+0.010, +0.055] |
| refuse_harm | N_negative [W8_W4] - mean(2 random) | +0.028 | [+0.005, +0.052] |
| refuse_harm | G_gradattr [W8_W4] - uniform W4 | +0.010 | [-0.015, +0.035] |
| refuse_harm | G_gradattr [W8_W4] - mean(2 random) | +0.007 | [-0.015, +0.030] |
| gap_harm | E1 uniform W4 - E0 FP16 | -0.170 | [-0.295, -0.045] |
| gap_harm | A_importance [FP16_W4] - uniform W4 | -0.124 | [-0.187, -0.061] |
| gap_harm | A_importance [FP16_W4] - mean(4 random) | -0.099 | [-0.166, -0.031] |
| gap_harm | B_fragility [FP16_W4] - uniform W4 | +0.460 | [+0.389, +0.528] |
| gap_harm | B_fragility [FP16_W4] - mean(4 random) | +0.486 | [+0.422, +0.546] |
| gap_harm | C_refusalDrop [FP16_W4] - uniform W4 | +0.706 | [+0.635, +0.774] |
| gap_harm | C_refusalDrop [FP16_W4] - mean(4 random) | +0.731 | [+0.674, +0.785] |
| gap_harm | D_composite [FP16_W4] - uniform W4 | +0.202 | [+0.148, +0.258] |
| gap_harm | D_composite [FP16_W4] - mean(4 random) | +0.228 | [+0.173, +0.283] |
| gap_harm | N_negative [FP16_W4] - uniform W4 | +0.015 | [-0.023, +0.052] |
| gap_harm | N_negative [FP16_W4] - mean(4 random) | +0.041 | [+0.009, +0.072] |
| gap_harm | G_gradattr [FP16_W4] - uniform W4 | -0.022 | [-0.063, +0.020] |
| gap_harm | G_gradattr [FP16_W4] - mean(4 random) | +0.004 | [-0.044, +0.052] |
| gap_harm | A_importance [W8_W4] - uniform W4 | -0.181 | [-0.251, -0.110] |
| gap_harm | A_importance [W8_W4] - mean(2 random) | -0.013 | [-0.094, +0.065] |
| gap_harm | B_fragility [W8_W4] - uniform W4 | +0.460 | [+0.387, +0.529] |
| gap_harm | B_fragility [W8_W4] - mean(2 random) | +0.628 | [+0.557, +0.699] |
| gap_harm | C_refusalDrop [W8_W4] - uniform W4 | +0.623 | [+0.557, +0.688] |
| gap_harm | C_refusalDrop [W8_W4] - mean(2 random) | +0.791 | [+0.731, +0.852] |
| gap_harm | D_composite [W8_W4] - uniform W4 | +0.215 | [+0.156, +0.276] |
| gap_harm | D_composite [W8_W4] - mean(2 random) | +0.383 | [+0.315, +0.452] |
| gap_harm | N_negative [W8_W4] - uniform W4 | -0.008 | [-0.050, +0.031] |
| gap_harm | N_negative [W8_W4] - mean(2 random) | +0.160 | [+0.115, +0.205] |
| gap_harm | G_gradattr [W8_W4] - uniform W4 | -0.013 | [-0.057, +0.032] |
| gap_harm | G_gradattr [W8_W4] - mean(2 random) | +0.155 | [+0.099, +0.213] |
| refuse_benign | E1 uniform W4 - E0 FP16 | -0.015 | [-0.040, +0.005] |
| refuse_benign | A_importance [FP16_W4] - uniform W4 | +0.005 | [-0.010, +0.025] |
| refuse_benign | A_importance [FP16_W4] - mean(4 random) | +0.005 | [-0.007, +0.021] |
| refuse_benign | B_fragility [FP16_W4] - uniform W4 | +0.015 | [-0.005, +0.040] |
| refuse_benign | B_fragility [FP16_W4] - mean(4 random) | +0.015 | [-0.004, +0.037] |
| refuse_benign | C_refusalDrop [FP16_W4] - uniform W4 | +0.005 | [+0.000, +0.015] |
| refuse_benign | C_refusalDrop [FP16_W4] - mean(4 random) | +0.005 | [-0.003, +0.018] |
| refuse_benign | D_composite [FP16_W4] - uniform W4 | +0.005 | [-0.010, +0.025] |
| refuse_benign | D_composite [FP16_W4] - mean(4 random) | +0.005 | [-0.007, +0.021] |
| refuse_benign | N_negative [FP16_W4] - uniform W4 | +0.000 | [+0.000, +0.000] |
| refuse_benign | N_negative [FP16_W4] - mean(4 random) | +0.000 | [-0.004, +0.004] |
| refuse_benign | G_gradattr [FP16_W4] - uniform W4 | -0.005 | [-0.015, +0.000] |
| refuse_benign | G_gradattr [FP16_W4] - mean(4 random) | -0.005 | [-0.015, +0.000] |
| refuse_benign | A_importance [W8_W4] - uniform W4 | +0.005 | [-0.010, +0.025] |
| refuse_benign | A_importance [W8_W4] - mean(2 random) | +0.007 | [-0.005, +0.025] |
| refuse_benign | B_fragility [W8_W4] - uniform W4 | +0.015 | [-0.005, +0.040] |
| refuse_benign | B_fragility [W8_W4] - mean(2 random) | +0.018 | [+0.000, +0.040] |
| refuse_benign | C_refusalDrop [W8_W4] - uniform W4 | +0.005 | [+0.000, +0.015] |
| refuse_benign | C_refusalDrop [W8_W4] - mean(2 random) | +0.007 | [+0.000, +0.020] |
| refuse_benign | D_composite [W8_W4] - uniform W4 | +0.010 | [-0.010, +0.030] |
| refuse_benign | D_composite [W8_W4] - mean(2 random) | +0.013 | [-0.003, +0.030] |
| refuse_benign | N_negative [W8_W4] - uniform W4 | +0.000 | [+0.000, +0.000] |
| refuse_benign | N_negative [W8_W4] - mean(2 random) | +0.003 | [+0.000, +0.007] |
| refuse_benign | G_gradattr [W8_W4] - uniform W4 | -0.005 | [-0.015, +0.000] |
| refuse_benign | G_gradattr [W8_W4] - mean(2 random) | -0.003 | [-0.007, +0.000] |

Limitations: phrase-matcher refusal (no judge model), one model, one quantizer family, random sets may contain fragile layers by chance, CI width reflects the sample sizes above.
