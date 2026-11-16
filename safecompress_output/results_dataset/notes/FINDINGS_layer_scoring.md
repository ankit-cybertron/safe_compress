# Layer scoring findings (Tiny Aya Global, 36 layers)

Method: Score A = activation patching on equal-length harmful/benign prompt pairs (EN+HI, PolyGuardMix), measured on the refusal-minus-compliance first-token logit gap. Score B = one layer at a time round-tripped through NF4/INT8, damage vs FP16 measured by KL, gap shift and MMLU letter accuracy on a held-out calibration set.

- baseline gap (refusal minus compliance logit): hi/harmful: 3.14, hi/unharmful: -2.43, en/harmful: 4.90, en/unharmful: -2.60
- Spearman importance vs fragility (KL, NF4): 0.34; importance vs refusal drop: 0.28
- top importance (patch_block): L0 (1.001), L22 (0.309), L17 (0.266), L19 (0.240), L5 (0.233)
- top fragility (KL safety, NF4): L35 (0.011), L0 (0.006), L12 (0.005), L16 (0.005), L7 (0.005)
- top MMLU KL under NF4: L16 (0.012), L12 (0.011), L13 (0.010), L11 (0.010), L7 (0.010)
- largest refusal erosion on harmful prompts (most negative gap shift): L16 (-0.194), L35 (-0.126), L27 (-0.100), L7 (-0.076), L14 (-0.074)
- max single-layer NF4 KL on safety prompts: 0.0110; max INT8 KL: 0.00025

Candidate sets (5 layers each):
- A_importance_patching: [0, 5, 17, 19, 22]
- B_fragility_kl: [0, 7, 12, 16, 35]
- C_refusal_drop_under_nf4: [7, 14, 16, 27, 35]
- D_composite_importance_x_fragility: [0, 14, 16, 17, 19]
- N_negative_control_low_both: [6, 25, 31, 33, 34]

Caveats: patching swaps a component's output at every token position, so very early layers (L0) can score high because they carry prompt content, not because they hold a refusal mechanism. Single-layer quantization damage is small, so rankings are first-order; interactions only show in full-model evaluation. INT8 here is a per-row absmax approximation. Calibration prompts and patching pairs share a pool and must not be reused as test data.