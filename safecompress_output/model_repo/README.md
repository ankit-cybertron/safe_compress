---
base_model: CohereLabs/tiny-aya-global
tags: [safety, quantization, mixed-precision, interpretability, multilingual, hindi]
---
# SafeCompress recipes for Tiny Aya Global

Safety-aware mixed-precision compression: keep five decoder layers at FP16 or INT8 and quantize the rest to NF4 (bitsandbytes). This repo holds the layer recipes and a loader, not weights (the base model is gated; load it yourself).

Headline (Global-MMLU EN+HI, Belebele EN+HI, 200 harmful and 200 benign EN/HI safety prompts): FP16 harmful-refusal 0.655, uniform 4-bit 0.605; best protected arm `S_C_refusalDrop_FP16_W4` 0.655 at 3.12 GB (FP16 6.70 GB, uniform 4-bit 2.53 GB).
Full tables and paired CIs: see the results dataset. Layer sets: `recipes.json`. Loader: `load_safecompress.py`.
Lexsi tools used: AuditKIT (independent refusal scoring).
