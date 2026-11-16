# Figures Directory — SafeCompress

All figures generated across the SafeCompress research experiments, categorized by analytical domain.

---

## Directory Organization

### `figures/methodology/` (System Architecture & Pipeline)
- **`safecompress_architecture_diagram.png`:** Comprehensive four-stage system architecture diagram rendered at 300 DPI on a clean white background, depicting base model layer topology, multi-signal profiling, layer selection strategies, mixed-precision allocation, and the dual-scorer evaluation protocol.

### `figures/tradeoffs/` (Memory vs. Safety vs. Capability)
- **`safety_capability_tradeoff_20arms.png`:** The 20-arm scatter plot of Harmful-prompt refusal rate (%) versus Global-MMLU accuracy (%) with numbered markers matching the configuration key, generated from `results/tables/results_table.csv`.
- **`final_results.png`:** Four-panel bar chart overview displaying Global-MMLU, Belebele, harmful refusal, and first-token refusal logit gap with 95% bootstrap confidence intervals across all 20 arms.
- **`fig1_pareto_safety_vs_compression.png`:** Pareto frontier plot from the earlier pilot experiment.
- **`05_safety_capability_tradeoff_clean.png`:** Clean publication-ready scatter plot of safety refusal vs. MMLU capability.
- **`tiny_aya_18_arm_results_table.png`:** Visual summary table comparing arm metrics.

### `figures/safety/` (Safety & Refusal Dynamics)
- **`auditkit_refusal.png`:** Independent evaluation results from Lexsi AuditKIT comparing refusal rates and bootstrap confidence intervals across tested arms.
- **`fig2_refusal_vs_false_refusal.png`:** ROC-style trade-off between harmful request refusal and benign request over-refusal.
- **`fig4_circuit_vs_random.png`:** Comparative bar chart contrasting targeted layer selection strategies against random layer baselines.

### `figures/layer_analysis/` (Layer Profiling & Interpretability)
- **`layer_scores.png`:** Composite 36-layer bar chart displaying activation patching importance, NF4 KL divergence fragility, and refusal drop for each transformer layer.
- **`fig3_layer_attribution_map.png`:** Heatmap showing layer-by-layer attribution across sequence positions.
