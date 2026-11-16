# Layer Selection Methodology — SafeCompress

SafeCompress systematically tested six analytical scoring methods to rank the 36 decoder layers of `CohereForAI/aya-23-8B`.

---

## 1. Candidate Strategy Layer Sets

Each strategy selects exactly $K=5$ layers based on a distinct empirical criterion:

| Strategy Name | Selection Criterion | Protected Layer Indices | Architectural Breakdown |
|---|---|---|---|
| **`A_importance`** | Top causal mediation under activation patching (`patch_block`) | `[0, 5, 17, 19, 22]` | 1 full-attn (`19`), 4 sliding-attn |
| **`B_fragility`** | Highest output KL divergence under single-layer NF4 (`kl_safety_nf4`) | `[0, 7, 12, 16, 35]` | 2 full-attn (`7, 35`), 3 sliding-attn |
| **`C_refusalDrop`** | Largest decrease in refusal logit gap under single-layer NF4 | `[7, 14, 16, 27, 35]` | 3 full-attn (`7, 27, 35`), 2 sliding-attn |
| **`D_composite`** | Highest product of normalized importance and normalized fragility | `[0, 14, 16, 17, 19]` | 1 full-attn (`19`), 4 sliding-attn |
| **`G_gradattr`** | Top gradient attribution on refusal gap (`gradattr_score`) | `[17, 19, 22, 27, 29]` | 2 full-attn (`19, 27`), 3 sliding-attn |
| **`N_negative`** | Bottom ranked layers on both importance and fragility (control) | `[6, 25, 31, 33, 34]` | 1 full-attn (`31`), 4 sliding-attn |

---

## 2. Quantitative Layer Ranking Profile

From `results/processed/layer_scores.csv` and `FINDINGS_layer_scoring.md`:

### Top 5 Layers by Causal Importance (`patch_block`)
- **Layer 0:** 1.001 (Early embedding integration)
- **Layer 22:** 0.309 (Mid-to-late semantic processing)
- **Layer 17:** 0.266 (Mid-layer representations)
- **Layer 19:** 0.240 (Full attention global routing)
- **Layer 5:** 0.233 (Early syntactic extraction)

### Top 5 Layers by Quantization Fragility (KL divergence under NF4)
- **Layer 35:** 0.011 (Final pre-unembed layer; extremely sensitive to precision)
- **Layer 0:** 0.006 (First layer; accumulates downstream quantization noise)
- **Layer 12:** 0.005
- **Layer 16:** 0.005
- **Layer 7:** 0.005 (Full attention global routing)

### Top 5 Layers by Refusal Drop Under NF4 (Reduction in refusal log-odds)
- **Layer 16:** $-0.194$ (Largest single-layer safety erosion)
- **Layer 35:** $-0.126$ (Substantial refusal degradation)
- **Layer 27:** $-0.100$ (Full attention layer)
- **Layer 7:** $-0.076$ (Full attention layer)
- **Layer 14:** $-0.074$

---

## 3. Metric Independence and Correlations

Analysis of rank correlations across all 36 layers reveals that causal importance and quantization fragility capture distinct phenomena:

- **Spearman $\rho(\text{Importance}, \text{Fragility KL}):$** $0.34$ ($p < 0.05$)  
  Moderate positive correlation. Layers that are causally critical for refusal are somewhat more likely to suffer logit shifts under quantization, but the overlap is far from complete.
- **Spearman $\rho(\text{Importance}, \text{Refusal Gap Drop}):$** $0.28$ ($p > 0.05$)  
  Weak correlation. Layers whose perturbation specifically damages refusal are distinct from layers that carry high clean-to-corrupted mediation.
- **Full Attention Over-representation:** In Strategy C (`refusalDrop`), 3 of 5 layers are full-attention layers (`7, 27, 35`), despite full-attention layers constituting only 25% of the total network. This indicates that global context layers are disproportionately vulnerable to 4-bit quantization collapse regarding safety refusal.
