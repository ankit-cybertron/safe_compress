# Limitations & Open Questions — SafeCompress

In accordance with rigorous scientific practice, this document details known limitations, potential confounding factors, and negative results from the SafeCompress investigation.

---

## 1. Lexsi Toolkit Integration & Roles

- **AuditKIT:** Successfully executed and completed (Cells 15 & 16 in `safe-compress.ipynb`). Results provided in `results/tables/auditkit_results.csv` and `results/statistics/auditkit_comparisons.csv`.
- **CircuitKIT:** Integrated into the pipeline via model hook architecture and attribution patching (`scripts/ck_discover.py`, `scripts/ck_inspect.py`, `scripts/ck_diag.py`). Provided the foundational formulation for the refusal-circuit gradient attribution analysis (`scripts/w_attr.py` and `results/processed/gradattr_layer_scores.csv`), directly yielding Strategy G (`[17, 19, 22, 27, 29]`), which achieved 65.0% refusal (+4.5 pp over uniform NF4).
- **SafeTune / CuratorKIT:** Not applicable for post-training quantization without parameter fine-tuning.

---

## 2. Scientific Limitations

### A. Non-Specificity of the Selection Signal (Negative Control Performance)
The negative control arm `S_N_negative_FP16_W4` (layers `[6, 25, 31, 33, 34]`, selected because they ranked *lowest* in both causal importance and quantization fragility) achieved **64.0%** harmful refusal—only $1.5\,\text{pp}$ below the top-performing strategy (`S_C_refusalDrop` at $65.5\%$) and $+3.5\,\text{pp}$ above uniform NF4 ($60.5\%$).
- **Implication:** Protecting *any* 5 layers in FP16 restores a meaningful fraction of safety performance. While targeted selection (`S_C_refusalDrop`) achieves the highest absolute refusal and the strongest AuditKIT score, we cannot claim that safety refusal is localized exclusively to a sparse circuit.

### B. Fixed Layer Budget ($K=5$)
The choice of $K=5$ layers (~14% of the 36-layer network) was held constant across all arms to control parameter volume strictly. We did not conduct a full budget sweep (e.g., $K \in \{1, 2, 3, 5, 8, 12\}$) to determine the minimal sufficient number of protected layers.

### C. Single Base Model Architecture
All findings were obtained on `CohereLabs/tiny-aya-global`. While Tiny Aya Global provides an interesting architectural blend of sliding-window and global attention, findings may not generalize identically to LLaMA, Mistral, or Qwen architectures without re-evaluating layer sensitivities.

### D. Refusal Phrase Matching Limitations
The primary evaluation relies on a 300-character prefix regular expression matcher. While validated against English and Hindi test sets, phrase matching can theoretically yield false positives (e.g., "I cannot provide harmful steps, but here is...") or false negatives (creative refusals). The AuditKIT independent scoring provides external validation but was evaluated on a 200-sample subset due to computational constraints.

### E. Static Post-Training Quantization
SafeCompress evaluated post-training quantization via `bitsandbytes` (NF4 and INT8). We did not evaluate quantization-aware training (QAT), weight-only GPTQ, or AWQ, which might alter which layers are most fragile.
