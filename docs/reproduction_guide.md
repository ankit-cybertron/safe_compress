# Reproduction Guide — SafeCompress

This guide details the step-by-step procedure for reproducing all figures, tables, and model artifacts in SafeCompress.

---

## 1. Hardware & Environment Requirements

- **Local / Verification Environment (Evaluation of Tables & Tests):**
  - Any standard CPU (Linux, macOS, Windows).
  - Python $\geq 3.10$.
  - Requires no GPU and no gated model access.
- **Full Model Execution & Quantization Environment:**
  - Linux with NVIDIA GPU ($\geq 16\,\text{GB}$ VRAM, e.g., T4 $\times 2$, A10G, or A100).
  - CUDA $\geq 12.0$.
  - Hugging Face access token with access to `CohereForAI/aya-23-8B`.

---

## 2. Quickstart: Verifying Results & Running Unit Tests

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ankittiwarifr/safecompress.git
   cd safecompress
   ```

2. **Install local dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

3. **Run the automated test suite:**
   ```bash
   python -m pytest tests/ -v
   ```
   All 14 tests verify schema integrity, layer definitions, bootstrap calculation, refusal phrase matching, and score table consistency without requiring GPU hardware.

4. **Inspect canonical tables and figures:**
   - Results table: `results/tables/results_table.csv`
   - AuditKIT table: `results/tables/auditkit_results.csv`
   - Main tradeoff figure: `figures/tradeoffs/final_results.png`

---

## 3. Loading a Pre-Configured Mixed-Precision Model

You can load any of the 20 experimental arms using the clean loader API:

```python
import torch
from safecompress.model_loader import load_arm

# Load Strategy C (Refusal Drop FP16+NF4 mixed precision)
# Protects layers [7, 14, 16, 27, 35] in FP16, remaining 31 layers in NF4
model = load_arm("S_C_refusalDrop_FP16_W4", device="cuda:0")

print(model)
```

Alternatively, run the standalone loader directly:
```bash
python artifacts/loaders/load_safecompress.py S_C_refusalDrop_FP16_W4
```

---

## 4. Full Experimental Workflow on Kaggle / Slurm

The primary 20-arm experimental pipeline was executed via `notebooks/safe-compress.ipynb`.

### Execution Order:
1. **Cells 1–3:** Setup environment, install pinned dependencies, pull benchmark datasets (`HarmBench`, `PolyGuardMix`, `Global-MMLU`, `Belebele`).
2. **Cell 4:** Generate layer anatomy (`provenance/anatomy.csv`) and probe first-token distributions (`scores/first_tokens.json`).
3. **Cell 5 (`scripts/w_patch.py`):** Causal activation patching on paired refusal prompts.
4. **Cell 6 (`scripts/w_sens.py`):** One-layer-at-a-time NF4 sensitivity profiling.
5. **Cell 7:** Compute composite rankings, correlation statistics, and candidate sets (`scores/candidates.json`).
6. **Cell 8 / 13:** CircuitKIT execution attempts (note: fails with GPU OOM on T4).
7. **Cell 10:** Freeze experimental matrix (`artifacts/manifests/arms.json`).
8. **Cell 11 (`scripts/w_eval.py`):** Evaluate all 20 arms on safety, MMLU, and Belebele. Results saved to `results/raw/<ARM>.json`.
9. **Cell 12:** Aggregate results table, compute bootstrap confidence intervals, and generate final tradeoff figures.
10. **Cells 15–16:** Run Lexsi AuditKIT multilingual refusal scoring on generated outputs.
11. **Cell 17–19:** Publish models and dataset to Hugging Face Hub.
