# Project Audit — SafeCompress

> [!NOTE]
> This audit documents the full file inventory, the dual-experiment structure, canonical sources, and known issues in the `lexis_submission` workspace.

---

## 1. Overview

The submission contains **two distinct experiments**:

| Label | Directory | Notebook | Status |
|---|---|---|---|
| **Earlier experiment** | `safecompress/` | (Kaggle notebooks, pre-final) | Pilot / archived |
| **Final / published experiment** | `safecompress_output/` | `safe-compress.ipynb` (20-cell Kaggle execution) | Canonical |

The final experiment supersedes the earlier one in every respect. The `safecompress/` directory is retained for provenance only.

---

## 2. The Notebooks

| Notebook | Description | Status |
|---|---|---|
| `safe-compress.ipynb` | Full 20-cell Kaggle execution notebook for the **final** experiment. Covers all phases: environment setup, layer scoring, arm construction, evaluation, AuditKIT, HF publication. | **Canonical** |

---

## 3. Full File Inventory

### 3.1 Top-Level

| Path | Size / Status | Role |
|---|---|---|
| `safe-compress.ipynb` | Working | Final experiment notebook (Kaggle, 20 cells) |
| `README.md` | Working | Submission overview |

### 3.2 `safecompress/` — Earlier Experiment

| Path | Size / Status | Role |
|---|---|---|
| `safecompress/scripts/load_safecompress.py` | **EMPTY (0 bytes)** | Stale placeholder; working version is in `safecompress_output/` |
| `safecompress/scripts/safety_part_worker.py` | **EMPTY** | Stale placeholder |
| `safecompress/scripts/patching_worker.py` | **EMPTY** | Stale placeholder |
| `safecompress/results/tables/main_table.md` | **EMPTY** | Stale placeholder |
| `safecompress/results/tables/main_table.tex` | **EMPTY** | Stale placeholder |
| `safecompress/` (other files) | Various | Earlier-experiment artefacts (pilot) |

> [!WARNING]
> All five files above are empty. Do not use them as source material. The working equivalents are under `safecompress_output/`.

### 3.3 `safecompress_output/` — Final Experiment (Canonical)

| Path | Size / Status | Role |
|---|---|---|
| `safecompress_output/model_repo/load_safecompress.py` | Working | **Canonical** model loader |
| `safecompress_output/model_repo/final_results.png` | Working | Results figure (duplicate of figs copy, same SHA) |
| `safecompress_output/results_dataset/figs/final_results.png` | Working | Results figure (duplicate of model_repo copy, same SHA) |
| `safecompress_output/results_dataset/code/` | Working | **Canonical** experiment scripts |
| `safecompress_output/results_dataset/results_table.csv` | Working | **Canonical** 20-arm results table |
| `safecompress_output/results_dataset/results_comparisons.csv` | Working | Pairwise bootstrap CIs |
| `safecompress_output/results_dataset/auditkit_results.csv` | Working | AuditKIT scores (13 arms) |
| `safecompress_output/results_dataset/auditkit_comparisons.csv` | Working | AuditKIT pairwise comparisons (14 rows) |
| `safecompress_output/results_dataset/FINDINGS_layer_scoring.md` | Working | Layer scoring narrative |
| `safecompress_output/results_dataset/candidates.json` | Working | Layer set definitions |

---

## 4. The Two Experiments Distinguished

### 4.1 Earlier Experiment (`safecompress/`)

- **Layer selection method:** Activation patching only  
- **Layer sets defined:**
  - `circuit` = [0, 3, 5, 17, 22]  
  - `least-effect` = [20, 28, 30, 31, 35]  
- **Arms:** E0_FP16, E1_uniform_W4, E3_circuit_FP16_W4, E4_least_FP16_W4, E5–E12 variants  
- **Evaluation scale:** 1,477 safety prompts, 500 paired MMLU per language  
- **Main finding:** Circuit layers did **not** help refusal; FP16_W4 circuit arm was −1.2 pp (95% CI: −2.4 to −0.1) versus baseline — worse than random in that scheme.  
- **Disk sizes (state_dict):** FP16 = 6.70 GB, uniform W4 = 2.53 GB, circuit FP16+W4 = 3.11 GB, circuit W8+W4 = 2.72 GB  

### 4.2 Final / Published Experiment (`safecompress_output/`)

- **Layer selection methods:** Activation patching, NF4 KL-sensitivity (fragility), refusal-gap erosion, composite scoring, negative control, gradient attribution  
- **Layer sets defined:** 6 named strategies + 4 random control sets (see `docs/layer_selection.md`)  
- **Arms:** 20 total (2 baseline + 12 strategy × 2 precision + 6 random)  
- **Evaluation scale:** 400 safety prompts, 600 MMLU, 200 Belebele  
- **Statistical method:** Paired bootstrap CI, n = 10,000 resamples  
- **HF artifacts:** `ankittiwarifr/safecompress` (model collection), `ankittiwarifr/safecompress-results` (results dataset)  

---

## 5. Canonical Sources

| Artefact | Canonical Location |
|---|---|
| Model loader | `safecompress_output/model_repo/load_safecompress.py` |
| Experiment scripts | `safecompress_output/results_dataset/code/` |
| Results table | `safecompress_output/results_dataset/results_table.csv` |
| AuditKIT results | `safecompress_output/results_dataset/auditkit_results.csv` |
| Layer scoring findings | `safecompress_output/results_dataset/FINDINGS_layer_scoring.md` |
| Layer set definitions | `safecompress_output/results_dataset/candidates.json` |
| Results figure | `safecompress_output/results_dataset/figs/final_results.png` |

---

## 6. Known Issues

| Issue | Detail |
|---|---|
| Five empty files in `safecompress/scripts/` and `safecompress/results/tables/` | Stale placeholders; working versions are elsewhere |
| Duplicate `final_results.png` | Identical file (SHA-256: f9d0f5fb…) exists at both `model_repo/` and `figs/`; either is usable |
| `safecompress/` scripts are empty | Do not cite or execute these files |
