# Duplicate and Archive Audit — SafeCompress

> [!NOTE]
> This document lists all duplicate files identified in the workspace, their SHA-256 hashes where known, and recommended disposition for each copy.

---

## Duplicate Files

### 1. `final_results.png`

| Property | Value |
|---|---|
| **File A** | `safecompress_output/model_repo/final_results.png` |
| **File B** | `safecompress_output/results_dataset/figs/final_results.png` |
| **SHA-256** | `f9d0f5fb…` (identical for both) |
| **Content** | Results figure for the final 20-arm experiment |
| **Are they identical?** | ✅ Yes — byte-for-byte identical |
| **Recommended disposition** | Keep **File B** (`figs/`) as the canonical copy (it is in the results dataset alongside other figures). Keep **File A** (`model_repo/`) as a convenience copy for the HF model repository upload. No action strictly required; both are safe to retain. |

---

## Mismatched Files (Same Name, Different Content)

### 2. `load_safecompress.py`

| Property | Value |
|---|---|
| **File A** | `safecompress/scripts/load_safecompress.py` |
| **File B** | `safecompress_output/model_repo/load_safecompress.py` |
| **Are they identical?** | ❌ No |
| **File A status** | **EMPTY (0 bytes)** — stale placeholder from the earlier experiment |
| **File B status** | Working implementation — canonical model loader for the final experiment |
| **Recommended disposition** | Use **File B** exclusively. File A should be clearly marked as deprecated or removed to avoid confusion. Do not execute or cite File A. |

---

## Empty Files (Stale Placeholders)

The following files exist in the `safecompress/` directory but are empty. They are listed here for completeness. None of them are duplicates of working files in the same location; their working equivalents (where they exist) are in `safecompress_output/`.

| Path | Status | Recommended Action |
|---|---|---|
| `safecompress/scripts/load_safecompress.py` | EMPTY (0 bytes) | Archive or delete; working version at `safecompress_output/model_repo/load_safecompress.py` |
| `safecompress/scripts/safety_part_worker.py` | EMPTY | Archive or delete; working equivalent in `safecompress_output/results_dataset/code/` |
| `safecompress/scripts/patching_worker.py` | EMPTY | Archive or delete; working equivalent in `safecompress_output/results_dataset/code/` |
| `safecompress/results/tables/main_table.md` | EMPTY | Archive or delete; canonical results in `results_table.csv` |
| `safecompress/results/tables/main_table.tex` | EMPTY | Archive or delete; no working equivalent (LaTeX table was not generated in final experiment) |

---

## Summary

| Category | Count | Action Required |
|---|---|---|
| Byte-for-byte duplicates | 1 (final_results.png) | Low priority; keep both |
| Name collision, different content | 1 (load_safecompress.py) | High priority: use `safecompress_output/` version only |
| Empty placeholders | 5 | Medium priority: archive or delete to avoid confusion |
