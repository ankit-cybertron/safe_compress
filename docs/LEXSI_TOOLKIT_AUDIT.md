# Lexsi Toolkit Integration & Audit — SafeCompress

> [!NOTE]
> SafeCompress integrates two primary Lexsi toolkits—**CircuitKIT** and **AuditKIT**—into an end-to-end safety-aware compression research pipeline.

---

## Summary Table

| Tool | Status | Role & Evidence in Research Pipeline |
|---|---|---|
| **CircuitKIT** | ✅ COMPLETED & INTEGRATED | Refusal circuit localization, Edge Attribution Patching with Integrated Gradients (`eap-ig`), and attribution analysis on GPU (peak 24 GB VRAM). Discovered the refusal-circuit layer set `[0, 14, 16, 20, 35]` (`S_CK_circuitkit`). Scripts: `scripts/ck_job.py`, `scripts/ck_discover.py`, `scripts/ck_inspect.py`, `scripts/ck_diag.py`; scores: `results/processed/ck_layer_scores.json`, `results/processed/ck_scores_sidecar.json`. |
| **AuditKIT** | ✅ COMPLETED & INTEGRATED | Independent multilingual refusal evaluation and validation of model replies across experimental arms. Results: `results/tables/auditkit_results.csv`, `results/statistics/auditkit_comparisons.csv`; Cells 15–16. |
| **SafeTune** | ⬜ NOT APPLICABLE | Post-training quantization directly compresses representation precision without parameter fine-tuning. |
| **CuratorKIT** | ⬜ NOT APPLICABLE | Standard established public benchmarks (`HarmBench`, `PolyGuardMix`) were used directly. |

---

## CircuitKIT Integration Details

| Field | Value |
|---|---|
| **Tool** | CircuitKIT (`lexsi/circuitkit`) |
| **Repository** | Hackathon Lexsi Toolkit |
| **Functionality Used** | Refusal circuit analysis, model hook architecture, and attribution-based circuit extraction |
| **Status** | ✅ COMPLETED & INTEGRATED |
| **Evidence Paths** | `scripts/ck_job.py`, `scripts/ck_discover.py`, `scripts/ck_inspect.py`, `scripts/ck_diag.py`, `results/processed/ck_layer_scores.json`, `results/processed/ck_scores_sidecar.json`, `results/tables/results_table.csv` |
| **Notebook Execution** | Cells 8 & 13 in `notebooks/safe-compress.ipynb` and HF execution job `scripts/ck_job.py` |
| **Downstream Impact** | Generated the CircuitKIT EAP-IG refusal circuit set (`[0, 14, 16, 20, 35]`). In full-model evaluation, `S_CK_circuitkit_FP16_W4` achieved **66.0%** harmful refusal (+5.5 pp over uniform NF4) and **0.477** Global-MMLU accuracy (outperforming unquantized FP16 baseline 0.475). |

---

## AuditKIT Integration Details

| Field | Value |
|---|---|
| **Tool** | AuditKIT (`lexsi/auditkit`) |
| **Repository** | Hackathon Lexsi Toolkit |
| **Functionality Used** | Independent multilingual refusal scoring and cross-validation |
| **Status** | ✅ COMPLETED & INTEGRATED |
| **Evidence Paths** | `results/tables/auditkit_results.csv`, `results/statistics/auditkit_comparisons.csv` |
| **Notebook Execution** | Cells 15 and 16 in `notebooks/safe-compress.ipynb` |
| **Downstream Impact** | Evaluated 12 key arms on 100 harmful + 100 benign probes across English and Hindi. Confirmed that mixed-precision layer protection restores safety refusal to **56.0%** vs. **44.0%** for uniform NF4 ($+12.0\,\text{pp}$, 95% bootstrap CI: $[+8.0, +17.0]$). |

---

## Multi-Toolkit Pipeline Synergy

SafeCompress combines both toolkits in a complementary dual-stage workflow:
1. **CircuitKIT (Identification Stage):** Leveraged for model hook instrumentation, activation analysis, and attribution-guided refusal circuit identification.
2. **AuditKIT (Validation Stage):** Provides independent, external evaluation of the resulting mixed-precision models to confirm that safety boundaries are preserved across languages.
