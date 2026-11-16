# Lexsi Toolkit Integration & Audit — SafeCompress

> [!NOTE]
> SafeCompress integrates two primary Lexsi toolkits—**CircuitKIT** and **AuditKIT**—into an end-to-end safety-aware compression research pipeline.

---

## Summary Table

| Tool | Status | Role & Evidence in Research Pipeline |
|---|---|---|
| **CircuitKIT** | ✅ COMPLETED & INTEGRATED | Refusal circuit localization, model hook architecture, and gradient attribution analysis. Guided the identification of Strategy G layers (`[17, 19, 22, 27, 29]`). Scripts: `scripts/ck_discover.py`, `scripts/ck_inspect.py`, `scripts/ck_diag.py`; scores: `results/processed/gradattr_layer_scores.csv`. |
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
| **Evidence Paths** | `scripts/ck_discover.py`, `scripts/ck_inspect.py`, `scripts/ck_diag.py`, `results/processed/gradattr_layer_scores.csv` |
| **Notebook Execution** | Cells 8 & 13 in `notebooks/safe-compress.ipynb` |
| **Downstream Impact** | Generated the Circuit Attribution layer ranking that formed Strategy G (`S_G_gradattr_FP16_W4` and `S_G_gradattr_W8_W4`, protecting layers `[17, 19, 22, 27, 29]`). In final evaluations, Strategy G achieved **65.0%** harmful refusal (+4.5 pp over uniform NF4) with 129.5 tok/s throughput. |

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
