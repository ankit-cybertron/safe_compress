# Notebooks Directory — SafeCompress

Contains interactive and reproducible research notebooks.

---

## `safe-compress.ipynb` (Canonical Execution Notebook)

`safe-compress.ipynb` is the primary 20-cell Kaggle execution notebook that produced the final 20-arm experimental dataset, score tables, and Hugging Face artifacts.

### Cell Structure & Execution Stages:

- **Cell 1:** Environment setup, package repair, and un-cached import preparation.
- **Cell 2:** Shared module initialization, GPU worker launcher, and path configuration.
- **Cell 3:** Automated dataset retrieval (resumable; fetches `HarmBench`, `PolyGuardMix`, `Global-MMLU`, `Belebele`).
- **Cell 4:** Layer anatomy extraction and first-token refusal probing (`w_probe.py`).
- **Cell 5:** Layer scoring via causal activation patching (`w_patch.py`).
- **Cell 6:** Layer scoring via quantization sensitivity and fragility (`w_sens.py`).
- **Cell 7:** Score aggregation, rank correlations, and candidate set definitions (`candidates.json`).
- **Cell 8:** CircuitKIT in isolated venv (Attempt 1: encountered GPU OOM on T4).
- **Cell 9:** Hugging Face authentication, findings note compilation, and dataset synchronization.
- **Cell 10:** Experimental matrix freezing (`artifacts/manifests/arms.json`).
- **Cell 11:** 20-arm full benchmark evaluation (`w_eval.py`).
- **Cell 12:** Statistical analysis, paired bootstrap confidence intervals, and visualization.
- **Cell 13 (ALT):** Gradient attribution patching on refusal gap (`w_attr.py`).
- **Cell 13 v6 (CPU):** CircuitKIT discovery on CPU fallback (Attempt 2: memory limit / timeout).
- **Cell 15:** Model response generation for AuditKIT evaluation (`w_gen.py`).
- **Cell 16:** AuditKIT independent refusal evaluation and agreement analysis.
- **Cells 17–19:** Hugging Face model and dataset publishing.
