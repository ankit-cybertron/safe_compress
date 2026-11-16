# Scripts Directory — SafeCompress

This directory contains standalone execution scripts used for layer profiling, evaluation, and Lexsi toolkit experiments. These scripts were executed in the Kaggle environment via `notebooks/safe-compress.ipynb`.

---

## Script Inventory

| Script | Purpose | Status | Execution Environment |
|---|---|---|---|
| `lab_common.py` | Shared laboratory utilities: tokenizer helper, dataset loader, left-padding, prompt hashing. | Functional | GPU / CPU |
| `w_probe.py` | Extracts model layer anatomy (`provenance/anatomy.csv`) and probes first-token distributions on refusal vs. compliance. | Functional | 1 GPU (T4) |
| `w_patch.py` | Performs activation patching (block denoising and corrupting) across all 36 layers on paired refusal prompts. | Functional | 1-2 GPUs (T4) |
| `w_sens.py` | Measures single-layer quantization sensitivity by fake-quantizing each layer to NF4/INT8 and computing KL divergence. | Functional | 1-2 GPUs (T4) |
| `w_attr.py` | Computes first-order gradient attribution of the refusal gap with respect to layer outputs. | Functional | 1 GPU (T4) |
| `w_eval.py` | Evaluates experimental arms on the 400-item safety benchmark, 600 MMLU questions, and 200 Belebele passages. | Functional | 1-2 GPUs (T4) |
| `w_gen.py` | Generates full model replies on harmful and benign safety prompts for evaluation by external judges. | Functional | 1 GPU (T4) |
| `ck_discover.py` | Automated circuit discovery script using `circuitkit.discover()`. | ⚠️ Incomplete (OOM on T4) | 1 GPU (T4) |
| `ck_inspect.py` | Inspection script for CircuitKIT output objects. | Auxiliary | CPU |
| `ck_diag.py` | Diagnostic script for checking memory allocation during CircuitKIT runs. | Auxiliary | CPU |
