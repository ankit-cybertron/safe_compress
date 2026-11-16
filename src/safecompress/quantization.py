"""
quantization.py – Quantization configuration constants and thin documentation
wrappers for SafeCompress.

SafeCompress uses `bitsandbytes` to quantize transformer layer weights.  Two
precision modes are supported:

**NF4 (4-bit NormalFloat)**
    ``bitsandbytes.nn.Linear4bit`` with ``quant_type="nf4"``.  Double
    quantization is intentionally disabled (``bnb_4bit_use_double_quant=False``)
    to keep the analysis clean.  Compute is promoted to FP16 at run-time.

**INT8 (LLM.int8)**
    ``bitsandbytes.nn.Linear8bitLt`` with ``threshold=6.0``.  Weights with
    absolute value above the threshold are kept in FP16 (outlier handling),
    all others are stored in INT8.

The actual in-place quantization loop lives in
:func:`safecompress.model_loader._quant`; this module provides reference
constants that document the parameter choices used throughout the experiments.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Reference configuration dicts – for documentation / reproducibility checks.
# These are NOT passed directly to bitsandbytes; the actual logic is in
# model_loader._quant which constructs the bitsandbytes objects explicitly.
# ---------------------------------------------------------------------------

NF4_CONFIG: dict = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": "float16",
    "bnb_4bit_use_double_quant": False,
    "bnb_4bit_quant_storage": "uint8",
}
"""NF4 (4-bit NormalFloat) quantization parameters.

``bnb_4bit_use_double_quant`` is ``False`` throughout the SafeCompress
experiments – the second quantization step is omitted so that memory savings
are easier to attribute to the first quantization pass alone.
"""

INT8_CONFIG: dict = {
    "load_in_8bit": True,
    "llm_int8_threshold": 6.0,
    "llm_int8_has_fp16_weight": False,
}
"""INT8 (LLM.int8) quantization parameters.

``llm_int8_threshold`` = 6.0 is the bitsandbytes default: weight values with
absolute magnitude above this threshold are kept in FP16 to handle outliers
while the remaining weights are stored in INT8.
"""

# Convenience mapping from bit-width integer to config name.
QUANT_CONFIG_MAP: dict[int, dict] = {
    4: NF4_CONFIG,
    8: INT8_CONFIG,
}
"""Map from bit-width (4 or 8) to the corresponding reference config dict."""
