"""
model_loader.py – Load a SafeCompress arm from the canonical recipes.json.

A "arm" in this project is a named mixed-precision quantization configuration:
  - a set of transformer layer indices to *protect* (kept in higher precision)
  - protect_bits: 16 (FP16) or 8 (INT8 via bitsandbytes)
  - rest_bits:    4  (NF4  via bitsandbytes) for all other layers

The loader reads ``artifacts/recipes/recipes.json`` to resolve the arm name
to its layer indices and bit-widths, then applies in-place quantization using
bitsandbytes before moving the model to the target device.

This module refactors ``artifacts/loaders/load_safecompress.py``; all
quantization logic is identical to the original script.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import torch
try:
    import bitsandbytes as bnb
    _HAS_BNB = True
except ImportError:
    bnb = None
    _HAS_BNB = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
    _HAS_TRANSFORMERS = True
except Exception:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    PreTrainedModel = None
    PreTrainedTokenizerBase = None
    _HAS_TRANSFORMERS = False


# Default model identifier used throughout the SafeCompress experiments.
DEFAULT_MODEL_ID = "CohereLabs/tiny-aya-global"

# Path to the canonical recipes file (resolved relative to the package root).
_RECIPES_PATH = Path(__file__).resolve().parents[2].parent / "artifacts" / "recipes" / "recipes.json"


def _quant(layer: torch.nn.Module, bits: int) -> None:
    """Quantize all ``torch.nn.Linear`` sub-modules within *layer* in-place.

    Applies NF4 (4-bit) or INT8 (8-bit LLM.int8) quantization via bitsandbytes.
    Weights are first cast to FP16 before the bitsandbytes wrapper replaces
    each ``Linear`` with the quantized equivalent.

    Args:
        layer: A transformer layer module whose Linear sub-modules will be
               replaced with bitsandbytes quantized equivalents.
        bits:  Target precision – 4 for NF4, 8 for INT8 (LLM.int8).
    """
    if not _HAS_BNB:
        raise ImportError(
            "bitsandbytes is required to quantize layers. Please install bitsandbytes: "
            "pip install bitsandbytes"
        )
    for name, mod in list(layer.named_modules()):
        if not isinstance(mod, torch.nn.Linear):
            continue
        w = mod.weight.data.to(torch.float16)
        if bits == 4:
            new = bnb.nn.Linear4bit(
                mod.in_features,
                mod.out_features,
                bias=(mod.bias is not None),
                quant_type="nf4",
                compute_dtype=torch.float16,
            )
            new.weight = bnb.nn.Params4bit(w, requires_grad=False, quant_type="nf4")
        else:
            new = bnb.nn.Linear8bitLt(
                mod.in_features,
                mod.out_features,
                bias=(mod.bias is not None),
                has_fp16_weights=False,
                threshold=6.0,
            )
            new.weight = bnb.nn.Int8Params(w, requires_grad=False, has_fp16_weights=False)
        if mod.bias is not None:
            new.bias = torch.nn.Parameter(mod.bias.data.clone().to(torch.float16))
        pn, _, at = name.rpartition(".")
        par = layer
        if pn:
            for part in pn.split("."):
                par = getattr(par, part)
        setattr(par, at, new)


def load_safecompress(
    protect_layers: list[int],
    protect_bits: int = 8,
    rest_bits: int = 4,
    model_id: str = DEFAULT_MODEL_ID,
    token: Optional[str] = None,
    device: str = "cuda:0",
) -> PreTrainedModel:
    """Load a mixed-precision quantized model (SafeCompress arm).

    Protected layers are kept at *protect_bits* precision; all other layers
    are quantized to *rest_bits* (NF4 by default).  FP16 layers (bits == 16)
    are left unchanged.

    This is a direct refactor of ``artifacts/loaders/load_safecompress.py``;
    all logic is identical.

    Args:
        protect_layers: Indices of transformer layers to protect (higher bits).
        protect_bits:   Precision for protected layers – 16 (FP16), 8 (INT8),
                        or 4 (NF4).
        rest_bits:      Precision for remaining layers – usually 4 (NF4).
        model_id:       HuggingFace model identifier.
        token:          HuggingFace access token (or ``None`` for public models).
        device:         Target device string, e.g. ``"cuda:0"``.

    Returns:
        Quantized ``AutoModelForCausalLM`` in eval mode, moved to *device*.
    """
    if not _HAS_TRANSFORMERS:
        raise RuntimeError("transformers is required to load models.")
    m = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=token,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).eval()
    protect_set = set(protect_layers)
    for i, ly in enumerate(m.model.layers):
        b = protect_bits if i in protect_set else rest_bits
        if b < 16:
            _quant(ly, b)
    m.model.embed_tokens = m.model.embed_tokens.to(device)
    for ly in m.model.layers:
        ly.to(device)
    m.model.norm = m.model.norm.to(device)
    m.lm_head = m.lm_head.to(device)
    return m


def load_arm(
    arm_name: str,
    *,
    recipes_path: Optional[str | Path] = None,
    token: Optional[str] = None,
    device: str = "cuda:0",
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a named arm from ``artifacts/recipes/recipes.json``.

    Resolves the arm name to its layer set, protect_bits, and rest_bits from
    the canonical recipes file, loads the tokenizer, then calls
    :func:`load_safecompress` to build the quantized model.

    Args:
        arm_name:      Name of the arm, e.g. ``"S_C_refusalDrop_FP16_W4"``.
        recipes_path:  Override path to ``recipes.json``.  If *None*, the
                       package default is used.
        token:         HuggingFace access token.
        device:        Target device string.

    Returns:
        ``(model, tokenizer)`` tuple ready for inference.

    Raises:
        KeyError: If *arm_name* is not found in recipes.json.
    """
    rp = Path(recipes_path) if recipes_path else _RECIPES_PATH
    with open(rp, encoding="utf-8") as fh:
        recipes = json.load(fh)

    # recipes.json structure: {"arms": [[name, protect_layers, pbits, rbits], ...]}
    arm_map = {entry[0]: entry for entry in recipes["arms"]}
    if arm_name not in arm_map:
        raise KeyError(f"Arm {arm_name!r} not found in {rp}. Available: {list(arm_map)}")

    _name, protect_layers, protect_bits, rest_bits = arm_map[arm_name]
    model_id = recipes.get("model_id", DEFAULT_MODEL_ID)
    hf_token = token or os.environ.get("HF_TOKEN")

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = load_safecompress(
        protect_layers=protect_layers,
        protect_bits=protect_bits,
        rest_bits=rest_bits,
        model_id=model_id,
        token=hf_token,
        device=device,
    )
    return model, tokenizer
