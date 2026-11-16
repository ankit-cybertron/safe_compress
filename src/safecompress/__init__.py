"""SafeCompress: Mixed-precision quantization with layer-informed protection for AI safety preservation.

This package provides tools for evaluating whether protecting safety-critical
transformer layers from aggressive quantization (NF4/INT8) can preserve
refusal behaviour while retaining most of the memory and throughput savings
of uniform quantization.

Experiment model: CohereLabs/tiny-aya-global (36 transformer layers).
"""

__version__ = "0.1.0"
__author__ = "Ankit Kumar Tiwari"

# Export key public API
from .model_loader import load_arm
from .layer_selection import LAYER_SETS
