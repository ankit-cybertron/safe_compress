"""Layer-level scoring and interpretability methods for SafeCompress.

Documents the three primary analytical methods used to measure layer significance:
1. Activation Patching (causal importance)
2. One-layer-at-a-time Quantization Sensitivity (NF4 KL divergence & refusal drop)
3. First-order Gradient Attribution (attribution patching)

The underlying computational workflows are executed via standalone scripts in
`scripts/` (e.g., `w_patch.py`, `w_sens.py`, `w_attr.py`). This module provides
data structures, constants, and loading utilities for analysis.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Optional

# Summary statistics from layer scoring (from FINDINGS_layer_scoring.md)
TOP_IMPORTANCE_LAYERS = [0, 22, 17, 19, 5]  # Highest patch_block score
TOP_FRAGILITY_LAYERS = [35, 0, 12, 16, 7]   # Highest KL divergence under single-layer NF4
TOP_REFUSAL_DROP_LAYERS = [16, 35, 27, 7, 14]  # Largest reduction in refusal gap under single-layer NF4

# Rank correlations between layer metrics (Spearman rank correlation)
CORRELATIONS = {
    ("importance", "fragility_kl"): 0.34,
    ("importance", "refusal_drop"): 0.28,
}


def load_layer_scores(path: Optional[Path | str] = None) -> pd.DataFrame:
    """Load per-layer composite scores from CSV.
    
    Args:
        path: Path to layer_scores.csv. Defaults to 'results/processed/layer_scores.csv'.
        
    Returns:
        DataFrame containing 36 rows with columns:
        - layer: Layer index (0..35)
        - patch_block: Causal activation patching effect
        - kl_safety_nf4: KL divergence on safety prompts under NF4
        - gap_drop_nf4: Drop in refusal gap under NF4
        - composite: Normalized product of importance and fragility
    """
    if path is None:
        path = Path("results/processed/layer_scores.csv")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Layer scores file not found at: {path}")
    return pd.read_csv(path)


def load_gradattr_scores(path: Optional[Path | str] = None) -> pd.DataFrame:
    """Load gradient attribution layer scores from CSV.
    
    Args:
        path: Path to gradattr_layer_scores.csv. Defaults to 'results/processed/gradattr_layer_scores.csv'.
    """
    if path is None:
        path = Path("results/processed/gradattr_layer_scores.csv")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Gradient attribution scores not found at: {path}")
    return pd.read_csv(path)
