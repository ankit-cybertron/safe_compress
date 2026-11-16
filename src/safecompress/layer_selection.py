"""Layer selection strategies and layer definitions for SafeCompress.

Provides the canonical protected layer indices identified across different
interpretability and sensitivity scoring methods, as well as random and negative
control layer sets.

Model: CohereForAI/aya-23-8B (36 decoder layers, indices 0..35).
K = 5 layers protected per mixed-precision arm.
"""

from __future__ import annotations

NUM_LAYERS: int = 36
K: int = 5  # Number of layers protected in each mixed-precision configuration

# Canonical strategy layer sets (each containing K=5 layer indices in [0, 35])
LAYER_SETS: dict[str, list[int]] = {
    # Strategy A: Causal importance via activation patching (block patching score)
    "A_importance": [0, 5, 17, 19, 22],
    
    # Strategy B: Quantization fragility via one-layer-at-a-time NF4 KL divergence
    "B_fragility": [0, 7, 12, 16, 35],
    
    # Strategy C: Refusal drop under NF4 quantization (erosion of refusal gap)
    "C_refusalDrop": [7, 14, 16, 27, 35],
    
    # Strategy D: Composite score (normalized importance × normalized fragility)
    "D_composite": [0, 14, 16, 17, 19],
    
    # Negative control: Lowest ranked layers on both importance and fragility
    "N_negative": [6, 25, 31, 33, 34],
    
    # Strategy G: First-order gradient attribution (attribution patching on refusal gap)
    "G_gradattr": [17, 19, 22, 27, 29],
}

# Random control sets (sampled without replacement, K=5 layers each)
RANDOM_SETS: dict[str, list[int]] = {
    "R0": [7, 13, 15, 24, 32],
    "R1": [11, 15, 22, 27, 30],
    "R2": [13, 14, 27, 31, 34],
    "R3": [1, 5, 19, 27, 33],
}

# Layer attention type mapping (from anatomy.csv)
# 0-indexed: sliding attention on most layers; full attention every 4th layer
FULL_ATTENTION_LAYERS: list[int] = [3, 7, 11, 15, 19, 23, 27, 31, 35]
SLIDING_ATTENTION_LAYERS: list[int] = [
    i for i in range(NUM_LAYERS) if i not in FULL_ATTENTION_LAYERS
]


def get_layer_set(name: str) -> list[int]:
    """Retrieve layer indices for a named strategy or control set.
    
    Args:
        name: Name of the set (e.g., 'A_importance', 'C_refusalDrop', 'R0', 'N_negative').
        
    Returns:
        Sorted list of integer layer indices.
        
    Raises:
        KeyError: If the set name is not recognized.
    """
    if name in LAYER_SETS:
        return sorted(LAYER_SETS[name])
    if name in RANDOM_SETS:
        return sorted(RANDOM_SETS[name])
    raise KeyError(f"Unknown layer set: '{name}'. Available: {list(LAYER_SETS.keys()) + list(RANDOM_SETS.keys())}")


def validate_layer_set(indices: list[int], max_layers: int = NUM_LAYERS) -> bool:
    """Validate that a layer set contains valid, unique indices within bounds."""
    if not isinstance(indices, (list, tuple, set)):
        return False
    if len(indices) == 0:
        return True
    return (
        len(set(indices)) == len(indices)
        and all(isinstance(x, int) and 0 <= x < max_layers for x in indices)
    )
