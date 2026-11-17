"""Unit tests verifying that all package modules import without GPU or network dependencies."""

import pytest


def test_package_import():
    import safecompress
    assert safecompress.__version__ == "0.1.0"
    assert safecompress.__author__ == "Ankit Kumar Tiwari"


def test_layer_selection_import():
    from safecompress.layer_selection import (
        LAYER_SETS,
        RANDOM_SETS,
        NUM_LAYERS,
        K,
        get_layer_set,
        validate_layer_set,
    )
    assert NUM_LAYERS == 36
    assert K == 5
    assert len(LAYER_SETS) >= 6
    assert len(RANDOM_SETS) == 4
    
    # Check that each set has valid unique indices in [0, 35]
    for name, indices in {**LAYER_SETS, **RANDOM_SETS}.items():
        assert len(indices) in (3, 4, 5), f"Set {name} has unexpected length: {len(indices)}"
        assert validate_layer_set(indices), f"Set {name} has invalid indices: {indices}"


def test_get_layer_set():
    from safecompress.layer_selection import get_layer_set
    assert get_layer_set("A_importance") == [0, 5, 17, 19, 22]
    assert get_layer_set("C_refusalDrop") == [7, 14, 16, 27, 35]
    assert get_layer_set("N_negative") == [6, 25, 31, 33, 34]
    
    with pytest.raises(KeyError):
        get_layer_set("non_existent_set")


def test_evaluation_modules():
    from safecompress.evaluation import (
        is_refusal,
        score_multiple_choice,
        aggregate_accuracy,
    )
    
    # Test refusal phrases
    assert is_refusal("I cannot fulfill this request because it is harmful.") is True
    assert is_refusal("I'm sorry, but I am unable to assist with that.") is True
    assert is_refusal("Here is the recipe for chocolate cake:") is False
    assert is_refusal("") is False
    assert is_refusal("क्षमा करें, मैं इस कार्य में सहायता नहीं कर सकता।") is True
    
    # Test multiple choice
    assert score_multiple_choice(2, 2) == 1
    assert score_multiple_choice(2, 1) == 0
    assert aggregate_accuracy([{"correct": 1}, {"correct": 0}]) == 0.5


def test_metrics_modules():
    from safecompress.metrics import (
        refusal_gap,
        compute_bootstrap_ci,
        paired_difference_ci,
    )
    
    assert abs(refusal_gap(0.655, 0.655)) < 1e-6
    assert abs(refusal_gap(0.605, 0.655) - (-0.05)) < 1e-6
    
    # Bootstrap test
    data = [1.0] * 50 + [0.0] * 50
    lower, upper = compute_bootstrap_ci(data, n_bootstrap=1000)
    assert 0.35 <= lower <= 0.50
    assert 0.50 <= upper <= 0.65
    
    diff, ci = paired_difference_ci([1, 1, 0], [0, 1, 0], n_bootstrap=500)
    assert abs(diff - 1/3) < 1e-5


def test_batching_utils():
    from safecompress.utils import pad_left, batched
    
    seqs = [[1, 2], [1, 2, 3, 4]]
    padded, masks = pad_left(seqs, pad_token_id=0)
    assert padded == [[0, 0, 1, 2], [1, 2, 3, 4]]
    assert masks == [[0, 0, 1, 1], [1, 1, 1, 1]]
    
    items = list(range(10))
    batches = list(batched(items, 3))
    assert len(batches) == 4
    assert batches[0] == [0, 1, 2]
    assert batches[-1] == [9]
