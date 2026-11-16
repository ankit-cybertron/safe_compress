"""General capability evaluation utilities (MMLU and Belebele)."""

from __future__ import annotations

from typing import Any, Sequence


def score_multiple_choice(gold: int, predicted: int) -> int:
    """Score a multiple-choice item.
    
    Args:
        gold: Integer index of gold answer.
        predicted: Integer index of predicted answer.
        
    Returns:
        1 if predicted matches gold, 0 otherwise.
    """
    return 1 if gold == predicted else 0


def aggregate_accuracy(records: Sequence[dict[str, Any]]) -> float:
    """Calculate mean accuracy over a collection of evaluation records.
    
    Args:
        records: List or sequence of dicts containing a 'correct' key (0 or 1).
        
    Returns:
        Mean accuracy in [0.0, 1.0]. Returns 0.0 if empty.
    """
    if not records:
        return 0.0
    correct_count = sum(int(r.get("correct", 0)) for r in records)
    return correct_count / len(records)
