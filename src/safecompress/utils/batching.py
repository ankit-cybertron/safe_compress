"""Batching and sequence padding utilities for decoder-only inference."""

from __future__ import annotations

import itertools
from typing import Any, Iterable, Iterator, List, Sequence


def batched(iterable: Iterable[Any], n: int) -> Iterator[List[Any]]:
    """Yield batches of size n from iterable.
    
    Args:
        iterable: Sequence or iterator to chunk.
        n: Batch size.
    """
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, n))
        if not chunk:
            break
        yield chunk


def pad_left(
    sequences: Sequence[Sequence[int]],
    pad_token_id: int,
) -> tuple[list[list[int]], list[list[int]]]:
    """Left-pad token sequences to equal length and produce attention masks.
    
    Left-padding is required for causal decoder-only models so that the final
    position across all items corresponds to the generation prompt boundary.
    
    Args:
        sequences: List of token ID sequences.
        pad_token_id: Token ID used for padding.
        
    Returns:
        (padded_ids, attention_masks)
    """
    if not sequences:
        return [], []
        
    max_len = max(len(seq) for seq in sequences)
    padded_seqs = []
    attention_masks = []
    
    for seq in sequences:
        pad_len = max_len - len(seq)
        padded = [pad_token_id] * pad_len + list(seq)
        mask = [0] * pad_len + [1] * len(seq)
        padded_seqs.append(padded)
        attention_masks.append(mask)
        
    return padded_seqs, attention_masks
