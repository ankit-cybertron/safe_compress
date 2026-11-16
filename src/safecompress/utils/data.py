"""Loading helpers for SafeCompress published evaluation tables and scores."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

try:
    import pandas as pd
    _HAS_PANDAS = True
except Exception:
    pd = None
    _HAS_PANDAS = False


def _read_csv(path: Path) -> Any:
    if _HAS_PANDAS and pd is not None:
        return pd.read_csv(path)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_results_table(path: Path | str = "results/tables/results_table.csv") -> Any:
    """Load the canonical 20-arm evaluation results table."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return _read_csv(p)


def load_auditkit_results(path: Path | str = "results/tables/auditkit_results.csv") -> Any:
    """Load the independent AuditKIT evaluation results table (12 arms)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return _read_csv(p)


def load_layer_scores_table(path: Path | str = "results/processed/layer_scores.csv") -> Any:
    """Load the 36-layer scoring and sensitivity table."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return _read_csv(p)


def load_results_comparisons(path: Path | str = "results/statistics/results_comparisons.csv") -> Any:
    """Load paired bootstrap statistical comparisons table."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return _read_csv(p)
