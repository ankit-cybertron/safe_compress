"""Unit tests verifying that result tables and score artifacts load with expected schemas."""

import csv
from pathlib import Path


def _read_csv_dicts(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def test_results_table():
    path = Path("results/tables/results_table.csv")
    assert path.exists()
    rows = _read_csv_dicts(path)
    
    # 20 experimental arms
    assert len(rows) == 20
    
    # Core columns matching results_table.csv schema
    required_cols = [
        "arm", "protect", "pbits", "GB", "peak_GB",
        "tok_s", "mmlu", "bele", "refuse_harm", "refuse_benign"
    ]
    for col in required_cols:
        assert col in rows[0], f"Missing column: {col}"
        
    # Check baseline row
    fp16_rows = [r for r in rows if r["arm"] == "E0_FP16"]
    assert len(fp16_rows) == 1
    fp16 = fp16_rows[0]
    assert float(fp16["refuse_harm"]) == 0.655
    assert float(fp16["refuse_benign"]) == 0.025
    assert abs(float(fp16["GB"]) - 6.698) < 0.01


def test_auditkit_results():
    path = Path("results/tables/auditkit_results.csv")
    assert path.exists()
    rows = _read_csv_dicts(path)
    
    # 12 arms evaluated by AuditKIT
    assert len(rows) == 12
    assert "arm" in rows[0]
    assert any(k in rows[0] for k in ["ak_refusal_harmful", "auditkit_refusal", "refuse_harm", "refuse"])


def test_layer_scores():
    path = Path("results/processed/layer_scores.csv")
    assert path.exists()
    rows = _read_csv_dicts(path)
    
    # 36 transformer layers
    assert len(rows) == 36
    assert "layer" in rows[0]


def test_results_comparisons():
    path = Path("results/statistics/results_comparisons.csv")
    assert path.exists()
    rows = _read_csv_dicts(path)
    assert len(rows) > 0
    assert any(k in rows[0] for k in ["arm", "comparison", "diff", "ci_lo"])
