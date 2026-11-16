"""Unit tests verifying configuration files and manifests."""

import json
from pathlib import Path

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


def test_eval_cfg():
    cfg_path = Path("configs/evaluation_config.json")
    assert cfg_path.exists()
    with open(cfg_path) as f:
        cfg = json.load(f)
    assert cfg["n_group"] == 100
    assert cfg["n_mmlu"] == 300
    assert cfg["n_bel"] == 100
    assert cfg["max_arms"] == 99


def test_recipes_json():
    recipes_path = Path("artifacts/recipes/recipes.json")
    assert recipes_path.exists()
    with open(recipes_path) as f:
        data = json.load(f)
    assert "arms" in data
    assert len(data["arms"]) == 20
    
    # Check that baseline arms exist
    arm_names = [arm["name"] if isinstance(arm, dict) else arm[0] for arm in data["arms"]]
    assert "E0_FP16" in arm_names
    assert "E1_uniform_W4" in arm_names
    assert "S_C_refusalDrop_FP16_W4" in arm_names
    assert "S_N_negative_FP16_W4" in arm_names


def test_model_config_yaml():
    yaml_path = Path("configs/model_config.yaml")
    assert yaml_path.exists()
    content = yaml_path.read_text()
    assert "CohereLabs/tiny-aya-global" in content
    assert "num_layers: 36" in content
    if _HAS_YAML:
        cfg = yaml.safe_load(content)
        assert cfg["model"]["id"] == "CohereLabs/tiny-aya-global"
        assert cfg["model"]["num_layers"] == 36


def test_quantization_config_yaml():
    yaml_path = Path("configs/quantization_config.yaml")
    assert yaml_path.exists()
    content = yaml_path.read_text()
    assert "bnb_4bit_quant_type: \"nf4\"" in content or "nf4" in content
    assert "k_protected: 5" in content
    if _HAS_YAML:
        cfg = yaml.safe_load(content)
        assert cfg["nf4"]["bnb_4bit_quant_type"] == "nf4"
        assert cfg["mixed_precision"]["k_protected"] == 5
