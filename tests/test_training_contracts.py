import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(relative: str) -> str:
    text = (ROOT / relative).read_text(encoding="utf-8")
    ast.parse(text)
    return text


def test_training_exposes_frozen_split_run_loss_and_formal_mamba_contracts():
    source = _source("model/train_optimized.py")
    for option in ("--split_file", "--run_dir", "--loss", "--require_official_mamba", "--test_ratio"):
        assert option in source
    assert "best_macro_f1" in source
    assert "SequenceNormalizer" in source
    assert "load_group_split" in source


def test_model_uses_configured_state_size_and_explicit_official_guard():
    model_source = _source("model/model.py")
    mamba_source = _source("model/mamba_branch.py")
    assert "d_state=cfg.d_state" in model_source
    assert "require_official" in mamba_source
    assert "官方 Mamba" in mamba_source


def test_export_defaults_to_test_split_and_requires_split_file():
    source = _source("model/export_results.py")
    assert "--split_file" in source
    assert 'default="test"' in source
    assert "select_task_labels" in source

