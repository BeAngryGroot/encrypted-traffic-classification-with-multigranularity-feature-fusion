from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.configuration import load_experiment_config  # noqa: E402


def test_all_shipped_configs_parse_and_have_unique_ids():
    paths = sorted((ROOT / "experiments/configs").rglob("*.yaml"))
    assert len(paths) == 8
    configs = [load_experiment_config(path, repo_root=ROOT) for path in paths]
    ids = [config.experiment_id for config in configs]
    assert len(ids) == len(set(ids))
    assert {config.task for config in configs} == {"application8", "tor_binary"}


def test_segment15_smoke_config_targets_new_versioned_features():
    path = ROOT / "experiments/configs/smoke/application8_segment15_burstp95_smoke_v1.yaml"

    config = load_experiment_config(path, repo_root=ROOT)

    assert config.task == "application8"
    assert config.fusion == "gated"
    assert config.feature_id == "segment15_burstp95_v1"
    assert config.split_file.name == "split_seed42.npz"
