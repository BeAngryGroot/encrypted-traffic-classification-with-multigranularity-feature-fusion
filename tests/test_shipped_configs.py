from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.configuration import load_experiment_config  # noqa: E402


def test_all_shipped_configs_parse_and_have_unique_ids():
    paths = sorted((ROOT / "experiments/configs").rglob("*.yaml"))
    assert len(paths) == 7
    configs = [load_experiment_config(path, repo_root=ROOT) for path in paths]
    ids = [config.experiment_id for config in configs]
    assert len(ids) == len(set(ids))
    assert {config.task for config in configs} == {"application8", "tor_binary"}

