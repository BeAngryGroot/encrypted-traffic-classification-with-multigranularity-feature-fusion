from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.configuration import load_experiment_config  # noqa: E402


def _write_config(path: Path, experiment_id="E00_smoke_v1", task="application8", fusion="gated", loss="focal"):
    path.write_text(
        "\n".join([
            f"experiment_id: {experiment_id}",
            f"task: {task}",
            f"fusion: {fusion}",
            f"loss: {loss}",
            "seed: 42",
            "features_dir: artifacts/features/demo",
            "split_file: artifacts/splits/demo.npz",
        ]),
        encoding="utf-8",
    )


def test_config_requires_safe_id_and_known_contract_values(tmp_path):
    path = tmp_path / "config.yaml"
    _write_config(path)
    config = load_experiment_config(path, repo_root=ROOT)
    assert config.experiment_id == "E00_smoke_v1"
    assert config.task == "application8"
    assert config.features_dir == ROOT / "artifacts/features/demo"

    _write_config(path, experiment_id="bad id")
    with pytest.raises(ValueError, match="experiment_id"):
        load_experiment_config(path, repo_root=ROOT)
    _write_config(path, task="combined")
    with pytest.raises(ValueError, match="task"):
        load_experiment_config(path, repo_root=ROOT)

