from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.configuration import ExperimentConfig  # noqa: E402
from experiments.run_experiment import prepare_run  # noqa: E402


def _config(experiment_id="E00_smoke_v1"):
    return ExperimentConfig.from_mapping({
        "experiment_id": experiment_id,
        "task": "application8",
        "fusion": "gated",
        "loss": "focal",
        "seed": 42,
        "feature_id": "iscxtor_pilot_p64_b32_v1",
        "split_id": "iscxtor_group_seed42_v1",
        "features_dir": "artifacts/features/demo",
        "split_file": "artifacts/splits/demo.npz",
    }, repo_root=ROOT)


def test_dry_run_contains_reproducibility_metadata_without_creating_directory(tmp_path):
    prepared = prepare_run(_config(), repo_root=tmp_path, dry_run=True)
    assert prepared.metadata["seed"] == 42
    assert prepared.metadata["feature_id"] == "iscxtor_pilot_p64_b32_v1"
    assert prepared.metadata["split_id"] == "iscxtor_group_seed42_v1"
    assert "git_commit" in prepared.metadata
    assert "git_dirty" in prepared.metadata
    assert not prepared.run_dir.exists()


def test_existing_run_directory_is_rejected_unless_resume(tmp_path):
    expected = tmp_path / "artifacts/runs/E00_smoke_v1/seed_42"
    expected.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        prepare_run(_config(), repo_root=tmp_path, dry_run=False)
    assert prepare_run(_config(), repo_root=tmp_path, dry_run=False, resume=True).run_dir == expected

