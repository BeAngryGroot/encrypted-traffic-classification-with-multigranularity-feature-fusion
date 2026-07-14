from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.splits import create_group_split, load_group_split, save_group_split  # noqa: E402


def test_group_split_has_no_overlap_and_is_reproducible(tmp_path):
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    groups = np.asarray(["a", "a", "b", "b", "c", "c", "d", "d"])
    first = create_group_split(labels, groups, val_ratio=0.25, test_ratio=0.25, seed=42)
    second = create_group_split(labels, groups, val_ratio=0.25, test_ratio=0.25, seed=42)
    assert first == second
    assert not set(groups[first.train]) & set(groups[first.val])
    assert not set(groups[first.train]) & set(groups[first.test])
    assert not set(groups[first.val]) & set(groups[first.test])
    assert sorted(np.concatenate([first.train, first.val, first.test]).tolist()) == list(range(8))

    path = tmp_path / "split.npz"
    save_group_split(first, path, labels=labels, groups=groups)
    assert load_group_split(path) == first
    assert path.with_suffix(".json").exists()

