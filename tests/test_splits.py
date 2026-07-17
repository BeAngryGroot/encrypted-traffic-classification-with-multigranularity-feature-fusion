from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.splits import (  # noqa: E402
    create_group_split,
    create_stratified_group_assignment,
    indices_from_group_assignment,
    load_group_split,
    save_group_split,
)


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


def test_stratified_group_assignment_covers_each_class_when_three_groups_exist():
    labels = {
        f"{label}-{index}": label
        for label in ["A", "B"]
        for index in range(5)
    }

    first = create_stratified_group_assignment(
        labels,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=42,
        require_class_coverage=True,
    )
    second = create_stratified_group_assignment(
        labels,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=42,
        require_class_coverage=True,
    )

    assert first == second
    for label in ["A", "B"]:
        observed = {first[group] for group, value in labels.items() if value == label}
        assert observed == {"train", "val", "test"}


def test_indices_from_group_assignment_preserves_all_samples_without_overlap():
    groups = np.asarray(["a", "a", "b", "c", "c"])
    assignment = {"a": "train", "b": "val", "c": "test"}

    split = indices_from_group_assignment(groups, assignment, seed=42)

    assert split.train.tolist() == [0, 1]
    assert split.val.tolist() == [2]
    assert split.test.tolist() == [3, 4]
