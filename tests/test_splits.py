from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.splits import (  # noqa: E402
    create_variable_weighted_group_assignment,
    create_group_split,
    create_stratified_group_assignment,
    create_weighted_group_assignment,
    evaluate_weighted_assignment,
    indices_from_group_assignment,
    load_group_split,
    save_group_split,
)


def test_variable_optimizer_can_use_multiple_holdout_groups_for_weight_balance():
    labels = {
        f"{label}-{index}": label
        for label in ["A", "B"]
        for index in range(8)
    }
    per_class_weights = [60, 20, 6, 5, 5, 5, 5, 4]
    weights = {
        group: float(per_class_weights[int(group.rsplit("-", 1)[1])])
        for group in labels
    }
    primary = {
        group: ("TOR" if index % 2 else "NONTOR")
        for index, group in enumerate(labels)
    }

    first = create_variable_weighted_group_assignment(
        labels,
        weights,
        primary,
        val_ratio=0.10,
        test_ratio=0.10,
        seed=42,
        trials=1,
    )
    second = create_variable_weighted_group_assignment(
        labels,
        weights,
        primary,
        val_ratio=0.10,
        test_ratio=0.10,
        seed=42,
        trials=1,
    )
    report = evaluate_weighted_assignment(
        labels,
        weights,
        primary,
        first,
        val_ratio=0.10,
        test_ratio=0.10,
        overall_tolerance=0.03,
        min_class_holdout=0.05,
    )

    assert first == second
    assert report.passed
    for label in ["A", "B"]:
        val_groups = sum(
            first[group] == "val" for group, value in labels.items() if value == label
        )
        test_groups = sum(
            first[group] == "test" for group, value in labels.items() if value == label
        )
        assert val_groups + test_groups >= 3
        assert max(val_groups, test_groups) >= 2


def test_quality_report_rejects_file_like_holdout_collapse():
    report = evaluate_weighted_assignment(
        {"big": "FILE", "small-v": "FILE", "small-t": "FILE"},
        {"big": 970.0, "small-v": 10.0, "small-t": 20.0},
        {"big": "TOR", "small-v": "NONTOR", "small-t": "NONTOR"},
        {"big": "train", "small-v": "val", "small-t": "test"},
        val_ratio=0.10,
        test_ratio=0.10,
        overall_tolerance=0.03,
        min_class_holdout=0.05,
    )

    assert not report.passed
    assert any("FILE" in violation for violation in report.violations)


def test_weighted_group_assignment_is_deterministic_and_balances_samples():
    """带权划分应兼顾应用覆盖和实际片段数，而不是只平分文件个数。"""

    labels = {
        f"{label}-{index}": label
        for label in ["A", "B"]
        for index in range(7)
    }
    weights = {
        group: float([70, 15, 15, 35, 8, 7, 50][int(group.rsplit("-", 1)[1])])
        for group in labels
    }
    primary = {
        group: ("TOR" if int(group.rsplit("-", 1)[1]) % 2 == 0 else "NONTOR")
        for group in labels
    }

    first = create_weighted_group_assignment(
        labels,
        weights,
        primary,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
        trials=2000,
    )
    second = create_weighted_group_assignment(
        labels,
        weights,
        primary,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
        trials=2000,
    )

    assert first == second
    assert set(first) == set(labels)
    for label in {"A", "B"}:
        assert {first[group] for group in labels if labels[group] == label} == {
            "train",
            "val",
            "test",
        }
    for primary_label in {"TOR", "NONTOR"}:
        assert {first[group] for group in labels if primary[group] == primary_label} == {
            "train",
            "val",
            "test",
        }

    totals = {
        split: sum(weights[group] for group in labels if first[group] == split)
        for split in ("train", "val", "test")
    }
    total = sum(totals.values())
    assert abs(totals["train"] / total - 0.70) < 0.10
    assert abs(totals["val"] / total - 0.15) < 0.08
    assert abs(totals["test"] / total - 0.15) < 0.08


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
