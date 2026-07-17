"""按采集组划分训练、验证和测试集，防止同源流量跨集合泄漏。"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass(frozen=True, eq=False)
class GroupSplit:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    seed: int

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GroupSplit) and self.seed == other.seed and all(
            np.array_equal(getattr(self, name), getattr(other, name))
            for name in ("train", "val", "test")
        )


def create_stratified_group_assignment(
    group_labels: Mapping[str, str],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    *,
    require_class_coverage: bool = True,
) -> dict[str, str]:
    """按类别在采集组层面冻结划分，避免同一源文件跨集合泄漏。"""

    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be non-negative and sum to less than 1")
    if not group_labels:
        raise ValueError("group_labels must not be empty")

    by_label: dict[str, list[str]] = {}
    for group, label in group_labels.items():
        by_label.setdefault(str(label), []).append(str(group))

    rng = np.random.default_rng(int(seed))
    assignment: dict[str, str] = {}
    for label in sorted(by_label):
        groups = sorted(set(by_label[label]))
        if require_class_coverage and (val_ratio > 0 or test_ratio > 0) and len(groups) < 3:
            raise ValueError(f"class {label} needs at least 3 capture groups")
        shuffled = [groups[index] for index in rng.permutation(len(groups))]

        if len(groups) < 3 and not require_class_coverage:
            for group in shuffled:
                assignment[group] = "train"
            continue

        test_count = max(1, int(round(len(groups) * test_ratio))) if test_ratio else 0
        val_count = max(1, int(round(len(groups) * val_ratio))) if val_ratio else 0
        while test_count + val_count >= len(groups):
            if val_count >= test_count and val_count > 0:
                val_count -= 1
            elif test_count > 0:
                test_count -= 1
        for group in shuffled[:test_count]:
            assignment[group] = "test"
        for group in shuffled[test_count:test_count + val_count]:
            assignment[group] = "val"
        for group in shuffled[test_count + val_count:]:
            assignment[group] = "train"

    # smoke 数据类别可能不足三组；仍保证总体三个集合非空，便于链路测试。
    if not require_class_coverage:
        for missing in ("val", "test"):
            if missing in assignment.values():
                continue
            donors = [
                group
                for group in sorted(assignment)
                if assignment[group] == "train"
                and sum(
                    assignment.get(candidate) == "train"
                    for candidate, label in group_labels.items()
                    if str(label) == str(group_labels[group])
                ) > 1
            ]
            if not donors:
                raise ValueError("smoke split needs at least 3 movable capture groups")
            assignment[donors[-1]] = missing
    return assignment


def indices_from_group_assignment(
    groups: np.ndarray,
    assignment: Mapping[str, str],
    *,
    seed: int = 42,
) -> GroupSplit:
    """将冻结的采集组归属转换成与特征数组对齐的样本索引。"""

    groups = np.asarray(groups).astype(str)
    missing = sorted(set(groups) - {str(group) for group in assignment})
    if missing:
        raise ValueError(f"groups missing from assignment: {', '.join(missing)}")
    normalized = {str(group): str(split) for group, split in assignment.items()}
    unknown = sorted(set(normalized.values()) - {"train", "val", "test"})
    if unknown:
        raise ValueError(f"unknown split names: {', '.join(unknown)}")

    def selected(name: str) -> np.ndarray:
        return np.asarray(
            [index for index, group in enumerate(groups) if normalized[group] == name],
            dtype=np.int64,
        )

    return GroupSplit(selected("train"), selected("val"), selected("test"), int(seed))


def create_group_split(
    labels: np.ndarray,
    groups: np.ndarray,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> GroupSplit:
    """一次性冻结采集组划分；同一 capture group 不会出现在多个集合。"""

    labels = np.asarray(labels)
    groups = np.asarray(groups).astype(str)
    if len(labels) != len(groups):
        raise ValueError("labels and groups must have the same length")
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be non-negative and sum to less than 1")
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        raise ValueError("分组划分至少需要 3 个不同 capture group")

    # 以组为单位随机化；标签仅用于稳定的预排序，使相同数据在不同机器上结果一致。
    dominant = {}
    for group in unique_groups:
        values, counts = np.unique(labels[groups == group], return_counts=True)
        dominant[group] = int(values[np.argmax(counts)])
    ordered = sorted(unique_groups.tolist(), key=lambda group: (dominant[group], group))
    rng = np.random.default_rng(int(seed))
    shuffled = [ordered[index] for index in rng.permutation(len(ordered))]

    test_count = max(1, int(round(len(shuffled) * test_ratio))) if test_ratio else 0
    val_count = max(1, int(round(len(shuffled) * val_ratio))) if val_ratio else 0
    while test_count + val_count >= len(shuffled):
        if val_count >= test_count and val_count > 0:
            val_count -= 1
        elif test_count > 0:
            test_count -= 1
    test_groups = set(shuffled[:test_count])
    val_groups = set(shuffled[test_count:test_count + val_count])
    train_groups = set(shuffled[test_count + val_count:])

    def indices(selected: set[str]) -> np.ndarray:
        return np.flatnonzero(np.isin(groups, list(selected))).astype(np.int64)

    return GroupSplit(indices(train_groups), indices(val_groups), indices(test_groups), int(seed))


def save_group_split(split: GroupSplit, path: str | Path, *, labels: np.ndarray, groups: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, train_idx=split.train, val_idx=split.val, test_idx=split.test, seed=split.seed)
    labels = np.asarray(labels)
    groups = np.asarray(groups).astype(str)
    summary = {"seed": split.seed, "splits": {}}
    for name, idx in (("train", split.train), ("val", split.val), ("test", split.test)):
        values, counts = np.unique(labels[idx], return_counts=True)
        summary["splits"][name] = {
            "samples": int(len(idx)),
            "groups": int(len(np.unique(groups[idx]))),
            "class_counts": {str(value): int(count) for value, count in zip(values, counts)},
        }
    path.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def load_group_split(path: str | Path) -> GroupSplit:
    with np.load(Path(path)) as data:
        return GroupSplit(
            train=np.asarray(data["train_idx"], dtype=np.int64),
            val=np.asarray(data["val_idx"], dtype=np.int64),
            test=np.asarray(data["test_idx"], dtype=np.int64),
            seed=int(data["seed"]),
        )
