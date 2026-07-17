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


def create_weighted_group_assignment(
    group_labels: Mapping[str, str],
    group_weights: Mapping[str, float],
    group_primary: Mapping[str, str],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    *,
    trials: int = 5000,
    require_class_coverage: bool = True,
) -> dict[str, str]:
    """按有效片段数搜索采集组划分，同时保持应用类别覆盖。

    每个采集组始终整体进入一个集合；权重只影响候选方案的评分，不会把
    同一源文件拆开。这样既阻止同源泄漏，又能缓解大文件集中到测试集的情况。
    """

    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be non-negative and sum to less than 1")
    if not group_labels:
        raise ValueError("group_labels must not be empty")
    if int(trials) < 1:
        raise ValueError("trials must be at least 1")

    labels = {str(group): str(label) for group, label in group_labels.items()}
    weights = {str(group): float(weight) for group, weight in group_weights.items()}
    primary = {str(group): str(label) for group, label in group_primary.items()}
    expected = set(labels)
    if set(weights) != expected or set(primary) != expected:
        raise ValueError("group_labels, group_weights and group_primary must contain identical groups")
    if any(not np.isfinite(weight) or weight < 0 for weight in weights.values()):
        raise ValueError("group weights must be finite and non-negative")
    if sum(weights.values()) <= 0:
        raise ValueError("at least one group weight must be positive")

    by_label: dict[str, list[str]] = {}
    for group, label in labels.items():
        by_label.setdefault(label, []).append(group)
    for label, groups in by_label.items():
        if require_class_coverage and (val_ratio > 0 or test_ratio > 0) and len(groups) < 3:
            raise ValueError(f"class {label} needs at least 3 capture groups")

    target = {
        "train": 1.0 - float(val_ratio) - float(test_ratio),
        "val": float(val_ratio),
        "test": float(test_ratio),
    }
    split_names = ("train", "val", "test")
    active_split_names = tuple(name for name in split_names if target[name] > 0)

    def split_counts(size: int) -> tuple[int, int]:
        test_count = max(1, int(round(size * test_ratio))) if test_ratio else 0
        val_count = max(1, int(round(size * val_ratio))) if val_ratio else 0
        while test_count + val_count >= size:
            if val_count >= test_count and val_count > 0:
                val_count -= 1
            elif test_count > 0:
                test_count -= 1
        return val_count, test_count

    def ratio_error(groups: list[str], assignment: Mapping[str, str]) -> float:
        total = sum(weights[group] for group in groups)
        if total <= 0:
            return 0.0
        return sum(
            (
                sum(weights[group] for group in groups if assignment[group] == split) / total
                - target[split]
            ) ** 2
            for split in split_names
        )

    def score(assignment: Mapping[str, str]) -> float:
        # 总体比例最重要，同时兼顾八类应用及 Tor/Non-Tor 的分布。
        value = 2.0 * ratio_error(sorted(labels), assignment)
        value += float(np.mean([
            ratio_error(sorted(groups), assignment)
            for groups in by_label.values()
        ]))
        primary_groups = {
            name: sorted(group for group, value in primary.items() if value == name)
            for name in sorted(set(primary.values()))
        }
        value += 0.5 * float(np.mean([
            ratio_error(groups, assignment) for groups in primary_groups.values()
        ]))
        return value

    def has_primary_coverage(assignment: Mapping[str, str]) -> bool:
        for name in set(primary.values()):
            groups = [group for group, value in primary.items() if value == name]
            if (
                len(groups) >= len(active_split_names)
                and {assignment[group] for group in groups} != set(active_split_names)
            ):
                return False
        return True

    rng = np.random.default_rng(int(seed))
    best: dict[str, str] | None = None
    best_key: tuple[float, tuple[tuple[str, str], ...]] | None = None
    fallback: dict[str, str] | None = None
    fallback_key: tuple[float, tuple[tuple[str, str], ...]] | None = None
    for _ in range(int(trials)):
        candidate: dict[str, str] = {}
        for label in sorted(by_label):
            groups = sorted(set(by_label[label]))
            shuffled = [groups[index] for index in rng.permutation(len(groups))]
            val_count, test_count = split_counts(len(groups))
            for group in shuffled[:test_count]:
                candidate[group] = "test"
            for group in shuffled[test_count:test_count + val_count]:
                candidate[group] = "val"
            for group in shuffled[test_count + val_count:]:
                candidate[group] = "train"

        deterministic = tuple(sorted(candidate.items()))
        candidate_key = (score(candidate), deterministic)
        if fallback_key is None or candidate_key < fallback_key:
            fallback, fallback_key = candidate.copy(), candidate_key
        if has_primary_coverage(candidate) and (best_key is None or candidate_key < best_key):
            best, best_key = candidate.copy(), candidate_key

    if best is None and require_class_coverage:
        raise ValueError(
            "no candidate split covers every required Tor/NonTor subset; "
            "inspect capture-group labels or increase split_search_trials"
        )
    # 极小 smoke 集可能无法同时覆盖 Tor/Non-Tor；此时保留应用覆盖并采用最佳比例。
    result = best if best is not None else fallback
    if result is None:  # pragma: no cover - trials 已校验，属于防御分支
        raise RuntimeError("failed to create a weighted group assignment")
    return result


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
