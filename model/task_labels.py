"""将统一特征标签映射为论文主任务和辅助任务。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from data.label_schema import APPLICATION_LABELS


@dataclass(frozen=True)
class TaskSelection:
    labels: np.ndarray
    keep_mask: np.ndarray
    class_names: list[str]
    num_classes: int


def _decode(values: np.ndarray, mapping: dict[str, Any]) -> np.ndarray:
    raw = mapping.get("id_to_label")
    if raw is None:
        raw = {idx: name for name, idx in mapping["label_to_id"].items()}
    decoded = []
    for value in np.asarray(values).astype(int):
        decoded.append(raw.get(int(value), raw.get(str(int(value)), "UNKNOWN")))
    return np.asarray(decoded, dtype=str)


def select_task_labels(
    mode: str,
    primary_labels: np.ndarray,
    application_labels: np.ndarray,
    mappings: dict[str, Any],
) -> TaskSelection:
    """筛选正式任务样本并重映射为连续类别编号。

    主任务跨 Tor/Non-Tor 识别八类应用；辅助任务只判断封装类型。所有未知或
    非 ISCXTor 标签都被排除，避免把 UNKNOWN 当成可学习类别。
    """

    primary = _decode(primary_labels, mappings["primary"])
    application = _decode(application_labels, mappings["secondary"])

    if mode == "application8":
        class_names = list(APPLICATION_LABELS)
        valid_primary = np.isin(primary, ["TOR", "NONTOR"])
        keep = valid_primary & np.isin(application, class_names)
        lookup = {name: idx for idx, name in enumerate(class_names)}
        labels = np.asarray([lookup[name] for name in application[keep]], dtype=np.int64)
    elif mode == "tor_binary":
        class_names = ["NONTOR", "TOR"]
        keep = np.isin(primary, class_names)
        lookup = {name: idx for idx, name in enumerate(class_names)}
        labels = np.asarray([lookup[name] for name in primary[keep]], dtype=np.int64)
    else:
        raise ValueError(f"Unsupported task mode: {mode}. Use application8 or tor_binary.")

    return TaskSelection(labels=labels, keep_mask=keep, class_names=class_names, num_classes=len(class_names))

