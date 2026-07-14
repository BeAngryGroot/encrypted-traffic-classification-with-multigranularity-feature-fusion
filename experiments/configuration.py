"""实验 YAML 配置的加载、校验和路径解析。"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping


VALID_TASKS = {"application8", "tor_binary"}
VALID_FUSIONS = {"gated", "concat", "fixed", "micro_only", "burst_only"}
VALID_LOSSES = {"cross_entropy", "focal"}


def _scalar(value: str) -> Any:
    value = value.strip()
    aliases = {"true": True, "false": False, "null": None, "none": None}
    if value.lower() in aliases:
        return aliases[value.lower()]
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value.strip("\"'")


def _load_yaml_text(text: str) -> dict[str, Any]:
    """优先使用 PyYAML；本地未安装时支持本项目扁平配置的最小解析。"""

    try:
        import yaml  # type: ignore
    except ImportError:
        result: dict[str, Any] = {}
        for line_number, raw in enumerate(text.splitlines(), start=1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if ":" not in line:
                raise ValueError(f"无法解析 YAML 第 {line_number} 行：{raw}")
            key, value = line.split(":", 1)
            result[key.strip()] = _scalar(value)
        return result
    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError("experiment config must be a mapping")
    return loaded


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    task: str
    fusion: str
    loss: str
    seed: int
    feature_id: str
    split_id: str
    features_dir: Path
    split_file: Path
    values: dict[str, Any]

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any], *, repo_root: str | Path) -> "ExperimentConfig":
        values = dict(mapping)
        required = ("experiment_id", "task", "fusion", "loss", "features_dir", "split_file")
        missing = [key for key in required if key not in values]
        if missing:
            raise ValueError(f"missing required config keys: {', '.join(missing)}")
        experiment_id = str(values["experiment_id"])
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", experiment_id):
            raise ValueError("experiment_id 只能包含字母、数字、点、下划线和短横线")
        task = str(values["task"])
        fusion = str(values["fusion"])
        loss = str(values["loss"])
        if task not in VALID_TASKS:
            raise ValueError(f"task must be one of {sorted(VALID_TASKS)}")
        if fusion not in VALID_FUSIONS:
            raise ValueError(f"fusion must be one of {sorted(VALID_FUSIONS)}")
        if loss not in VALID_LOSSES:
            raise ValueError(f"loss must be one of {sorted(VALID_LOSSES)}")

        root = Path(repo_root).resolve()
        def resolve(value: Any) -> Path:
            path = Path(str(value))
            return path.resolve() if path.is_absolute() else (root / path).resolve()

        return cls(
            experiment_id=experiment_id,
            task=task,
            fusion=fusion,
            loss=loss,
            seed=int(values.get("seed", 42)),
            feature_id=str(values.get("feature_id", "unspecified")),
            split_id=str(values.get("split_id", "unspecified")),
            features_dir=resolve(values["features_dir"]),
            split_file=resolve(values["split_file"]),
            values=values,
        )

    def resolved_mapping(self) -> dict[str, Any]:
        resolved = dict(self.values)
        resolved.update({
            "experiment_id": self.experiment_id,
            "task": self.task,
            "fusion": self.fusion,
            "loss": self.loss,
            "seed": self.seed,
            "feature_id": self.feature_id,
            "split_id": self.split_id,
            "features_dir": str(self.features_dir),
            "split_file": str(self.split_file),
        })
        return resolved


def load_experiment_config(path: str | Path, *, repo_root: str | Path | None = None) -> ExperimentConfig:
    path = Path(path)
    mapping = _load_yaml_text(path.read_text(encoding="utf-8"))
    return ExperimentConfig.from_mapping(mapping, repo_root=repo_root or path.resolve().parents[2])


def dump_resolved_config(config: ExperimentConfig, path: str | Path) -> None:
    Path(path).write_text(json.dumps(config.resolved_mapping(), ensure_ascii=False, indent=2), encoding="utf-8")

