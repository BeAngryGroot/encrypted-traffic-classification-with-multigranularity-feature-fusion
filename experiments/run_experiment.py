#!/usr/bin/env python3
"""统一实验入口：准备不可覆盖的 run 目录并启动训练脚本。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.configuration import ExperimentConfig, dump_resolved_config, load_experiment_config


@dataclass(frozen=True)
class PreparedRun:
    run_dir: Path
    command: list[str]
    metadata: dict[str, Any]


def _git_metadata(repo_root: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=repo_root, check=True, capture_output=True, text=True).stdout.strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN", False


def build_training_command(config: ExperimentConfig, run_dir: Path, repo_root: Path) -> list[str]:
    values = config.values
    command = [
        sys.executable,
        str(repo_root / "model/train_optimized.py"),
        "--features_dir", str(config.features_dir),
        "--split_file", str(config.split_file),
        "--run_dir", str(run_dir),
        "--classification_mode", config.task,
        "--fusion_mode", config.fusion,
        "--loss", config.loss,
        "--seed", str(config.seed),
        "--epochs", str(values.get("epochs", 80)),
        "--batch_size", str(values.get("batch_size", 64)),
        "--learning_rate", str(values.get("learning_rate", 0.0002)),
        "--val_ratio", str(values.get("val_ratio", 0.15)),
        "--test_ratio", str(values.get("test_ratio", 0.15)),
        "--fixed_fusion_weight", str(values.get("fixed_fusion_weight", 0.5)),
    ]
    for key, default in (
        ("micro_d_model", 384), ("micro_layers", 4), ("d_state", 128),
        ("macro_d_model", 96), ("macro_layers", 3), ("macro_heads", 6),
        ("fusion_hidden", 192), ("num_workers", 0),
    ):
        command.extend([f"--{key}", str(values.get(key, default))])
    if bool(values.get("require_official_mamba", True)):
        command.append("--require_official_mamba")
    return command


def prepare_run(
    config: ExperimentConfig,
    *,
    repo_root: str | Path,
    dry_run: bool = False,
    resume: bool = False,
) -> PreparedRun:
    repo_root = Path(repo_root).resolve()
    run_dir = repo_root / "artifacts" / "runs" / config.experiment_id / f"seed_{config.seed}"
    if run_dir.exists() and not resume and not dry_run:
        raise FileExistsError(f"运行目录已存在，拒绝覆盖：{run_dir}")
    commit, dirty = _git_metadata(repo_root)
    command = build_training_command(config, run_dir, repo_root)
    metadata = {
        "experiment_id": config.experiment_id,
        "task": config.task,
        "feature_id": config.feature_id,
        "split_id": config.split_id,
        "seed": config.seed,
        "git_commit": commit,
        "git_dirty": dirty,
        "command": command,
    }
    if not dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
        dump_resolved_config(config, run_dir / "resolved_config.json")
        (run_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return PreparedRun(run_dir=run_dir, command=command, metadata=metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description="运行版本化论文实验")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config = load_experiment_config(args.config, repo_root=repo_root)
    prepared = prepare_run(config, repo_root=repo_root, dry_run=args.dry_run, resume=args.resume)
    print(json.dumps({"run_dir": str(prepared.run_dir), **prepared.metadata}, ensure_ascii=False, indent=2))
    if not args.dry_run:
        subprocess.run(prepared.command, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
