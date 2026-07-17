#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ISCXTor 15 秒流片段 + 自适应 burst 特征一键生成入口。

服务器使用时只需要修改下方三个配置项，然后直接运行本文件：

    python data/run_segment_feature_pipeline.py
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import pickle
from typing import Any, Iterator

import numpy as np
import pandas as pd

try:
    from data.build_features import _records_for_group
    from data.burst_features import BURST_FEATURES, PACKET_FEATURES, build_flow_features
    from data.label_schema import APPLICATION_LABELS, infer_labels
    from data.segment_features import (
        assign_bursts_with_reasons,
        collect_mult_packet_burst_durations,
        pack_by_burst_capacity,
        time_segment_packets,
    )
    from data.splits import (
        create_stratified_group_assignment,
        indices_from_group_assignment,
        save_group_split,
    )
except ImportError:  # pragma: no cover - 支持直接执行 data 目录下的脚本
    from build_features import _records_for_group
    from burst_features import BURST_FEATURES, PACKET_FEATURES, build_flow_features
    from label_schema import APPLICATION_LABELS, infer_labels
    from segment_features import (
        assign_bursts_with_reasons,
        collect_mult_packet_burst_durations,
        pack_by_burst_capacity,
        time_segment_packets,
    )
    from splits import (
        create_stratified_group_assignment,
        indices_from_group_assignment,
        save_group_split,
    )


# =============================================================================
# 用户配置区：服务器运行时只修改这三项
# =============================================================================
CSV_DIR = Path("/data3/wsb_workspace/study/data/Dual_data/csv/full_session60_v1")
OUTPUT_DIR = Path("/data3/wsb_workspace/study/data/Dual_data/processed/segment15_burstp95_v1")
RUN_MODE = "smoke"  # 第一次保持 smoke；检查成功后改成 full

# =============================================================================
# 论文首版固定参数：后续敏感性实验通过新数据版本修改，不覆盖本版本
# =============================================================================
WINDOW_SECONDS = 15.0
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42
ALPHA = 1.0
D_MAX_QUANTILE = 0.95
MIN_MODEL_PACKETS = 2
MAX_PACKETS = 64
MAX_BURSTS = 32


@dataclass(frozen=True)
class SegmentPipelineSettings:
    """一次片段特征生成运行的完整、可记录配置。"""

    csv_dir: Path
    output_dir: Path
    run_mode: str
    window_seconds: float = WINDOW_SECONDS
    val_ratio: float = VAL_RATIO
    test_ratio: float = TEST_RATIO
    seed: int = SEED
    alpha: float = ALPHA
    dmax_quantile: float = D_MAX_QUANTILE
    min_model_packets: int = MIN_MODEL_PACKETS
    max_packets: int = MAX_PACKETS
    max_bursts: int = MAX_BURSTS


@dataclass(frozen=True)
class SourceInfo:
    path: Path
    source_key: str
    capture_group: str
    primary: str
    application: str


@dataclass(frozen=True)
class InitialSegment:
    source: SourceInfo
    parent_flow_id: str
    segment_index: int
    is_tail_segment: bool
    packets: list[dict[str, Any]]

    @property
    def segment_id(self) -> str:
        return f"{self.source.source_key}::{self.parent_flow_id}::seg{self.segment_index}"


REQUIRED_COLUMNS = {
    "flow_id",
    "timestamp",
    "packet_length",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "protocol",
}


def _validate_settings(settings: SegmentPipelineSettings) -> None:
    if settings.run_mode.lower() not in {"smoke", "full"}:
        raise ValueError('run_mode must be "smoke" or "full"')
    if not Path(settings.csv_dir).is_dir():
        raise FileNotFoundError(f"CSV directory does not exist: {settings.csv_dir}")
    if settings.window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if settings.min_model_packets < 2:
        raise ValueError("min_model_packets must be at least 2")
    if settings.max_packets < settings.min_model_packets or settings.max_bursts < 1:
        raise ValueError("invalid packet or burst capacity")
    if not 0.0 < settings.dmax_quantile <= 1.0:
        raise ValueError("dmax_quantile must be in (0, 1]")


def _source_info(path: Path, csv_root: Path) -> SourceInfo:
    relative = path.relative_to(csv_root).as_posix()
    labels = infer_labels(Path(relative))
    return SourceInfo(path, relative, relative, labels.primary, labels.application)


def _discover_sources(csv_root: Path) -> list[SourceInfo]:
    paths = sorted(csv_root.rglob("*_packets.csv"), key=lambda path: str(path).lower())
    if not paths:
        raise FileNotFoundError(f"No *_packets.csv files found under {csv_root}")
    return [_source_info(path, csv_root) for path in paths]


def _select_sources(sources: list[SourceInfo], mode: str) -> list[SourceInfo]:
    if mode.lower() == "full":
        return sources

    # 每个可识别应用选择最小的三个 CSV，使 smoke 既较快又能形成三个集合。
    selected: list[SourceInfo] = []
    for application in APPLICATION_LABELS:
        candidates = [source for source in sources if source.application == application]
        candidates.sort(key=lambda source: (source.path.stat().st_size, source.source_key))
        selected.extend(candidates[:3])
    if len(selected) < 3:
        known = [source for source in sources if source.application in APPLICATION_LABELS]
        known.sort(key=lambda source: (source.path.stat().st_size, source.source_key))
        selected = known[:max(3, min(len(known), 9))]
    if len(selected) < 3:
        raise ValueError("smoke mode needs at least 3 recognized packet CSV files")
    return sorted({source.source_key: source for source in selected}.values(), key=lambda source: source.source_key)


def _validate_source_labels(sources: list[SourceInfo], mode: str) -> None:
    unknown = [
        source.source_key
        for source in sources
        if source.primary not in {"TOR", "NONTOR"}
        or source.application not in APPLICATION_LABELS
    ]
    if unknown:
        raise ValueError(f"unrecognized ISCXTor labels: {', '.join(unknown[:5])}")
    if mode.lower() == "full":
        applications = {source.application for source in sources}
        missing = sorted(set(APPLICATION_LABELS) - applications)
        if missing:
            raise ValueError(f"full mode is missing application classes: {', '.join(missing)}")
        primary = {source.primary for source in sources}
        if primary != {"TOR", "NONTOR"}:
            raise ValueError("full mode must contain both Tor and NonTor sources")


def _read_packet_frame(source: SourceInfo) -> pd.DataFrame:
    frame = pd.read_csv(source.path)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"{source.source_key} missing required column: {', '.join(missing)}")
    if frame["timestamp"].isna().any():
        raise ValueError(f"{source.source_key} contains invalid timestamp")
    if "frame_index" in frame.columns and frame["frame_index"].duplicated().any():
        raise ValueError(f"{source.source_key} contains duplicate frame_index")
    return frame


def _iter_initial_segments(
    sources: list[SourceInfo],
    *,
    window_seconds: float,
) -> Iterator[InitialSegment]:
    for source in sources:
        frame = _read_packet_frame(source)
        for flow_id, group in frame.groupby("flow_id", sort=False):
            records = _records_for_group(group)
            if not records:
                continue
            flow_start = float(records[0]["timestamp"])
            segments = time_segment_packets(records, window_seconds)
            for position, packets in enumerate(segments):
                segment_index = int(
                    max(0.0, float(packets[0]["timestamp"]) - flow_start)
                    // float(window_seconds)
                )
                yield InitialSegment(
                    source=source,
                    parent_flow_id=str(flow_id),
                    segment_index=segment_index,
                    is_tail_segment=position == len(segments) - 1,
                    packets=packets,
                )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array)
    temporary.replace(path)


def _atomic_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream)
    temporary.replace(path)


def _label_mappings() -> dict[str, Any]:
    primary_names = ["NONTOR", "TOR"]
    secondary_names = list(APPLICATION_LABELS)

    def mapping(names: list[str]) -> dict[str, Any]:
        return {
            "label_to_id": {name: index for index, name in enumerate(names)},
            "id_to_label": {index: name for index, name in enumerate(names)},
            "num_classes": len(names),
        }

    return {
        "primary": mapping(primary_names),
        "secondary": mapping(secondary_names),
        "packet_features": PACKET_FEATURES,
        "burst_features": BURST_FEATURES,
    }


def run_segment_pipeline(settings: SegmentPipelineSettings) -> dict[str, Any]:
    """执行两遍 CSV 处理：训练集拟合 D_max，再生成全部冻结特征。"""

    _validate_settings(settings)
    csv_root = Path(settings.csv_dir)
    output_root = Path(settings.output_dir)
    success_path = output_root / ".pipeline_success.json"
    success_path.unlink(missing_ok=True)

    sources = _select_sources(_discover_sources(csv_root), settings.run_mode)
    _validate_source_labels(sources, settings.run_mode)
    group_labels = {source.capture_group: source.application for source in sources}
    group_assignment = create_stratified_group_assignment(
        group_labels,
        settings.val_ratio,
        settings.test_ratio,
        settings.seed,
        require_class_coverage=settings.run_mode.lower() == "full",
    )

    # 第一遍只建立初始片段审计，并且只从训练组收集自然 burst 时长。
    segment_rows: list[dict[str, Any]] = []
    natural_durations: list[float] = []
    input_packets = 0
    ineligible_packets = 0
    for segment in _iter_initial_segments(sources, window_seconds=settings.window_seconds):
        packet_count = len(segment.packets)
        input_packets += packet_count
        eligible = packet_count >= settings.min_model_packets
        if not eligible:
            ineligible_packets += packet_count
        split_name = group_assignment[segment.source.capture_group]
        natural = assign_bursts_with_reasons(segment.packets, alpha=settings.alpha)
        if eligible and split_name == "train":
            natural_durations.extend(
                collect_mult_packet_burst_durations(segment.packets, natural)
            )
        timestamps = [float(packet["timestamp"]) for packet in segment.packets]
        segment_rows.append(
            {
                "segment_id": segment.segment_id,
                "source_key": segment.source.source_key,
                "capture_group": segment.source.capture_group,
                "parent_flow_id": segment.parent_flow_id,
                "segment_index": segment.segment_index,
                "start_time": min(timestamps),
                "end_time": max(timestamps),
                "duration": max(timestamps) - min(timestamps),
                "packet_count": packet_count,
                "natural_burst_count": len(set(natural.burst_ids)),
                "adaptive_threshold": natural.adaptive_threshold,
                "primary": segment.source.primary,
                "application": segment.source.application,
                "split": split_name,
                "is_tail_segment": segment.is_tail_segment,
                "eligible_for_model": eligible,
            }
        )

    if not natural_durations:
        raise ValueError("training split has no multi-packet natural bursts for D_max")
    dmax = float(np.quantile(np.asarray(natural_durations, dtype=np.float64), settings.dmax_quantile))

    # 第二遍应用冻结 D_max，并在 burst 边界做容量拆分后生成双分支特征。
    packet_arrays: list[np.ndarray] = []
    packet_masks: list[np.ndarray] = []
    burst_arrays: list[np.ndarray] = []
    burst_masks: list[np.ndarray] = []
    primary_labels: list[int] = []
    secondary_labels: list[int] = []
    sample_keys: list[str] = []
    group_ids: list[str] = []
    sample_rows: list[dict[str, Any]] = []
    capacity_reasons: Counter[str] = Counter()
    modeled_packets = 0
    mappings = _label_mappings()

    for segment in _iter_initial_segments(sources, window_seconds=settings.window_seconds):
        if len(segment.packets) < settings.min_model_packets:
            continue
        final_assignment = assign_bursts_with_reasons(
            segment.packets,
            alpha=settings.alpha,
            max_duration=dmax,
        )
        capacity_samples = pack_by_burst_capacity(
            segment.packets,
            final_assignment,
            max_packets=settings.max_packets,
            max_bursts=settings.max_bursts,
        )
        for subsegment_index, capacity_sample in enumerate(capacity_samples):
            result = build_flow_features(
                capacity_sample.packets,
                max_packets=settings.max_packets,
                max_bursts=settings.max_bursts,
                alpha=settings.alpha,
                precomputed_burst_ids=capacity_sample.burst_ids,
                truncate=False,
            )
            sample_id = f"{segment.segment_id}::sub{subsegment_index}"
            packet_arrays.append(result.packet_seq)
            packet_masks.append(result.packet_mask)
            burst_arrays.append(result.burst_seq)
            burst_masks.append(result.burst_mask)
            primary_labels.append(mappings["primary"]["label_to_id"][segment.source.primary])
            secondary_labels.append(mappings["secondary"]["label_to_id"][segment.source.application])
            sample_keys.append(sample_id)
            group_ids.append(segment.source.capture_group)
            modeled_packets += len(capacity_sample.packets)
            capacity_reasons[capacity_sample.split_reason] += 1
            timestamps = [float(packet["timestamp"]) for packet in capacity_sample.packets]
            sample_rows.append(
                {
                    "sample_id": sample_id,
                    "parent_segment_id": segment.segment_id,
                    "subsegment_index": subsegment_index,
                    "source_key": segment.source.source_key,
                    "capture_group": segment.source.capture_group,
                    "parent_flow_id": segment.parent_flow_id,
                    "start_time": min(timestamps),
                    "end_time": max(timestamps),
                    "packet_count": len(capacity_sample.packets),
                    "burst_count": len(set(capacity_sample.burst_ids)),
                    "capacity_split_reason": capacity_sample.split_reason,
                    "primary": segment.source.primary,
                    "application": segment.source.application,
                    "split": group_assignment[segment.source.capture_group],
                }
            )

    if not packet_arrays:
        raise ValueError("no eligible model samples were generated")
    if modeled_packets + ineligible_packets != input_packets:
        raise AssertionError(
            "packet conservation failed: modeled + ineligible must equal input"
        )

    packet_seq = np.stack(packet_arrays).astype(np.float32)
    packet_mask = np.stack(packet_masks).astype(np.float32)
    burst_seq = np.stack(burst_arrays).astype(np.float32)
    burst_mask = np.stack(burst_masks).astype(np.float32)
    primary_array = np.asarray(primary_labels, dtype=np.int64)
    secondary_array = np.asarray(secondary_labels, dtype=np.int64)
    group_array = np.asarray(group_ids, dtype=str)
    key_array = np.asarray(sample_keys, dtype=str)

    features_dir = output_root / "features"
    arrays = {
        "packet_seq.npy": packet_seq,
        "packet_mask.npy": packet_mask,
        "burst_seq.npy": burst_seq,
        "burst_mask.npy": burst_mask,
        "primary_labels.npy": primary_array,
        "secondary_labels.npy": secondary_array,
        "tor_labels.npy": primary_array,
        "application_labels.npy": secondary_array,
        "sample_keys.npy": key_array,
        "group_ids.npy": group_array,
    }
    for name, array in arrays.items():
        _atomic_npy(features_dir / name, array)
    _atomic_pickle(features_dir / "label_mappings.pkl", mappings)

    split = indices_from_group_assignment(
        group_array,
        group_assignment,
        seed=settings.seed,
    )
    if not len(split.train) or not len(split.val) or not len(split.test):
        raise ValueError("frozen split must contain train, val and test samples")
    save_group_split(
        split,
        features_dir / f"split_seed{settings.seed}.npz",
        labels=secondary_array,
        groups=group_array,
    )

    manifests_dir = output_root / "manifests"
    segment_frame = pd.DataFrame(segment_rows)
    sample_frame = pd.DataFrame(sample_rows)
    _atomic_csv(manifests_dir / "segment_manifest.csv", segment_frame)
    _atomic_csv(manifests_dir / "sample_manifest.csv", sample_frame)
    _atomic_csv(
        manifests_dir / "split_manifest.csv",
        pd.DataFrame(
            [
                {
                    "capture_group": source.capture_group,
                    "source_key": source.source_key,
                    "primary": source.primary,
                    "application": source.application,
                    "split": group_assignment[source.capture_group],
                }
                for source in sources
            ]
        ),
    )
    class_summary = (
        sample_frame.groupby(["split", "application", "primary"], dropna=False)
        .size()
        .reset_index(name="sample_count")
    )
    _atomic_csv(manifests_dir / "class_summary.csv", class_summary)

    statistics_dir = output_root / "statistics"
    _atomic_json(
        statistics_dir / "dmax_summary.json",
        {
            "source_split": "train",
            "quantile": float(settings.dmax_quantile),
            "value_seconds": dmax,
            "natural_burst_count": len(natural_durations),
        },
    )
    _atomic_json(
        statistics_dir / "natural_burst_summary.json",
        {
            "count": len(natural_durations),
            "min_seconds": float(np.min(natural_durations)),
            "median_seconds": float(np.median(natural_durations)),
            "max_seconds": float(np.max(natural_durations)),
        },
    )
    _atomic_json(
        statistics_dir / "capacity_split_summary.json",
        {"sample_counts": dict(sorted(capacity_reasons.items()))},
    )

    summary = {
        "run_mode": settings.run_mode.lower(),
        "source_files": len(sources),
        "initial_segments": len(segment_rows),
        "model_samples": len(sample_rows),
        "input_packets": int(input_packets),
        "modeled_packets": int(modeled_packets),
        "ineligible_packets": int(ineligible_packets),
        "dmax_seconds": dmax,
        "packet_shape": list(packet_seq.shape),
        "burst_shape": list(burst_seq.shape),
        "split_samples": {
            "train": int(len(split.train)),
            "val": int(len(split.val)),
            "test": int(len(split.test)),
        },
        "config": {
            "window_seconds": float(settings.window_seconds),
            "alpha": float(settings.alpha),
            "dmax_quantile": float(settings.dmax_quantile),
            "min_model_packets": int(settings.min_model_packets),
            "max_packets": int(settings.max_packets),
            "max_bursts": int(settings.max_bursts),
            "seed": int(settings.seed),
        },
    }
    _atomic_json(statistics_dir / "segmentation_summary.json", summary)
    _atomic_json(success_path, {"status": "success", **summary})
    return summary


def main() -> None:
    settings = SegmentPipelineSettings(
        csv_dir=CSV_DIR,
        output_dir=OUTPUT_DIR,
        run_mode=RUN_MODE,
    )
    print("=" * 72)
    print("ISCXTor 15秒片段 + 自适应 Burst 特征生成")
    print(f"模式：{settings.run_mode}")
    print(f"输入：{settings.csv_dir}")
    print(f"输出：{settings.output_dir}")
    print("=" * 72)
    try:
        summary = run_segment_pipeline(settings)
    except KeyboardInterrupt:
        print("\n用户中断：未写入新的成功标记，可以安全重新运行。")
        raise SystemExit(130)
    except Exception as exc:
        print(f"\n处理失败：{exc}")
        raise SystemExit(1)

    print("\n处理完成")
    print(f"输入包数：{summary['input_packets']}")
    print(f"进入模型的包数：{summary['modeled_packets']}")
    print(f"审计保留但不训练的单包数：{summary['ineligible_packets']}")
    print(f"D_max(P95)：{summary['dmax_seconds']:.6f} 秒")
    print(f"包序列形状：{summary['packet_shape']}")
    print(f"Burst序列形状：{summary['burst_shape']}")
    print(f"集合样本数：{summary['split_samples']}")
    if settings.run_mode.lower() == "smoke":
        print('\nsmoke 已通过。检查输出后，把文件顶部 RUN_MODE 改为 "full" 再运行。')


if __name__ == "__main__":
    main()
