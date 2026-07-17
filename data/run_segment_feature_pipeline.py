#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ISCXTor 15 秒流片段 + 自适应 burst 特征一键生成入口。

服务器使用时只需要修改下方三个配置项，然后直接运行本文件：

    python data/run_segment_feature_pipeline.py
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Iterator

# 服务器可能在仓库根目录或 data 目录启动本脚本。这里先固定加入仓库根目录，
# 后续模块统一使用 data.xxx 导入，避免两套导入路径造成同名模块状态不一致。
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

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
    create_weighted_group_assignment,
    indices_from_group_assignment,
    save_group_split,
)


# =============================================================================
# 用户配置区：服务器运行时只需要确认这四项
# =============================================================================
CSV_DIR = Path("/data3/wsb_workspace/study/data/Dual_data/csv/full_session60_v1")
OUTPUT_DIR = Path("/data3/wsb_workspace/study/data/Dual_data/processed/segment15_burstp95_v1_1")
RUN_MODE = "smoke"  # 第一次保持 smoke；检查成功后改成 full
WORKERS = 2  # 服务器首版建议 2；排错时改为 1，不建议直接超过 4

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
SMOKE_FLOWS_PER_FILE = 5
SPLIT_SEARCH_TRIALS = 5000
READ_CHUNKSIZE = 200_000


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
    workers: int = WORKERS
    smoke_flows_per_file: int = SMOKE_FLOWS_PER_FILE
    split_search_trials: int = SPLIT_SEARCH_TRIALS
    read_chunksize: int = READ_CHUNKSIZE


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


@dataclass(frozen=True)
class SourceProfile:
    """第一遍扫描得到的源文件权重及 D_max 候选统计。"""

    source: SourceInfo
    input_packets: int
    initial_segments: int
    eligible_segments: int
    ineligible_packets: int
    natural_durations: tuple[float, ...]
    selected_flow_count: int
    elapsed_seconds: float


@dataclass(frozen=True)
class SourceFeatureBatch:
    """单个源文件生成的连续特征批次，供主进程按源顺序合并。"""

    source: SourceInfo
    packet_seq: np.ndarray
    packet_mask: np.ndarray
    burst_seq: np.ndarray
    burst_mask: np.ndarray
    primary_labels: np.ndarray
    secondary_labels: np.ndarray
    sample_keys: np.ndarray
    group_ids: np.ndarray
    segment_rows: tuple[dict[str, Any], ...]
    sample_rows: tuple[dict[str, Any], ...]
    capacity_counts: dict[str, int]
    modeled_packets: int
    elapsed_seconds: float


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
    if settings.workers < 1 or settings.workers > 32:
        raise ValueError("workers must be between 1 and 32")
    if settings.smoke_flows_per_file < 1:
        raise ValueError("smoke_flows_per_file must be at least 1")
    if settings.split_search_trials < 1 or settings.read_chunksize < 1:
        raise ValueError("split_search_trials and read_chunksize must be positive")


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


def _flow_summary_path(packet_path: Path) -> Path:
    suffix = "_packets.csv"
    if packet_path.name.endswith(suffix):
        return packet_path.with_name(packet_path.name[:-len(suffix)] + "_flows.csv")
    return packet_path.with_name(packet_path.stem + "_flows.csv")


def _select_smoke_flow_ids(source: SourceInfo, *, limit: int) -> tuple[str, ...]:
    """选择包数最少的完整父流；优先读取已有流级汇总，避免全表扫描。"""

    summary_path = _flow_summary_path(source.path)
    if summary_path.is_file():
        summary = pd.read_csv(summary_path, usecols=["flow_id", "packet_count"])
        counts = (
            summary.assign(flow_id=summary["flow_id"].astype(str))
            .groupby("flow_id", as_index=False)["packet_count"]
            .sum()
        )
    else:
        parts = []
        for chunk in pd.read_csv(source.path, usecols=["flow_id"], chunksize=READ_CHUNKSIZE):
            parts.append(chunk["flow_id"].astype(str).value_counts())
        if not parts:
            return ()
        counts_series = pd.concat(parts, axis=1).fillna(0).sum(axis=1)
        counts = counts_series.rename_axis("flow_id").reset_index(name="packet_count")
    counts = counts.sort_values(["packet_count", "flow_id"], kind="stable")
    return tuple(counts.head(int(limit))["flow_id"].astype(str).tolist())


def _read_packet_frame(
    source: SourceInfo,
    *,
    selected_flow_ids: tuple[str, ...] | None = None,
    chunksize: int = READ_CHUNKSIZE,
) -> pd.DataFrame:
    if selected_flow_ids is None:
        frame = pd.read_csv(source.path)
    else:
        selected = set(selected_flow_ids)
        chunks: list[pd.DataFrame] = []
        for chunk in pd.read_csv(source.path, chunksize=int(chunksize)):
            flow_ids = chunk["flow_id"].astype(str)
            filtered = chunk.loc[flow_ids.isin(selected)].copy()
            if not filtered.empty:
                filtered["flow_id"] = filtered["flow_id"].astype(str)
                chunks.append(filtered)
        frame = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        observed = set(frame["flow_id"].astype(str)) if "flow_id" in frame else set()
        missing_flows = sorted(selected - observed)
        if missing_flows:
            raise ValueError(
                f"{source.source_key} missing selected flows: {', '.join(missing_flows[:5])}"
            )
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
    selected_flow_ids: dict[str, tuple[str, ...] | None] | None = None,
    chunksize: int = READ_CHUNKSIZE,
) -> Iterator[InitialSegment]:
    for source in sources:
        selected = None if selected_flow_ids is None else selected_flow_ids[source.source_key]
        frame = _read_packet_frame(
            source,
            selected_flow_ids=selected,
            chunksize=chunksize,
        )
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


def _profile_source_task(
    task: tuple[SourceInfo, SegmentPipelineSettings, tuple[str, ...] | None],
) -> SourceProfile:
    """第一遍工作进程：统计完整父流的片段权重和自然 burst 时长。"""

    source, settings, selected_flows = task
    started = time.perf_counter()
    input_packets = 0
    initial_segments = 0
    eligible_segments = 0
    ineligible_packets = 0
    natural_durations: list[float] = []
    observed_flows: set[str] = set()
    selection = {source.source_key: selected_flows}
    for segment in _iter_initial_segments(
        [source],
        window_seconds=settings.window_seconds,
        selected_flow_ids=selection,
        chunksize=settings.read_chunksize,
    ):
        packet_count = len(segment.packets)
        observed_flows.add(segment.parent_flow_id)
        input_packets += packet_count
        initial_segments += 1
        if packet_count < settings.min_model_packets:
            ineligible_packets += packet_count
            continue
        eligible_segments += 1
        natural = assign_bursts_with_reasons(segment.packets, alpha=settings.alpha)
        natural_durations.extend(
            collect_mult_packet_burst_durations(segment.packets, natural)
        )
    return SourceProfile(
        source=source,
        input_packets=input_packets,
        initial_segments=initial_segments,
        eligible_segments=eligible_segments,
        ineligible_packets=ineligible_packets,
        natural_durations=tuple(natural_durations),
        selected_flow_count=len(observed_flows),
        elapsed_seconds=time.perf_counter() - started,
    )


def _empty_feature_array(rows: int, columns: int) -> np.ndarray:
    return np.empty((0, rows, columns), dtype=np.float32)


def _build_source_task(
    task: tuple[
        SourceInfo,
        SegmentPipelineSettings,
        tuple[str, ...] | None,
        float,
        str,
    ],
) -> SourceFeatureBatch:
    """第二遍工作进程：应用冻结 D_max，生成一个源文件的连续特征批。"""

    source, settings, selected_flows, dmax, split_name = task
    started = time.perf_counter()
    mappings = _label_mappings()
    packet_arrays: list[np.ndarray] = []
    packet_masks: list[np.ndarray] = []
    burst_arrays: list[np.ndarray] = []
    burst_masks: list[np.ndarray] = []
    primary_labels: list[int] = []
    secondary_labels: list[int] = []
    sample_keys: list[str] = []
    group_ids: list[str] = []
    segment_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    capacity_reasons: Counter[str] = Counter()
    modeled_packets = 0
    selection = {source.source_key: selected_flows}

    for segment in _iter_initial_segments(
        [source],
        window_seconds=settings.window_seconds,
        selected_flow_ids=selection,
        chunksize=settings.read_chunksize,
    ):
        packet_count = len(segment.packets)
        eligible = packet_count >= settings.min_model_packets
        natural = assign_bursts_with_reasons(segment.packets, alpha=settings.alpha)
        timestamps = [float(packet["timestamp"]) for packet in segment.packets]
        segment_rows.append(
            {
                "segment_id": segment.segment_id,
                "source_key": source.source_key,
                "capture_group": source.capture_group,
                "parent_flow_id": segment.parent_flow_id,
                "segment_index": segment.segment_index,
                "start_time": min(timestamps),
                "end_time": max(timestamps),
                "duration": max(timestamps) - min(timestamps),
                "packet_count": packet_count,
                "natural_burst_count": len(set(natural.burst_ids)),
                "adaptive_threshold": natural.adaptive_threshold,
                "primary": source.primary,
                "application": source.application,
                "split": split_name,
                "is_tail_segment": segment.is_tail_segment,
                "eligible_for_model": eligible,
            }
        )
        if not eligible:
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
            primary_labels.append(mappings["primary"]["label_to_id"][source.primary])
            secondary_labels.append(mappings["secondary"]["label_to_id"][source.application])
            sample_keys.append(sample_id)
            group_ids.append(source.capture_group)
            modeled_packets += len(capacity_sample.packets)
            capacity_reasons[capacity_sample.split_reason] += 1
            sample_timestamps = [float(packet["timestamp"]) for packet in capacity_sample.packets]
            sample_rows.append(
                {
                    "sample_id": sample_id,
                    "parent_segment_id": segment.segment_id,
                    "subsegment_index": subsegment_index,
                    "source_key": source.source_key,
                    "capture_group": source.capture_group,
                    "parent_flow_id": segment.parent_flow_id,
                    "start_time": min(sample_timestamps),
                    "end_time": max(sample_timestamps),
                    "packet_count": len(capacity_sample.packets),
                    "burst_count": len(set(capacity_sample.burst_ids)),
                    "capacity_split_reason": capacity_sample.split_reason,
                    "primary": source.primary,
                    "application": source.application,
                    "split": split_name,
                }
            )

    packet_seq = (
        np.stack(packet_arrays).astype(np.float32)
        if packet_arrays
        else _empty_feature_array(settings.max_packets, len(PACKET_FEATURES))
    )
    packet_mask = (
        np.stack(packet_masks).astype(np.float32)
        if packet_masks
        else np.empty((0, settings.max_packets), dtype=np.float32)
    )
    burst_seq = (
        np.stack(burst_arrays).astype(np.float32)
        if burst_arrays
        else _empty_feature_array(settings.max_bursts, len(BURST_FEATURES))
    )
    burst_mask = (
        np.stack(burst_masks).astype(np.float32)
        if burst_masks
        else np.empty((0, settings.max_bursts), dtype=np.float32)
    )
    return SourceFeatureBatch(
        source=source,
        packet_seq=packet_seq,
        packet_mask=packet_mask,
        burst_seq=burst_seq,
        burst_mask=burst_mask,
        primary_labels=np.asarray(primary_labels, dtype=np.int64),
        secondary_labels=np.asarray(secondary_labels, dtype=np.int64),
        sample_keys=np.asarray(sample_keys, dtype=str),
        group_ids=np.asarray(group_ids, dtype=str),
        segment_rows=tuple(segment_rows),
        sample_rows=tuple(sample_rows),
        capacity_counts=dict(capacity_reasons),
        modeled_packets=modeled_packets,
        elapsed_seconds=time.perf_counter() - started,
    )


def _run_source_tasks(worker: Any, tasks: list[Any], workers: int, stage: str) -> list[Any]:
    """按源文件并行处理，并保持输入顺序返回，确保结果可复现。"""

    if workers == 1:
        results = []
        for index, task in enumerate(tasks, start=1):
            result = worker(task)
            results.append(result)
            print(
                f"[{stage}] {index}/{len(tasks)} {result.source.source_key} "
                f"耗时 {result.elapsed_seconds:.2f}s",
                flush=True,
            )
        return results

    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for index, result in enumerate(executor.map(worker, tasks), start=1):
            results.append(result)
            print(
                f"[{stage}] {index}/{len(tasks)} {result.source.source_key} "
                f"耗时 {result.elapsed_seconds:.2f}s",
                flush=True,
            )
    return results


def _write_split_balance_audit(
    output_root: Path,
    profiles: list[SourceProfile],
    sample_frame: pd.DataFrame,
    assignment: dict[str, str],
    settings: SegmentPipelineSettings,
) -> None:
    """同时保存划分前估计权重和生成后的真实样本数，便于论文审计。"""

    targets = {
        "train": 1.0 - settings.val_ratio - settings.test_ratio,
        "val": settings.val_ratio,
        "test": settings.test_ratio,
    }
    estimated = pd.DataFrame(
        [
            {
                "basis": "eligible_initial_segments",
                "capture_group": profile.source.capture_group,
                "application": profile.source.application,
                "primary": profile.source.primary,
                "split": assignment[profile.source.capture_group],
                "count": profile.eligible_segments,
            }
            for profile in profiles
        ]
    )
    final = sample_frame[["capture_group", "application", "primary", "split"]].copy()
    final["basis"] = "final_model_samples"
    final["count"] = 1
    combined = pd.concat([estimated, final], ignore_index=True)
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"target_ratios": targets, "bases": {}}
    for basis, basis_frame in combined.groupby("basis", sort=True):
        total = float(basis_frame["count"].sum())
        basis_summary = {}
        for split_name in ("train", "val", "test"):
            count = int(basis_frame.loc[basis_frame["split"] == split_name, "count"].sum())
            ratio = count / total if total else 0.0
            deviation = ratio - targets[split_name]
            rows.append(
                {
                    "basis": basis,
                    "split": split_name,
                    "count": count,
                    "ratio": ratio,
                    "target_ratio": targets[split_name],
                    "deviation": deviation,
                }
            )
            basis_summary[split_name] = {
                "count": count,
                "ratio": ratio,
                "deviation": deviation,
            }
        summary["bases"][basis] = basis_summary
    _atomic_csv(output_root / "manifests" / "split_balance.csv", pd.DataFrame(rows))
    _atomic_json(output_root / "statistics" / "split_balance_summary.json", summary)


def run_segment_pipeline(settings: SegmentPipelineSettings) -> dict[str, Any]:
    """两遍处理：先估计带权划分和训练 D_max，再并行生成冻结特征。"""

    _validate_settings(settings)
    csv_root = Path(settings.csv_dir)
    output_root = Path(settings.output_dir)
    success_path = output_root / ".pipeline_success.json"
    success_path.unlink(missing_ok=True)

    sources = _select_sources(_discover_sources(csv_root), settings.run_mode)
    _validate_source_labels(sources, settings.run_mode)
    selected_flows = {
        source.source_key: (
            _select_smoke_flow_ids(source, limit=settings.smoke_flows_per_file)
            if settings.run_mode.lower() == "smoke"
            else None
        )
        for source in sources
    }
    if any(value == () for value in selected_flows.values()):
        raise ValueError("at least one smoke source contains no selectable flows")

    profile_tasks = [
        (source, settings, selected_flows[source.source_key]) for source in sources
    ]
    profiles: list[SourceProfile] = _run_source_tasks(
        _profile_source_task,
        profile_tasks,
        settings.workers,
        "统计",
    )
    group_labels = {profile.source.capture_group: profile.source.application for profile in profiles}
    group_weights = {
        profile.source.capture_group: float(max(1, profile.eligible_segments))
        for profile in profiles
    }
    group_primary = {profile.source.capture_group: profile.source.primary for profile in profiles}
    group_assignment = create_weighted_group_assignment(
        group_labels,
        group_weights,
        group_primary,
        settings.val_ratio,
        settings.test_ratio,
        settings.seed,
        trials=settings.split_search_trials,
        require_class_coverage=settings.run_mode.lower() == "full",
    )

    # 只有已划入训练集的采集组可以参与 D_max 拟合，避免验证/测试信息泄漏。
    natural_durations = [
        duration
        for profile in profiles
        if group_assignment[profile.source.capture_group] == "train"
        for duration in profile.natural_durations
    ]
    if not natural_durations:
        raise ValueError("training split has no multi-packet natural bursts for D_max")
    dmax = float(
        np.quantile(np.asarray(natural_durations, dtype=np.float64), settings.dmax_quantile)
    )

    build_tasks = [
        (
            source,
            settings,
            selected_flows[source.source_key],
            dmax,
            group_assignment[source.capture_group],
        )
        for source in sources
    ]
    batches: list[SourceFeatureBatch] = _run_source_tasks(
        _build_source_task,
        build_tasks,
        settings.workers,
        "特征",
    )
    nonempty = [batch for batch in batches if len(batch.sample_keys)]
    if not nonempty:
        raise ValueError("no eligible model samples were generated")

    packet_seq = np.concatenate([batch.packet_seq for batch in nonempty]).astype(np.float32)
    packet_mask = np.concatenate([batch.packet_mask for batch in nonempty]).astype(np.float32)
    burst_seq = np.concatenate([batch.burst_seq for batch in nonempty]).astype(np.float32)
    burst_mask = np.concatenate([batch.burst_mask for batch in nonempty]).astype(np.float32)
    primary_array = np.concatenate([batch.primary_labels for batch in nonempty]).astype(np.int64)
    secondary_array = np.concatenate([batch.secondary_labels for batch in nonempty]).astype(np.int64)
    key_array = np.concatenate([batch.sample_keys for batch in nonempty]).astype(str)
    group_array = np.concatenate([batch.group_ids for batch in nonempty]).astype(str)
    segment_rows = [row for batch in batches for row in batch.segment_rows]
    sample_rows = [row for batch in batches for row in batch.sample_rows]
    input_packets = sum(profile.input_packets for profile in profiles)
    ineligible_packets = sum(profile.ineligible_packets for profile in profiles)
    modeled_packets = sum(batch.modeled_packets for batch in batches)
    capacity_reasons: Counter[str] = Counter()
    for batch in batches:
        capacity_reasons.update(batch.capacity_counts)
    if modeled_packets + ineligible_packets != input_packets:
        raise AssertionError("packet conservation failed: modeled + ineligible must equal input")
    mappings = _label_mappings()

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

    _write_split_balance_audit(
        output_root,
        profiles,
        sample_frame,
        group_assignment,
        settings,
    )

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
        "data_version": "segment15_burstp95_v1_1",
        "source_files": len(sources),
        "selected_parent_flows": int(sum(profile.selected_flow_count for profile in profiles)),
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
            "workers": int(settings.workers),
            "smoke_flows_per_file": int(settings.smoke_flows_per_file),
            "split_search_trials": int(settings.split_search_trials),
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
    print(f"处理父流数：{summary['selected_parent_flows']}")
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
