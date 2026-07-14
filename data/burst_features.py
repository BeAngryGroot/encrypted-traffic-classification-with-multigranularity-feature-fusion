from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


PACKET_FEATURES = [
    "packet_length",
    "payload_length",
    "direction",
    "signed_length",
    "iat",
    "log_iat",
    "tcp_flags",
    "ip_ttl",
    "protocol",
    "burst_id",
    "pos_in_burst",
    "burst_size",
    "burst_bytes",
    "is_burst_start",
    "is_burst_end",
    "burst_duration",
]

BURST_FEATURES = [
    "burst_direction",
    "packet_count",
    "byte_sum",
    "duration",
    "mean_len",
    "std_len",
    "mean_iat",
    "std_iat",
    "max_iat",
    "gap_to_previous_burst",
    "is_first_burst",
    "is_last_burst",
]


@dataclass(frozen=True)
class FlowFeatureResult:
    packet_seq: np.ndarray
    packet_mask: np.ndarray
    burst_seq: np.ndarray
    burst_mask: np.ndarray
    burst_ids: list[int]


def _get_number(packet: Mapping[str, Any], *names: str, default: float = 0.0) -> float:
    for name in names:
        value = packet.get(name)
        if value is None or value == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return float(default)


def _tcp_flags_to_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        pass
    try:
        return float(int(text, 16))
    except ValueError:
        return 0.0


def _sorted_packets(packets: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(
        packets,
        key=lambda p: _get_number(p, "timestamp", "packet_time", "time", "ts", default=0.0),
    )


def _timestamps(packets: Sequence[Mapping[str, Any]]) -> list[float]:
    return [_get_number(p, "timestamp", "packet_time", "time", "ts", default=0.0) for p in packets]


def _directions(packets: Sequence[Mapping[str, Any]]) -> list[float]:
    return [_get_number(p, "direction", "packet_direction", default=0.0) for p in packets]


def _lengths(packets: Sequence[Mapping[str, Any]]) -> list[float]:
    return [_get_number(p, "packet_length", "len", "length", default=0.0) for p in packets]


def _payload_lengths(packets: Sequence[Mapping[str, Any]]) -> list[float]:
    return [_get_number(p, "payload_length", "payload_len", default=0.0) for p in packets]


def compute_iats(timestamps: Sequence[float]) -> list[float]:
    if not timestamps:
        return []
    iats = [0.0]
    for i in range(1, len(timestamps)):
        iats.append(max(0.0, float(timestamps[i]) - float(timestamps[i - 1])))
    return iats


def compute_adaptive_threshold(iats: Sequence[float], alpha: float = 1.0) -> float:
    positive = np.asarray([float(x) for x in iats if float(x) > 0.0], dtype=np.float64)
    if positive.size == 0:
        return 0.0
    q1 = float(np.percentile(positive, 25))
    q3 = float(np.percentile(positive, 75))
    median = float(np.median(positive))
    return median + float(alpha) * (q3 - q1)


def assign_bursts(
    packets: Sequence[Mapping[str, Any]],
    *,
    alpha: float = 1.0,
    fixed_threshold: float | None = None,
) -> list[int]:
    ordered = _sorted_packets(packets)
    if not ordered:
        return []

    timestamps = _timestamps(ordered)
    directions = _directions(ordered)
    iats = compute_iats(timestamps)
    threshold = float(fixed_threshold) if fixed_threshold is not None else compute_adaptive_threshold(iats, alpha)

    burst_ids = [0]
    current = 0
    for i in range(1, len(ordered)):
        same_direction = directions[i] == directions[i - 1]
        close_enough = iats[i] <= threshold
        if same_direction and close_enough:
            burst_ids.append(current)
        else:
            current += 1
            burst_ids.append(current)
    return burst_ids


def _burst_slices(burst_ids: Sequence[int]) -> list[tuple[int, int, int]]:
    if not burst_ids:
        return []
    slices: list[tuple[int, int, int]] = []
    start = 0
    current = burst_ids[0]
    for i, burst_id in enumerate(burst_ids[1:], start=1):
        if burst_id != current:
            slices.append((current, start, i))
            current = burst_id
            start = i
    slices.append((current, start, len(burst_ids)))
    return slices


def build_flow_features(
    packets: Sequence[Mapping[str, Any]],
    *,
    max_packets: int = 64,
    max_bursts: int = 32,
    alpha: float = 1.0,
    fixed_threshold: float | None = None,
) -> FlowFeatureResult:
    # 两个分支必须共享同一个可观察前缀。先截断再计算阈值和 burst，避免
    # packet_seq 偷看 max_packets 之后的包，也避免 burst_seq 与包级视图错位。
    ordered = _sorted_packets(packets)[:max_packets]
    packet_seq = np.zeros((max_packets, len(PACKET_FEATURES)), dtype=np.float32)
    packet_mask = np.zeros((max_packets,), dtype=np.float32)
    burst_seq = np.zeros((max_bursts, len(BURST_FEATURES)), dtype=np.float32)
    burst_mask = np.zeros((max_bursts,), dtype=np.float32)

    if not ordered:
        return FlowFeatureResult(packet_seq, packet_mask, burst_seq, burst_mask, [])

    timestamps = _timestamps(ordered)
    lengths = _lengths(ordered)
    payload_lengths = _payload_lengths(ordered)
    directions = _directions(ordered)
    iats = compute_iats(timestamps)
    burst_ids = assign_bursts(ordered, alpha=alpha, fixed_threshold=fixed_threshold)
    slices = _burst_slices(burst_ids)

    burst_lookup: dict[int, dict[str, float]] = {}
    previous_end_time: float | None = None
    for sequence_index, (burst_id, start, end) in enumerate(slices):
        idx = list(range(start, end))
        burst_lengths = np.asarray([lengths[i] for i in idx], dtype=np.float32)
        burst_iats = np.asarray([iats[i] for i in idx if iats[i] > 0], dtype=np.float32)
        start_time = timestamps[start]
        end_time = timestamps[end - 1]
        duration = max(0.0, end_time - start_time)
        gap = 0.0 if previous_end_time is None else max(0.0, start_time - previous_end_time)
        previous_end_time = end_time

        stats = {
            "sequence_index": float(sequence_index),
            "direction": float(directions[start]),
            "packet_count": float(end - start),
            "byte_sum": float(np.sum(burst_lengths)),
            "duration": float(duration),
            "mean_len": float(np.mean(burst_lengths)) if burst_lengths.size else 0.0,
            "std_len": float(np.std(burst_lengths)) if burst_lengths.size else 0.0,
            "mean_iat": float(np.mean(burst_iats)) if burst_iats.size else 0.0,
            "std_iat": float(np.std(burst_iats)) if burst_iats.size else 0.0,
            "max_iat": float(np.max(burst_iats)) if burst_iats.size else 0.0,
            "gap_to_previous_burst": float(gap),
            "start": float(start),
            "end": float(end),
        }
        burst_lookup[burst_id] = stats

    for i, packet in enumerate(ordered):
        burst_id = burst_ids[i]
        burst = burst_lookup[burst_id]
        pos_in_burst = i - int(burst["start"])
        burst_size = int(burst["packet_count"])
        direction = directions[i]
        packet_length = lengths[i]
        payload_length = payload_lengths[i]
        packet_seq[i] = np.asarray(
            [
                packet_length,
                payload_length,
                direction,
                packet_length * direction,
                iats[i],
                np.log1p(iats[i]),
                _tcp_flags_to_float(packet.get("tcp_flags", packet.get("flags"))),
                _get_number(packet, "ip_ttl", "ttl", default=0.0),
                _get_number(packet, "protocol", "proto", default=0.0),
                float(burst_id),
                float(pos_in_burst),
                float(burst_size),
                burst["byte_sum"],
                1.0 if pos_in_burst == 0 else 0.0,
                1.0 if pos_in_burst == burst_size - 1 else 0.0,
                burst["duration"],
            ],
            dtype=np.float32,
        )
        packet_mask[i] = 1.0

    for output_index, (burst_id, _start, _end) in enumerate(slices[:max_bursts]):
        burst = burst_lookup[burst_id]
        burst_seq[output_index] = np.asarray(
            [
                burst["direction"],
                burst["packet_count"],
                burst["byte_sum"],
                burst["duration"],
                burst["mean_len"],
                burst["std_len"],
                burst["mean_iat"],
                burst["std_iat"],
                burst["max_iat"],
                burst["gap_to_previous_burst"],
                1.0 if output_index == 0 else 0.0,
                1.0 if output_index == min(len(slices), max_bursts) - 1 else 0.0,
            ],
            dtype=np.float32,
        )
        burst_mask[output_index] = 1.0

    return FlowFeatureResult(packet_seq, packet_mask, burst_seq, burst_mask, burst_ids)
