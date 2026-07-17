"""流片段、训练集自适应 burst 与无损容量拆分的纯函数。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from data.burst_features import compute_adaptive_threshold


Packet = Mapping[str, Any]


@dataclass(frozen=True)
class BurstAssignment:
    """一个已按时间排序片段的 burst 编号和逐包切分原因。"""

    burst_ids: list[int]
    split_reasons: list[str]
    adaptive_threshold: float


@dataclass(frozen=True)
class CapacitySample:
    """满足模型容量约束且没有丢包的最终样本。"""

    packets: list[dict[str, Any]]
    burst_ids: list[int]
    split_reason: str


def _timestamp(packet: Packet) -> float:
    for name in ("timestamp", "packet_time", "time", "ts"):
        value = packet.get(name)
        if value is not None and value != "":
            return float(value)
    raise ValueError("packet is missing timestamp")


def _direction(packet: Packet) -> float:
    for name in ("direction", "packet_direction"):
        value = packet.get(name)
        if value is not None and value != "":
            return float(value)
    raise ValueError("packet is missing direction")


def time_segment_packets(
    packets: Sequence[Packet],
    window_seconds: float,
) -> list[list[dict[str, Any]]]:
    """按父流首包对齐的固定时间窗切片，每个输入包恰好保留一次。"""

    if float(window_seconds) <= 0:
        raise ValueError("window_seconds must be positive")
    ordered = sorted((dict(packet) for packet in packets), key=_timestamp)
    if not ordered:
        return []

    first_timestamp = _timestamp(ordered[0])
    buckets: dict[int, list[dict[str, Any]]] = {}
    for packet in ordered:
        offset = max(0.0, _timestamp(packet) - first_timestamp)
        segment_index = int(offset // float(window_seconds))
        buckets.setdefault(segment_index, []).append(packet)
    return [buckets[index] for index in sorted(buckets)]


def assign_bursts_with_reasons(
    packets: Sequence[Packet],
    *,
    alpha: float = 1.0,
    max_duration: float | None = None,
    fixed_threshold: float | None = None,
) -> BurstAssignment:
    """按方向、IAT 和最大持续时间生成同向 burst。"""

    ordered = sorted(packets, key=_timestamp)
    if not ordered:
        return BurstAssignment([], [], 0.0)
    if max_duration is not None and float(max_duration) < 0:
        raise ValueError("max_duration must be non-negative")

    timestamps = [_timestamp(packet) for packet in ordered]
    directions = [_direction(packet) for packet in ordered]
    iats = [0.0]
    for index in range(1, len(timestamps)):
        iats.append(max(0.0, timestamps[index] - timestamps[index - 1]))
    threshold = (
        float(fixed_threshold)
        if fixed_threshold is not None
        else compute_adaptive_threshold(iats, alpha)
    )

    burst_ids = [0]
    split_reasons = ["flow_start"]
    current_burst = 0
    current_start_time = timestamps[0]
    for index in range(1, len(ordered)):
        reason = "continuation"
        if directions[index] != directions[index - 1]:
            reason = "direction_change"
        elif iats[index] > threshold:
            reason = "iat_gap"
        elif (
            max_duration is not None
            and timestamps[index] - current_start_time > float(max_duration)
        ):
            reason = "duration_cap"

        if reason != "continuation":
            current_burst += 1
            current_start_time = timestamps[index]
        burst_ids.append(current_burst)
        split_reasons.append(reason)

    return BurstAssignment(burst_ids, split_reasons, float(threshold))


def collect_mult_packet_burst_durations(
    packets: Sequence[Packet],
    assignment: BurstAssignment,
) -> list[float]:
    """收集至少含两个包的自然 burst 时长，排除零信息单包 burst。"""

    ordered = sorted(packets, key=_timestamp)
    if len(ordered) != len(assignment.burst_ids):
        raise ValueError("packet count and burst assignment length must match")
    if not ordered:
        return []

    durations: list[float] = []
    start = 0
    for index in range(1, len(ordered) + 1):
        boundary = (
            index == len(ordered)
            or assignment.burst_ids[index] != assignment.burst_ids[start]
        )
        if boundary:
            if index - start >= 2:
                durations.append(max(0.0, _timestamp(ordered[index - 1]) - _timestamp(ordered[start])))
            start = index
    return durations


def pack_by_burst_capacity(
    packets: Sequence[Packet],
    assignment: BurstAssignment,
    *,
    max_packets: int,
    max_bursts: int,
) -> list[CapacitySample]:
    """优先在 burst 边界拆分；单个超长 burst 才按包容量切分。"""

    if int(max_packets) < 1 or int(max_bursts) < 1:
        raise ValueError("max_packets and max_bursts must be positive")
    ordered = sorted((dict(packet) for packet in packets), key=_timestamp)
    if len(ordered) != len(assignment.burst_ids):
        raise ValueError("packet count and burst assignment length must match")
    if not ordered:
        return []

    # 先形成完整 burst 单元；只有单个 burst 自身超限时才建立容量子 burst。
    units: list[tuple[list[dict[str, Any]], bool]] = []
    start = 0
    for index in range(1, len(ordered) + 1):
        boundary = (
            index == len(ordered)
            or assignment.burst_ids[index] != assignment.burst_ids[start]
        )
        if not boundary:
            continue
        group = ordered[start:index]
        if len(group) > int(max_packets):
            for chunk_start in range(0, len(group), int(max_packets)):
                units.append((group[chunk_start:chunk_start + int(max_packets)], True))
        else:
            units.append((group, False))
        start = index

    samples: list[CapacitySample] = []
    current_packets: list[dict[str, Any]] = []
    current_ids: list[int] = []
    current_burst_count = 0
    current_forced = False

    def flush() -> None:
        nonlocal current_packets, current_ids, current_burst_count, current_forced
        if not current_packets:
            return
        samples.append(
            CapacitySample(
                packets=current_packets,
                burst_ids=current_ids,
                split_reason=("packet_capacity_cap" if current_forced else "burst_capacity_boundary"),
            )
        )
        current_packets = []
        current_ids = []
        current_burst_count = 0
        current_forced = False

    for unit_packets, forced in units:
        would_overflow = current_packets and (
            len(current_packets) + len(unit_packets) > int(max_packets)
            or current_burst_count + 1 > int(max_bursts)
        )
        if would_overflow:
            flush()
        new_burst_id = current_burst_count
        current_packets.extend(unit_packets)
        current_ids.extend([new_burst_id] * len(unit_packets))
        current_burst_count += 1
        current_forced = current_forced or forced
    flush()
    if len(samples) == 1 and samples[0].split_reason == "burst_capacity_boundary":
        only = samples[0]
        samples[0] = CapacitySample(only.packets, only.burst_ids, "none")
    return samples
