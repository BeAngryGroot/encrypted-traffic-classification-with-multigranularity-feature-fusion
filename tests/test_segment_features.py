from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.segment_features import (  # noqa: E402
    BurstAssignment,
    assign_bursts_with_reasons,
    collect_mult_packet_burst_durations,
    pack_by_burst_capacity,
    time_segment_packets,
)


def test_time_segments_are_non_overlapping_and_keep_tail():
    packets = [
        {"timestamp": value, "frame_index": index}
        for index, value in enumerate([0.0, 14.9, 15.0, 31.0])
    ]

    segments = time_segment_packets(packets, 15.0)

    assert [[packet["frame_index"] for packet in segment] for segment in segments] == [
        [0, 1],
        [2],
        [3],
    ]
    assert sum(map(len, segments)) == len(packets)


def test_time_segmentation_rejects_non_positive_window():
    with pytest.raises(ValueError, match="window_seconds"):
        time_segment_packets([{"timestamp": 0.0}], 0.0)


def test_bursts_split_on_direction_iat_and_duration_cap():
    packets = [
        {"timestamp": 0.0, "direction": 1.0},
        {"timestamp": 0.1, "direction": 1.0},
        {"timestamp": 0.2, "direction": -1.0},
        {"timestamp": 0.3, "direction": -1.0},
        {"timestamp": 0.8, "direction": -1.0},
        {"timestamp": 1.2, "direction": -1.0},
    ]

    result = assign_bursts_with_reasons(
        packets,
        alpha=1.0,
        max_duration=0.3,
        fixed_threshold=0.6,
    )

    assert result.burst_ids == [0, 0, 1, 1, 2, 3]
    assert result.split_reasons == [
        "flow_start",
        "continuation",
        "direction_change",
        "continuation",
        "duration_cap",
        "duration_cap",
    ]


def test_natural_burst_duration_collection_excludes_single_packet_bursts():
    packets = [
        {"timestamp": 0.0, "direction": 1.0},
        {"timestamp": 0.1, "direction": 1.0},
        {"timestamp": 0.2, "direction": -1.0},
    ]
    assignment = assign_bursts_with_reasons(packets, alpha=1.0)

    assert collect_mult_packet_burst_durations(packets, assignment) == pytest.approx([0.1])


def test_zero_iat_segment_has_zero_adaptive_threshold():
    packets = [
        {"timestamp": 1.0, "direction": 1.0},
        {"timestamp": 1.0, "direction": 1.0},
    ]

    result = assign_bursts_with_reasons(packets, alpha=1.0)

    assert result.adaptive_threshold == 0.0
    assert result.burst_ids == [0, 0]


def test_capacity_packing_preserves_every_packet_once():
    packets = [
        {
            "frame_index": index,
            "timestamp": index * 0.01,
            "direction": 1.0 if index < 3 else -1.0,
        }
        for index in range(8)
    ]
    assignment = BurstAssignment(
        burst_ids=[0, 0, 0, 1, 1, 1, 2, 2],
        split_reasons=[
            "flow_start",
            "continuation",
            "continuation",
            "direction_change",
            "continuation",
            "continuation",
            "direction_change",
            "continuation",
        ],
        adaptive_threshold=1.0,
    )

    samples = pack_by_burst_capacity(
        packets,
        assignment,
        max_packets=5,
        max_bursts=2,
    )

    observed = [packet["frame_index"] for sample in samples for packet in sample.packets]
    assert observed == list(range(8))
    assert all(len(sample.packets) <= 5 for sample in samples)
    assert all(len(set(sample.burst_ids)) <= 2 for sample in samples)


def test_single_oversized_burst_uses_packet_capacity_cap():
    packets = [
        {"frame_index": index, "timestamp": index * 0.001, "direction": 1.0}
        for index in range(7)
    ]
    assignment = BurstAssignment(
        burst_ids=[0] * 7,
        split_reasons=["flow_start"] + ["continuation"] * 6,
        adaptive_threshold=1.0,
    )

    samples = pack_by_burst_capacity(
        packets,
        assignment,
        max_packets=3,
        max_bursts=2,
    )

    assert [len(sample.packets) for sample in samples] == [3, 3, 1]
    assert [packet["frame_index"] for sample in samples for packet in sample.packets] == list(range(7))
    assert all(sample.split_reason == "packet_capacity_cap" for sample in samples)


def test_sample_within_capacity_is_not_reported_as_capacity_split():
    packets = [
        {"frame_index": 0, "timestamp": 0.0, "direction": 1.0},
        {"frame_index": 1, "timestamp": 0.1, "direction": 1.0},
    ]
    assignment = BurstAssignment(
        burst_ids=[0, 0],
        split_reasons=["flow_start", "continuation"],
        adaptive_threshold=1.0,
    )

    samples = pack_by_burst_capacity(
        packets,
        assignment,
        max_packets=4,
        max_bursts=2,
    )

    assert len(samples) == 1
    assert samples[0].split_reason == "none"
