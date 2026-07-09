from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.burst_features import (  # noqa: E402
    BURST_FEATURES,
    PACKET_FEATURES,
    assign_bursts,
    build_flow_features,
)


def sample_packets():
    return [
        {"timestamp": 0.000, "packet_length": 100, "payload_length": 60, "direction": 1, "tcp_flags": 16, "ip_ttl": 64, "protocol": 6},
        {"timestamp": 0.010, "packet_length": 120, "payload_length": 80, "direction": 1, "tcp_flags": 16, "ip_ttl": 64, "protocol": 6},
        {"timestamp": 0.020, "packet_length": 300, "payload_length": 250, "direction": -1, "tcp_flags": 24, "ip_ttl": 63, "protocol": 6},
        {"timestamp": 0.021, "packet_length": 280, "payload_length": 240, "direction": -1, "tcp_flags": 24, "ip_ttl": 63, "protocol": 6},
        {"timestamp": 0.500, "packet_length": 90, "payload_length": 50, "direction": -1, "tcp_flags": 16, "ip_ttl": 63, "protocol": 6},
    ]


def test_assign_bursts_splits_on_direction_and_large_gap():
    burst_ids = assign_bursts(sample_packets(), alpha=1.0)

    assert burst_ids == [0, 0, 1, 1, 2]


def test_fixed_threshold_can_be_used_for_ablation():
    burst_ids = assign_bursts(sample_packets(), fixed_threshold=0.005)

    assert burst_ids == [0, 1, 2, 2, 3]


def test_build_flow_features_returns_packet_and_burst_sequences():
    result = build_flow_features(sample_packets(), max_packets=6, max_bursts=4, alpha=1.0)

    assert result.packet_seq.shape == (6, len(PACKET_FEATURES))
    assert result.packet_mask.tolist() == [1, 1, 1, 1, 1, 0]
    assert result.burst_seq.shape == (4, len(BURST_FEATURES))
    assert result.burst_mask.tolist() == [1, 1, 1, 0]

    packet_burst_id = PACKET_FEATURES.index("burst_id")
    packet_pos = PACKET_FEATURES.index("pos_in_burst")
    burst_packet_count = BURST_FEATURES.index("packet_count")
    burst_gap = BURST_FEATURES.index("gap_to_previous_burst")

    np.testing.assert_allclose(result.packet_seq[:5, packet_burst_id], [0, 0, 1, 1, 2])
    np.testing.assert_allclose(result.packet_seq[:5, packet_pos], [0, 1, 0, 1, 0])
    np.testing.assert_allclose(result.burst_seq[:3, burst_packet_count], [2, 2, 1])
    assert result.burst_seq[2, burst_gap] > 0.1
