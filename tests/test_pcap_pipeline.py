from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.pcap_to_csv import (  # noqa: E402
    FlowSessionizer,
    discover_pcaps,
    transport_payload_length,
)


def test_discover_pcaps_deduplicates_and_ignores_similar_names(tmp_path):
    (tmp_path / "a.pcap").write_bytes(b"")
    (tmp_path / "b.pcapng").write_bytes(b"")
    (tmp_path / "notes.pcap.txt").write_bytes(b"")
    assert [p.name for p in discover_pcaps(tmp_path)] == ["a.pcap", "b.pcapng"]


def test_sessionizer_splits_after_timeout_and_fin():
    sessionizer = FlowSessionizer(timeout_seconds=60)
    assert sessionizer.assign("flow", 0.0, 0) == "flow_S0"
    assert sessionizer.assign("flow", 61.0, 0) == "flow_S1"
    assert sessionizer.assign("flow", 62.0, 0x01) == "flow_S1"
    assert sessionizer.assign("flow", 62.1, 0) == "flow_S2"


def test_transport_payload_length_excludes_transport_headers():
    tcp = bytearray(80)
    tcp[40 + 12] = 5 << 4  # TCP data offset=5，即 20 字节首部
    assert transport_payload_length(bytes(tcp), 20, 40, 6, 80) == 20
    assert transport_payload_length(bytes(80), 20, 40, 17, 80) == 32
    assert transport_payload_length(bytes(45), 20, 40, 17, 80) == 0

