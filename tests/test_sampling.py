from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.sample_flows_by_ratio import sample_one_pair, select_flow_ids  # noqa: E402


def test_select_flow_ids_is_deterministic_and_respects_cap():
    ids = [f"flow-{index}" for index in range(20)]
    first = select_flow_ids(ids, ratio=0.8, seed=7, max_flows=5, min_flows=1)
    second = select_flow_ids(ids, ratio=0.8, seed=7, max_flows=5, min_flows=1)
    assert first == second
    assert len(first) == 5


def test_sampling_keeps_complete_flows_and_writes_manifest(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "sampled"
    source.mkdir()
    flow_file = source / "Tor_Chat_flows.csv"
    packet_file = source / "Tor_Chat_packets.csv"
    pd.DataFrame({"flow_id": ["a", "b", "c"], "packet_count": [2, 2, 2]}).to_csv(flow_file, index=False)
    pd.DataFrame({"flow_id": ["a", "a", "b", "b", "c", "c"], "packet_length": [1] * 6}).to_csv(packet_file, index=False)

    record = sample_one_pair(flow_file, 0.5, output, source, seed=11, max_flows=1, min_flows=1)
    sampled_packets = pd.read_csv(output / packet_file.name)
    assert sampled_packets["flow_id"].nunique() == 1
    assert len(sampled_packets) == 2
    assert record["seed"] == 11
    assert record["selected_flow_ids"] in {"a", "b", "c"}

