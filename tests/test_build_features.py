from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.build_features import build_features_from_csv_dir  # noqa: E402
from data.burst_features import BURST_FEATURES, PACKET_FEATURES  # noqa: E402


def test_build_features_from_csv_dir_writes_new_feature_schema(tmp_path):
    csv_dir = tmp_path / "csv"
    out_dir = tmp_path / "features"
    csv_dir.mkdir()

    rows = [
        {
            "flow_id": "flow-a",
            "timestamp": 0.000,
            "packet_length": 100,
            "payload_length": 60,
            "src_ip": "10.0.0.1",
            "dst_ip": "10.0.0.2",
            "src_port": 1234,
            "dst_port": 443,
            "protocol": 6,
            "ip_ttl": 64,
            "tcp_flags": 16,
        },
        {
            "flow_id": "flow-a",
            "timestamp": 0.010,
            "packet_length": 120,
            "payload_length": 80,
            "src_ip": "10.0.0.1",
            "dst_ip": "10.0.0.2",
            "src_port": 1234,
            "dst_port": 443,
            "protocol": 6,
            "ip_ttl": 64,
            "tcp_flags": 16,
        },
        {
            "flow_id": "flow-a",
            "timestamp": 0.020,
            "packet_length": 280,
            "payload_length": 240,
            "src_ip": "10.0.0.2",
            "dst_ip": "10.0.0.1",
            "src_port": 443,
            "dst_port": 1234,
            "protocol": 6,
            "ip_ttl": 63,
            "tcp_flags": 24,
        },
    ]
    pd.DataFrame(rows).to_csv(csv_dir / "VPN_AUDIO_packets.csv", index=False)

    summary = build_features_from_csv_dir(
        csv_dir,
        out_dir,
        max_packets=4,
        max_bursts=3,
        alpha=1.0,
    )

    assert summary["num_samples"] == 1
    assert np.load(out_dir / "packet_seq.npy").shape == (1, 4, len(PACKET_FEATURES))
    assert np.load(out_dir / "packet_mask.npy").shape == (1, 4)
    assert np.load(out_dir / "burst_seq.npy").shape == (1, 3, len(BURST_FEATURES))
    assert np.load(out_dir / "burst_mask.npy").shape == (1, 3)
    assert np.load(out_dir / "combined_labels.npy").tolist() == [0]

    with (out_dir / "label_mappings.pkl").open("rb") as f:
        mappings = pickle.load(f)
    assert mappings["combined"]["id_to_label"][0] == "VPN:AUDIO"
    assert np.load(out_dir / "sample_keys.npy", allow_pickle=True).tolist() == ["VPN_AUDIO:flow-a"]
    assert np.load(out_dir / "group_ids.npy", allow_pickle=True).tolist() == ["VPN_AUDIO_packets.csv"]
    manifest = pd.read_csv(out_dir / "sample_manifest.csv")
    assert manifest.loc[0, "sample_key"] == "VPN_AUDIO:flow-a"
    assert manifest.loc[0, "capture_group"] == "VPN_AUDIO_packets.csv"
