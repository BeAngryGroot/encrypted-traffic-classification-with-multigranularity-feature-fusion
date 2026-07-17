from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.run_segment_feature_pipeline import (  # noqa: E402
    SegmentPipelineSettings,
    _read_packet_frame,
    _select_smoke_flow_ids,
    _source_info,
    run_segment_pipeline,
)


def test_pipeline_entry_can_be_loaded_from_data_directory():
    """服务器从 data 目录直接运行脚本时，也必须能找到 data 包。"""

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import runpy; "
                "runpy.run_path('run_segment_feature_pipeline.py', run_name='not_main')"
            ),
        ],
        cwd=ROOT / "data",
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def build_synthetic_iscxtor_csv_tree(tmp_path: Path) -> Path:
    root = tmp_path / "csv"
    for application in ["BROWSING", "EMAIL"]:
        for group_index in range(3):
            transport = "Tor" if group_index % 2 == 0 else "NonTor"
            path = root / transport / application / f"capture_{group_index}_packets.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            rows = []
            for frame, timestamp in enumerate([0.0, 0.1, 15.0, 15.1, 15.2]):
                forward = frame % 3 != 2
                rows.append(
                    {
                        "flow_id": f"flow-{group_index}",
                        "frame_index": frame,
                        "timestamp": timestamp,
                        "packet_length": 100 + frame,
                        "payload_length": 60 + frame,
                        "src_ip": "10.0.0.1" if forward else "10.0.0.2",
                        "dst_ip": "10.0.0.2" if forward else "10.0.0.1",
                        "src_port": 1234 if forward else 443,
                        "dst_port": 443 if forward else 1234,
                        "protocol": 6,
                        "ip_ttl": 64,
                        "tcp_flags": 16,
                    }
                )
            pd.DataFrame(rows).to_csv(path, index=False)
    return root


def make_settings(csv_root: Path, output: Path) -> SegmentPipelineSettings:
    return SegmentPipelineSettings(
        csv_dir=csv_root,
        output_dir=output,
        run_mode="smoke",
        max_packets=4,
        max_bursts=3,
        window_seconds=15.0,
        min_model_packets=2,
        seed=42,
    )


def test_segment_pipeline_writes_compatible_features_without_packet_loss(tmp_path):
    csv_root = build_synthetic_iscxtor_csv_tree(tmp_path)
    output = tmp_path / "processed"

    summary = run_segment_pipeline(make_settings(csv_root, output))

    assert summary["input_packets"] == 30
    assert summary["input_packets"] == summary["modeled_packets"] + summary["ineligible_packets"]
    assert np.load(output / "features/packet_seq.npy").shape[1:] == (4, 16)
    assert np.load(output / "features/burst_seq.npy").shape[1:] == (3, 12)
    assert np.array_equal(
        np.load(output / "features/primary_labels.npy"),
        np.load(output / "features/tor_labels.npy"),
    )
    assert np.array_equal(
        np.load(output / "features/secondary_labels.npy"),
        np.load(output / "features/application_labels.npy"),
    )
    assert (output / "features/split_seed42.npz").exists()
    assert (output / "manifests/segment_manifest.csv").exists()
    assert (output / "manifests/sample_manifest.csv").exists()
    assert (output / "manifests/split_balance.csv").exists()
    assert (output / "statistics/split_balance_summary.json").exists()
    assert (output / ".pipeline_success.json").exists()

    dmax = json.loads((output / "statistics/dmax_summary.json").read_text(encoding="utf-8"))
    assert dmax["source_split"] == "train"
    assert dmax["quantile"] == 0.95
    assert dmax["natural_burst_count"] > 0


def test_segment_pipeline_is_deterministic_on_rerun(tmp_path):
    csv_root = build_synthetic_iscxtor_csv_tree(tmp_path)
    output = tmp_path / "processed"
    settings = make_settings(csv_root, output)

    run_segment_pipeline(settings)
    first_keys = np.load(output / "features/sample_keys.npy", allow_pickle=True).copy()
    first_dmax = (output / "statistics/dmax_summary.json").read_text(encoding="utf-8")
    run_segment_pipeline(settings)

    np.testing.assert_array_equal(
        first_keys,
        np.load(output / "features/sample_keys.npy", allow_pickle=True),
    )
    assert first_dmax == (output / "statistics/dmax_summary.json").read_text(encoding="utf-8")


def test_segment_pipeline_rejects_missing_timestamp(tmp_path):
    csv_root = build_synthetic_iscxtor_csv_tree(tmp_path)
    target = next(csv_root.rglob("*_packets.csv"))
    frame = pd.read_csv(target).drop(columns=["timestamp"])
    frame.to_csv(target, index=False)

    with pytest.raises(ValueError, match="timestamp"):
        run_segment_pipeline(make_settings(csv_root, tmp_path / "broken"))


def test_smoke_selects_smallest_complete_flows(tmp_path):
    csv_root = tmp_path / "csv"
    packet_path = csv_root / "Tor" / "BROWSING" / "many_packets.csv"
    packet_path.parent.mkdir(parents=True)
    rows = []
    counts = {"large": 9, "small": 2, "middle": 5}
    for flow_id, count in counts.items():
        for frame in range(count):
            rows.append(
                {
                    "flow_id": flow_id,
                    "frame_index": f"{flow_id}-{frame}",
                    "timestamp": frame * 0.1,
                    "packet_length": 100,
                    "src_ip": "10.0.0.1",
                    "dst_ip": "10.0.0.2",
                    "src_port": 1234,
                    "dst_port": 443,
                    "protocol": 6,
                }
            )
    pd.DataFrame(rows).to_csv(packet_path, index=False)
    pd.DataFrame(
        [{"flow_id": flow_id, "packet_count": count} for flow_id, count in counts.items()]
    ).to_csv(packet_path.with_name("many_flows.csv"), index=False)

    source = _source_info(packet_path, csv_root)
    selected = _select_smoke_flow_ids(source, limit=2)
    frame = _read_packet_frame(source, selected_flow_ids=selected, chunksize=3)

    assert selected == ("small", "middle")
    assert frame.groupby("flow_id").size().to_dict() == {"middle": 5, "small": 2}


def test_single_and_two_worker_pipeline_outputs_are_identical(tmp_path):
    csv_root = build_synthetic_iscxtor_csv_tree(tmp_path)
    single_output = tmp_path / "single"
    parallel_output = tmp_path / "parallel"
    single = make_settings(csv_root, single_output)
    parallel = SegmentPipelineSettings(
        **{**single.__dict__, "output_dir": parallel_output, "workers": 2}
    )

    single_summary = run_segment_pipeline(single)
    parallel_summary = run_segment_pipeline(parallel)

    assert single_summary["split_samples"] == parallel_summary["split_samples"]
    for name in [
        "packet_seq.npy",
        "packet_mask.npy",
        "burst_seq.npy",
        "burst_mask.npy",
        "primary_labels.npy",
        "secondary_labels.npy",
        "sample_keys.npy",
        "group_ids.npy",
    ]:
        np.testing.assert_array_equal(
            np.load(single_output / "features" / name, allow_pickle=True),
            np.load(parallel_output / "features" / name, allow_pickle=True),
        )
