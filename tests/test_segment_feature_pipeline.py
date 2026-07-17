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
    SourceInfo,
    SourceProfile,
    SourceSampleCount,
    _build_source_task,
    _count_source_samples_task,
    _discover_sources,
    _read_packet_frame,
    _refine_group_assignment,
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


def test_count_only_pass_matches_built_source_sample_count(tmp_path):
    csv_root = build_synthetic_iscxtor_csv_tree(tmp_path)
    source = _discover_sources(csv_root)[0]
    settings = make_settings(csv_root, tmp_path / "out")

    count = _count_source_samples_task((source, settings, None, 0.2))
    batch = _build_source_task((source, settings, None, 0.2, "train"))

    assert count.final_sample_count == len(batch.sample_keys)
    assert count.modeled_packets == batch.modeled_packets


def test_refinement_uses_final_counts_and_final_train_only_dmax(tmp_path):
    sources = []
    profiles = []
    expected_counts = {}
    weights = [60, 20, 6, 5, 5, 5, 5, 4]
    for application in ["A", "B"]:
        for index, weight in enumerate(weights):
            source = SourceInfo(
                path=tmp_path / f"{application}-{index}.csv",
                source_key=f"{application}-{index}",
                capture_group=f"{application}-{index}",
                primary="TOR" if index % 2 else "NONTOR",
                application=application,
            )
            sources.append(source)
            profiles.append(
                SourceProfile(
                    source=source,
                    input_packets=10,
                    initial_segments=1,
                    eligible_segments=1,
                    ineligible_packets=0,
                    natural_durations=(0.01 * (index + 1),),
                    selected_flow_count=1,
                    elapsed_seconds=0.0,
                )
            )
            expected_counts[source.capture_group] = weight

    settings = SegmentPipelineSettings(
        csv_dir=tmp_path,
        output_dir=tmp_path / "out",
        run_mode="full",
        val_ratio=0.10,
        test_ratio=0.10,
        split_search_trials=4000,
        max_split_iterations=3,
        workers=1,
    )

    def count_runner(current_sources, _settings, _selected_flows, _dmax):
        return [
            SourceSampleCount(
                source=source,
                final_sample_count=expected_counts[source.capture_group],
                modeled_packets=expected_counts[source.capture_group],
                elapsed_seconds=0.0,
            )
            for source in current_sources
        ]

    result = _refine_group_assignment(
        sources,
        profiles,
        settings,
        {source.source_key: None for source in sources},
        count_runner=count_runner,
    )

    assert result.group_sample_counts == expected_counts
    assert len(result.history) <= 3
    assert result.converged
    assert set(result.dmax_train_groups) == {
        group for group, split in result.assignment.items() if split == "train"
    }
    expected_durations = [
        duration
        for profile in profiles
        if result.assignment[profile.source.capture_group] == "train"
        for duration in profile.natural_durations
    ]
    assert result.dmax == pytest.approx(np.quantile(expected_durations, 0.95))


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


def build_unbalanceable_full_csv_tree(tmp_path: Path) -> Path:
    """每类一个大组、两个小组，保持PCAP完整时无法通过80/10/10门槛。"""

    root = tmp_path / "full_csv"
    applications = ["AUDIO", "BROWSING", "CHAT", "EMAIL", "FILE", "P2P", "VIDEO", "VOIP"]
    for application in applications:
        for group_index, packet_count in enumerate([80, 2, 2]):
            transport = "NonTor" if group_index == 1 else "Tor"
            path = root / transport / application / f"capture_{group_index}_packets.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            rows = []
            for frame in range(packet_count):
                # 每两个包改变一次方向，既产生可计算时长的Burst，又触发容量拆分。
                forward = (frame // 2) % 2 == 0
                rows.append(
                    {
                        "flow_id": f"{application}-{group_index}",
                        "frame_index": frame,
                        "timestamp": frame * 0.01,
                        "packet_length": 100 + frame,
                        "payload_length": 60,
                        "src_ip": "10.0.0.1" if forward else "10.0.0.2",
                        "dst_ip": "10.0.0.2" if forward else "10.0.0.1",
                        "src_port": 1234 if forward else 443,
                        "dst_port": 443 if forward else 1234,
                        "protocol": 6,
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
    assert (output / "manifests/split_iteration_history.csv").exists()
    assert (output / "manifests/group_weight_audit.csv").exists()
    assert (output / "statistics/split_balance_summary.json").exists()
    assert (output / ".pipeline_success.json").exists()

    dmax = json.loads((output / "statistics/dmax_summary.json").read_text(encoding="utf-8"))
    assert dmax["source_split"] == "train"
    assert dmax["quantile"] == 0.95
    assert dmax["natural_burst_count"] > 0
    balance = json.loads(
        (output / "statistics/split_balance_summary.json").read_text(encoding="utf-8")
    )
    assert balance["target_ratios"] == {"train": 0.8, "val": 0.1, "test": 0.1}
    assert balance["quality"]["status"] in {"passed", "smoke_not_enforced"}


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


def test_full_pipeline_rejects_unreachable_split_quality_without_success_marker(tmp_path):
    csv_root = build_unbalanceable_full_csv_tree(tmp_path)
    output = tmp_path / "quality_failure"
    settings = SegmentPipelineSettings(
        csv_dir=csv_root,
        output_dir=output,
        run_mode="full",
        val_ratio=0.10,
        test_ratio=0.10,
        max_packets=4,
        max_bursts=3,
        workers=1,
        split_search_trials=500,
        max_split_iterations=1,
    )

    with pytest.raises(ValueError, match="split quality"):
        run_segment_pipeline(settings)

    assert not (output / ".pipeline_success.json").exists()


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
