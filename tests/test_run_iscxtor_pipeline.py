from pathlib import Path
import json
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.run_iscxtor_pipeline import (  # noqa: E402
    PipelineSettings,
    is_completed,
    output_paths,
    select_pcaps,
    validate_settings,
    write_done_marker,
)
import data.run_iscxtor_pipeline as pipeline  # noqa: E402


def test_smoke_selects_smallest_pcap_and_full_keeps_all(tmp_path):
    large = tmp_path / "large.pcap"
    small = tmp_path / "small.pcap"
    large.write_bytes(b"x" * 20)
    small.write_bytes(b"x" * 5)

    assert select_pcaps([large, small], "smoke") == [small]
    assert select_pcaps([large, small], "full") == [large, small]
    with pytest.raises(ValueError, match="RUN_MODE"):
        select_pcaps([large], "invalid")


def test_output_paths_preserve_tor_and_application_directories(tmp_path):
    raw = tmp_path / "raw"
    pcap = raw / "Tor" / "CHAT" / "demo.pcap"
    pcap.parent.mkdir(parents=True)
    pcap.write_bytes(b"pcap")
    paths = output_paths(pcap, raw, tmp_path / "csv")

    assert paths.packet_csv == tmp_path / "csv/Tor/CHAT/demo_packets.csv"
    assert paths.flow_csv == tmp_path / "csv/Tor/CHAT/demo_flows.csv"
    assert paths.done_marker == tmp_path / "csv/Tor/CHAT/demo.conversion_done.json"


def test_completion_requires_csv_files_and_success_marker(tmp_path):
    raw = tmp_path / "raw"
    pcap = raw / "NonTor" / "AUDIO" / "demo.pcap"
    pcap.parent.mkdir(parents=True)
    pcap.write_bytes(b"pcap")
    paths = output_paths(pcap, raw, tmp_path / "csv")
    paths.packet_csv.parent.mkdir(parents=True)

    paths.packet_csv.write_text("header\nrow\n", encoding="utf-8")
    paths.flow_csv.write_text("header\nrow\n", encoding="utf-8")
    assert not is_completed(paths)

    write_done_marker(paths, {"status": "success", "source": "demo.pcap"})
    assert is_completed(paths)
    assert json.loads(paths.done_marker.read_text(encoding="utf-8"))["status"] == "success"


def test_validate_settings_checks_mode_directory_workers_and_disk(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    settings = PipelineSettings(raw_dir=raw, output_dir=tmp_path / "out", run_mode="smoke")
    validate_settings(settings, free_bytes=1)

    with pytest.raises(ValueError, match="RUN_MODE"):
        validate_settings(PipelineSettings(raw, tmp_path / "out", "bad"), free_bytes=10**12)
    with pytest.raises(ValueError, match="WORKERS"):
        validate_settings(PipelineSettings(raw, tmp_path / "out", "full", workers=0), free_bytes=10**12)
    with pytest.raises(RuntimeError, match="磁盘空间"):
        validate_settings(PipelineSettings(raw, tmp_path / "out", "full", min_free_gib=2), free_bytes=1024**3)


def test_resume_skip_preserves_metrics_from_done_marker(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    pcap = raw / "Tor" / "CHAT" / "demo.pcap"
    pcap.parent.mkdir(parents=True)
    pcap.write_bytes(b"pcap")
    settings = PipelineSettings(raw, tmp_path / "out", "smoke")
    paths = output_paths(pcap, raw, tmp_path / "out/csv/smoke_parse_v1")
    paths.packet_csv.parent.mkdir(parents=True)
    paths.packet_csv.write_text("header\nrow\n", encoding="utf-8")
    paths.flow_csv.write_text("header\nrow\n", encoding="utf-8")
    write_done_marker(paths, {
        "status": "success",
        "source": pcap.as_posix(),
        "kept_packets": 17,
        "total_sessions": 3,
        "kept_sessions": 3,
    })
    monkeypatch.setattr(pipeline, "RawPcapReader", object())

    records = pipeline.run_pipeline(
        settings,
        converter=lambda *args: pytest.fail("完成文件不应再次转换"),
    )

    assert records[0]["status"] == "skipped_completed"
    assert records[0]["kept_packets"] == 17
    assert records[0]["total_sessions"] == 3
