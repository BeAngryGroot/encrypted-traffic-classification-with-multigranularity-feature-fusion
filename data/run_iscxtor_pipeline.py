#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ISCXTor PCAP 一键转换入口。

普通使用只修改下面三项配置，然后运行：

    python data/run_iscxtor_pipeline.py

第一次保持 RUN_MODE="smoke"，确认成功后改为 RUN_MODE="full"。
"""

from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
import shutil
import sys
from typing import Any, Callable

try:
    from data.pcap_to_csv import RawPcapReader, discover_pcaps, process_one
except ImportError:  # 支持直接在 data 目录中运行当前文件。
    from pcap_to_csv import RawPcapReader, discover_pcaps, process_one


# ============================================================================
# 用户配置区：正常使用只需要修改下面三项
# ============================================================================
RAW_DIR = Path(
    "/data3/wsb_workspace/study/data/"
    "ISCX-VPN-NonVPN/ISCX-Tor-NonTor-2017/Pcaps"
)
OUTPUT_DIR = Path("/data3/wsb_workspace/study/data/Dual_data")
RUN_MODE = "smoke"  # 第一次用 smoke；确认成功后改成 full。


# ============================================================================
# 高级配置区：当前论文实验保持默认值即可
# ============================================================================
FLOW_TIMEOUT = 60.0
MIN_PKTS = 1
MIN_BYTES = 0
WORKERS = 2
MIN_FREE_GIB = 60.0


@dataclass(frozen=True)
class PipelineSettings:
    """一次转换运行所需的完整配置。"""

    raw_dir: Path
    output_dir: Path
    run_mode: str
    flow_timeout: float = FLOW_TIMEOUT
    min_pkts: int = MIN_PKTS
    min_bytes: int = MIN_BYTES
    workers: int = WORKERS
    min_free_gib: float = MIN_FREE_GIB


@dataclass(frozen=True)
class ConversionPaths:
    """一个源 PCAP 对应的两个 CSV 和完成标记。"""

    packet_csv: Path
    flow_csv: Path
    done_marker: Path


def select_pcaps(pcaps: list[Path], mode: str) -> list[Path]:
    """smoke 自动选择最小文件，full 保留全部文件。"""

    normalized_mode = mode.strip().lower()
    if normalized_mode == "smoke":
        return [min(pcaps, key=lambda path: path.stat().st_size)] if pcaps else []
    if normalized_mode == "full":
        return list(pcaps)
    raise ValueError('RUN_MODE 只能是 "smoke" 或 "full"')


def output_paths(pcap: Path, raw_root: Path, csv_root: Path) -> ConversionPaths:
    """在输出侧保留 Tor/NonTor/应用类别目录结构。"""

    relative_parent = pcap.parent.relative_to(raw_root)
    output_directory = csv_root / relative_parent
    return ConversionPaths(
        packet_csv=output_directory / f"{pcap.stem}_packets.csv",
        flow_csv=output_directory / f"{pcap.stem}_flows.csv",
        done_marker=output_directory / f"{pcap.stem}.conversion_done.json",
    )


def is_completed(paths: ConversionPaths) -> bool:
    """只有两个 CSV 与成功标记同时存在时，才允许断点跳过。"""

    if not paths.packet_csv.is_file() or paths.packet_csv.stat().st_size == 0:
        return False
    if not paths.flow_csv.is_file() or paths.flow_csv.stat().st_size == 0:
        return False
    if not paths.done_marker.is_file():
        return False
    try:
        payload = json.loads(paths.done_marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("status") == "success"


def write_done_marker(paths: ConversionPaths, payload: dict[str, Any]) -> None:
    """原子写入完成标记，防止中断产生“假成功”。"""

    if not paths.packet_csv.is_file() or not paths.flow_csv.is_file():
        raise FileNotFoundError("两个 CSV 尚未全部生成，不能写完成标记")
    paths.done_marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = paths.done_marker.with_suffix(paths.done_marker.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(paths.done_marker)


def _nearest_existing(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def validate_settings(settings: PipelineSettings, *, free_bytes: int | None = None) -> None:
    """在读取大文件前集中检查配置，避免运行很久后才发现错误。"""

    if settings.run_mode.strip().lower() not in {"smoke", "full"}:
        raise ValueError('RUN_MODE 只能是 "smoke" 或 "full"')
    if not settings.raw_dir.is_dir():
        raise FileNotFoundError(f"原始 PCAP 目录不存在：{settings.raw_dir}")
    if settings.workers < 1:
        raise ValueError("WORKERS 必须大于或等于 1")
    if settings.flow_timeout <= 0:
        raise ValueError("FLOW_TIMEOUT 必须大于 0")
    if free_bytes is None:
        disk_root = _nearest_existing(settings.output_dir)
        free_bytes = shutil.disk_usage(disk_root).free
    free_gib = free_bytes / 1024**3
    if settings.run_mode.strip().lower() == "full" and free_gib < settings.min_free_gib:
        raise RuntimeError(
            f"可用磁盘空间只有 {free_gib:.2f} GiB，低于 full 模式要求的 "
            f"{settings.min_free_gib:.2f} GiB"
        )


def _csv_root(settings: PipelineSettings) -> Path:
    version = "smoke_parse_v1" if settings.run_mode.lower() == "smoke" else "full_session60_v1"
    return settings.output_dir / "csv" / version


def _setup_logger(settings: PipelineSettings) -> logging.Logger:
    log_directory = settings.output_dir / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("iscxtor_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    log_path = log_directory / f"pcap_conversion_{settings.run_mode.lower()}.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _result_record(pcap: Path, result: tuple[Any, ...], status: str) -> dict[str, Any]:
    return {
        "source_key": pcap.as_posix(),
        "status": status,
        "kept_packets": int(result[2]),
        "total_sessions": int(result[3]),
        "kept_sessions": int(result[4]),
        "filtered_sessions": int(result[5]),
        "tcp_packets": int(result[6]),
        "udp_packets": int(result[7]),
        "elapsed_sec": float(result[8]),
        "error": str(result[9]),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def _skipped_record(pcap: Path, paths: ConversionPaths) -> dict[str, Any]:
    """从完成标记恢复历史指标，使断点汇总不被零值覆盖。"""

    payload = json.loads(paths.done_marker.read_text(encoding="utf-8"))
    return {
        "source_key": pcap.as_posix(),
        "status": "skipped_completed",
        "kept_packets": int(payload.get("kept_packets", 0)),
        "total_sessions": int(payload.get("total_sessions", 0)),
        "kept_sessions": int(payload.get("kept_sessions", 0)),
        "filtered_sessions": int(payload.get("filtered_sessions", 0)),
        "tcp_packets": int(payload.get("tcp_packets", 0)),
        "udp_packets": int(payload.get("udp_packets", 0)),
        "elapsed_sec": float(payload.get("elapsed_sec", 0.0)),
        "error": "",
        "updated_at": str(payload.get("completed_at", datetime.now().isoformat(timespec="seconds"))),
    }


def _write_summary(csv_root: Path, records: list[dict[str, Any]]) -> Path:
    summary_path = csv_root / "conversion_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_key", "status", "kept_packets", "total_sessions",
        "kept_sessions", "filtered_sessions", "tcp_packets", "udp_packets",
        "elapsed_sec", "error", "updated_at",
    ]
    temporary = summary_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(summary_path)
    return summary_path


def _handle_result(
    pcap: Path,
    result: tuple[Any, ...],
    paths: ConversionPaths,
    logger: logging.Logger,
) -> dict[str, Any]:
    error = str(result[9])
    if error:
        logger.error("转换失败：%s | %s", pcap, error)
        return _result_record(pcap, result, "failed")

    payload = {
        "status": "success",
        "source": pcap.as_posix(),
        "kept_packets": int(result[2]),
        "total_sessions": int(result[3]),
        "kept_sessions": int(result[4]),
        "filtered_sessions": int(result[5]),
        "tcp_packets": int(result[6]),
        "udp_packets": int(result[7]),
        "elapsed_sec": float(result[8]),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_done_marker(paths, payload)
    logger.info(
        "转换完成：%s | 包=%d | 会话=%d/%d | %.2fs",
        pcap.name,
        int(result[2]),
        int(result[4]),
        int(result[3]),
        float(result[8]),
    )
    return _result_record(pcap, result, "success")


def run_pipeline(
    settings: PipelineSettings,
    *,
    converter: Callable[..., tuple[Any, ...]] = process_one,
) -> list[dict[str, Any]]:
    """执行 smoke 或 full 转换，并返回本次运行汇总。"""

    validate_settings(settings)
    if RawPcapReader is None:
        raise RuntimeError("Scapy 未安装或不可用，请先安装 requirements.txt 中的依赖")

    all_pcaps = discover_pcaps(settings.raw_dir)
    if not all_pcaps:
        raise FileNotFoundError(f"未找到 PCAP/PCAPNG：{settings.raw_dir}")
    selected = select_pcaps(all_pcaps, settings.run_mode)
    csv_root = _csv_root(settings)
    csv_root.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(settings)

    records: list[dict[str, Any]] = []
    pending: list[tuple[Path, ConversionPaths]] = []
    for pcap in selected:
        paths = output_paths(pcap, settings.raw_dir, csv_root)
        if is_completed(paths):
            logger.info("断点跳过：%s", pcap.relative_to(settings.raw_dir))
            records.append(_skipped_record(pcap, paths))
        else:
            pending.append((pcap, paths))

    logger.info(
        "模式=%s | 发现=%d | 本次处理=%d | 已完成跳过=%d | 输出=%s",
        settings.run_mode,
        len(selected),
        len(pending),
        len(selected) - len(pending),
        csv_root,
    )

    if settings.run_mode.lower() == "smoke":
        for index, (pcap, paths) in enumerate(pending, start=1):
            logger.info("[%d/%d] 开始：%s", index, len(pending), pcap)
            result = converter(
                pcap,
                settings.raw_dir,
                csv_root,
                settings.flow_timeout,
                settings.min_pkts,
                settings.min_bytes,
            )
            records.append(_handle_result(pcap, result, paths, logger))
    else:
        # 全量模式使用有限进程数，避免多个大 PCAP 同时占满内存。
        with ProcessPoolExecutor(max_workers=settings.workers) as pool:
            futures = {
                pool.submit(
                    converter,
                    pcap,
                    settings.raw_dir,
                    csv_root,
                    settings.flow_timeout,
                    settings.min_pkts,
                    settings.min_bytes,
                ): (pcap, paths)
                for pcap, paths in pending
            }
            total = len(futures)
            for finished, future in enumerate(as_completed(futures), start=1):
                pcap, paths = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # 防止单个子进程异常终止整个批次。
                    result = (pcap.name, "", 0, 0, 0, 0, 0, 0, 0.0, str(exc))
                logger.info("[%d/%d] 收到结果：%s", finished, total, pcap.name)
                records.append(_handle_result(pcap, result, paths, logger))

    summary_path = _write_summary(csv_root, records)
    success_count = sum(record["status"] == "success" for record in records)
    failed_count = sum(record["status"] == "failed" for record in records)
    logger.info("本次结束：成功=%d | 失败=%d | 汇总=%s", success_count, failed_count, summary_path)
    return records


def main() -> None:
    settings = PipelineSettings(
        raw_dir=RAW_DIR,
        output_dir=OUTPUT_DIR,
        run_mode=RUN_MODE,
    )
    print("=" * 72)
    print("ISCXTor PCAP 一键转换")
    print(f"运行模式：{settings.run_mode}")
    print(f"原始目录：{settings.raw_dir}")
    print(f"数据目录：{settings.output_dir}")
    print("=" * 72)
    try:
        records = run_pipeline(settings)
    except KeyboardInterrupt:
        print("\n用户中断。已经写入完成标记的文件下次会自动跳过。")
        raise SystemExit(130)
    except Exception as exc:
        print(f"\n运行失败：{exc}")
        raise SystemExit(1)

    if any(record["status"] == "failed" for record in records):
        print("\n存在转换失败文件，请查看日志后重新运行；成功文件会自动跳过。")
        raise SystemExit(1)
    if settings.run_mode.lower() == "smoke":
        print('\nsmoke 转换成功。下一步把文件顶部 RUN_MODE 改为 "full" 后再次运行。')
    else:
        print("\n全量 PCAP 转换完成，可以进入数据划分与特征构建。")


if __name__ == "__main__":
    main()
