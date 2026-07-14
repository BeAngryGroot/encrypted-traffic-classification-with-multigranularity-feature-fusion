#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将 PCAP/PCAPNG 转换为按会话组织的包级 CSV 与流级 CSV。"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import multiprocessing
from pathlib import Path
import struct
import time
from typing import Any

try:  # 只使用轻量帮助函数或运行单元测试时，不强制要求已安装 Scapy。
    from scapy.utils import RawPcapNgReader, RawPcapReader
except ImportError:  # pragma: no cover - 由真正读取 pcap 的路径给出明确错误
    RawPcapReader = None
    RawPcapNgReader = None

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def discover_pcaps(root: Path) -> list[Path]:
    """递归发现标准 pcap 文件，避免重叠通配符造成同一文件被重复处理。"""

    root = Path(root)
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".pcap", ".pcapng"}),
        key=lambda path: str(path).lower(),
    )


@dataclass
class _SessionState:
    index: int
    last_timestamp: float
    closed: bool = False


class FlowSessionizer:
    """依据空闲超时和 TCP FIN/RST 将复用的五元组拆成独立会话。"""

    def __init__(self, timeout_seconds: float = 60.0):
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.timeout_seconds = float(timeout_seconds)
        self._states: dict[str, _SessionState] = {}

    def assign(self, base_flow_id: str, timestamp: float, tcp_flags: int = 0) -> str:
        state = self._states.get(base_flow_id)
        if state is None:
            state = _SessionState(index=0, last_timestamp=float(timestamp))
            self._states[base_flow_id] = state
        elif state.closed or float(timestamp) - state.last_timestamp > self.timeout_seconds:
            state = _SessionState(index=state.index + 1, last_timestamp=float(timestamp))
            self._states[base_flow_id] = state
        else:
            state.last_timestamp = float(timestamp)

        session_id = f"{base_flow_id}_S{state.index}"
        # FIN/RST 包仍属于当前会话，下一包才进入新会话。
        if int(tcp_flags) & (0x01 | 0x04):
            state.closed = True
        return session_id


def detect_l2(packet: bytes) -> tuple[int, int, int]:
    size = len(packet)
    vlan_id = -1
    if size >= 14:
        ether_type = struct.unpack_from("!H", packet, 12)[0]
        if ether_type in (0x8100, 0x88A8) and size >= 18:
            vlan_id = struct.unpack_from("!H", packet, 14)[0] & 0x0FFF
            inner_type = struct.unpack_from("!H", packet, 16)[0]
            if inner_type in (0x0800, 0x86DD):
                return inner_type, 18, vlan_id
        if ether_type in (0x0800, 0x86DD):
            return ether_type, 14, vlan_id
    if size >= 16:
        guessed = struct.unpack_from("!H", packet, 14)[0]
        if guessed in (0x0800, 0x86DD):
            return guessed, 16, vlan_id
    if size and packet[0] >> 4 in (4, 6):
        return (0x0800 if packet[0] >> 4 == 4 else 0x86DD), 0, vlan_id
    return -1, 0, vlan_id


def ipv6_walk(packet: bytes, offset: int) -> tuple[int, int]:
    next_header = packet[offset + 6]
    current = offset + 40
    while next_header in (0, 43, 44, 60, 51) and current < len(packet):
        if next_header == 44 and len(packet) >= current + 8:
            next_header, current = packet[current], current + 8
        elif next_header == 51 and len(packet) >= current + 2:
            length = (packet[current + 1] + 2) * 4
            next_header, current = packet[current], current + length
        elif len(packet) >= current + 2:
            length = (packet[current + 1] + 1) * 8
            next_header, current = packet[current], current + length
        else:
            break
    return next_header, current


def transport_payload_length(
    raw: bytes,
    l3_offset: int,
    l4_offset: int,
    protocol: int,
    packet_end: int,
) -> int:
    """计算 TCP/UDP 应用负载长度，不把传输层首部错误计入载荷。"""

    captured_end = min(len(raw), max(l4_offset, int(packet_end)))
    if protocol == 6:
        if captured_end < l4_offset + 20:
            return 0
        header_length = (raw[l4_offset + 12] >> 4) * 4
        if header_length < 20:
            return 0
    elif protocol == 17:
        header_length = 8
        if captured_end < l4_offset + header_length:
            return 0
    else:
        header_length = 0
    return max(0, captured_end - l4_offset - header_length)


def make_fid(packet: dict[str, Any]) -> tuple[str, str]:
    src_ip, dst_ip = packet.get("sip", ""), packet.get("dip", "")
    src_port, dst_port = packet.get("sport", -1), packet.get("dport", -1)
    protocol = packet.get("proto", 0)
    if src_port != -1 and dst_port != -1 and src_ip and dst_ip:
        endpoint_a, endpoint_b = sorted([src_ip, dst_ip])
        port_a, port_b = (src_port, dst_port) if src_ip == endpoint_a else (dst_port, src_port)
        return f"5T_{endpoint_a}_{endpoint_b}_{port_a}_{port_b}_{protocol}", "5T"
    if src_ip and dst_ip:
        endpoint_a, endpoint_b = sorted([src_ip, dst_ip])
        return f"3T_{endpoint_a}_{endpoint_b}_{protocol}", "3T"
    return f"P_{protocol}", "P"


def reader_for(path: Path):
    if RawPcapReader is None:
        raise RuntimeError("读取 pcap 需要安装 scapy：pip install scapy")
    with Path(path).open("rb") as stream:
        magic = stream.read(4)
    if magic == b"\x0a\x0d\x0d\x0a":
        if RawPcapNgReader is None:
            raise RuntimeError("当前 Scapy 不支持 PCAPNG，请升级 scapy")
        return RawPcapNgReader(str(path))
    return RawPcapReader(str(path))


def _timestamp(meta: Any) -> float:
    seconds = getattr(meta, "sec", getattr(meta, "ts_sec", 0))
    micros = getattr(meta, "usec", getattr(meta, "ts_usec", 0))
    return float(seconds) + float(micros) / 1e6


def parse_pcap(path: Path, *, flow_timeout: float = 60.0):
    flows: dict[str, dict[str, Any]] = defaultdict(lambda: {"idx": [], "key_type": None})
    packets: list[dict[str, Any]] = []
    protocol_counter: dict[int, int] = defaultdict(int)
    sessionizer = FlowSessionizer(flow_timeout)

    for frame_index, (raw, meta) in enumerate(reader_for(path), start=1):
        if not raw:
            continue
        timestamp = _timestamp(meta)
        ether_type, l3, vlan_id = detect_l2(raw)
        if ether_type == 0x0800 and len(raw) >= l3 + 20:
            ihl = (raw[l3] & 0x0F) * 4
            if ihl < 20 or len(raw) < l3 + ihl:
                continue
            protocol = raw[l3 + 9]
            src_ip = ".".join(map(str, raw[l3 + 12:l3 + 16]))
            dst_ip = ".".join(map(str, raw[l3 + 16:l3 + 20]))
            ttl = raw[l3 + 8]
            l4 = l3 + ihl
            total_length = struct.unpack_from("!H", raw, l3 + 2)[0]
            packet_end = l3 + total_length
            ip_version = 4
        elif ether_type == 0x86DD and len(raw) >= l3 + 40:
            protocol, l4 = ipv6_walk(raw, l3)
            src_ip = ":".join(f"{raw[l3+i]:02x}{raw[l3+i+1]:02x}" for i in range(8, 24, 2))
            dst_ip = ":".join(f"{raw[l3+i]:02x}{raw[l3+i+1]:02x}" for i in range(24, 40, 2))
            ttl = raw[l3 + 7]
            payload_length = struct.unpack_from("!H", raw, l3 + 4)[0]
            packet_end = l3 + 40 + payload_length
            ip_version = 6
        else:
            continue

        src_port = dst_port = -1
        tcp_flags = tcp_window = 0
        if protocol in (6, 17) and len(raw) >= l4 + 4:
            src_port, dst_port = struct.unpack_from("!HH", raw, l4)
        if protocol == 6 and len(raw) >= l4 + 20:
            tcp_flags = raw[l4 + 13]
            tcp_window = struct.unpack_from("!H", raw, l4 + 14)[0]

        packet = {
            "frame_index": frame_index,
            "ts": timestamp,
            "len": len(raw),
            "payload_length": transport_payload_length(raw, l3, l4, protocol, packet_end),
            "sip": src_ip,
            "dip": dst_ip,
            "sport": src_port,
            "dport": dst_port,
            "proto": protocol,
            "ipv": ip_version,
            "ip_ttl": ttl,
            "tcp_flags": tcp_flags,
            "tcp_window": tcp_window,
            "vlan_id": vlan_id,
            "ether_type": ether_type,
        }
        base_id, key_type = make_fid(packet)
        flow_id = sessionizer.assign(base_id, timestamp, tcp_flags if protocol == 6 else 0)
        packets.append(packet)
        flows[flow_id]["idx"].append(len(packets) - 1)
        flows[flow_id]["key_type"] = key_type
        protocol_counter[protocol] += 1
    return flows, packets, protocol_counter


def write_csv(flows, packets, output_dir: Path, stem: str, *, min_pkts: int = 1, min_bytes: int = 0):
    output_dir.mkdir(parents=True, exist_ok=True)
    kept_ids = []
    for flow_id, meta in flows.items():
        indices = meta["idx"]
        byte_count = sum(packets[index]["len"] for index in indices)
        if len(indices) >= min_pkts and byte_count >= min_bytes:
            kept_ids.append(flow_id)

    packet_file = output_dir / f"{stem}_packets.csv"
    with packet_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["flow_id", "frame_index", "timestamp", "packet_length", "payload_length", "src_ip", "dst_ip", "src_port", "dst_port", "protocol", "ip_version", "ip_ttl", "tcp_flags", "tcp_window", "vlan_id", "ether_type"])
        for flow_id in kept_ids:
            for index in flows[flow_id]["idx"]:
                packet = packets[index]
                writer.writerow([flow_id, packet["frame_index"], packet["ts"], packet["len"], packet["payload_length"], packet["sip"], packet["dip"], packet["sport"], packet["dport"], packet["proto"], packet["ipv"], packet["ip_ttl"], packet["tcp_flags"], packet["tcp_window"], packet["vlan_id"], packet["ether_type"]])

    flow_file = output_dir / f"{stem}_flows.csv"
    with flow_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["flow_id", "key_type", "protocol", "packet_count", "byte_count", "start_time", "end_time", "duration"])
        for flow_id in kept_ids:
            indices = flows[flow_id]["idx"]
            timestamps = [packets[index]["ts"] for index in indices]
            byte_count = sum(packets[index]["len"] for index in indices)
            writer.writerow([flow_id, flows[flow_id]["key_type"], packets[indices[0]]["proto"], len(indices), byte_count, min(timestamps), max(timestamps), max(timestamps) - min(timestamps)])
    return len(flows), len(kept_ids), sum(len(flows[flow_id]["idx"]) for flow_id in kept_ids)


def process_one(pcap: Path, root: Path, output_root: Path, flow_timeout: float, min_pkts: int, min_bytes: int):
    relative_dir = pcap.parent.relative_to(root)
    started = time.time()
    try:
        flows, packets, protocols = parse_pcap(pcap, flow_timeout=flow_timeout)
        total, kept, packet_count = write_csv(flows, packets, output_root / relative_dir, pcap.stem, min_pkts=min_pkts, min_bytes=min_bytes)
        return (pcap.name, str(relative_dir), packet_count, total, kept, total - kept, protocols.get(6, 0), protocols.get(17, 0), round(time.time() - started, 2), "")
    except Exception as exc:  # 进程池必须返回可序列化错误，避免单文件中断整个批次。
        return (pcap.name, str(relative_dir), 0, 0, 0, 0, 0, 0, round(time.time() - started, 2), str(exc))


def main() -> None:
    parser = argparse.ArgumentParser(description="将 PCAP/PCAPNG 转为会话化 CSV")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pcap")
    source.add_argument("--input_dir")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--flow_timeout", type=float, default=60.0)
    parser.add_argument("--min_pkts", type=int, default=1)
    parser.add_argument("--min_bytes", type=int, default=0)
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 1))
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    if args.pcap:
        pcap = Path(args.pcap)
        flows, packets, _ = parse_pcap(pcap, flow_timeout=args.flow_timeout)
        print(write_csv(flows, packets, output_root, pcap.stem, min_pkts=args.min_pkts, min_bytes=args.min_bytes))
        return

    root = Path(args.input_dir)
    pcaps = discover_pcaps(root)
    if not pcaps:
        raise FileNotFoundError(f"未在 {root} 下发现 .pcap/.pcapng 文件")
    summary_path = output_root / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["file_name", "relative_path", "kept_packets", "total_sessions", "kept_sessions", "filtered_sessions", "tcp_packets", "udp_packets", "elapsed_sec", "error"])
        # Windows 使用 spawn；全部阈值显式传入，避免子进程读到不同的模块全局值。
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_one, path, root, output_root, args.flow_timeout, args.min_pkts, args.min_bytes) for path in pcaps]
            iterator = as_completed(futures)
            if tqdm is not None:
                iterator = tqdm(iterator, total=len(futures), desc="PCAP 转换", unit="file")
            for future in iterator:
                writer.writerow(future.result())
                stream.flush()
    print(f"转换完成：{summary_path.resolve()}")


if __name__ == "__main__":
    main()
