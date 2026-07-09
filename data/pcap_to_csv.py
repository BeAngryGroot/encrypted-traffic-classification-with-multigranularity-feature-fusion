# -*- coding: utf-8 -*-
"""
pcap_to_csv.py
--------------
多进程版 PCAP → CSV 转换器
特征提取友好型：保留最大数量的字段，不过滤短流/加密流。

输出：
  - *_packets.csv : 每个包的详细字段
  - *_flows.csv   : 聚合流的统计信息
  - summary.csv   : 各文件统计汇总
"""

import argparse, csv, struct, hashlib, time, multiprocessing
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from scapy.utils import RawPcapReader
try:
    from scapy.utils import RawPcapNgReader
except ImportError:
    RawPcapNgReader = None

# ------------------------ 全局参数 ------------------------
MIN_PKTS = 1
MIN_BYTES = 0

# ------------------------ L2 识别 ------------------------
def detect_l2(pkt: bytes):
    n = len(pkt)
    vlan_id = -1
    if n >= 14:
        et = struct.unpack_from("!H", pkt, 12)[0]
        if et in (0x8100, 0x88A8) and n >= 18:
            vlan_raw = struct.unpack_from("!H", pkt, 14)[0]
            vlan_id = vlan_raw & 0x0FFF
            et2 = struct.unpack_from("!H", pkt, 16)[0]
            if et2 in (0x0800, 0x86DD):
                return et2, 18, vlan_id
            if (pkt[18] >> 4) in (4, 6):
                return (0x0800 if (pkt[18] >> 4) == 4 else 0x86DD), 18, vlan_id
        if et in (0x0800, 0x86DD):
            return et, 14, vlan_id
        if n > 14 and (pkt[14] >> 4) in (4, 6):
            et2 = 0x0800 if (pkt[14] >> 4) == 4 else 0x86DD
            return et2, 14, vlan_id
    if n >= 16:
        proto_guess = struct.unpack_from("!H", pkt, 14)[0]
        if proto_guess in (0x0800, 0x86DD):
            return proto_guess, 16, vlan_id
    if n >= 1 and (pkt[0] >> 4) in (4, 6):
        et = 0x0800 if (pkt[0] >> 4) == 4 else 0x86DD
        return et, 0, vlan_id
    return -1, 0, vlan_id

# ------------------------ IPv6 扩展头解析 ------------------------
def ipv6_walk(pkt: bytes, off: int):
    nh = pkt[off + 6]
    cur = off + 40
    n = len(pkt)
    while nh in (0, 43, 44, 60, 51):
        if nh == 44 and n >= cur + 8:
            nh = pkt[cur]; cur += 8
        elif nh == 51 and n >= cur + 2:
            l = (pkt[cur + 1] + 2) * 4; nh = pkt[cur]; cur += l
        elif n >= cur + 2:
            l = (pkt[cur + 1] + 1) * 8; nh = pkt[cur]; cur += l
        else:
            break
        if cur >= n:
            break
    return nh, cur

# ------------------------ Flow ID ------------------------
def make_fid(pkt):
    sip = pkt.get("sip") or pkt.get("src_ip", "")
    dip = pkt.get("dip") or pkt.get("dst_ip", "")
    sp  = pkt.get("sport", pkt.get("src_port", -1))
    dp  = pkt.get("dport", pkt.get("dst_port", -1))
    proto = pkt.get("proto", 0)
    if sp != -1 and dp != -1 and sip and dip:
        a, b = sorted([sip, dip])
        asp, bdp = (sp, dp) if sip == a else (dp, sp)
        return f"5T_{a}_{b}_{asp}_{bdp}_{proto}", "5T"
    elif sip and dip:
        a, b = sorted([sip, dip])
        return f"3T_{a}_{b}_{proto}", "3T"
    else:
        return f"P_{proto}", "P"

# ------------------------ Reader ------------------------
def reader_for(p: Path):
    with p.open("rb") as f:
        magic = f.read(4)
    if magic in (b"\xd4\xc3\xb2\xa1", b"\xa1\xb2\xc3\xd4"):
        return RawPcapReader(str(p))
    if magic == b"\r\n\r\n" and RawPcapNgReader:
        return RawPcapNgReader(str(p))
    return RawPcapReader(str(p))

# ------------------------ 主解析 ------------------------
def parse_pcap(p: Path):
    flows = defaultdict(lambda: {"idx": [], "key_type": None})
    pkts = []
    proto_counter = defaultdict(int)
    frame_index = 0
    for raw, meta in reader_for(p):
        if not raw:
            continue
        frame_index += 1
        ts = getattr(meta, "sec", getattr(meta, "ts_sec", 0)) + getattr(meta, "usec", getattr(meta, "ts_usec", 0)) / 1e6
        et, l3, vlan_id = detect_l2(raw)
        if et not in (0x0800, 0x86DD):  # 非 IPv4/6
            continue
        if et == 0x0800 and len(raw) >= l3 + 20:
            ihl = (raw[l3] & 0x0F) * 4
            proto = raw[l3 + 9]
            sip = ".".join(map(str, raw[l3 + 12:l3 + 16]))
            dip = ".".join(map(str, raw[l3 + 16:l3 + 20]))
            ttl = raw[l3 + 8]
            l4 = l3 + ihl
        elif et == 0x86DD and len(raw) >= l3 + 40:
            proto, l4 = ipv6_walk(raw, l3)
            sip = ":".join(f"{raw[l3+i]:02x}{raw[l3+i+1]:02x}" for i in range(8,24,2))
            dip = ":".join(f"{raw[l3+i]:02x}{raw[l3+i+1]:02x}" for i in range(24,40,2))
            ttl = 0
        else:
            continue
        sp = dp = -1
        tcp_flags = tcp_window = 0
        if proto in (6, 17) and len(raw) >= l4 + 4:
            sp, dp = struct.unpack_from("!HH", raw, l4)
            if proto == 6 and len(raw) >= l4 + 20:
                tcp_flags = raw[l4 + 13]
                tcp_window = struct.unpack_from("!H", raw, l4 + 14)[0]
        pkt = {
            "frame_index": frame_index, "ts": ts,
            "len": len(raw), "payload_length": max(0, len(raw) - l4),
            "sip": sip, "dip": dip, "sport": sp, "dport": dp,
            "proto": proto, "ipv": 4 if et==0x0800 else 6,
            "ip_ttl": ttl, "tcp_flags": tcp_flags, "tcp_window": tcp_window,
            "vlan_id": vlan_id, "ether_type": et,
        }
        pkts.append(pkt)
        fid, kt = make_fid(pkt)
        flows[fid]["idx"].append(len(pkts)-1)
        flows[fid]["key_type"] = kt
        proto_counter[proto]+=1
    return flows, pkts, proto_counter

# ------------------------ 写 CSV ------------------------
def write_csv(flows, pkts, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    pfile = out_dir / f"{stem}_packets.csv"
    ffile = out_dir / f"{stem}_flows.csv"
    with pfile.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["flow_id","frame_index","timestamp","packet_length","payload_length",
                    "src_ip","dst_ip","src_port","dst_port","protocol","ip_version",
                    "ip_ttl","tcp_flags","tcp_window","vlan_id","ether_type"])
        for fid, meta in flows.items():
            for i in meta["idx"]:
                p = pkts[i]
                w.writerow([fid,p["frame_index"],p["ts"],p["len"],p["payload_length"],
                            p["sip"],p["dip"],p["sport"],p["dport"],p["proto"],
                            p["ipv"],p["ip_ttl"],p["tcp_flags"],p["tcp_window"],
                            p["vlan_id"],p["ether_type"]])
    total_flows=kept_flows=0
    with ffile.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["flow_id","key_type","protocol","packet_count","byte_count",
                    "start_time","end_time","duration"])
        for fid, meta in flows.items():
            total_flows += 1
            idxs = meta["idx"]
            if len(idxs)<MIN_PKTS: continue
            bsum = sum(pkts[i]["len"] for i in idxs)
            if bsum<MIN_BYTES: continue
            kept_flows += 1
            st=min(pkts[i]["ts"] for i in idxs)
            et=max(pkts[i]["ts"] for i in idxs)
            proto=pkts[idxs[0]]["proto"]
            w.writerow([fid,meta["key_type"],proto,len(idxs),bsum,st,et,et-st])
    return total_flows, kept_flows, len(pkts)

# ------------------------ 并行处理单文件 ------------------------
def process_one(pcap, root, out_root):
    rel = pcap.parent.relative_to(root)
    out_dir = out_root / rel
    start_t = time.time()
    try:
        flows, pkts, proto_count = parse_pcap(pcap)
        total_flows, kept_flows, total_pkts = write_csv(flows, pkts, out_dir, pcap.stem)
        filtered = total_flows - kept_flows
        avg = total_pkts / kept_flows if kept_flows else 0
        tcp_pkts = proto_count.get(6, 0)
        udp_pkts = proto_count.get(17, 0)
        icmp_pkts = proto_count.get(1, 0)
        other_pkts = sum(v for k,v in proto_count.items() if k not in (1,6,17))
        elapsed = time.time()-start_t
        return (pcap.name, str(rel), total_pkts, total_flows, kept_flows,
                filtered, round(avg,2), tcp_pkts, udp_pkts, icmp_pkts, other_pkts, f"{elapsed:.2f}")
    except Exception as e:
        return (pcap.name, str(rel), "ERROR", "-", "-", "-", "-", "-", "-", "-", "-", str(e))

# ------------------------ 主函数 ------------------------
def main():
    ap = argparse.ArgumentParser(description="Convert PCAP(s) to CSV with multiprocessing")
    ap.add_argument("--pcap", help="单个 pcap 文件路径")
    ap.add_argument("--input_dir", help="输入目录（批量递归）")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--min_pkts", type=int, default=1)
    ap.add_argument("--min_bytes", type=int, default=0)
    ap.add_argument("--workers", type=int, default=multiprocessing.cpu_count())
    args = ap.parse_args()
    global MIN_PKTS, MIN_BYTES
    MIN_PKTS, MIN_BYTES = args.min_pkts, args.min_bytes
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.pcap:
        start = time.time()
        pcap = Path(args.pcap)
        print(f"[INFO] 单文件模式: {pcap}")
        flows, pkts, proto_count = parse_pcap(pcap)
        total_flows, kept_flows, total_pkts = write_csv(flows, pkts, out_root, pcap.stem)
        print(f"✅ 完成 {pcap.name}: {total_pkts} 包, {kept_flows}/{total_flows} 流, 用时 {time.time()-start:.2f}s")
        return

    if args.input_dir:
        root = Path(args.input_dir)
        exts = ("*.pcap","*.pcapng","*.pcap*")
        all_files=[]
        for e in exts: all_files.extend(sorted(root.rglob(e)))
        if not all_files:
            print(f"[WARN] 未找到任何 pcap 文件: {root}")
            return
        print(f"[INFO] 启动多进程模式，发现 {len(all_files)} 个文件，使用 {args.workers} 核心。\n")

        summary = out_root / "summary.csv"
        with summary.open("w", newline="") as sf:
            w = csv.writer(sf)
            w.writerow(["file_name","relative_path","total_pkts","total_flows","kept_flows",
                        "filtered_flows","avg_pkts_per_flow","tcp_pkts","udp_pkts",
                        "icmp_pkts","other_pkts","elapsed_sec"])

        start_all = time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_one, p, root, out_root): p for p in all_files}
            with tqdm(total=len(futures), ncols=110, desc="🚀 转换进度", unit="file") as bar:
                with summary.open("a", newline="") as sf:
                    w = csv.writer(sf)
                    for fut in as_completed(futures):
                        res = fut.result()
                        w.writerow(res)
                        if "ERROR" in res:
                            tqdm.write(f"[ERROR] {res[0]}: {res[-1]}")
                        else:
                            fname, _, pkts, flows, kept, filt, avg, *_rest, t = res
                            tqdm.write(f"[OK] {fname:<40} 包:{pkts:<8} 流:{kept}/{flows:<6} 耗时:{t}s")
                        bar.update(1)
        print(f"\n✅ 全部完成，总耗时 {time.time()-start_all:.2f}s")
        print(f"📊 汇总表: {summary.resolve()}")

if __name__ == "__main__":
    main()
