
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# PCAP to CSV (Preserve+Debug)
# - IPv4/IPv6 + TCP/UDP + ICMP(v4/v6) + ESP + AH + GRE
# - 5-tuple for TCP/UDP; 3-tuple for other IP protocols; optional L2 pseudo-flows
# - Very loose filtering (min_pkts=1, min_bytes=1)
# - Stats and skip reasons saved to JSON per PCAP and a global summary
# - Labeling from file path: traffic_label (VPN/NONVPN/UNKNOWN), app_label (AUDIO/CHAT/FILE/VIDEO/VOIP/... if present)

import argparse, csv, json, struct, sys, gc, os
from collections import defaultdict, Counter
from pathlib import Path
from scapy.utils import RawPcapReader

MIN_PKTS = 1
MIN_BYTES = 1

def parse_vlan_chain_and_ethertype(pkt: bytes):
    vlan_ids = []
    eth_type = struct.unpack_from("!H", pkt, 12)[0]
    idx = 14
    while eth_type in (0x8100, 0x88A8):
        if len(pkt) < idx + 4:
            return eth_type, vlan_ids, len(pkt)
        tci = struct.unpack_from("!H", pkt, idx)[0]
        vlan_ids.append(tci & 0x0FFF)
        eth_type = struct.unpack_from("!H", pkt, idx + 2)[0]
        idx += 4
    return eth_type, vlan_ids, idx

def norm_pair(a, b):
    return (a, b) if a <= b else (b, a)

def key5(src_ip, dst_ip, sp, dp, proto):
    a, b = norm_pair(src_ip, dst_ip)
    if a == src_ip:
        asp, bdp = sp, dp
    else:
        asp, bdp = dp, sp
    return (a, b, asp, bdp, proto)

def key3(src_ip, dst_ip, proto):
    a, b = norm_pair(src_ip, dst_ip)
    return (a, b, proto)

def mac_addr(pkt, off):
    return ":".join(f"{b:02x}" for b in pkt[off:off+6])

def infer_labels_from_path(pcap_path: Path):
    p = str(pcap_path)
    low = p.lower()
    traffic = "UNKNOWN"
    if "nonvpn" in low:
        traffic = "NONVPN"
    if "vpn" in low and "nonvpn" not in low:
        traffic = "VPN"
    app = "UNKNOWN"
    for name in p.split(os.sep):
        upper = name.upper()
        for cand in ["AUDIO","CHAT","FILE","VIDEO","VOIP","BROWSING","MAIL","P2P","STREAM","SOCIAL","GAMING"]:
            if cand in upper:
                app = cand
    return traffic, app

def parse_pcap(pcap_path: Path, keep_l2=False):
    flows = defaultdict(lambda: {
        "key_type": "",
        "packet_count": 0,
        "byte_count": 0,
        "start_time": None,
        "end_time": None,
        "packet_indices": []
    })
    packets = []
    stats = {
        "total_packets": 0,
        "valid_packets": 0,
        "error_packets": 0,
        "proto_counter": Counter(),
        "ethertype_counter": Counter(),
        "ip_version_counter": Counter(),
        "flows_by_keytype": Counter(),
        "l2_only_packets": 0,
        "reasons": Counter(),
    }

    try:
        for raw, meta in RawPcapReader(str(pcap_path)):
            stats["total_packets"] += 1
            if len(raw) < 14:
                stats["error_packets"] += 1
                stats["reasons"]["truncated_eth"] += 1
                continue

            eth_type, vlan_ids, l3_off = parse_vlan_chain_and_ethertype(raw)
            stats["ethertype_counter"][hex(eth_type)] += 1
            ts = meta.sec + meta.usec / 1e6

            base = {
                "timestamp": ts,
                "ethertype": eth_type,
                "vlan_ids": "|".join(map(str, vlan_ids)) if vlan_ids else "",
                "packet_length": len(raw),
                "src_mac": mac_addr(raw, 6),
                "dst_mac": mac_addr(raw, 0),
            }

            # IPv4
            if eth_type == 0x0800 and len(raw) >= l3_off + 20:
                ver_ihl = raw[l3_off]
                ihl = (ver_ihl & 0x0F) * 4
                if len(raw) < l3_off + ihl:
                    stats["error_packets"] += 1
                    stats["reasons"]["truncated_ipv4"] += 1
                    continue
                ip_proto = raw[l3_off + 9]
                src_ip = ".".join(map(str, raw[l3_off+12:l3_off+16]))
                dst_ip = ".".join(map(str, raw[l3_off+16:l3_off+20]))
                ip_ttl = raw[l3_off + 8]
                l4_off = l3_off + ihl
                stats["ip_version_counter"][4] += 1
                stats["proto_counter"][ip_proto] += 1

                pkt_row = dict(base, **{
                    "ip_version": 4, "src_ip": src_ip, "dst_ip": dst_ip, "protocol": ip_proto,
                    "ip_ttl": ip_ttl, "src_port": -1, "dst_port": -1,
                    "tcp_seq": 0, "tcp_ack": 0, "tcp_flags": 0, "tcp_window": 0,
                    "icmp_type": -1, "icmp_code": -1, "esp_spi": 0, "ah_spi": 0, "gre_protocol": 0,
                    "header_length": 0, "payload_length": 0, "direction": 0
                })

                def upd(fkey, ktype, direction):
                    if fkey not in flows:
                        flows[fkey]["key_type"] = ktype
                    f = flows[fkey]
                    f["packet_count"] += 1
                    f["byte_count"] += len(raw)
                    f["start_time"] = ts if f["start_time"] is None else min(f["start_time"], ts)
                    f["end_time"]   = ts if f["end_time"]   is None else max(f["end_time"], ts)
                    f["packet_indices"].append(len(packets))
                    p = pkt_row.copy()
                    p["direction"] = direction
                    packets.append(p)
                    stats["valid_packets"] += 1

                if ip_proto in (6,17) and len(raw) >= l4_off + 4:
                    sp, dp = struct.unpack_from("!HH", raw, l4_off)
                    pkt_row["src_port"] = sp; pkt_row["dst_port"] = dp
                    if ip_proto == 6 and len(raw) >= l4_off + 20:
                        data_off = (raw[l4_off + 12] >> 4) & 0x0F
                        pkt_row["tcp_seq"] = struct.unpack_from("!I", raw, l4_off + 4)[0]
                        pkt_row["tcp_ack"] = struct.unpack_from("!I", raw, l4_off + 8)[0]
                        pkt_row["tcp_flags"] = raw[l4_off + 13]
                        pkt_row["tcp_window"] = struct.unpack_from("!H", raw, l4_off + 14)[0]
                        hdr_len = l4_off + data_off * 4
                    else:
                        hdr_len = l4_off + 8
                    pkt_row["header_length"] = hdr_len
                    pkt_row["payload_length"] = max(0, len(raw) - hdr_len)
                    fkey = key5(src_ip, dst_ip, sp, dp, ip_proto)
                    a_ip, b_ip, a_sp, b_dp, _ = fkey
                    direction = 0 if (src_ip == a_ip and sp == a_sp) else 1
                    upd(fkey, "5T", direction)

                elif ip_proto == 1 and len(raw) >= l4_off + 4:
                    pkt_row["icmp_type"] = raw[l4_off]
                    pkt_row["icmp_code"] = raw[l4_off+1]
                    hdr_len = l4_off + 4
                    pkt_row["header_length"] = hdr_len
                    pkt_row["payload_length"] = max(0, len(raw) - hdr_len)
                    fkey = key3(src_ip, dst_ip, ip_proto); a_ip, b_ip, _ = fkey
                    direction = 0 if src_ip == a_ip else 1
                    upd(fkey, "3T", direction)

                elif ip_proto == 50 and len(raw) >= l4_off + 8:
                    pkt_row["esp_spi"] = struct.unpack_from("!I", raw, l4_off)[0]
                    hdr_len = l4_off + 8
                    pkt_row["header_length"] = hdr_len
                    pkt_row["payload_length"] = max(0, len(raw) - hdr_len)
                    fkey = key3(src_ip, dst_ip, ip_proto); a_ip, b_ip, _ = fkey
                    direction = 0 if src_ip == a_ip else 1
                    upd(fkey, "3T", direction)

                elif ip_proto == 51 and len(raw) >= l4_off + 8:
                    pkt_row["ah_spi"] = struct.unpack_from("!I", raw, l4_off + 4)[0]
                    hdr_len = l4_off + 8
                    pkt_row["header_length"] = hdr_len
                    pkt_row["payload_length"] = max(0, len(raw) - hdr_len)
                    fkey = key3(src_ip, dst_ip, ip_proto); a_ip, b_ip, _ = fkey
                    direction = 0 if src_ip == a_ip else 1
                    upd(fkey, "3T", direction)

                elif ip_proto == 47 and len(raw) >= l4_off + 4:
                    pkt_row["gre_protocol"] = struct.unpack_from("!H", raw, l4_off + 2)[0]
                    hdr_len = l4_off + 4
                    pkt_row["header_length"] = hdr_len
                    pkt_row["payload_length"] = max(0, len(raw) - hdr_len)
                    fkey = key3(src_ip, dst_ip, ip_proto); a_ip, b_ip, _ = fkey
                    direction = 0 if src_ip == a_ip else 1
                    upd(fkey, "3T", direction)

                else:
                    hdr_len = l4_off
                    pkt_row["header_length"] = hdr_len
                    pkt_row["payload_length"] = max(0, len(raw) - hdr_len)
                    fkey = key3(src_ip, dst_ip, ip_proto); a_ip, b_ip, _ = fkey
                    direction = 0 if src_ip == a_ip else 1
                    upd(fkey, "3T", direction)

            # IPv6
            elif eth_type == 0x86DD and len(raw) >= l3_off + 40:
                next_header = raw[l3_off + 6]
                src_ip = ":".join(f"{raw[l3_off+i]:02x}{raw[l3_off+i+1]:02x}" for i in range(8,24,2))
                dst_ip = ":".join(f"{raw[l3_off+i]:02x}{raw[l3_off+i+1]:02x}" for i in range(24,40,2))
                l4_off = l3_off + 40
                stats["ip_version_counter"][6] += 1
                stats["proto_counter"][next_header] += 1

                pkt_row = dict(base, **{
                    "ip_version": 6, "src_ip": src_ip, "dst_ip": dst_ip, "protocol": next_header,
                    "src_port": -1, "dst_port": -1,
                    "tcp_seq": 0, "tcp_ack": 0, "tcp_flags": 0, "tcp_window": 0,
                    "icmp_type": -1, "icmp_code": -1, "esp_spi": 0, "ah_spi": 0, "gre_protocol": 0,
                    "header_length": 0, "payload_length": 0, "direction": 0
                })

                def upd6(fkey, ktype, direction):
                    if fkey not in flows:
                        flows[fkey]["key_type"] = ktype
                    f = flows[fkey]
                    f["packet_count"] += 1
                    f["byte_count"] += len(raw)
                    ts = base["timestamp"]
                    f["start_time"] = ts if f["start_time"] is None else min(f["start_time"], ts)
                    f["end_time"]   = ts if f["end_time"]   is None else max(f["end_time"], ts)
                    f["packet_indices"].append(len(packets))
                    p = pkt_row.copy()
                    p["direction"] = direction
                    packets.append(p)
                    stats["valid_packets"] += 1

                if next_header in (6,17) and len(raw) >= l4_off + 4:
                    sp, dp = struct.unpack_from("!HH", raw, l4_off)
                    pkt_row["src_port"] = sp; pkt_row["dst_port"] = dp
                    if next_header == 6 and len(raw) >= l4_off + 20:
                        data_off = (raw[l4_off + 12] >> 4) & 0x0F
                        pkt_row["tcp_seq"] = struct.unpack_from("!I", raw, l4_off + 4)[0]
                        pkt_row["tcp_ack"] = struct.unpack_from("!I", raw, l4_off + 8)[0]
                        pkt_row["tcp_flags"] = raw[l4_off + 13]
                        pkt_row["tcp_window"] = struct.unpack_from("!H", raw, l4_off + 14)[0]
                        hdr_len = l4_off + data_off * 4
                    else:
                        hdr_len = l4_off + 8
                    pkt_row["header_length"] = hdr_len
                    pkt_row["payload_length"] = max(0, len(raw) - hdr_len)
                    fkey = key5(src_ip, dst_ip, sp, dp, next_header)
                    a_ip, b_ip, a_sp, b_dp, _ = fkey
                    direction = 0 if (src_ip == a_ip and sp == a_sp) else 1
                    upd6(fkey, "5T", direction)

                elif next_header in (58,) and len(raw) >= l4_off + 4:
                    pkt_row["icmp_type"] = raw[l4_off]; pkt_row["icmp_code"] = raw[l4_off+1]
                    hdr_len = l4_off + 4
                    pkt_row["header_length"] = hdr_len
                    pkt_row["payload_length"] = max(0, len(raw) - hdr_len)
                    fkey = key3(src_ip, dst_ip, next_header); a_ip, b_ip, _ = fkey
                    direction = 0 if src_ip == a_ip else 1
                    upd6(fkey, "3T", direction)
                else:
                    hdr_len = l4_off
                    pkt_row["header_length"] = hdr_len
                    pkt_row["payload_length"] = max(0, len(raw) - hdr_len)
                    fkey = key3(src_ip, dst_ip, next_header); a_ip, b_ip, _ = fkey
                    direction = 0 if src_ip == a_ip else 1
                    upd6(fkey, "3T", direction)

            else:
                # Non-IP L2 packet
                stats["l2_only_packets"] += 1
                stats["reasons"]["non_ip_ethertype"] += 1
                if keep_l2:
                    smac = mac_addr(raw, 6); dmac = mac_addr(raw, 0)
                    fkey = ("L2", smac, dmac, eth_type)
                    if fkey not in flows:
                        flows[fkey]["key_type"] = "L2"
                    f = flows[fkey]
                    f["packet_count"] += 1
                    f["byte_count"] += len(raw)
                    ts = base["timestamp"]
                    f["start_time"] = ts if f["start_time"] is None else min(f["start_time"], ts)
                    f["end_time"]   = ts if f["end_time"]   is None else max(f["end_time"], ts)
                    f["packet_indices"].append(len(packets))
                    packets.append({
                        **base,
                        "ip_version": 0, "src_ip": "", "dst_ip": "", "protocol": -1,
                        "src_port": -1, "dst_port": -1, "direction": 0,
                        "ip_ttl": 0, "tcp_seq": 0, "tcp_ack": 0, "tcp_flags": 0, "tcp_window": 0,
                        "icmp_type": -1, "icmp_code": -1, "esp_spi": 0, "ah_spi": 0, "gre_protocol": 0,
                        "header_length": 14 + 4*len(vlan_ids),
                        "payload_length": max(0, len(raw) - (14 + 4*len(vlan_ids)))
                    })
                    stats["valid_packets"] += 1

    except Exception as e:
        print(f"[ERROR] parse failed: {e}", file=sys.stderr)

    for meta in flows.values():
        stats["flows_by_keytype"][meta["key_type"]] += 1
    return flows, packets, stats

def write_flows_csv(flows, packets, out_path: Path, stem: str, traffic_label: str, app_label: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "flow_id","key_type",
        "src_ip","dst_ip","src_port","dst_port","protocol",
        "packet_count","byte_count","start_time","end_time","duration",
        "ip_version","traffic_label","app_label","pcap_file"
    ]
    with out_path.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for fkey, meta in flows.items():
            if meta["packet_count"] < MIN_PKTS or meta["byte_count"] < MIN_BYTES:
                continue
            ktype = meta["key_type"]
            if ktype == "5T":
                src_ip, dst_ip, sp, dp, proto = fkey
            elif ktype == "3T":
                src_ip, dst_ip, proto = fkey; sp = dp = -1
            else:
                src_ip = dst_ip = ""; sp = dp = -1; proto = -1
            dur = (meta["end_time"] - meta["start_time"]) if (meta["start_time"] and meta["end_time"]) else 0.0
            start_us = int(meta["start_time"]*1e6) if meta["start_time"] else 0
            if ktype == "5T":
                fid = f"5T_{src_ip}_{dst_ip}_{sp}_{dp}_{proto}_{start_us}"
            elif ktype == "3T":
                fid = f"3T_{src_ip}_{dst_ip}_{proto}_{start_us}"
            else:
                tag, smac, dmac, et = fkey
                fid = f"L2_{smac}_{dmac}_{et}_{start_us}"
            ipver = 0
            if meta["packet_indices"]:
                ipver = packets[meta["packet_indices"][0]].get("ip_version", 0)
            w.writerow([fid, ktype, src_ip, dst_ip, sp, dp, proto,
                        meta["packet_count"], meta["byte_count"], meta["start_time"], meta["end_time"], dur,
                        ipver, traffic_label, app_label, stem])

def write_packets_csv(flows, packets, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "flow_id","key_type","packet_index",
        "timestamp","packet_length","direction",
        "ethertype","vlan_ids","src_mac","dst_mac",
        "ip_version","src_ip","dst_ip","protocol",
        "src_port","dst_port","tcp_seq","tcp_ack","tcp_flags","tcp_window",
        "icmp_type","icmp_code","esp_spi","ah_spi","gre_protocol",
        "ip_ttl","header_length","payload_length"
    ]
    with out_path.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        pkt_flowid = [("", "")] * len(packets)
        for fkey, meta in flows.items():
            if meta["packet_count"] < MIN_PKTS or meta["byte_count"] < MIN_BYTES:
                continue
            ktype = meta["key_type"]
            start_us = int(meta["start_time"]*1e6) if meta["start_time"] else 0
            if ktype == "5T":
                src_ip, dst_ip, sp, dp, proto = fkey
                fid = f"5T_{src_ip}_{dst_ip}_{sp}_{dp}_{proto}_{start_us}"
            elif ktype == "3T":
                src_ip, dst_ip, proto = fkey
                fid = f"3T_{src_ip}_{dst_ip}_{proto}_{start_us}"
            else:
                tag, smac, dmac, et = fkey
                fid = f"L2_{smac}_{dmac}_{et}_{start_us}"
            for idx in meta["packet_indices"]:
                pkt_flowid[idx] = (fid, ktype)

        for i, p in enumerate(packets):
            fid, ktype = pkt_flowid[i]
            if not fid:
                fid = f"ORPHAN_{i}"
                ktype = "ORPHAN"
            w.writerow([
                fid, ktype, i,
                p.get("timestamp",0.0), p.get("packet_length",0), p.get("direction",0),
                p.get("ethertype",0), p.get("vlan_ids",""), p.get("src_mac",""), p.get("dst_mac",""),
                p.get("ip_version",0), p.get("src_ip",""), p.get("dst_ip",""), p.get("protocol",-1),
                p.get("src_port",-1), p.get("dst_port",-1), p.get("tcp_seq",0), p.get("tcp_ack",0), p.get("tcp_flags",0), p.get("tcp_window",0),
                p.get("icmp_type",-1), p.get("icmp_code",-1), p.get("esp_spi",0), p.get("ah_spi",0), p.get("gre_protocol",0),
                p.get("ip_ttl",0), p.get("header_length",0), p.get("payload_length",0),
            ])

def main():
    ap = argparse.ArgumentParser(description="PCAP to CSV (Preserve+Debug)")
    ap.add_argument("--pcap", help="Single PCAP or PCAPNG")
    ap.add_argument("--input_dir", help="Directory of pcaps")
    ap.add_argument("--output_dir", required=True, help="Output directory")
    ap.add_argument("--keep_l2", action="store_true", help="Aggregate L2-only packets to pseudo flows")
    ap.add_argument("--min_pkts", type=int, default=1)
    ap.add_argument("--min_bytes", type=int, default=1)
    args = ap.parse_args()

    global MIN_PKTS, MIN_BYTES
    MIN_PKTS, MIN_BYTES = args.min_pkts, args.min_bytes

    out_root = Path(args.output_dir); out_root.mkdir(parents=True, exist_ok=True)
    summary = []

    def process_one(pcap_path: Path):
        flows, packets, s = parse_pcap(pcap_path, keep_l2=args.keep_l2)
        stem = pcap_path.name
        traffic, app = infer_labels_from_path(pcap_path)
        write_flows_csv(flows, packets, out_root / f"{pcap_path.stem}_flows.csv", stem, traffic, app)
        write_packets_csv(flows, packets, out_root / f"{pcap_path.stem}_packets.csv")
        # stats file
        stat_path = out_root / f"{pcap_path.stem}_stats.json"
        with stat_path.open("w", encoding="utf-8") as f:
            json.dump(s, f, ensure_ascii=False, indent=2, default=lambda x: dict(x) if isinstance(x, Counter) else x)
        summary.append({
            "pcap": str(pcap_path),
            "traffic_label_inferred": traffic,
            "app_label_inferred": app,
            "total_packets": s["total_packets"],
            "valid_packets": s["valid_packets"],
            "l2_only_packets": s["l2_only_packets"],
            "flows_by_keytype": dict(s["flows_by_keytype"]),
            "proto_counter": dict(s["proto_counter"]),
            "ethertype_counter": dict(s["ethertype_counter"]),
            "ip_version_counter": dict(s["ip_version_counter"]),
            "reasons": dict(s["reasons"]),
        })
        flows.clear(); packets.clear(); gc.collect()

    if args.pcap:
        process_one(Path(args.pcap))
    elif args.input_dir:
        for p in sorted(Path(args.input_dir).rglob("*.pcap*")):
            process_one(p)
    else:
        print("Please provide --pcap or --input_dir")
        sys.exit(1)

    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
