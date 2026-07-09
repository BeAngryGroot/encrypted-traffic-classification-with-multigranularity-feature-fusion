#!/usr/bin/env python3
"""
features_test.py
----------------
从 *_packets.csv 提取微观与宏观特征，生成训练所需的 NPY 文件。
支持：
  1) primary / secondary / combined 三层标签；
  2) label_mappings.pkl 自动匹配实际类；
  3) seq_len.npy / window_keys.npy 调试。
"""

import argparse, csv, pickle, re
from pathlib import Path
import numpy as np

# ========================= 顶层类别（primary） =========================
TRAFFIC_ORDER = ['VPN', 'OTHER', 'TOR', 'QUIC']
ENCRYPTED_SET = {'VPN', 'TOR', 'QUIC'}

# ========================= 二级细粒度类别（secondary） =================
APP_ORDER = [
    'AUDIO','CHAT','FILE','VIDEO','VOIP','BROWSING','MAIL',
    'P2P','STREAM','SOCIAL','GAMING','UNKNOWN'
]

APP_TOKEN2CANON = {
    # CHAT
    'CHAT':'CHAT','ICQ':'CHAT','WHATSAPP':'CHAT','WECHAT':'CHAT','QQ':'CHAT',
    'HANGOUT':'CHAT','IRC':'CHAT','IM':'CHAT','MESSENGER':'CHAT',
    # VIDEO / STREAM
    'VIDEO':'VIDEO','VIMEO':'VIDEO','YOUTUBE':'VIDEO','NETFLIX':'VIDEO',
    'STREAM':'STREAM','LIVE':'STREAM',
    # VOIP
    'VOIP':'VOIP','SIP':'VOIP','CALL':'VOIP','TELEPHONY':'VOIP',
    'SKYPE_VOIP':'VOIP','VOIP_SKYPE':'VOIP',
    # AUDIO
    'AUDIO':'AUDIO','SPOTIFY':'AUDIO','MUSIC':'AUDIO','MP3':'AUDIO',
    # FILE / P2P
    'FILE':'FILE','FTP':'FILE','FTPS':'FILE','SFTP':'FILE','DROPBOX':'FILE',
    'BITTORRENT':'P2P','TORRENT':'P2P','P2P':'P2P','EMULE':'P2P',
    # WEB
    'BROWSING':'BROWSING','HTTP':'BROWSING','HTTPS':'BROWSING','WEB':'BROWSING',
    # MAIL
    'MAIL':'MAIL','SMTP':'MAIL','IMAP':'MAIL','POP3':'MAIL','GMAIL':'MAIL','OUTLOOK':'MAIL',
    # SOCIAL
    'SOCIAL':'SOCIAL','FACEBOOK':'SOCIAL','INSTAGRAM':'SOCIAL','TWITTER':'SOCIAL',
    # GAMING
    'GAMING':'GAMING','GAME':'GAMING','STEAM':'GAMING'
}

MICRO_FEATURES = [
    'packet_length','payload_length','src_port','dst_port','protocol',
    'ip_version','ip_ttl','tcp_flags','tcp_window','icmp_type','icmp_code',
    'esp_spi','ah_spi','gre_protocol','direction','normalized_time',
    'inter_arrival','length_ratio','payload_ratio','is_tcp'
]
MACRO_FEATURES = [
    'packet_count','byte_count','duration','avg_pkt_len','avg_pay_len',
    'fwd_bytes','rev_bytes','fwd_pkts','rev_pkts','min_pkt','max_pkt',
    'std_pkt','min_pay','max_pay','std_pay','ratio_fwd_rev_bytes',
    'ratio_fwd_rev_pkts','ratio_len_pay','tcp_ratio','udp_ratio'
]
WINDOW_SEC = 5.0
STRIDE_SEC = 2.5
MAX_SEQ = 64

# ========================= 工具函数 ==========================
def _to_float(x, default=0.0):
    try: return float(x)
    except Exception: return float(default)

def _to_int(x, default=0):
    try: return int(float(x))
    except Exception: return int(default)

def load_packets(csv_file: Path):
    with csv_file.open() as f:
        return list(csv.DictReader(f))

def group_by_flow(packets):
    flows = {}
    for p in packets:
        fid = p.get('flow_id','')
        flows.setdefault(fid, []).append(p)
    for fid in flows:
        flows[fid].sort(key=lambda x: _to_float(x.get('timestamp',0)))
    return flows

def ensure_direction(plist):
    fwd_key = None
    out = []
    for p in plist:
        s = (p.get('src_ip',''), _to_int(p.get('src_port',-1)))
        if fwd_key is None:
            fwd_key = s
        q = dict(p)
        q['direction'] = '0' if s == fwd_key else '1'
        out.append(q)
    return out

def build_windows(flow_packets):
    if not flow_packets: return []
    t0 = _to_float(flow_packets[0].get('timestamp',0))
    t1 = _to_float(flow_packets[-1].get('timestamp',0))
    windows, cur = [], t0
    while cur <= t1:
        w_end = cur + WINDOW_SEC
        slice_pkts = [p for p in flow_packets if cur <= _to_float(p.get('timestamp',0)) < w_end]
        if slice_pkts:
            windows.append((cur, slice_pkts))
        cur += STRIDE_SEC
    return windows

def _tokens_from_path(path: Path):
    parts = [s.upper() for s in path.parts]
    extra = re.split(r'[^A-Z0-9]+', path.name.upper())
    return [t for t in parts + extra if t]

def infer_primary(pkt_csv: Path, in_root: Path, packets):
    for p in packets:
        tl = p.get('traffic_label')
        if tl:
            v = str(tl).upper()
            if v in TRAFFIC_ORDER:
                return v
    parts = [p.upper() for p in pkt_csv.relative_to(in_root).parts]
    for name in TRAFFIC_ORDER:
        if name in parts:
            return name
    return 'OTHER'

def infer_secondary(pkt_csv: Path, in_root: Path, packets, primary_label: str):
    if primary_label not in ENCRYPTED_SET:
        return 'UNKNOWN'
    csv_keys = ('app_label','application','app','appclass','label_app')
    for p in packets:
        for k in csv_keys:
            if k in p and p[k]:
                v = str(p[k]).upper()
                if v in APP_ORDER:
                    return v
                if v in APP_TOKEN2CANON:
                    return APP_TOKEN2CANON[v]
    rel = pkt_csv.relative_to(in_root)
    for tok in _tokens_from_path(rel):
        if tok in APP_TOKEN2CANON:
            return APP_TOKEN2CANON[tok]
    return 'UNKNOWN'

def micro_vector(pkts):
    seq, base = [], _to_float(pkts[0].get('timestamp',0))
    prev_t = None
    for p in pkts[:MAX_SEQ]:
        plen = _to_int(p.get('packet_length',0))
        pay  = _to_int(p.get('payload_length',0))
        proto= _to_int(p.get('protocol',0))
        t = _to_float(p.get('timestamp',0)) - base
        inter = 0.0 if prev_t is None else t - prev_t
        prev_t = t
        seq.append([
            plen, pay,
            _to_int(p.get('src_port',-1)), _to_int(p.get('dst_port',-1)), proto,
            _to_int(p.get('ip_version',4)), _to_int(p.get('ip_ttl',0)),
            _to_int(p.get('tcp_flags',0)), _to_int(p.get('tcp_window',0)),
            _to_int(p.get('icmp_type',-1)), _to_int(p.get('icmp_code',-1)),
            _to_int(p.get('esp_spi',0)), _to_int(p.get('ah_spi',0)), _to_int(p.get('gre_protocol',0)),
            _to_int(p.get('direction',0)),
            t, inter,
            (pay/plen) if plen else 0.0,
            pay / MAX_SEQ,
            1 if proto == 6 else 0
        ])
    while len(seq) < MAX_SEQ:
        seq.append([0]*len(MICRO_FEATURES))
    return np.array(seq, dtype=np.float32), min(len(pkts), MAX_SEQ)

def macro_vector(pkts):
    lens = [_to_int(p.get('packet_length',0)) for p in pkts]
    pays = [_to_int(p.get('payload_length',0)) for p in pkts]
    protos = [_to_int(p.get('protocol',0)) for p in pkts]
    dirs   = [_to_int(p.get('direction',0)) for p in pkts]
    fwd_bytes = sum(l for l,d in zip(lens,dirs) if d==0)
    rev_bytes = sum(l for l,d in zip(lens,dirs) if d==1)
    fwd_pkts  = dirs.count(0); rev_pkts = dirs.count(1)
    total_bytes = sum(lens); total_pays = sum(pays)
    duration = max(0.0, _to_float(pkts[-1].get('timestamp',0)) - _to_float(pkts[0].get('timestamp',0)))
    return np.array([
        len(pkts), total_bytes, duration,
        float(np.mean(lens)) if lens else 0.0,
        float(np.mean(pays)) if pays else 0.0,
        fwd_bytes, rev_bytes, fwd_pkts, rev_pkts,
        min(lens) if lens else 0, max(lens) if lens else 0,
        float(np.std(lens)) if lens else 0.0,
        min(pays) if pays else 0, max(pays) if pays else 0,
        float(np.std(pays)) if pays else 0.0,
        (fwd_bytes/rev_bytes) if rev_bytes>0 else 0.0,
        (fwd_pkts/rev_pkts) if rev_pkts>0 else 0.0,
        (total_bytes/total_pays) if total_pays>0 else 0.0,
        (protos.count(6)/len(protos)) if protos else 0.0,
        (protos.count(17)/len(protos)) if protos else 0.0
    ], dtype=np.float32)

# ========================= 主流程 ===============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir', required=True)
    ap.add_argument('--output_dir', required=True)
    args = ap.parse_args()
    in_root = Path(args.input_dir)
    out_root = Path(args.output_dir)

    micro, macro = [], []
    primary_labels, secondary_labels, combined_labels = [], [], []
    seqlen, keys = [], []

    seen_secondary = set(['UNKNOWN'])
    seen_combined  = set()

    for pkt_csv in sorted(in_root.rglob('*_packets.csv')):
        pkts_raw = load_packets(pkt_csv)
        flows = group_by_flow(pkts_raw)
        p_primary = infer_primary(pkt_csv, in_root, pkts_raw)

        for fid, plist in flows.items():
            plist = ensure_direction(plist)
            for start_ts, wpkts in build_windows(plist):
                mv, sl = micro_vector(wpkts)
                micro.append(mv)
                macro.append(macro_vector(wpkts))
                seqlen.append(sl)
                keys.append(f"{fid}|{pkt_csv.name}|{start_ts:.6f}")

                y_primary = TRAFFIC_ORDER.index(p_primary) if p_primary in TRAFFIC_ORDER else TRAFFIC_ORDER.index('OTHER')
                primary_labels.append(y_primary)

                sec_name = infer_secondary(pkt_csv, in_root, pkts_raw, p_primary)
                if sec_name not in APP_ORDER:
                    sec_name = 'UNKNOWN'
                seen_secondary.add(sec_name)

                if p_primary in ENCRYPTED_SET:
                    comb_name = f"{p_primary}:{sec_name}"
                else:
                    comb_name = "OTHER:UNKNOWN"
                seen_combined.add(comb_name)

                secondary_labels.append(sec_name)
                combined_labels.append(comb_name)

    micro_arr = np.stack(micro) if micro else np.zeros((0, MAX_SEQ, len(MICRO_FEATURES)), dtype=np.float32)
    macro_arr = np.stack(macro) if macro else np.zeros((0, len(MACRO_FEATURES)), dtype=np.float32)
        # ======= 数值清理与归一化，防止训练阶段溢出/NaN =======
    micro_arr = np.nan_to_num(micro_arr, nan=0.0, posinf=0.0, neginf=0.0)
    macro_arr = np.nan_to_num(macro_arr, nan=0.0, posinf=0.0, neginf=0.0)

    # ---------- 宏观特征归一化改进版 ----------
    if macro_arr.size > 0:
        # 1️⃣ log 压缩计数类特征
        log_cols = ['packet_count','byte_count','duration',
                    'fwd_bytes','rev_bytes','fwd_pkts','rev_pkts']
        log_idxs = [MACRO_FEATURES.index(c) for c in log_cols if c in MACRO_FEATURES]
        macro_arr[:, log_idxs] = np.log1p(np.clip(macro_arr[:, log_idxs], 0, None))

        # 2️⃣ 标准化所有非比率列
        ratio_cols = [c for c in MACRO_FEATURES if 'ratio' in c]
        ratio_idxs = [MACRO_FEATURES.index(c) for c in ratio_cols]
        non_ratio_idxs = [i for i in range(len(MACRO_FEATURES)) if i not in ratio_idxs]

        mean = np.mean(macro_arr[:, non_ratio_idxs], axis=0, keepdims=True)
        std  = np.std(macro_arr[:, non_ratio_idxs], axis=0, keepdims=True) + 1e-6
        macro_arr[:, non_ratio_idxs] = (macro_arr[:, non_ratio_idxs] - mean) / std

        # 3️⃣ 限制比率类特征范围 [-10,10]
        macro_arr[:, ratio_idxs] = np.clip(macro_arr[:, ratio_idxs], -10, 10)

        # 4️⃣ 全局裁剪 [-5,5]
        macro_arr = np.clip(macro_arr, -5, 5)

    # micro 数值范围裁剪
    micro_arr = np.clip(micro_arr, -1e3, 1e3)

    print("[DEBUG] macro_bag stats:",
      "mean=", np.mean(macro_arr),
      "std=", np.std(macro_arr),
      "min=", np.min(macro_arr),
      "max=", np.max(macro_arr))

    seqlen_np = np.array(seqlen, dtype=np.int64)
    keys_np   = np.array(keys, dtype=np.str_)
    primary_np= np.array(primary_labels, dtype=np.int64)

    # ===== micro_mask =====
    micro_mask = np.zeros((len(seqlen_np), MAX_SEQ), dtype=np.float32)
    for i, l in enumerate(seqlen_np):
        micro_mask[i, :l] = 1.0

    # ===== macro_mask =====
    # 动态生成掩码：对应每个 flow/window 的有效窗口数，而不是全 1
    if len(seqlen_np):
        max_valid = max(1, int(np.percentile(seqlen_np, 95)))  # 大部分样本的有效窗口
    else:
        max_valid = 1

    macro_mask = np.zeros((len(macro_arr), MAX_SEQ), dtype=np.float32)
    for i, l in enumerate(seqlen_np):
        macro_mask[i, :min(l, MAX_SEQ)] = 1.0

    # 如果模型只接受 [N, 1] 掩码，则可改为平均有效率：
    macro_mask_mean = (macro_mask.sum(axis=1, keepdims=True) / MAX_SEQ).astype(np.float32)


    # ===== secondary & combined =====
    def ordered_subset(order_list, seen_set):
        return [x for x in order_list if x in seen_set]

    secondary_classes = ordered_subset(APP_ORDER, seen_secondary)
    sec_label_to_id = {name: i for i, name in enumerate(secondary_classes)}
    sec_id_to_label = {i: name for name, i in sec_label_to_id.items()}
    secondary_np = np.array([sec_label_to_id[s] for s in secondary_labels], dtype=np.int64)

    comb_order = []
    for p_name in TRAFFIC_ORDER:
        for s_name in secondary_classes:
            cand = f"{p_name}:{s_name}" if p_name in ENCRYPTED_SET else "OTHER:UNKNOWN"
            if cand in seen_combined and cand not in comb_order:
                comb_order.append(cand)
    comb_label_to_id = {name: i for i, name in enumerate(comb_order)}
    comb_id_to_label = {i: name for name, i in comb_label_to_id.items()}
    combined_np = np.array([comb_label_to_id[c] for c in combined_labels], dtype=np.int64)

    # ===== 保存 =====
    out_root.mkdir(parents=True, exist_ok=True)
    np.save(out_root/'micro_seq.npy',  micro_arr)
    np.save(out_root/'micro_mask.npy', micro_mask)
    np.save(out_root/'macro_bag.npy',  macro_arr)
    np.save(out_root/'macro_mask.npy', macro_mask)
    np.save(out_root/'primary_labels.npy', primary_np)
    np.save(out_root/'secondary_labels.npy', secondary_np)
    np.save(out_root/'combined_labels.npy',  combined_np)
    np.save(out_root/'seq_len.npy',    seqlen_np)
    np.save(out_root/'window_keys.npy',keys_np)

    label_mappings = {
        "primary": {
            "num_classes": len(TRAFFIC_ORDER),
            "label_to_id": {n:i for i,n in enumerate(TRAFFIC_ORDER)},
            "id_to_label": {i:n for i,n in enumerate(TRAFFIC_ORDER)},
            "class_names": TRAFFIC_ORDER[:],
        },
        "secondary": {
            "num_classes": len(secondary_classes),
            "label_to_id": sec_label_to_id,
            "id_to_label": sec_id_to_label,
            "class_names": secondary_classes[:],
        },
        "combined": {
            "num_classes": len(comb_order),
            "label_to_id": comb_label_to_id,
            "id_to_label": comb_id_to_label,
            "class_names": comb_order[:],
        }
    }
    with open(out_root/'label_mappings.pkl', 'wb') as f:
        pickle.dump(label_mappings, f)

    print(f"[INFO] samples={len(primary_np)}, micro_seq={micro_arr.shape}, macro_bag={macro_arr.shape}")
    if len(primary_np):
        u, c = np.unique(primary_np, return_counts=True)
        print("[INFO] primary 分布:", {int(k): int(v) for k,v in zip(u,c)})
    if len(secondary_np):
        u, c = np.unique(secondary_np, return_counts=True)
        print("[INFO] secondary 分布:", {int(k): int(v) for k,v in zip(u,c)}, "classes=", secondary_classes)
    if len(combined_np):
        u, c = np.unique(combined_np, return_counts=True)
        print("[INFO] combined 分布:", {int(k): int(v) for k,v in zip(u,c)}, "classes=", comb_order)

if __name__ == '__main__':
    main()
