#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/audit_features.py
--------------------------------------
审计 features 目录的完整脚本：
- 输出 primary / secondary / combined 的类名与分布
- 统计各协议样本数、各协议下的细粒度类分布
- 输出评估建议
- 结果自动写入 audit_report.txt 文件
"""

import os, pickle, argparse, numpy as np
from collections import Counter, defaultdict
from datetime import datetime
import sys

def load_np(path):
    return np.load(path) if os.path.exists(path) else None


def write_both(f, text=""):
    """同步输出到终端和文件"""
    print(text)
    f.write(text + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", required=True, help="特征文件目录")
    ap.add_argument("--output", default="audit_report.txt", help="输出报告文件名")
    args = ap.parse_args()

    fd = args.features_dir
    out_path = os.path.join(fd, args.output)

    with open(out_path, "w", encoding="utf-8") as fout:
        write_both(fout, f"🕒 审计时间: {datetime.now()}")
        write_both(fout, f"📁 特征目录: {fd}")
        write_both(fout, "-" * 60)

        # --- 检查文件 ---
        required = [
            "micro_seq.npy", "micro_mask.npy", "macro_bag.npy", "macro_mask.npy",
            "primary_labels.npy", "secondary_labels.npy",
            "combined_labels.npy", "label_mappings.pkl"
        ]
        missing = [f for f in required if not os.path.exists(os.path.join(fd, f))]
        if missing:
            write_both(fout, f"❌ 缺少文件: {missing}")
            return

        # --- 读文件 ---
        prim = load_np(os.path.join(fd, "primary_labels.npy"))
        sec = load_np(os.path.join(fd, "secondary_labels.npy"))
        comb = load_np(os.path.join(fd, "combined_labels.npy"))
        with open(os.path.join(fd, "label_mappings.pkl"), "rb") as f:
            lm = pickle.load(f)

        # --- 获取类名 ---
        def names(d):
            return d.get("labels") or d.get("class_names") or []

        prim_names = names(lm.get("primary", {}))
        sec_names = names(lm.get("secondary", {}))
        comb_names = names(lm.get("combined", {}))

        write_both(fout, "\n=== 基本信息 ===")
        write_both(fout, f"样本数: {len(sec)}")
        write_both(fout, f"primary 类数: {len(prim_names)}")
        write_both(fout, f"secondary 类数: {len(sec_names)}")
        write_both(fout, f"combined 类数: {len(comb_names)}")

        # --- 打印分布函数 ---
        def dist(arr):
            c = Counter(arr.tolist())
            return sorted(c.items(), key=lambda x: x[0])

        write_both(fout, "\n=== primary 分布(id:count -> name) ===")
        for k, v in dist(prim):
            nm = prim_names[k] if k < len(prim_names) else f"id_{k}"
            write_both(fout, f"{k}:{v} -> {nm}")

        write_both(fout, "\n=== secondary 分布(id:count -> name) ===")
        for k, v in dist(sec):
            nm = sec_names[k] if k < len(sec_names) else f"id_{k}"
            write_both(fout, f"{k}:{v} -> {nm}")

        write_both(fout, "\n=== combined 分布(id:count -> name) ===")
        for k, v in dist(comb):
            nm = comb_names[k] if k < len(comb_names) else f"id_{k}"
            write_both(fout, f"{k}:{v} -> {nm}")

        # --- 协议划分 ---
        if not comb_names:
            write_both(fout, "\n⚠️ combined 类名缺失，无法做协议分析")
            return

        comb_proto = np.array([n.split(":")[0].upper() for n in comb_names])
        proto_counts = Counter(comb_proto[comb])
        write_both(fout, "\n=== 协议分布 ===")
        for p, cnt in proto_counts.items():
            write_both(fout, f"{p}: {cnt}")

        # --- 各协议下 secondary 分布 ---
        write_both(fout, "\n=== 各协议下的 secondary 分布 ===")
        proto_to_ids = defaultdict(list)
        for cid, name in enumerate(comb_names):
            p = name.split(":")[0].upper()
            proto_to_ids[p].append(cid)

        for p, ids in proto_to_ids.items():
            mask = np.isin(comb, ids)
            sec_sub = sec[mask]
            if sec_sub.size == 0:
                write_both(fout, f"[{p}] 0 条样本")
                continue
            c = Counter(sec_sub.tolist())
            write_both(fout, f"[{p}] 共 {sec_sub.size} 条：")
            for sid, cnt in sorted(c.items(), key=lambda x: x[0]):
                n = sec_names[sid] if sid < len(sec_names) else f"id_{sid}"
                write_both(fout, f"  {sid}:{cnt} -> {n}")

        # --- 评估建议 ---
        write_both(fout, "\n=== 评估建议 ===")
        wanted = ["AUDIO", "CHAT", "FILE", "VIDEO", "VOIP", "BROWSING"]
        present = [w for w in wanted if w in [s.upper() for s in sec_names]]
        write_both(fout, f"存在的 6 类: {present}")
        if len(present) < 6:
            write_both(fout, "⚠️ secondary 类名不完整，请确认 eval_classes 参数。")

        protos = sorted(list(proto_counts.keys()))
        write_both(fout, f"可用协议: {protos}")
        write_both(fout, "\n=== 示例命令 ===")
        write_both(fout, "  # 仅评 VPN 细粒度（自动重映射）:")
        write_both(fout, f"  python model/export_results.py "
                  f"--checkpoint .../best_secondary.pt "
                  f"--features_dir {fd} "
                  f"--output_dir .../evaluation_results "
                  f"--protocol VPN")
        write_both(fout, "  # 指定 6 类:")
        write_both(fout, f"  python model/export_results.py "
                  f"--checkpoint .../best_secondary.pt "
                  f"--features_dir {fd} "
                  f"--output_dir .../evaluation_results "
                  f"--protocol VPN "
                  "--eval_classes AUDIO CHAT FILE VIDEO VOIP BROWSING")

        write_both(fout, "-" * 60)
        write_both(fout, f"✅ 报告已保存: {out_path}")

if __name__ == "__main__":
    main()
