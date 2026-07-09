#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sample_flows_by_ratio.py
-------------------------
按类别目录结构采样流量：
每个 *_flows.csv 按比例采样流，并同步筛选对应 *_packets.csv。
保留原目录层级到输出目录，并输出统计汇总。
"""

import pandas as pd
from pathlib import Path
import argparse

# -----------------------------------------------------------
# 单文件采样逻辑
# -----------------------------------------------------------
def sample_one_pair(flow_file: Path, ratio: float, out_root: Path, input_root: Path, global_stats: list):
    pkt_file = flow_file.with_name(flow_file.name.replace("_flows", "_packets"))
    if not pkt_file.exists():
        print(f"[WARN] 找不到对应的 packets 文件: {pkt_file}")
        return

    flows = pd.read_csv(flow_file)
    if flows.empty:
        print(f"[WARN] 空文件跳过: {flow_file}")
        return

    total_flows = len(flows)
    n = max(1, int(total_flows * ratio))
    sampled_flows = flows.sample(n=n, random_state=42)
    flow_ids = set(sampled_flows["flow_id"].astype(str))

    pkts = pd.read_csv(pkt_file)
    pkts_sampled = pkts[pkts["flow_id"].astype(str).isin(flow_ids)]

    # 输出路径
    out_dir = out_root / flow_file.parent.relative_to(input_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    sampled_flows.to_csv(out_dir / flow_file.name, index=False)
    pkts_sampled.to_csv(out_dir / pkt_file.name, index=False)

    kept_flows = len(sampled_flows)
    kept_pkts = len(pkts_sampled)
    kept_ratio = kept_flows / total_flows * 100 if total_flows else 0

    print(f"[OK] {flow_file.name:<30} "
          f"原流={total_flows:>5}, 采样流={kept_flows:>5}, "
          f"采样包={kept_pkts:>7}, 保留率={kept_ratio:5.2f}%")

    global_stats.append({
        "category": str(flow_file.parent.relative_to(input_root)),
        "flow_file": flow_file.name,
        "flows_before": total_flows,
        "flows_after": kept_flows,
        "packets_after": kept_pkts,
        "ratio_percent": kept_ratio
    })


# -----------------------------------------------------------
# 主入口
# -----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="按比例采样每个类别流量数据（带统计汇总）")
    parser.add_argument("--input_dir", required=True, help="原始 CSV 数据目录")
    parser.add_argument("--output_dir", required=True, help="输出采样目录")
    parser.add_argument("--ratio", type=float, default=0.1, help="采样比例 (默认10%)")
    args = parser.parse_args()

    input_root = Path(args.input_dir)
    out_root = Path(args.output_dir)
    ratio = args.ratio

    print(f"[INFO] 开始采样：输入={input_root} 输出={out_root} 采样比例={ratio:.2f}")
    print("="*70)
    global_stats = []

    for flow_file in sorted(input_root.rglob("*_flows.csv")):
        try:
            sample_one_pair(flow_file, ratio, out_root, input_root, global_stats)
        except Exception as e:
            print(f"[ERROR] {flow_file}: {e}")

    # -----------------------------------------------------------
    # 汇总统计
    # -----------------------------------------------------------
    if not global_stats:
        print("[WARN] 未采样到任何文件。")
        return

    df_stats = pd.DataFrame(global_stats)
    df_stats["category_root"] = df_stats["category"].apply(lambda x: x.split('/')[0] if '/' in x else x)

    total_before = df_stats["flows_before"].sum()
    total_after = df_stats["flows_after"].sum()
    total_pkts = df_stats["packets_after"].sum()
    avg_ratio = total_after / total_before * 100 if total_before else 0

    print("\n========== 汇总统计 ==========")
    print(f"[INFO] 总类别数: {df_stats['category_root'].nunique()}")
    print(f"[INFO] 总文件数: {len(df_stats)}")
    print(f"[INFO] 总流数(原): {total_before}")
    print(f"[INFO] 总流数(采样): {total_after}")
    print(f"[INFO] 总包数(采样): {total_pkts}")
    print(f"[INFO] 平均采样率: {avg_ratio:5.2f}%")

    # 分类别统计
    print("\n[INFO] 按上层类别统计：")
    cat_sum = df_stats.groupby("category_root")[["flows_before", "flows_after"]].sum()
    cat_sum["ratio(%)"] = cat_sum["flows_after"] / cat_sum["flows_before"] * 100
    print(cat_sum.round(2))

    # 输出汇总 CSV
    summary_file = out_root / "sampling_summary.csv"
    df_stats.to_csv(summary_file, index=False)
    print(f"\n✅ 采样完成，详细统计已保存到: {summary_file.resolve()}")

if __name__ == "__main__":
    main()
