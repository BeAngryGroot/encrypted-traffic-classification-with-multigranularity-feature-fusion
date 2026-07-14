#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按完整 flow 生成 smoke/pilot 数据集，并记录可复现采样清单。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def select_flow_ids(
    flow_ids: Iterable[str],
    *,
    ratio: float,
    seed: int,
    max_flows: int | None,
    min_flows: int = 1,
) -> list[str]:
    """确定性选择完整流，数量同时受比例、下限和上限约束。"""

    unique = sorted({str(flow_id) for flow_id in flow_ids})
    if not unique:
        return []
    if not 0 < ratio <= 1:
        raise ValueError("ratio must be in (0, 1]")
    requested = max(int(min_flows), int(np.ceil(len(unique) * ratio)))
    if max_flows is not None:
        requested = min(requested, int(max_flows))
    requested = min(requested, len(unique))
    rng = np.random.default_rng(int(seed))
    selected_indices = sorted(rng.choice(len(unique), size=requested, replace=False).tolist())
    return [unique[index] for index in selected_indices]


def sample_one_pair(
    flow_file: Path,
    ratio: float,
    output_root: Path,
    input_root: Path,
    *,
    seed: int = 42,
    max_flows: int | None = None,
    min_flows: int = 1,
) -> dict[str, object]:
    """同步筛选 flow/packet 文件；绝不从一个 flow 中只截取部分数据包。"""

    flow_file = Path(flow_file)
    packet_file = flow_file.with_name(flow_file.name.replace("_flows", "_packets"))
    if not packet_file.exists():
        raise FileNotFoundError(f"找不到配对的包文件：{packet_file}")

    flows = pd.read_csv(flow_file)
    packets = pd.read_csv(packet_file)
    if "flow_id" not in flows or "flow_id" not in packets:
        raise ValueError("flow/packet CSV 均必须包含 flow_id 列")
    selected = select_flow_ids(
        flows["flow_id"].astype(str),
        ratio=ratio,
        seed=seed,
        max_flows=max_flows,
        min_flows=min_flows,
    )
    selected_set = set(selected)
    sampled_flows = flows[flows["flow_id"].astype(str).isin(selected_set)].copy()
    sampled_packets = packets[packets["flow_id"].astype(str).isin(selected_set)].copy()

    relative_parent = flow_file.parent.relative_to(input_root)
    output_dir = Path(output_root) / relative_parent
    output_dir.mkdir(parents=True, exist_ok=True)
    sampled_flows.to_csv(output_dir / flow_file.name, index=False)
    sampled_packets.to_csv(output_dir / packet_file.name, index=False)

    return {
        "source_key": str(flow_file.relative_to(input_root)).replace("\\", "/"),
        "category": str(relative_parent).replace("\\", "/"),
        "flow_file": flow_file.name,
        "flows_before": int(len(flows)),
        "flows_after": int(len(sampled_flows)),
        "packets_after": int(len(sampled_packets)),
        "ratio": float(ratio),
        "seed": int(seed),
        "selected_flow_ids": "|".join(selected),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="按完整 flow 构建可复现的测试/先导数据集")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_flows_per_file", type=int)
    parser.add_argument("--min_flows_per_file", type=int, default=1)
    args = parser.parse_args()

    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)
    records = []
    for flow_file in sorted(input_root.rglob("*_flows.csv")):
        records.append(
            sample_one_pair(
                flow_file,
                args.ratio,
                output_root,
                input_root,
                seed=args.seed,
                max_flows=args.max_flows_per_file,
                min_flows=args.min_flows_per_file,
            )
        )
    if not records:
        raise FileNotFoundError(f"未在 {input_root} 下发现 *_flows.csv")
    manifest = output_root / "sampling_manifest.csv"
    pd.DataFrame(records).to_csv(manifest, index=False)
    print(f"采样完成：{manifest.resolve()}")


if __name__ == "__main__":
    main()
