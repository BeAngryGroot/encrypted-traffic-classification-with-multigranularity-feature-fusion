#!/usr/bin/env python
"""
diagnose_alignment.py - 诊断宏观和微观特征对齐问题
-------------------------------------------------
快速定位flow_id不匹配的根本原因
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

def diagnose_flow_id_mismatch():
    """诊断flow_id不匹配问题"""
    
    # 路径配置
    sampled_dir = Path("/data3/wsb_workspace/study/data/data_test/sampled_data")
    features_dir = Path("/data3/wsb_workspace/study/models/flowcls-test/features")
    
    print("=" * 60)
    print("Flow ID 对齐问题诊断")
    print("=" * 60)
    
    # 1. 检查采样数据
    flow_csv = sampled_dir / "sampled_flows.csv"
    pkt_csv = sampled_dir / "sampled_flows.pkt.csv"
    
    if not flow_csv.exists():
        print(f"❌ 流级CSV不存在: {flow_csv}")
        return
    
    if not pkt_csv.exists():
        print(f"❌ 包级CSV不存在: {pkt_csv}")
        return
    
    # 读取数据
    print("读取采样数据...")
    flow_df = pd.read_csv(flow_csv)
    pkt_df = pd.read_csv(pkt_csv)
    
    print(f"流级数据: {len(flow_df)} 行")
    print(f"包级数据: {len(pkt_df)} 行")
    
    # 2. flow_id分析
    flow_ids = set(flow_df['flow_id'].values)
    pkt_flow_ids = set(pkt_df['flow_id'].values)
    
    print(f"\nFlow ID 统计:")
    print(f"流级唯一flow_id: {len(flow_ids)}")
    print(f"包级唯一flow_id: {len(pkt_flow_ids)}")
    print(f"交集: {len(flow_ids & pkt_flow_ids)}")
    print(f"流级独有: {len(flow_ids - pkt_flow_ids)}")
    print(f"包级独有: {len(pkt_flow_ids - flow_ids)}")
    
    # 3. 分析缺失的flow_id
    missing_in_pkt = flow_ids - pkt_flow_ids
    missing_in_flow = pkt_flow_ids - flow_ids
    
    print(f"\n缺失flow_id分析:")
    print(f"包级数据中缺失的flow_id数量: {len(missing_in_pkt)}")
    print(f"流级数据中缺失的flow_id数量: {len(missing_in_flow)}")
    
    # 4. 抽样检查格式差异
    print(f"\nflow_id格式检查:")
    if flow_ids:
        sample_flow_ids = list(flow_ids)[:5]
        print("流级flow_id样例:")
        for fid in sample_flow_ids:
            print(f"  {fid}")
    
    if pkt_flow_ids:
        sample_pkt_ids = list(pkt_flow_ids)[:5]
        print("包级flow_id样例:")
        for fid in sample_pkt_ids:
            print(f"  {fid}")
    
    # 5. 分析缺失流的特征
    if missing_in_pkt:
        print(f"\n分析包级数据中缺失的流:")
        missing_flows = flow_df[flow_df['flow_id'].isin(missing_in_pkt)]
        
        print(f"缺失流的统计:")
        if 'packets' in missing_flows.columns:
            packets_stats = missing_flows['packets'].describe()
            print(f"  包数统计: {packets_stats.to_dict()}")
        
        if 'bytes' in missing_flows.columns:
            bytes_stats = missing_flows['bytes'].describe()
            print(f"  字节数统计: {bytes_stats.to_dict()}")
        
        if 'duration' in missing_flows.columns:
            duration_stats = missing_flows['duration'].describe()
            print(f"  持续时间统计: {duration_stats.to_dict()}")
        
        # 检查是否是短流
        if 'packets' in missing_flows.columns:
            short_flows = missing_flows[missing_flows['packets'] <= 5]
            print(f"  短流(<=5包): {len(short_flows)} / {len(missing_flows)} ({len(short_flows)/len(missing_flows)*100:.1f}%)")
    
    # 6. 检查特征文件
    print(f"\n检查特征文件:")
    if features_dir.exists():
        macro_file = features_dir / "macro_bag.npy"
        micro_file = features_dir / "micro_seq.npy"
        
        if macro_file.exists() and micro_file.exists():
            macro_bag = np.load(macro_file)
            micro_seq = np.load(micro_file)
            
            print(f"  宏观特征: {macro_bag.shape}")
            print(f"  微观特征: {micro_seq.shape}")
            print(f"  维度差异: {macro_bag.shape[0] - micro_seq.shape[0]}")
            
            # 计算匹配率
            match_rate = len(flow_ids & pkt_flow_ids) / len(flow_ids) * 100
            print(f"  Flow ID匹配率: {match_rate:.1f}%")
            
            if match_rate < 80:
                print(f"  ⚠️  匹配率过低，存在严重的flow_id不一致问题")
            elif match_rate < 95:
                print(f"  ⚠️  匹配率偏低，可能有数据丢失")
            else:
                print(f"  ✅ 匹配率正常")
    
    # 7. 给出修复建议
    print(f"\n修复建议:")
    
    if len(missing_in_pkt) > len(flow_ids) * 0.2:  # 超过20%缺失
        print("1. 【高优先级】包级数据大量缺失，建议检查:")
        print("   - PCAP到CSV转换过程中是否丢失了包数据")
        print("   - 采样过程中flow_id是否正确传递")
        print("   - 包级CSV的flow_id生成逻辑")
        
    if len(missing_in_flow) > 0:
        print("2. 包级数据中存在流级数据没有的flow_id，可能原因:")
        print("   - 采样过程中的不一致")
        print("   - 数据处理管道的异步问题")
    
    if missing_in_pkt:
        missing_flows = flow_df[flow_df['flow_id'].isin(missing_in_pkt)]
        if 'packets' in missing_flows.columns:
            short_flows_ratio = len(missing_flows[missing_flows['packets'] <= 5]) / len(missing_flows)
            if short_flows_ratio > 0.5:
                print("3. 缺失的flow主要是短流，建议:")
                print("   - 在特征构建时统一过滤短流")
                print("   - 或在包级数据处理时保留短流的空序列")
    
    print("\n建议使用修复脚本自动处理这些问题")

def check_flow_id_format():
    """检查flow_id格式的一致性"""
    sampled_dir = Path("/data3/wsb_workspace/study/data/sampled_data")
    
    flow_csv = sampled_dir / "sampled_flows.csv"
    pkt_csv = sampled_dir / "sampled_flows.pkt.csv"
    
    if not (flow_csv.exists() and pkt_csv.exists()):
        print("CSV文件不存在，跳过格式检查")
        return
    
    flow_df = pd.read_csv(flow_csv, nrows=1000)  # 只读前1000行
    pkt_df = pd.read_csv(pkt_csv, nrows=1000)
    
    print(f"\nFlow ID 格式一致性检查:")
    
    # 检查格式模式
    flow_id_patterns = set()
    for fid in flow_df['flow_id'].head(10):
        # 分析格式模式
        parts = str(fid).split('|')
        pattern = f"{len(parts)}部分"
        if len(parts) >= 5:
            # 检查最后一部分是否是时间戳
            last_part = parts[-1]
            if last_part.isdigit():
                if len(last_part) > 10:  # 微秒时间戳
                    pattern += "_微秒时间戳"
                else:  # 秒时间戳
                    pattern += "_秒时间戳"
        flow_id_patterns.add(pattern)
    
    pkt_id_patterns = set()
    for fid in pkt_df['flow_id'].head(10):
        parts = str(fid).split('|')
        pattern = f"{len(parts)}部分"
        if len(parts) >= 5:
            last_part = parts[-1]
            if last_part.isdigit():
                if len(last_part) > 10:
                    pattern += "_微秒时间戳"
                else:
                    pattern += "_秒时间戳"
        pkt_id_patterns.add(pattern)
    
    print(f"流级flow_id格式模式: {flow_id_patterns}")
    print(f"包级flow_id格式模式: {pkt_id_patterns}")
    
    if flow_id_patterns == pkt_id_patterns:
        print("✅ flow_id格式一致")
    else:
        print("❌ flow_id格式不一致，这可能是问题根源")

if __name__ == "__main__":
    diagnose_flow_id_mismatch()
    check_flow_id_format()