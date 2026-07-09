#!/usr/bin/env python
"""
trace_tor_labels.py - 追踪TOR标签在各个处理阶段的变化
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import pickle

def check_stage1_csv_files(csv_dir):
    """检查第一阶段：原始CSV文件"""
    print("=" * 60)
    print("阶段1：检查原始CSV文件")
    print("=" * 60)
    
    csv_path = Path(csv_dir)
    all_labels = []
    file_details = []
    
    for csv_file in csv_path.rglob("*.csv"):
        if csv_file.name.endswith(".pkt.csv"):
            continue
            
        try:
            df = pd.read_csv(csv_file)
            if 'class_label' in df.columns:
                labels = df['class_label'].tolist()
                all_labels.extend(labels)
                
                label_counts = Counter(labels)
                file_details.append({
                    'file': csv_file.name,
                    'total': len(df),
                    'tor': label_counts.get('TOR', 0),
                    'nontor': label_counts.get('NONTOR', 0)
                })
                
        except Exception as e:
            print(f"读取失败: {csv_file.name} - {e}")
    
    # 显示结果
    total_tor = all_labels.count('TOR')
    total_nontor = all_labels.count('NONTOR')
    
    print(f"总计: {len(all_labels)} 个流")
    print(f"TOR: {total_tor} ({total_tor/len(all_labels)*100:.1f}%)")
    print(f"NONTOR: {total_nontor} ({total_nontor/len(all_labels)*100:.1f}%)")
    
    # 显示有TOR数据的文件
    print(f"\n包含TOR流的文件:")
    tor_files = [f for f in file_details if f['tor'] > 0]
    for f in tor_files[:10]:  # 只显示前10个
        print(f"  {f['file']}: TOR={f['tor']}, NONTOR={f['nontor']}")
    
    return total_tor, total_nontor

def check_stage2_extended_flows(features_dir):
    """检查第二阶段：扩展流数据"""
    print("\n" + "=" * 60)
    print("阶段2：检查扩展流数据 (extended_flows.csv)")
    print("=" * 60)
    
    features_path = Path(features_dir)
    extended_file = features_path / "extended_flows.csv"
    
    if not extended_file.exists():
        print("未找到 extended_flows.csv")
        return 0, 0
    
    df = pd.read_csv(extended_file)
    print(f"扩展流数据总数: {len(df)}")
    
    if 'class_label' in df.columns:
        label_counts = df['class_label'].value_counts()
        print("class_label分布:")
        for label, count in label_counts.items():
            pct = count/len(df)*100
            print(f"  {label}: {count} ({pct:.1f}%)")
        
        # 检查是否有重复的flow_id导致的问题
        if 'flow_id' in df.columns:
            unique_flows = df['flow_id'].nunique()
            total_rows = len(df)
            print(f"\n唯一flow_id: {unique_flows}")
            print(f"总行数: {total_rows}")
            print(f"平均每个flow_id: {total_rows/unique_flows:.1f} 行")
            
            # 检查TOR流的flow_id分布
            tor_data = df[df['class_label'] == 'TOR']
            if len(tor_data) > 0:
                tor_unique_flows = tor_data['flow_id'].nunique()
                print(f"TOR唯一flow_id: {tor_unique_flows}")
                print(f"TOR总行数: {len(tor_data)}")
        
        return label_counts.get('TOR', 0), label_counts.get('NONTOR', 0)
    
    return 0, 0

def check_stage3_final_features(features_dir):
    """检查第三阶段：最终特征和标签"""
    print("\n" + "=" * 60)
    print("阶段3：检查最终特征和标签")
    print("=" * 60)
    
    features_path = Path(features_dir)
    
    # 检查标签文件
    primary_labels = np.load(features_path / "primary_labels.npy")
    
    # 检查标签映射
    with open(features_path / "label_mappings.pkl", "rb") as f:
        label_mappings = pickle.load(f)
    
    print(f"最终样本数: {len(primary_labels)}")
    print(f"标签映射: {label_mappings['primary']}")
    
    # 统计分布
    label_counts = Counter(primary_labels)
    for label_id, count in label_counts.items():
        label_name = label_mappings['primary']['labels'][label_id]
        pct = count/len(primary_labels)*100
        print(f"  {label_name} (ID={label_id}): {count} ({pct:.1f}%)")
    
    # 检查标签编码是否正确
    print(f"\n标签编码检查:")
    for i, label_name in enumerate(label_mappings['primary']['labels']):
        count = (primary_labels == i).sum()
        print(f"  ID={i} -> {label_name}: {count} 样本")
    
    return label_counts

def check_label_encoding_process(features_dir):
    """检查标签编码过程"""
    print("\n" + "=" * 60)
    print("阶段4：检查标签编码过程")
    print("=" * 60)
    
    features_path = Path(features_dir)
    extended_file = features_path / "extended_flows.csv"
    
    if extended_file.exists():
        # 读取原始标签
        df = pd.read_csv(extended_file)
        original_labels = df['class_label'].tolist() if 'class_label' in df.columns else []
        
        # 读取编码后的标签
        encoded_labels = np.load(features_path / "primary_labels.npy")
        
        # 读取标签映射
        with open(features_path / "label_mappings.pkl", "rb") as f:
            label_mappings = pickle.load(f)
        
        print(f"原始标签样本: {len(original_labels)}")
        print(f"编码标签样本: {len(encoded_labels)}")
        
        if len(original_labels) == len(encoded_labels):
            # 检查编码是否正确
            print(f"\n编码验证 (前10个样本):")
            for i in range(min(10, len(original_labels))):
                orig = original_labels[i]
                encoded = encoded_labels[i]
                decoded = label_mappings['primary']['labels'][encoded]
                match = "✓" if orig == decoded else "✗"
                print(f"  {i:2d}: {orig:6s} -> {encoded} -> {decoded:6s} {match}")
                
            # 统计编码错误
            correct = 0
            for orig, enc in zip(original_labels, encoded_labels):
                decoded = label_mappings['primary']['labels'][enc]
                if orig == decoded:
                    correct += 1
            
            accuracy = correct / len(original_labels) * 100
            print(f"\n编码准确率: {accuracy:.2f}% ({correct}/{len(original_labels)})")
            
            if accuracy < 100:
                print("发现编码错误！检查标签编码逻辑...")

def main():
    # 配置路径
    csv_dir = "/data3/wsb_workspace/study/data/data_test/data_full_flow"
    features_dir = "/data3/wsb_workspace/study/models/flowcls-test/features_supplemented"
    
    print("TOR标签追踪工具")
    
    try:
        # 阶段1：原始CSV
        stage1_tor, stage1_nontor = check_stage1_csv_files(csv_dir)
        
        # 阶段2：扩展流数据  
        stage2_tor, stage2_nontor = check_stage2_extended_flows(features_dir)
        
        # 阶段3：最终特征
        stage3_counts = check_stage3_final_features(features_dir)
        
        # 阶段4：编码过程检查
        check_label_encoding_process(features_dir)
        
        # 总结
        print("\n" + "=" * 60)
        print("数据流向总结")
        print("=" * 60)
        print(f"阶段1 (原始CSV):     TOR={stage1_tor:,}, NONTOR={stage1_nontor:,}")
        print(f"阶段2 (扩展流):      TOR={stage2_tor:,}, NONTOR={stage2_nontor:,}")
        
        if stage3_counts:
            stage3_tor = 0
            stage3_nontor = 0
            # 需要根据标签映射确定哪个ID对应TOR
            
        # 分析数据损失
        if stage1_tor > 0:
            if stage2_tor == 0:
                print("\n❌ 问题发现：TOR标签在阶段1->2之间丢失")
                print("   检查 supplement_micro_samples 中的标签处理逻辑")
            elif stage2_tor > 0 and len([c for c in stage3_counts.values() if c > stage2_tor * 0.8]) == 0:
                print("\n❌ 问题发现：TOR标签在阶段2->3之间丢失")  
                print("   检查标签编码和映射逻辑")
        
    except Exception as e:
        print(f"检查过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()