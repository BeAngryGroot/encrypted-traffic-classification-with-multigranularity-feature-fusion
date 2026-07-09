#!/usr/bin/env python
"""
check_label_distribution.py - 检查标签分布
"""
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import pickle

def check_final_features(features_dir):
    """检查最终特征文件的标签分布"""
    features_path = Path(features_dir)
    
    print("=" * 60)
    print("检查最终特征数据的标签分布")
    print("=" * 60)
    
    # 加载标签
    primary_labels = np.load(features_path / "primary_labels.npy")
    
    # 加载标签映射
    with open(features_path / "label_mappings.pkl", "rb") as f:
        label_mappings = pickle.load(f)
    
    # 统计分布
    primary_counts = Counter(primary_labels)
    primary_mapping = label_mappings['primary']
    
    print(f"总样本数: {len(primary_labels):,}")
    print(f"类别分布:")
    for label_id, count in primary_counts.items():
        label_name = primary_mapping['labels'][label_id]
        percentage = count / len(primary_labels) * 100
        print(f"  {label_name}: {count:,} 样本 ({percentage:.2f}%)")
    
    return primary_counts, primary_mapping

def check_extended_flows(features_dir):
    """检查扩展流文件的标签分布"""
    features_path = Path(features_dir)
    extended_flows_path = features_path / "extended_flows.csv"
    
    if not extended_flows_path.exists():
        print("未找到 extended_flows.csv 文件")
        return None
    
    print("\n" + "=" * 60)
    print("检查扩展流数据的标签分布")
    print("=" * 60)
    
    df = pd.read_csv(extended_flows_path)
    print(f"扩展流数据总行数: {len(df):,}")
    
    if 'class_label' in df.columns:
        class_counts = df['class_label'].value_counts()
        print(f"class_label分布:")
        for label, count in class_counts.items():
            percentage = count / len(df) * 100
            print(f"  {label}: {count:,} 样本 ({percentage:.2f}%)")
    
    if 'activity_label' in df.columns:
        activity_counts = df['activity_label'].value_counts()
        print(f"\nactivity_label分布 (前10个):")
        for label, count in activity_counts.head(10).items():
            percentage = count / len(df) * 100
            print(f"  {label}: {count:,} 样本 ({percentage:.2f}%)")
    
    return df

def check_source_csv_files(input_dir):
    """检查源CSV文件的标签分布"""
    input_path = Path(input_dir)
    
    print("\n" + "=" * 60)
    print("检查源CSV文件的标签分布")
    print("=" * 60)
    
    all_class_labels = []
    all_activity_labels = []
    file_count = 0
    
    # 查找所有流级CSV文件
    for csv_file in input_path.rglob("*.csv"):
        if csv_file.name.endswith(".pkt.csv"):
            continue
            
        try:
            df = pd.read_csv(csv_file)
            if len(df) == 0:
                continue
                
            file_count += 1
            
            if 'class_label' in df.columns:
                all_class_labels.extend(df['class_label'].tolist())
            
            if 'activity_label' in df.columns:
                all_activity_labels.extend(df['activity_label'].tolist())
            
            # 显示前几个文件的信息
            if file_count <= 5:
                print(f"\n文件: {csv_file.name}")
                print(f"  样本数: {len(df)}")
                if 'class_label' in df.columns:
                    class_dist = df['class_label'].value_counts()
                    print(f"  class_label: {dict(class_dist)}")
                    
        except Exception as e:
            print(f"读取文件出错 {csv_file.name}: {e}")
            continue
    
    print(f"\n处理了 {file_count} 个CSV文件")
    
    if all_class_labels:
        class_counts = Counter(all_class_labels)
        total = len(all_class_labels)
        print(f"\n源数据总样本数: {total:,}")
        print(f"class_label分布:")
        for label, count in class_counts.items():
            percentage = count / total * 100
            print(f"  {label}: {count:,} 样本 ({percentage:.2f}%)")
    
    if all_activity_labels:
        activity_counts = Counter(all_activity_labels)
        print(f"\nactivity_label分布 (前10个):")
        for label, count in Counter(all_activity_labels).most_common(10):
            percentage = count / len(all_activity_labels) * 100
            print(f"  {label}: {count:,} 样本 ({percentage:.2f}%)")
    
    return all_class_labels, all_activity_labels

def main():
    # 配置路径
    features_dir = "/data3/wsb_workspace/study/models/flowcls-test/features_supplemented"
    input_dir = "/data3/wsb_workspace/study/data/data_test/data_vpn_flow"
    
    print("标签分布检查工具")
    
    try:
        # 1. 检查源CSV文件
        source_class, source_activity = check_source_csv_files(input_dir)
        
        # 2. 检查扩展流数据
        extended_df = check_extended_flows(features_dir)
        
        # 3. 检查最终特征数据
        final_counts, final_mapping = check_final_features(features_dir)
        
        # 4. 对比分析
        print("\n" + "=" * 60)
        print("对比分析")
        print("=" * 60)
        
        if source_class:
            source_tor = source_class.count('TOR')
            source_nontor = source_class.count('NONTOR')
            print(f"源数据: TOR={source_tor:,}, NONTOR={source_nontor:,}")
        
        if final_counts and final_mapping:
            final_tor = 0
            final_nontor = 0
            for label_id, count in final_counts.items():
                label_name = final_mapping['labels'][label_id]
                if label_name == 'TOR':
                    final_tor = count
                elif label_name == 'NONTOR':
                    final_nontor = count
            
            print(f"最终数据: TOR={final_tor:,}, NONTOR={final_nontor:,}")
            
            if source_class:
                print(f"\n数据损失:")
                print(f"  TOR: {source_tor} -> {final_tor} (损失 {source_tor - final_tor})")
                print(f"  NONTOR: {source_nontor} -> {final_nontor} (损失 {source_nontor - final_nontor})")
        
    except Exception as e:
        print(f"检查过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()