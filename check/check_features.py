#!/usr/bin/env python
"""
特征质量诊断脚本 - 检查当前特征是否适合VPN检测
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

def diagnose_feature_quality(features_dir):
    """全面诊断特征质量"""
    
    print("=" * 60)
    print("VPN分类特征质量诊断报告")
    print("=" * 60)
    
    # 加载数据
    macro_bag = np.load(f"{features_dir}/macro_bag.npy")
    micro_seq = np.load(f"{features_dir}/micro_seq.npy") 
    labels = np.load(f"{features_dir}/labels.npy")
    
    print(f"数据规模:")
    print(f"  宏观特征: {macro_bag.shape}")
    print(f"  微观特征: {micro_seq.shape}")
    print(f"  标签分布: {np.bincount(labels)}")
    
    # 1. 基础统计分析
    vpn_mask = labels == 1
    nonvpn_mask = labels == 0
    
    print(f"\n1. 基础分布分析:")
    print(f"  VPN样本: {vpn_mask.sum()} ({vpn_mask.mean()*100:.1f}%)")
    print(f"  NONVPN样本: {nonvpn_mask.sum()} ({nonvpn_mask.mean()*100:.1f}%)")
    
    # 2. 宏观特征分析
    print(f"\n2. 宏观特征区分度分析:")
    macro_features = macro_bag.reshape(len(macro_bag), -1)
    
    feature_importance = []
    for i in range(macro_features.shape[1]):
        feature = macro_features[:, i]
        
        # 计算VPN和NONVPN的均值差异
        vpn_mean = feature[vpn_mask].mean()
        nonvpn_mean = feature[nonvpn_mask].mean()
        
        # 计算效应大小 (Cohen's d)
        pooled_std = np.sqrt(((feature[vpn_mask].var() + feature[nonvpn_mask].var()) / 2))
        cohens_d = abs(vpn_mean - nonvpn_mean) / (pooled_std + 1e-8)
        
        feature_importance.append({
            'feature_idx': i,
            'vpn_mean': vpn_mean,
            'nonvpn_mean': nonvpn_mean,
            'cohens_d': cohens_d,
            'difference_ratio': abs(vpn_mean - nonvpn_mean) / (abs(nonvpn_mean) + 1e-8)
        })
        
        print(f"  特征 {i}: VPN={vpn_mean:.3f}, NONVPN={nonvpn_mean:.3f}, "
              f"Cohen's d={cohens_d:.3f}")
    
    # 找出最有区分度的特征
    feature_importance.sort(key=lambda x: x['cohens_d'], reverse=True)
    print(f"\n最有区分度的3个宏观特征:")
    for i in range(min(3, len(feature_importance))):
        feat = feature_importance[i]
        print(f"  特征 {feat['feature_idx']}: Cohen's d = {feat['cohens_d']:.3f}")
    
    # 3. 简单分类器基线测试
    print(f"\n3. 简单分类器基线测试:")
    
    # 数据准备
    X = macro_features
    y = labels
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 测试不同分类器
    classifiers = {
        'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    }
    
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, y_proba)
        
        print(f"\n{name} 结果:")
        print(f"  AUC: {auc:.4f}")
        print(classification_report(y_test, y_pred, target_names=['NONVPN', 'VPN'], digits=3))
    
    # 4. 特征重要性分析 (基于随机森林)
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    
    print(f"\n4. 随机森林特征重要性排序:")
    importance_scores = rf.feature_importances_
    sorted_idx = np.argsort(importance_scores)[::-1]
    
    for i in range(min(10, len(sorted_idx))):
        idx = sorted_idx[i]
        print(f"  特征 {idx}: 重要性 = {importance_scores[idx]:.4f}")
    
    # 5. 微观特征简单分析
    print(f"\n5. 微观特征简单分析:")
    
    # 对微观序列进行简单统计
    micro_stats = []
    for i in range(micro_seq.shape[0]):
        seq = micro_seq[i]  # (seq_len, features)
        stats = {
            'mean': seq.mean(axis=0),
            'std': seq.std(axis=0),
            'max': seq.max(axis=0),
            'min': seq.min(axis=0)
        }
        micro_stats.append(np.concatenate([stats['mean'], stats['std'], stats['max'] - stats['min']]))
    
    micro_features = np.array(micro_stats)
    
    # 测试微观特征的区分度
    X_micro_train, X_micro_test, y_micro_train, y_micro_test = train_test_split(
        StandardScaler().fit_transform(micro_features), y, test_size=0.2, stratify=y, random_state=42
    )
    
    rf_micro = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf_micro.fit(X_micro_train, y_micro_train)
    y_micro_pred = rf_micro.predict(X_micro_test)
    y_micro_proba = rf_micro.predict_proba(X_micro_test)[:, 1]
    
    micro_auc = roc_auc_score(y_micro_test, y_micro_proba)
    print(f"微观特征统计 AUC: {micro_auc:.4f}")
    
    # 6. 诊断结论
    print(f"\n" + "=" * 60)
    print("诊断结论:")
    print("=" * 60)
    
    best_macro_auc = max([roc_auc_score(y_test, classifiers[name].predict_proba(X_test)[:, 1]) 
                         for name in classifiers.keys()])
    
    if best_macro_auc < 0.6:
        print("❌ 严重问题: 宏观特征区分度很低 (AUC < 0.6)")
        print("   建议: 重新设计特征提取，专门针对VPN特征")
    elif best_macro_auc < 0.7:
        print("⚠️  中等问题: 宏观特征有一定区分度但不够强")
        print("   建议: 优化现有特征或添加VPN特定特征")
    else:
        print("✅ 宏观特征质量良好")
        print("   问题可能在模型架构或训练策略")
    
    if micro_auc < 0.6:
        print("❌ 微观特征对VPN检测帮助很小")
        print("   建议: 减少对包序列的依赖，专注流级特征")
    
    max_cohens_d = max([feat['cohens_d'] for feat in feature_importance])
    if max_cohens_d < 0.2:
        print("❌ 所有特征的效应大小都很小 (Cohen's d < 0.2)")
        print("   这解释了为什么深度模型表现不佳")
    
    print(f"\n推荐行动:")
    if best_macro_auc < 0.6:
        print("1. 🔥 立即重新设计特征提取")
        print("2. 简化模型架构，使用简单分类器")
        print("3. 检查数据质量和标签准确性")
    else:
        print("1. 简化深度模型，减少过拟合") 
        print("2. 调整训练策略和正则化")
        print("3. 考虑集成方法")

if __name__ == "__main__":
    features_dir = "/data3/wsb_workspace/study/models/flowcls-test/features"
    diagnose_feature_quality(features_dir)