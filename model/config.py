#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
config.py
------------------------------------------------------------
模型与数据配置文件
✅ 已与 train_optimized_fixed_stable.py / macro_transformer.py / mamba_branch.py 对齐
✅ 支持双分支结构 (Mamba + Transformer)
✅ 支持 binary / primary / secondary / combined 四种模式
"""

class ModelConfig:
    """模型结构与训练参数配置"""

    def __init__(self):
        # =====================================================
        # === 输入维度（来自 features_test.py） ===
        # =====================================================
        self.micro_d_in = 16        # packet_seq 每个包 token 的特征维度
        self.macro_d_in = 12        # burst_seq 每个突发段 token 的特征维度

        # =====================================================
        # === Micro 分支（Mamba 序列建模） ===
        # =====================================================
        self.micro_d_model = 384    # Mamba 隐层维度 (建议 256/384/512)
        self.micro_layers = 4       # Mamba 堆叠层数
        self.d_state = 128          # SSM 状态维度 (控制时序记忆长度)
        self.dropout = 0.2          # Dropout 防止过拟合

        # =====================================================
        # === Macro 分支（Transformer 序列建模） ===
        # =====================================================
        self.macro_d_model = 96     # Transformer 隐层维度
        self.macro_layers = 3       # Transformer Encoder 层数
        self.macro_heads = 6        # 多头注意力头数

        # =====================================================
        # === 融合层 (Gated Fusion) ===
        # =====================================================
        self.fusion_hidden = 192    # 融合后隐藏层维度，用于整合 micro/macro 输出
        self.fusion_mode = "gated"  # gated / concat / fixed / micro_only / burst_only
        self.fixed_fusion_weight = 0.5

        # =====================================================
        # === 分类相关 ===
        # =====================================================
        self.num_classes = 10       # 会由训练脚本自动覆盖
        self.learning_rate = 2e-4   # 默认学习率
        self.batch_size = 32        # 训练 batch
        self.epochs = 80            # 最大 epoch 数

        # =====================================================
        # === 训练优化相关 ===
        # =====================================================
        self.weight_decay = 3e-4    # AdamW 正则
        self.gamma = 1.0            # FocalLoss gamma
        self.warmup_epochs = 5      # CosineLR 预热轮数
        self.patience = 10          # EarlyStopping 容忍次数

class DataConfig:
    """数据与特征结构配置"""
    def __init__(self):
        # === 微观特征 ===
        self.max_seq_len = 64       # 包序列长度
        self.max_burst_len = 32     # 突发段序列长度
        self.feature_dim_micro = 16 # 每个包特征维度

        # === 宏观特征 ===
        self.feature_dim_macro = 12 # 每个突发段 token 特征维度
        self.alpha = 1.0            # 自适应同向突发段阈值系数
