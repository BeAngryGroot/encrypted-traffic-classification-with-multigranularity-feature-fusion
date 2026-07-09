#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model.py
------------------------------------------------------------
双分支特征提取 + 融合模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from macro_transformer import MacroTransformer   # 宏观分支 backbone + AttnPool
from mamba_branch     import MicroMambaBranch    # 微观分支 backbone + AttnPool
from fusion_head      import GatedFusion         # 门控融合


class TransformerEncoderBlock(nn.Module):
    """简化版 Transformer Encoder：供其它模块复用"""
    def __init__(self, d_model=64, nhead=4, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        self.ff1 = nn.Linear(d_model, dim_feedforward)
        self.ff2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.do = nn.Dropout(dropout)

    def forward(self, x, kpm=None):          # x:[B,L,D]  kpm:True 表示 pad
        attn, _ = self.self_attn(x, x, x, key_padding_mask=kpm)
        x = self.norm1(x + self.do(attn))
        ff = self.ff2(F.relu(self.ff1(x)))
        x = self.norm2(x + self.do(ff))
        return x


class DualBranchFlowClassifier(nn.Module):
    """
    输入:
      micro_seq : [B, T, D_micro]    micro_mask : [B, T]   (1=有效, 0=pad)
      macro_bag : [B, D_macro]       macro_mask : [B, 1]   (1=有效, 0=pad)
    输出:
      融合特征向量 [B, fusion_hidden]
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.micro = MicroMambaBranch(
            d_in=cfg.micro_d_in,
            d_model=cfg.micro_d_model,
            n_layers=cfg.micro_layers,
            d_state=64,
            dropout=cfg.dropout
        )

        self.macro = MacroTransformer(
            d_in=cfg.macro_d_in,
            d_model=cfg.macro_d_model,
            n_layers=cfg.macro_layers,
            n_heads=cfg.macro_heads,
            dropout=cfg.dropout
        )

        self.fusion = GatedFusion(
            d_micro=cfg.micro_d_model,
            d_macro=cfg.macro_d_model,
            d_hidden=cfg.fusion_hidden
        )

    @staticmethod
    def _to_bin_mask(mask):
        if mask is None:
            return None
        return (mask > 0).to(torch.float32)   # ⭐ 新增：把掩码统一成 0/1

    def forward_features(self,micro_seq, macro_bag, micro_mask=None,          macro_mask=None):  

        # ---- 统一掩码到 0/1 ----
        micro_mask = self._to_bin_mask(micro_mask)   # ⭐ 新增
        macro_mask = self._to_bin_mask(macro_mask)   # ⭐ 新增

        # 1️⃣ 微观分支
        p = self.micro(micro_seq, mask=micro_mask)           # [B, D_micro]

         # 2) 宏观分支：允许两种输入
        if macro_bag.dim() == 2:                         # [B, D] -> [B, 1, D]
            macro_seq = macro_bag.unsqueeze(1)
        elif macro_bag.dim() == 3:                       # [B, K, D]
            macro_seq = macro_bag
        else:
            raise ValueError(f"macro_bag shape not supported: {macro_bag.shape}")

        # 掩码：保持与 macro_seq 的时间维一致
        macro_seq_len = macro_seq.size(1)
        if macro_mask is None:
            macro_mask_seq = None
        else:
            # 允许 [B,1] 或 [B,K]
            if macro_mask.dim() != 2:
                raise ValueError(f"macro_mask must be [B, L], got {macro_mask.shape}")
            if macro_mask.size(1) == 1 and macro_seq_len > 1:
                macro_mask_seq = macro_mask.expand(-1, macro_seq_len).contiguous()  # [B,K]
            elif macro_mask.size(1) == macro_seq_len:
                macro_mask_seq = macro_mask
            else:
                raise ValueError(f"macro_mask length {macro_mask.size(1)} "
                                f"!= macro_seq length {macro_seq_len}")

        # 3) 宏观分支前向（内部会把 1/0 转成 key_padding_mask）
        g = self.macro(macro_seq, mask=macro_mask_seq)   # [B, D_macro]
       

        # 3️⃣ 融合
        z = self.fusion(p, g)                                # [B, fusion_hidden]
        return z

    def forward(self, micro_seq, macro_bag,
                micro_mask=None, macro_mask=None):
        return self.forward_features(micro_seq, macro_bag,
                                     micro_mask, macro_mask)
