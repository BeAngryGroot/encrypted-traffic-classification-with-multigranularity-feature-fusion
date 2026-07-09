# models/macro_transformer.py
import math, torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# RMSNorm
# ============================================================
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        rms = (x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        return x / rms * self.w


# ============================================================
# 位置编码 (batch_first 版本)
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        # [1, max_len, d] 保证 batch_first 对齐
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: [B, L, D]
        return x + self.pe[:, :x.size(1), :]


# ============================================================
# 稳定版注意力池化 AttnPool
# ============================================================
class AttnPool(nn.Module):
    def __init__(self, d, temperature: float = 1.0):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d))
        self.temperature = temperature

    def forward(self, x, mask=None):
        # x: [B, L, D], mask: [B, L] (1=有效, 0=pad)
        s = x @ self.q
        if mask is not None:
            s = s.masked_fill(mask == 0, -1e4)  # ✅ 防止半精度溢出
        s = torch.clamp(s, min=-1e4, max=1e4)
        s = s - s.max(dim=1, keepdim=True).values  # 稳定 softmax
        w = torch.softmax(s / self.temperature, dim=1).unsqueeze(-1)
        return (x * w).sum(dim=1)


# ============================================================
# Transformer Encoder Block
# ============================================================
class TransformerEncoderBlock(nn.Module):
    """标准 Transformer Encoder，支持 key_padding_mask=True 表示 pad"""
    def __init__(self, d_model=64, nhead=4, dim_ff=128, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.ff1 = nn.Linear(d_model, dim_ff)
        self.ff2 = nn.Linear(dim_ff, d_model)
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.do = nn.Dropout(dropout)

    def forward(self, x, kpm=None):  # x:[B,L,D], kpm:[B,L] bool, True=pad
        attn, _ = self.mha(x, x, x, key_padding_mask=kpm)
        x = self.n1(x + self.do(attn))
        ff = self.ff2(F.relu(self.ff1(x)))
        x = self.n2(x + self.do(ff))
        return x


# ============================================================
# 宏观分支主干 MacroTransformer
# ============================================================
class MacroTransformer(nn.Module):
    """
    宏观分支 backbone:
    - 输入  : x [B, K, D_in]  (K 可以为 1)
    - 掩码  : mask [B, K]     (1=有效, 0=pad)
    - 输出  : pooled [B, D_model]
    """
    def __init__(self, d_in, d_model=64, n_layers=2, n_heads=4, dropout=0.1):
        super().__init__()
        self.emb = nn.Linear(d_in, d_model)
        self.pos = PositionalEncoding(d_model)
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_model * 2, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.pool = AttnPool(d_model)

    def forward(self, x, mask=None):
        # x: [B,K,D], mask: [B,K] (1=有效,0=pad)
        # ✅ 保证 mask 与序列维度一致
        if mask is not None and mask.size(1) != x.size(1):
            mask = mask.expand(-1, x.size(1))

        kpm = (mask == 0) if mask is not None else None  # True=pad

        x = self.emb(x)
        x = self.pos(x)

        for blk in self.blocks:
            x = blk(x, kpm)

        x = self.norm(x)
        out = self.pool(x, mask)
        return out
