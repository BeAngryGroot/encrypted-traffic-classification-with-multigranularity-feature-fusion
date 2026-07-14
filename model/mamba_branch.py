"""包级微观分支：正式实验使用官方 Mamba2，smoke 可显式允许轻量回退。"""

from __future__ import annotations

import torch
import torch.nn as nn


try:
    from mamba_ssm import Mamba2
    OFFICIAL_MAMBA_AVAILABLE = True
    _MAMBA_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # CUDA/编译版本不匹配也应视为官方实现不可用。
    Mamba2 = None
    OFFICIAL_MAMBA_AVAILABLE = False
    _MAMBA_IMPORT_ERROR = exc


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dimension))

    def forward(self, inputs):
        rms = (inputs.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        return inputs / rms * self.weight


class SimpleSSMBlock(nn.Module):
    """仅用于快速跑通数据与张量契约，不作为论文正式实验结果。"""

    def __init__(self, d_model: int, d_state: int = 64, conv_kernel: int = 3, dropout: float = 0.1):
        super().__init__()
        self.depthwise = nn.Conv1d(d_model, d_model, conv_kernel, groups=d_model, padding=(conv_kernel - 1) // 2)
        self.input_projection = nn.Linear(d_model, 2 * d_model)
        self.A = nn.Parameter(torch.randn(d_state, d_state) * 0.01)
        self.B = nn.Parameter(torch.randn(d_state, d_model) * 0.01)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.output_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = RMSNorm(d_model)

    def forward(self, inputs, mask=None):
        residual = inputs
        values = self.depthwise(self.norm(inputs).transpose(1, 2)).transpose(1, 2)
        update, gate = self.input_projection(values).chunk(2, dim=-1)
        gate = torch.sigmoid(gate)
        batch, length, _ = update.shape
        state = torch.zeros(batch, self.A.size(0), device=values.device, dtype=values.dtype)
        outputs = []
        if mask is not None and mask.size(1) == 1 and length > 1:
            mask = mask.expand(-1, length)
        for index in range(length):
            state = state @ self.A.T + update[:, index, :] @ self.B.T
            current = state @ self.C.T
            if mask is not None:
                current = current * mask[:, index].unsqueeze(-1)
            outputs.append(current)
        values = torch.stack(outputs, dim=1) * gate
        return residual + self.dropout(self.output_projection(values))


class AttentionPool(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dimension))

    def forward(self, inputs, mask=None):
        scores = inputs @ self.query
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        scores = scores - scores.max(dim=1, keepdim=True).values
        weights = torch.softmax(scores.clamp(-1e4, 1e4), dim=1).unsqueeze(-1)
        return (inputs * weights).sum(dim=1)


class MambaStack(nn.Module):
    def __init__(self, d_model: int, n_layers: int, d_state: int, dropout: float, require_official: bool):
        super().__init__()
        if require_official and not OFFICIAL_MAMBA_AVAILABLE:
            raise RuntimeError(
                "正式实验要求官方 Mamba2，但 mamba_ssm 当前不可用。请检查 CUDA、PyTorch 与 mamba-ssm 安装。"
            ) from _MAMBA_IMPORT_ERROR
        self.implementation_name = "mamba_ssm.Mamba2" if OFFICIAL_MAMBA_AVAILABLE else "SimpleSSMBlock(smoke-only)"
        if OFFICIAL_MAMBA_AVAILABLE:
            factory = lambda: Mamba2(d_model=d_model, d_state=d_state, expand=2)
        else:
            factory = lambda: SimpleSSMBlock(d_model, d_state=d_state, dropout=dropout)
        self.blocks = nn.ModuleList([factory() for _ in range(n_layers)])
        self.norm = RMSNorm(d_model)

    def forward(self, inputs, mask=None):
        values = inputs
        for block in self.blocks:
            if isinstance(block, SimpleSSMBlock):
                values = block(values, mask)
            else:
                # 官方 Mamba2 不接收 padding mask；每层前后清零，最终池化仍使用 mask。
                if mask is not None:
                    values = values * mask.unsqueeze(-1)
                values = block(values)
        return self.norm(values)


class MicroMambaBranch(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_model: int = 256,
        n_layers: int = 3,
        d_state: int = 64,
        dropout: float = 0.1,
        require_official: bool = True,
    ):
        super().__init__()
        self.embedding = nn.Linear(d_in, d_model)
        self.backbone = MambaStack(d_model, n_layers, d_state, dropout, require_official)
        self.pool = AttentionPool(d_model)
        self.implementation_name = self.backbone.implementation_name

    def forward(self, inputs, mask=None):
        values = self.embedding(inputs)
        values = self.backbone(values, mask)
        return self.pool(values, mask)
