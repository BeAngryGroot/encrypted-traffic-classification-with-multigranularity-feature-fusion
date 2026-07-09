# models/fusion_head.py
import torch
import torch.nn as nn

class GatedFusion(nn.Module):
    def __init__(self, d_micro, d_macro, d_hidden, mode="gated", fixed_weight=0.5):
        super().__init__()
        self.mode = mode
        self.fixed_weight = float(fixed_weight)
        self.pn = nn.Linear(d_micro, d_hidden)
        self.gn = nn.Linear(d_macro, d_hidden)
        self.concat = nn.Linear(d_hidden * 2, d_hidden)
        self.act = nn.Sigmoid()
        self.last_gate = None

    def forward(self, micro_vec, macro_vec):
        a = self.pn(micro_vec)
        b = self.gn(macro_vec)
        if self.mode == "micro_only":
            self.last_gate = torch.ones_like(a)
            return a
        if self.mode == "burst_only":
            self.last_gate = torch.zeros_like(b)
            return b
        if self.mode == "concat":
            self.last_gate = None
            return self.concat(torch.cat([a, b], dim=-1))
        if self.mode == "fixed":
            gate = torch.full_like(a, self.fixed_weight)
        else:
            gate = self.act(a + b)
        self.last_gate = gate.detach()
        return gate * a + (1 - gate) * b


class ClassifierHead(nn.Module):
    def __init__(self, d_in, num_classes, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_in, num_classes)
        )

    def forward(self, x):
        return self.mlp(x)
