# models/fusion_head.py
import torch.nn as nn

class GatedFusion(nn.Module):
    def __init__(self, d_micro, d_macro, d_hidden):
        super().__init__()
        self.pn = nn.Linear(d_micro, d_hidden)
        self.gn = nn.Linear(d_macro, d_hidden)
        self.act = nn.Sigmoid()

    def forward(self, micro_vec, macro_vec):
        a = self.pn(micro_vec)
        b = self.gn(macro_vec)
        gate = self.act(a + b)  # 确保 a 和 b 维度一致

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
