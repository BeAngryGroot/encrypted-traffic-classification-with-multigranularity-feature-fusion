from __future__ import annotations

import torch.nn as nn

from model import DualBranchFlowClassifier


class ExperimentClassifier(nn.Module):
    """Task-specific classifier head on top of the dual-branch feature extractor."""

    def __init__(self, config, num_classes: int):
        super().__init__()
        self.feature_extractor = DualBranchFlowClassifier(config)
        hidden = max(1, config.fusion_hidden // 2)
        self.head = nn.Sequential(
            nn.Linear(config.fusion_hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, packet_seq, burst_seq, packet_mask=None, burst_mask=None):
        z = self.feature_extractor.forward_features(packet_seq, burst_seq, packet_mask, burst_mask)
        return self.head(z)
