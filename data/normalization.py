"""序列特征归一化：只用训练集有效 token 拟合统计量。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np


class SequenceNormalizer:
    DEFAULT_PACKET_CONTINUOUS = {
        "packet_length", "payload_length", "signed_length", "iat", "log_iat",
        "ip_ttl", "pos_in_burst", "burst_size", "burst_bytes", "burst_duration",
    }
    DEFAULT_BURST_CONTINUOUS = {
        "packet_count", "byte_sum", "duration", "mean_len", "std_len",
        "mean_iat", "std_iat", "max_iat", "gap_to_previous_burst",
    }

    def __init__(self, packet_feature_names: Sequence[str], burst_feature_names: Sequence[str]):
        self.packet_feature_names = list(packet_feature_names)
        self.burst_feature_names = list(burst_feature_names)
        self.packet_normalized_names = [name for name in self.packet_feature_names if name in self.DEFAULT_PACKET_CONTINUOUS]
        self.burst_normalized_names = [name for name in self.burst_feature_names if name in self.DEFAULT_BURST_CONTINUOUS]
        self.packet_mean = np.zeros(len(self.packet_feature_names), dtype=np.float32)
        self.packet_std = np.ones(len(self.packet_feature_names), dtype=np.float32)
        self.burst_mean = np.zeros(len(self.burst_feature_names), dtype=np.float32)
        self.burst_std = np.ones(len(self.burst_feature_names), dtype=np.float32)

    @staticmethod
    def _statistics(sequence: np.ndarray, mask: np.ndarray, indices: np.ndarray, selected_columns: list[int]) -> tuple[np.ndarray, np.ndarray]:
        selected = np.asarray(sequence)[indices]
        selected_mask = np.asarray(mask)[indices].astype(bool)
        tokens = selected[selected_mask]
        if tokens.size == 0:
            raise ValueError("训练划分没有有效 token，无法拟合归一化参数")
        mean = np.zeros(tokens.shape[-1], dtype=np.float64)
        std = np.ones(tokens.shape[-1], dtype=np.float64)
        if selected_columns:
            selected = tokens[:, selected_columns]
            mean[selected_columns] = selected.mean(axis=0, dtype=np.float64)
            selected_std = selected.std(axis=0, dtype=np.float64)
            std[selected_columns] = np.where(selected_std < 1e-8, 1.0, selected_std)
        return mean.astype(np.float32), std.astype(np.float32)

    def fit(self, packet_seq, packet_mask, burst_seq, burst_mask, *, train_indices: np.ndarray) -> "SequenceNormalizer":
        train_indices = np.asarray(train_indices, dtype=np.int64)
        packet_columns = [self.packet_feature_names.index(name) for name in self.packet_normalized_names]
        burst_columns = [self.burst_feature_names.index(name) for name in self.burst_normalized_names]
        self.packet_mean, self.packet_std = self._statistics(packet_seq, packet_mask, train_indices, packet_columns)
        self.burst_mean, self.burst_std = self._statistics(burst_seq, burst_mask, train_indices, burst_columns)
        return self

    def transform(self, packet_seq, packet_mask, burst_seq, burst_mask):
        packet_out = (np.asarray(packet_seq, dtype=np.float32) - self.packet_mean) / self.packet_std
        burst_out = (np.asarray(burst_seq, dtype=np.float32) - self.burst_mean) / self.burst_std
        # padding 必须恢复为零，否则模型会把归一化偏置当成真实 token。
        packet_out *= np.asarray(packet_mask, dtype=np.float32)[..., None]
        burst_out *= np.asarray(burst_mask, dtype=np.float32)[..., None]
        return packet_out.astype(np.float32), burst_out.astype(np.float32)

    def save(self, path: str | Path) -> None:
        payload = {
            "packet_feature_names": self.packet_feature_names,
            "burst_feature_names": self.burst_feature_names,
            "packet_normalized_names": self.packet_normalized_names,
            "burst_normalized_names": self.burst_normalized_names,
            "packet_mean": self.packet_mean.tolist(),
            "packet_std": self.packet_std.tolist(),
            "burst_mean": self.burst_mean.tolist(),
            "burst_std": self.burst_std.tolist(),
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SequenceNormalizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        normalizer = cls(payload["packet_feature_names"], payload["burst_feature_names"])
        normalizer.packet_normalized_names = payload.get("packet_normalized_names", normalizer.packet_normalized_names)
        normalizer.burst_normalized_names = payload.get("burst_normalized_names", normalizer.burst_normalized_names)
        for name in ("packet_mean", "packet_std", "burst_mean", "burst_std"):
            setattr(normalizer, name, np.asarray(payload[name], dtype=np.float32))
        return normalizer
