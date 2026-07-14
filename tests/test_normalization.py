from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.normalization import SequenceNormalizer  # noqa: E402


def test_normalizer_fits_only_training_tokens_and_preserves_padding():
    packet = np.asarray([[[1.0], [3.0]], [[1000.0], [0.0]]], dtype=np.float32)
    packet_mask = np.asarray([[1, 1], [1, 0]], dtype=np.float32)
    burst = np.asarray([[[2.0]], [[2000.0]]], dtype=np.float32)
    burst_mask = np.asarray([[1], [1]], dtype=np.float32)
    normalizer = SequenceNormalizer(packet_feature_names=["packet_length"], burst_feature_names=["byte_sum"])
    normalizer.fit(packet, packet_mask, burst, burst_mask, train_indices=np.asarray([0]))
    assert normalizer.packet_mean.tolist() == [2.0]
    assert normalizer.burst_mean.tolist() == [2.0]

    packet_out, burst_out = normalizer.transform(packet, packet_mask, burst, burst_mask)
    assert packet_out[1, 1, 0] == 0.0
    assert packet_out[0, :, 0].tolist() == [-1.0, 1.0]


def test_categorical_and_binary_columns_are_not_standardized():
    packet = np.asarray([[[1.0, -1.0], [3.0, 1.0]]], dtype=np.float32)
    mask = np.ones((1, 2), dtype=np.float32)
    burst = np.asarray([[[-1.0], [1.0]]], dtype=np.float32)
    normalizer = SequenceNormalizer(
        packet_feature_names=["packet_length", "direction"],
        burst_feature_names=["burst_direction"],
    )
    normalizer.fit(packet, mask, burst, mask, train_indices=np.asarray([0]))
    packet_out, burst_out = normalizer.transform(packet, mask, burst, mask)
    assert packet_out[0, :, 1].tolist() == [-1.0, 1.0]
    assert burst_out[0, :, 0].tolist() == [-1.0, 1.0]
