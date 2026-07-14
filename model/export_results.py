#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在冻结的指定集合上评估模型；默认只评估 test，防止训练集结果混入论文。"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from classifier import ExperimentClassifier
from data.normalization import SequenceNormalizer
from data.splits import load_group_split
from model.task_labels import select_task_labels


def confusion_matrix(y_true, y_pred, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_value, predicted_value in zip(y_true, y_pred):
        matrix[int(true_value), int(predicted_value)] += 1
    return matrix


def classification_metrics(y_true, y_pred, num_classes, class_names):
    matrix = confusion_matrix(y_true, y_pred, num_classes)
    per_class, f1_values, supports = [], [], []
    for class_id in range(num_classes):
        true_positive = matrix[class_id, class_id]
        false_positive = matrix[:, class_id].sum() - true_positive
        false_negative = matrix[class_id, :].sum() - true_positive
        support = matrix[class_id, :].sum()
        precision = float(true_positive / (true_positive + false_positive)) if true_positive + false_positive else 0.0
        recall = float(true_positive / (true_positive + false_negative)) if true_positive + false_negative else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        per_class.append({"class_id": class_id, "class_name": class_names[class_id], "precision": precision, "recall": recall, "f1": f1, "support": int(support)})
        f1_values.append(f1)
        supports.append(support)
    supports = np.asarray(supports, dtype=np.float64)
    f1_values = np.asarray(f1_values, dtype=np.float64)
    return {
        "accuracy": float(np.trace(matrix) / matrix.sum()) if matrix.sum() else 0.0,
        "macro_f1": float(f1_values.mean()) if f1_values.size else 0.0,
        "weighted_f1": float(np.average(f1_values, weights=supports)) if supports.sum() else 0.0,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def load_task_features(features_dir: Path, task: str):
    packet_seq = np.load(features_dir / "packet_seq.npy")
    packet_mask = np.load(features_dir / "packet_mask.npy")
    burst_seq = np.load(features_dir / "burst_seq.npy")
    burst_mask = np.load(features_dir / "burst_mask.npy")
    primary = np.load(features_dir / "primary_labels.npy")
    secondary = np.load(features_dir / "secondary_labels.npy")
    sample_keys = np.load(features_dir / "sample_keys.npy", allow_pickle=True).astype(str)
    with (features_dir / "label_mappings.pkl").open("rb") as stream:
        mappings = pickle.load(stream)
    selection = select_task_labels(task, primary, secondary, mappings)
    keep = selection.keep_mask
    return packet_seq[keep], packet_mask[keep], burst_seq[keep], burst_mask[keep], selection.labels, sample_keys[keep], selection


def main() -> None:
    parser = argparse.ArgumentParser(description="在冻结测试集上导出论文评估结果")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--features_dir", required=True)
    parser.add_argument("--split_file", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--normalizer")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    task = checkpoint["classification_mode"]
    packet_seq, packet_mask, burst_seq, burst_mask, labels, sample_keys, selection = load_task_features(Path(args.features_dir), task)
    split = load_group_split(args.split_file)
    indices = getattr(split, args.split)

    normalizer_path = Path(args.normalizer) if args.normalizer else checkpoint_path.parent / "normalizer.json"
    normalizer = SequenceNormalizer.load(normalizer_path)
    packet_seq, burst_seq = normalizer.transform(packet_seq, packet_mask, burst_seq, burst_mask)
    packet_seq, packet_mask = packet_seq[indices], packet_mask[indices]
    burst_seq, burst_mask = burst_seq[indices], burst_mask[indices]
    labels, sample_keys = labels[indices], sample_keys[indices]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ExperimentClassifier(checkpoint["config"], selection.num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    predictions, probabilities = [], []
    with torch.no_grad():
        for start in range(0, len(labels), args.batch_size):
            stop = min(start + args.batch_size, len(labels))
            logits = model(
                torch.from_numpy(packet_seq[start:stop]).float().to(device),
                torch.from_numpy(burst_seq[start:stop]).float().to(device),
                torch.from_numpy(packet_mask[start:stop]).float().to(device),
                torch.from_numpy(burst_mask[start:stop]).float().to(device),
            )
            probability = torch.softmax(logits, dim=-1).cpu().numpy()
            probabilities.append(probability)
            predictions.extend(probability.argmax(axis=-1).tolist())
    probabilities_array = np.concatenate(probabilities, axis=0) if probabilities else np.empty((0, selection.num_classes))
    predictions_array = np.asarray(predictions, dtype=np.int64)
    metrics = {"evaluated_split": args.split, "sample_count": int(len(labels)), **classification_metrics(labels, predictions_array, selection.num_classes, selection.class_names)}
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_key", "true_id", "true_label", "pred_id", "pred_label", *[f"prob_{name}" for name in selection.class_names]])
        for key, true_value, predicted_value, probability in zip(sample_keys, labels, predictions_array, probabilities_array):
            writer.writerow([key, int(true_value), selection.class_names[int(true_value)], int(predicted_value), selection.class_names[int(predicted_value)], *probability.tolist()])
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
