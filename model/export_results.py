#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate a trained checkpoint on packet_seq + burst_seq feature files."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from classifier import ExperimentClassifier
from config import ModelConfig


def load_feature_arrays(features_dir: Path, mode: str):
    packet_seq = np.load(features_dir / "packet_seq.npy")
    packet_mask = np.load(features_dir / "packet_mask.npy")
    burst_seq = np.load(features_dir / "burst_seq.npy")
    burst_mask = np.load(features_dir / "burst_mask.npy")
    labels = np.load(features_dir / f"{mode}_labels.npy")
    sample_keys_path = features_dir / "sample_keys.npy"
    sample_keys = np.load(sample_keys_path, allow_pickle=True) if sample_keys_path.exists() else np.asarray([str(i) for i in range(len(labels))])
    with (features_dir / "label_mappings.pkl").open("rb") as f:
        label_mappings = pickle.load(f)
    return packet_seq, packet_mask, burst_seq, burst_mask, labels, sample_keys, label_mappings


def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(y_true, y_pred):
        cm[int(true), int(pred)] += 1
    return cm


def classification_metrics(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, num_classes)
    total = cm.sum()
    accuracy = float(np.trace(cm) / total) if total else 0.0
    per_class = []
    f1_values = []
    supports = []
    for cls in range(num_classes):
        tp = cm[cls, cls]
        fp = cm[:, cls].sum() - tp
        fn = cm[cls, :].sum() - tp
        support = cm[cls, :].sum()
        precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class.append({"class_id": cls, "precision": precision, "recall": recall, "f1": f1, "support": int(support)})
        f1_values.append(f1)
        supports.append(support)
    supports = np.asarray(supports, dtype=np.float64)
    f1_values = np.asarray(f1_values, dtype=np.float64)
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_values)) if f1_values.size else 0.0,
        "weighted_f1": float(np.average(f1_values, weights=supports)) if supports.sum() else 0.0,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


class Evaluator:
    def __init__(self, checkpoint_path, features_dir, output_dir):
        self.checkpoint_path = Path(checkpoint_path)
        self.features_dir = Path(features_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        self.mode = checkpoint.get("classification_mode", "combined")
        self.num_classes = int(checkpoint.get("num_classes", 1))
        self.class_names = checkpoint.get("class_names") or [f"class_{i}" for i in range(self.num_classes)]
        self.config = checkpoint.get("config", ModelConfig())
        self.model = ExperimentClassifier(self.config, self.num_classes).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.model.eval()

    def load_features(self):
        packet_seq, packet_mask, burst_seq, burst_mask, labels, sample_keys, _label_mappings = load_feature_arrays(self.features_dir, self.mode)
        self.packet_seq = torch.tensor(packet_seq, dtype=torch.float32, device=self.device)
        self.packet_mask = torch.tensor(packet_mask > 0, dtype=torch.float32, device=self.device)
        self.burst_seq = torch.tensor(burst_seq, dtype=torch.float32, device=self.device)
        self.burst_mask = torch.tensor(burst_mask > 0, dtype=torch.float32, device=self.device)
        self.labels = labels.astype(np.int64)
        self.sample_keys = sample_keys.astype(str)

    def evaluate(self):
        preds = []
        probs = []
        with torch.no_grad():
            for i in range(self.packet_seq.shape[0]):
                logits = self.model(
                    self.packet_seq[i : i + 1],
                    self.burst_seq[i : i + 1],
                    self.packet_mask[i : i + 1],
                    self.burst_mask[i : i + 1],
                )
                prob = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                probs.append(prob)
                preds.append(int(np.argmax(prob)))
        self.preds = np.asarray(preds, dtype=np.int64)
        self.probs = np.asarray(probs, dtype=np.float32)
        self.metrics = classification_metrics(self.labels, self.preds, self.num_classes)
        return self.metrics

    def save_results(self):
        metrics_path = self.output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(self.metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        np.save(self.output_dir / "confusion_matrix.npy", np.asarray(self.metrics["confusion_matrix"], dtype=np.int64))
        with (self.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_key", "true_id", "true_label", "pred_id", "pred_label", *[f"prob_{name}" for name in self.class_names]])
            for key, true, pred, prob in zip(self.sample_keys, self.labels, self.preds, self.probs):
                writer.writerow([key, int(true), self.class_names[int(true)], int(pred), self.class_names[int(pred)], *prob.tolist()])


def main():
    parser = argparse.ArgumentParser(description="Evaluate thesis classifier checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--features_dir", default="artifacts/features")
    parser.add_argument("--output_dir", default="artifacts/results")
    args = parser.parse_args()

    evaluator = Evaluator(args.checkpoint, args.features_dir, args.output_dir)
    evaluator.load_model()
    evaluator.load_features()
    metrics = evaluator.evaluate()
    evaluator.save_results()
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
