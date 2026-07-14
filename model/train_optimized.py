#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""使用冻结分组划分训练 Mamba + burst Transformer 双分支分类器。"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import pickle
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from classifier import ExperimentClassifier
from config import ModelConfig
from data.burst_features import BURST_FEATURES, PACKET_FEATURES
from data.normalization import SequenceNormalizer
from data.splits import create_group_split, load_group_split, save_group_split
from model.task_labels import select_task_labels


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        log_prob = F.log_softmax(inputs.float(), dim=1)
        log_pt = log_prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp().clamp(1e-7, 1.0 - 1e-7)
        weight = self.alpha.gather(0, targets) if self.alpha is not None else 1.0
        return (-weight * (1 - pt).pow(self.gamma) * log_pt).mean()


class FlowFeatureDataset(Dataset):
    def __init__(self, packet_seq, packet_mask, burst_seq, burst_mask, labels, indices):
        indices = np.asarray(indices, dtype=np.int64)
        self.packet_seq = torch.from_numpy(np.asarray(packet_seq[indices])).float()
        self.packet_mask = torch.from_numpy(np.asarray(packet_mask[indices])).float()
        self.burst_seq = torch.from_numpy(np.asarray(burst_seq[indices])).float()
        self.burst_mask = torch.from_numpy(np.asarray(burst_mask[indices])).float()
        self.labels = torch.from_numpy(np.asarray(labels[indices])).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "packet_seq": self.packet_seq[index],
            "packet_mask": self.packet_mask[index],
            "burst_seq": self.burst_seq[index],
            "burst_mask": self.burst_mask[index],
            "labels": self.labels[index],
        }


def load_features(features_dir: str | Path):
    root = Path(features_dir)
    required = ["packet_seq.npy", "packet_mask.npy", "burst_seq.npy", "burst_mask.npy", "primary_labels.npy", "secondary_labels.npy", "group_ids.npy"]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"特征目录缺少文件：{', '.join(missing)}")
    packet_seq = np.load(root / "packet_seq.npy", mmap_mode="r")
    packet_mask = np.load(root / "packet_mask.npy", mmap_mode="r")
    burst_seq = np.load(root / "burst_seq.npy", mmap_mode="r")
    burst_mask = np.load(root / "burst_mask.npy", mmap_mode="r")
    if burst_seq.ndim != 3:
        raise ValueError("正式 Transformer 分支要求 burst_seq 形状为 [N,K,D]，不接受单个全局向量")
    primary = np.load(root / "primary_labels.npy")
    secondary = np.load(root / "secondary_labels.npy")
    groups = np.load(root / "group_ids.npy", allow_pickle=True).astype(str)
    with (root / "label_mappings.pkl").open("rb") as stream:
        mappings = pickle.load(stream)
    return packet_seq, packet_mask, burst_seq, burst_mask, primary, secondary, groups, mappings


def classification_scores(y_true, y_pred, num_classes):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    f1_values, supports = [], []
    for class_id in range(num_classes):
        true_positive = np.sum((y_true == class_id) & (y_pred == class_id))
        false_positive = np.sum((y_true != class_id) & (y_pred == class_id))
        false_negative = np.sum((y_true == class_id) & (y_pred != class_id))
        support = np.sum(y_true == class_id)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        supports.append(support)
    f1_values = np.asarray(f1_values, dtype=np.float64)
    supports = np.asarray(supports, dtype=np.float64)
    return {
        "accuracy": float(np.mean(y_true == y_pred)) if len(y_true) else 0.0,
        "macro_f1": float(f1_values.mean()) if len(f1_values) else 0.0,
        "weighted_f1": float(np.average(f1_values, weights=supports)) if supports.sum() else 0.0,
    }


def make_class_weights(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = np.zeros(num_classes, dtype=np.float32)
    nonzero = counts > 0
    weights[nonzero] = counts.sum() / (num_classes * counts[nonzero])
    return torch.tensor(weights)


def move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def run_epoch(model, loader, device, loss_fn, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss, labels_all, predictions_all = 0.0, [], []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(batch["packet_seq"], batch["burst_seq"], batch["packet_mask"], batch["burst_mask"])
                loss = loss_fn(logits, batch["labels"])
            if training:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            total_loss += float(loss.item()) * len(batch["labels"])
            labels_all.extend(batch["labels"].detach().cpu().tolist())
            predictions_all.extend(logits.argmax(dim=-1).detach().cpu().tolist())
    return total_loss / max(1, len(labels_all)), labels_all, predictions_all


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练论文双分支加密流量分类模型")
    parser.add_argument("--features_dir", required=True)
    parser.add_argument("--split_file", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--classification_mode", default="application8", choices=["application8", "tor_binary"])
    parser.add_argument("--fusion_mode", default="gated", choices=["gated", "concat", "fixed", "micro_only", "burst_only"])
    parser.add_argument("--fixed_fusion_weight", type=float, default=0.5)
    parser.add_argument("--loss", default="focal", choices=["focal", "cross_entropy"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--require_official_mamba", action="store_true")
    parser.add_argument("--micro_d_model", type=int, default=384)
    parser.add_argument("--micro_layers", type=int, default=4)
    parser.add_argument("--d_state", type=int, default=128)
    parser.add_argument("--macro_d_model", type=int, default=96)
    parser.add_argument("--macro_layers", type=int, default=3)
    parser.add_argument("--macro_heads", type=int, default=6)
    parser.add_argument("--fusion_hidden", type=int, default=192)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    packet_seq, packet_mask, burst_seq, burst_mask, primary, secondary, groups, mappings = load_features(args.features_dir)
    selection = select_task_labels(args.classification_mode, primary, secondary, mappings)
    keep = selection.keep_mask
    packet_seq, packet_mask = packet_seq[keep], packet_mask[keep]
    burst_seq, burst_mask = burst_seq[keep], burst_mask[keep]
    groups = groups[keep]
    labels = selection.labels

    split_path = Path(args.split_file)
    if split_path.exists():
        split = load_group_split(split_path)
    else:
        split = create_group_split(labels, groups, args.val_ratio, args.test_ratio, args.seed)
        save_group_split(split, split_path, labels=labels, groups=groups)
    if not len(split.train) or not len(split.val) or not len(split.test):
        raise ValueError("冻结划分中 train/val/test 均必须非空")

    normalizer = SequenceNormalizer(PACKET_FEATURES, BURST_FEATURES)
    normalizer.fit(packet_seq, packet_mask, burst_seq, burst_mask, train_indices=split.train)
    packet_seq, burst_seq = normalizer.transform(packet_seq, packet_mask, burst_seq, burst_mask)
    normalizer.save(run_dir / "normalizer.json")

    train_set = FlowFeatureDataset(packet_seq, packet_mask, burst_seq, burst_mask, labels, split.train)
    val_set = FlowFeatureDataset(packet_seq, packet_mask, burst_seq, burst_mask, labels, split.val)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_set, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    cfg = ModelConfig()
    cfg.micro_d_in, cfg.macro_d_in = int(packet_seq.shape[-1]), int(burst_seq.shape[-1])
    cfg.micro_d_model, cfg.micro_layers, cfg.d_state = args.micro_d_model, args.micro_layers, args.d_state
    cfg.macro_d_model, cfg.macro_layers, cfg.macro_heads = args.macro_d_model, args.macro_layers, args.macro_heads
    cfg.fusion_hidden, cfg.fusion_mode = args.fusion_hidden, args.fusion_mode
    cfg.fixed_fusion_weight = args.fixed_fusion_weight
    cfg.require_official_mamba = args.require_official_mamba
    cfg.num_classes = selection.num_classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ExperimentClassifier(cfg, selection.num_classes).to(device)
    class_weights = make_class_weights(labels[split.train], selection.num_classes).to(device)
    loss_fn = FocalLoss(class_weights, cfg.gamma) if args.loss == "focal" else nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = GradScaler(enabled=device.type == "cuda")

    history = []
    best_macro_f1 = -1.0
    stale_epochs = 0
    checkpoint_path = run_dir / "best_model.pt"
    mamba_implementation = model.feature_extractor.micro.implementation_name
    logger.info("task=%s classes=%s Mamba=%s", args.classification_mode, selection.class_names, mamba_implementation)

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_loss, train_true, train_pred = run_epoch(model, train_loader, device, loss_fn, optimizer, scaler)
        val_loss, val_true, val_pred = run_epoch(model, val_loader, device, loss_fn)
        train_scores = classification_scores(train_true, train_pred, selection.num_classes)
        val_scores = classification_scores(val_true, val_pred, selection.num_classes)
        scheduler.step()
        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **{f"train_{key}": value for key, value in train_scores.items()}, **{f"val_{key}": value for key, value in val_scores.items()}, "seconds": time.time() - started}
        history.append(record)
        (run_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("epoch=%03d train_loss=%.4f val_loss=%.4f val_macro_f1=%.4f", epoch, train_loss, val_loss, val_scores["macro_f1"])

        # 主任务类别不平衡明显，统一按验证集 Macro-F1 选模，而不是总体准确率或 Weighted-F1。
        if val_scores["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_scores["macro_f1"]
            stale_epochs = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "classification_mode": args.classification_mode,
                "class_names": selection.class_names,
                "num_classes": selection.num_classes,
                "best_macro_f1": best_macro_f1,
                "mamba_implementation": mamba_implementation,
            }, checkpoint_path)
            (run_dir / "metrics.json").write_text(json.dumps({"selection_split": "val", "best_epoch": epoch, **val_scores}, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            stale_epochs += 1
            if stale_epochs >= cfg.patience:
                break
        if epoch % 5 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    logger.info("训练完成：best_macro_f1=%.4f checkpoint=%s", best_macro_f1, checkpoint_path)


if __name__ == "__main__":
    main()
