#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train the thesis dual-branch classifier on packet_seq + burst_seq features."""

from __future__ import annotations

import argparse
import gc
import logging
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from classifier import ExperimentClassifier
from config import ModelConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.5, reduction="mean", eps=1e-7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, inputs, targets):
        logits = inputs.float()
        logp = F.log_softmax(logits, dim=1)
        logpt = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = logpt.exp().clamp(self.eps, 1.0 - self.eps)
        at = self.alpha.gather(0, targets) if self.alpha is not None else 1.0
        loss = -at * (1 - pt).pow(self.gamma) * logpt
        return loss.mean() if self.reduction == "mean" else loss.sum()


class FlowFeatureDataset(Dataset):
    def __init__(self, packet_seq, packet_mask, burst_seq, burst_mask, labels):
        self.packet_seq = torch.from_numpy(np.asarray(packet_seq)).float()
        self.packet_mask = torch.from_numpy(np.asarray(packet_mask)).float()
        self.burst_seq = torch.from_numpy(np.asarray(burst_seq)).float()
        self.burst_mask = torch.from_numpy(np.asarray(burst_mask)).float()
        self.labels = torch.from_numpy(np.asarray(labels)).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "packet_seq": self.packet_seq[idx],
            "packet_mask": self.packet_mask[idx],
            "burst_seq": self.burst_seq[idx],
            "burst_mask": self.burst_mask[idx],
            "labels": self.labels[idx],
        }


def _load_array(features_dir: Path, preferred: str, fallback: str | None = None, mmap=True):
    preferred_path = features_dir / preferred
    if preferred_path.exists():
        return np.load(preferred_path, mmap_mode="r" if mmap else None)
    if fallback:
        fallback_path = features_dir / fallback
        if fallback_path.exists():
            logger.warning("Using legacy feature file %s; prefer %s", fallback_path.name, preferred)
            return np.load(fallback_path, mmap_mode="r" if mmap else None)
    raise FileNotFoundError(f"Missing feature file: {preferred_path}")


def load_features(features_dir: str | Path):
    p = Path(features_dir)
    logger.info("Loading feature files from: %s", p)
    packet_seq = _load_array(p, "packet_seq.npy", "micro_seq.npy")
    packet_mask = _load_array(p, "packet_mask.npy", "micro_mask.npy")
    burst_seq = _load_array(p, "burst_seq.npy", "macro_bag.npy")
    burst_mask = _load_array(p, "burst_mask.npy", "macro_mask.npy")
    if burst_seq.ndim == 2:
        logger.warning("burst_seq is [N,D], not [N,K,D]. Transformer will see one token only.")
        burst_seq = burst_seq[:, None, :]
        if burst_mask.ndim == 2 and burst_mask.shape[1] != 1:
            burst_mask = np.ones((burst_seq.shape[0], 1), dtype=np.float32)

    primary_labels = np.load(p / "primary_labels.npy")
    secondary_labels = np.load(p / "secondary_labels.npy")
    combined_labels = np.load(p / "combined_labels.npy")
    with (p / "label_mappings.pkl").open("rb") as f:
        label_mappings = pickle.load(f)
    return packet_seq, packet_mask, burst_seq, burst_mask, primary_labels, secondary_labels, combined_labels, label_mappings


def class_names(mapping):
    id_to_label = mapping.get("id_to_label", {})
    return [id_to_label[i] for i in range(mapping["num_classes"])]


def remap_contiguous(labels, names):
    unique = sorted(np.unique(labels).tolist())
    old_to_new = {old: new for new, old in enumerate(unique)}
    remapped = np.asarray([old_to_new[int(label)] for label in labels], dtype=np.int64)
    remapped_names = [names[old] for old in unique]
    return remapped, len(remapped_names), remapped_names


def select_labels(mode, primary_labels, secondary_labels, combined_labels, label_mappings):
    primary_map = label_mappings["primary"]["label_to_id"]
    encrypted_ids = [primary_map[x] for x in ["VPN", "TOR", "QUIC"] if x in primary_map]
    if mode == "binary":
        labels = np.where(np.isin(primary_labels, encrypted_ids), 1, 0).astype(np.int64)
        return labels, 2, ["OTHER", "ENCRYPTED"], np.ones_like(labels, dtype=bool)
    if mode == "primary":
        return primary_labels, label_mappings["primary"]["num_classes"], class_names(label_mappings["primary"]), np.ones_like(primary_labels, dtype=bool)

    keep = np.isin(primary_labels, encrypted_ids) if encrypted_ids else np.ones_like(primary_labels, dtype=bool)
    if mode == "secondary":
        names = class_names(label_mappings["secondary"])
        labels, num_classes, names = remap_contiguous(secondary_labels[keep], names)
        return labels, num_classes, names, keep
    names = class_names(label_mappings["combined"])
    labels, num_classes, names = remap_contiguous(combined_labels[keep], names)
    return labels, num_classes, names, keep


def make_class_weights(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = np.zeros((num_classes,), dtype=np.float32)
    total = counts.sum()
    nonzero = counts > 0
    weights[nonzero] = total / (num_classes * counts[nonzero])
    return torch.tensor(weights, dtype=torch.float32)


def stratified_split(labels, val_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []
    labels = np.asarray(labels)
    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        rng.shuffle(idx)
        val_count = int(round(len(idx) * val_ratio))
        if len(idx) > 1:
            val_count = max(1, min(val_count, len(idx) - 1))
        else:
            val_count = 0
        val_idx.extend(idx[:val_count].tolist())
        train_idx.extend(idx[val_count:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64)


def weighted_f1_score(y_true, y_pred, num_classes):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    f1_values = []
    weights = []
    for cls in range(num_classes):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        support = np.sum(y_true == cls)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1_values.append(f1)
        weights.append(support)
    weights = np.asarray(weights, dtype=np.float64)
    f1_values = np.asarray(f1_values, dtype=np.float64)
    macro = float(np.mean(f1_values)) if f1_values.size else 0.0
    weighted = float(np.average(f1_values, weights=weights)) if weights.sum() else 0.0
    return macro, weighted


def move_batch(batch, device):
    return {
        "packet_seq": batch["packet_seq"].to(device),
        "packet_mask": (batch["packet_mask"] > 0).to(device),
        "burst_seq": batch["burst_seq"].to(device),
        "burst_mask": (batch["burst_mask"] > 0).to(device),
        "labels": batch["labels"].to(device),
    }


def train_one_epoch(model, loader, optimizer, scaler, device, loss_fn):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    use_cuda = device.type == "cuda"
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_cuda):
            logits = model(batch["packet_seq"], batch["burst_seq"], batch["packet_mask"], batch["burst_mask"])
            loss = loss_fn(logits, batch["labels"])
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        preds = logits.argmax(dim=-1)
        total_loss += loss.item() * batch["labels"].size(0)
        total_correct += (preds == batch["labels"]).sum().item()
        total_samples += batch["labels"].size(0)
    return total_loss / total_samples, total_correct / total_samples


def validate_one_epoch(model, loader, device, loss_fn, num_classes):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    preds_all = []
    labels_all = []
    use_cuda = device.type == "cuda"
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            with autocast(device_type=device.type, enabled=use_cuda):
                logits = model(batch["packet_seq"], batch["burst_seq"], batch["packet_mask"], batch["burst_mask"])
                loss = loss_fn(logits, batch["labels"])
            preds = logits.argmax(dim=-1)
            total_loss += loss.item() * batch["labels"].size(0)
            total_correct += (preds == batch["labels"]).sum().item()
            total_samples += batch["labels"].size(0)
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(batch["labels"].cpu().numpy())
    macro_f1, weighted_f1 = weighted_f1_score(labels_all, preds_all, num_classes)
    return total_loss / total_samples, total_correct / total_samples, macro_f1, weighted_f1


def configure_model(cfg, packet_seq, burst_seq, num_classes, args):
    cfg.micro_d_in = int(packet_seq.shape[-1])
    cfg.macro_d_in = int(burst_seq.shape[-1])
    cfg.num_classes = int(num_classes)
    cfg.fusion_mode = args.fusion_mode
    cfg.fixed_fusion_weight = args.fixed_fusion_weight
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Train Mamba + Transformer classifier on thesis burst features.")
    parser.add_argument("--features_dir", default="artifacts/features")
    parser.add_argument("--checkpoints_dir", default="artifacts/checkpoints")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--classification_mode", default="combined", choices=["binary", "primary", "secondary", "combined"])
    parser.add_argument("--fusion_mode", default="gated", choices=["gated", "concat", "fixed", "micro_only", "burst_only"])
    parser.add_argument("--fixed_fusion_weight", type=float, default=0.5)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    packet_seq, packet_mask, burst_seq, burst_mask, primary_labels, secondary_labels, combined_labels, label_mappings = load_features(args.features_dir)
    labels, num_classes, names, keep = select_labels(args.classification_mode, primary_labels, secondary_labels, combined_labels, label_mappings)

    packet_seq = packet_seq[keep]
    packet_mask = packet_mask[keep]
    burst_seq = burst_seq[keep]
    burst_mask = burst_mask[keep]
    train_idx, val_idx = stratified_split(labels, val_ratio=args.val_ratio, seed=args.seed)
    if len(val_idx) == 0:
        raise ValueError("Validation split is empty; provide more samples or lower --val_ratio.")

    train_set = FlowFeatureDataset(packet_seq[train_idx], packet_mask[train_idx], burst_seq[train_idx], burst_mask[train_idx], labels[train_idx])
    val_set = FlowFeatureDataset(packet_seq[val_idx], packet_mask[val_idx], burst_seq[val_idx], burst_mask[val_idx], labels[val_idx])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_set, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = configure_model(ModelConfig(), packet_seq, burst_seq, num_classes, args)
    model = ExperimentClassifier(cfg, num_classes).to(device)
    class_weights = make_class_weights(labels[train_idx], num_classes).to(device)
    loss_fn = FocalLoss(alpha=class_weights, gamma=cfg.gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = GradScaler(enabled=device.type == "cuda")

    os.makedirs(args.checkpoints_dir, exist_ok=True)
    best_weighted_f1 = 0.0
    best_path = Path(args.checkpoints_dir) / f"best_{args.classification_mode}_{args.fusion_mode}.pt"
    patience = 0

    logger.info("[MODE] %s | classes=%s | samples=%d | packet=%s | burst=%s", args.classification_mode, names, len(labels), packet_seq.shape, burst_seq.shape)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scaler, device, loss_fn)
        val_loss, val_acc, val_macro_f1, val_weighted_f1 = validate_one_epoch(model, val_loader, device, loss_fn, num_classes)
        scheduler.step()
        elapsed = time.time() - t0
        logger.info(
            "Epoch %03d | Train %.4f/%.3f | Val %.4f/%.3f macro=%.3f weighted=%.3f | %.1fs",
            epoch,
            tr_loss,
            tr_acc,
            val_loss,
            val_acc,
            val_macro_f1,
            val_weighted_f1,
            elapsed,
        )
        if val_weighted_f1 > best_weighted_f1:
            best_weighted_f1 = val_weighted_f1
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_weighted_f1": best_weighted_f1,
                    "label_mappings": label_mappings,
                    "classification_mode": args.classification_mode,
                    "class_names": names,
                    "num_classes": num_classes,
                    "config": cfg,
                    "feature_schema": {
                        "packet_shape": list(packet_seq.shape),
                        "burst_shape": list(burst_seq.shape),
                    },
                },
                best_path,
            )
            logger.info("Saved best checkpoint: %s (weighted F1=%.4f)", best_path, best_weighted_f1)
            patience = 0
        else:
            patience += 1
            if patience >= cfg.patience:
                logger.info("Early stopping after %d stale epochs", cfg.patience)
                break
        if epoch % 5 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    logger.info("Training complete. Best weighted F1=%.4f | checkpoint=%s", best_weighted_f1, best_path)


if __name__ == "__main__":
    main()
