#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_optimized_fixed_stable.py
------------------------------------------------------------
✅ 双分支架构 (Transformer + Mamba)
✅ 支持 binary / primary / secondary / combined 四种模式
✅ FocalLoss 数值稳定版
✅ Cosine + Warmup 调度，EarlyStopping
✅ 梯度裁剪 + TF32 加速
✅ 极简日志：每个 epoch 仅一行输出
"""

import os, time, gc, pickle, argparse, logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast  # ⚠️ 使用 torch.amp 版本
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

from config import ModelConfig
from model import DualBranchFlowClassifier

# ----------------------- CUDA/性能 -----------------------
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------- FocalLoss -----------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.5, reduction='mean', eps=1e-7):
        super().__init__()
        self.alpha, self.gamma, self.reduction, self.eps = alpha, gamma, reduction, eps

    def forward(self, inputs, targets):
        logits = inputs.float()
        logp = F.log_softmax(logits, dim=1)                          # [B,C]
        logpt = logp.gather(1, targets.unsqueeze(1)).squeeze(1)      # [B]
        pt = logpt.exp().clamp(self.eps, 1.0 - self.eps)
        at = self.alpha.gather(0, targets) if self.alpha is not None else 1.0
        loss = -at * (1 - pt).pow(self.gamma) * logpt
        return loss.mean() if self.reduction == 'mean' else loss.sum()

# ----------------------- Dataset ------------------------
class MultilevelFlowDataset(Dataset):
    def __init__(self, macro_bag, macro_mask, micro_seq, micro_mask,
                 primary_labels=None, secondary_labels=None, combined_labels=None):
        self.macro_bag  = torch.from_numpy(macro_bag).float()
        self.macro_mask = torch.from_numpy(macro_mask).float()
        self.micro_seq  = torch.from_numpy(micro_seq).float()
        self.micro_mask = torch.from_numpy(micro_mask).float()
        self.primary_labels   = torch.from_numpy(primary_labels).long() if primary_labels is not None else None
        self.secondary_labels = torch.from_numpy(secondary_labels).long() if secondary_labels is not None else None
        self.combined_labels  = torch.from_numpy(combined_labels).long()  if combined_labels  is not None else None

    def __len__(self): return len(self.macro_bag)

    def __getitem__(self, idx):
        b = {
            "macro_bag":  self.macro_bag[idx],
            "macro_mask": self.macro_mask[idx],
            "micro_seq":  self.micro_seq[idx],
            "micro_mask": self.micro_mask[idx],
        }
        if self.primary_labels   is not None: b["primary_labels"]   = self.primary_labels[idx]
        if self.secondary_labels is not None: b["secondary_labels"] = self.secondary_labels[idx]
        if self.combined_labels  is not None: b["combined_labels"]  = self.combined_labels[idx]
        return b

# ----------------------- 模型包装 ------------------------
class LightweightDualBranchClassifier(nn.Module):
    def __init__(self, config, label_mappings):
        super().__init__()
        self.feature_extractor = DualBranchFlowClassifier(config)
        self.shared_head = nn.Sequential(
            nn.Linear(config.fusion_hidden, config.fusion_hidden // 2),
            nn.ReLU(), nn.Dropout(0.1)
        )
        self.binary_classifier    = nn.Linear(config.fusion_hidden // 2, 2)
        self.primary_classifier   = nn.Linear(config.fusion_hidden // 2, label_mappings['primary']['num_classes'])
        self.secondary_classifier = nn.Linear(config.fusion_hidden // 2, label_mappings['secondary']['num_classes'])
        self.combined_classifier  = nn.Linear(config.fusion_hidden // 2, label_mappings['combined']['num_classes'])
        self.classification_mode = 'primary'

    def set_classification_mode(self, mode): self.classification_mode = mode

    def forward(self, micro_seq, macro_bag, micro_mask, macro_mask):
        # ⚠️ 必须带 device_type，否则就会报你截图里的错
        with autocast('cuda'):
            z = self.feature_extractor.forward_features(micro_seq, macro_bag, micro_mask, macro_mask)
            h = self.shared_head(z)
            if   self.classification_mode == 'binary':    return self.binary_classifier(h)
            elif self.classification_mode == 'primary':   return self.primary_classifier(h)
            elif self.classification_mode == 'secondary': return self.secondary_classifier(h)
            else:                                         return self.combined_classifier(h)

# ----------------------- I/O -----------------------------
def load_features(features_dir: str):
    p = Path(features_dir)
    logger.info(f"加载特征文件自: {p}")
    macro_bag  = np.load(p / "macro_bag.npy",  mmap_mode='r')
    macro_mask = np.load(p / "macro_mask.npy", mmap_mode='r')
    micro_seq  = np.load(p / "micro_seq.npy",  mmap_mode='r')
    micro_mask = np.load(p / "micro_mask.npy", mmap_mode='r')
    primary_labels   = np.load(p / "primary_labels.npy")
    secondary_labels = np.load(p / "secondary_labels.npy")
    combined_labels  = np.load(p / "combined_labels.npy")
    with open(p / "label_mappings.pkl", "rb") as f:
        label_mappings = pickle.load(f)
    return macro_bag, macro_mask, micro_seq, micro_mask, primary_labels, secondary_labels, combined_labels, label_mappings

def make_class_weights(labels, num_classes):
    uniq = np.unique(labels)
    weights = compute_class_weight('balanced', classes=uniq, y=labels)
    full = np.zeros((num_classes,), dtype=np.float32); full[uniq] = weights
    return torch.tensor(full, dtype=torch.float32)

# ----------------------- Train/Valid ---------------------
def train_one_epoch(model, loader, optimizer, scaler, device, epoch, loss_fn, mode):
    model.train(); model.set_classification_mode(mode)
    total_loss = 0.0; total_correct = 0; total_samples = 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        x_micro = batch["micro_seq"].to(device)
        x_macro = batch["macro_bag"].to(device)
        m_micro = (batch["micro_mask"] > 0).to(device)
        m_macro = (batch["macro_mask"] > 0).to(device)

        # 正确写法：构造与 x_macro 匹配的 [B, K] 掩码（全 1，表示 5 段都有效）
        K = 5
        B = x_macro.size(0)

        # 1) 升维 macro 特征到 [B, K, D]
        x_macro = batch["macro_bag"].to(device)              # [B, D]
        x_macro = x_macro.unsqueeze(1).repeat(1, K, 1)       # [B, K, D]

        # 2) 直接用全 1 的 [B, K] 掩码（MacroTransformer 里会把 0 当 pad）
        m_macro = torch.ones(B, K, device=device)            # [B, K]



        labels  = batch[f"{'primary' if mode=='binary' else mode}_labels"].to(device)

        with autocast('cuda'):
            logits = model(x_micro, x_macro, m_micro, m_macro)
            loss   = loss_fn(logits, labels)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer); scaler.update()

        preds = logits.argmax(dim=-1)
        total_loss   += loss.item() * labels.size(0)
        total_correct+= (preds == labels).sum().item()
        total_samples+= labels.size(0)

    return total_loss / total_samples, total_correct / total_samples

def validate_one_epoch(model, loader, device, loss_fn, mode):
    model.eval(); model.set_classification_mode(mode)
    total_loss = 0.0; total_correct = 0; total_samples = 0
    preds_all, labels_all = [], []
    with torch.no_grad():
        for batch in loader:
            x_micro = batch["micro_seq"].to(device)
            x_macro = batch["macro_bag"].to(device)
            m_micro = (batch["micro_mask"] > 0).to(device)
            m_macro = (batch["macro_mask"] > 0).to(device)

            # 正确写法：构造与 x_macro 匹配的 [B, K] 掩码（全 1，表示 5 段都有效）
            K = 5
            B = x_macro.size(0)

            # 1) 升维 macro 特征到 [B, K, D]
            x_macro = batch["macro_bag"].to(device)              # [B, D]
            x_macro = x_macro.unsqueeze(1).repeat(1, K, 1)       # [B, K, D]

            # 2) 直接用全 1 的 [B, K] 掩码（MacroTransformer 里会把 0 当 pad）
            m_macro = torch.ones(B, K, device=device)            # [B, K]

            labels  = batch[f"{'primary' if mode=='binary' else mode}_labels"].to(device)

            with autocast('cuda'):
                logits = model(x_micro, x_macro, m_micro, m_macro)
                loss   = loss_fn(logits, labels)

            preds = logits.argmax(dim=-1)
            total_loss   += loss.item() * labels.size(0)
            total_correct+= (preds == labels).sum().item()
            total_samples+= labels.size(0)
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    f1 = f1_score(labels_all, preds_all, average='weighted', zero_division=0)
    return total_loss / total_samples, total_correct / total_samples, f1

# ----------------------- Main ----------------------------
def main():
    ap = argparse.ArgumentParser(description="最终稳定版双分支分类器")
    ap.add_argument("--features_dir", required=True)
    ap.add_argument("--checkpoints_dir", required=True)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--learning_rate", type=float, default=0.0002)
    ap.add_argument("--classification_mode", default="binary",
                    choices=['binary','primary','secondary','combined'])
    args = ap.parse_args()

    # 载入特征
    (macro_bag, macro_mask, micro_seq, micro_mask,
     primary_labels, secondary_labels, combined_labels, label_mappings) = load_features(args.features_dir)

    primary_map = label_mappings['primary']['label_to_id']
    encrypted_ids = [primary_map[x] for x in ['VPN','TOR','QUIC'] if x in primary_map]
    mode = args.classification_mode

    # 标签与筛选
    def apply_mask(mask, *arrays): return [a[mask] if a is not None else None for a in arrays]
    if mode == 'binary':
        labels = np.where(np.isin(primary_labels, encrypted_ids), 1, 0); num_classes = 2
    elif mode == 'primary':
        labels = primary_labels; num_classes = label_mappings['primary']['num_classes']
    elif mode == 'secondary':
        mask = np.isin(primary_labels, encrypted_ids)
        (macro_bag, macro_mask, micro_seq, micro_mask,
         primary_labels, secondary_labels, combined_labels) = apply_mask(mask, macro_bag, macro_mask, micro_seq, micro_mask, primary_labels, secondary_labels, combined_labels)
        labels = secondary_labels; num_classes = label_mappings['secondary']['num_classes']
    else:
        mask = np.isin(primary_labels, encrypted_ids)
        (macro_bag, macro_mask, micro_seq, micro_mask,
         primary_labels, secondary_labels, combined_labels) = apply_mask(mask, macro_bag, macro_mask, micro_seq, micro_mask, primary_labels, secondary_labels, combined_labels)
        labels = combined_labels; num_classes = label_mappings['combined']['num_classes']

    # 划分
    idx = np.arange(len(labels))
    train_idx, tmp_idx = train_test_split(idx, test_size=0.2, stratify=labels, random_state=42)
    val_idx, _ = train_test_split(tmp_idx, test_size=0.5, stratify=labels[tmp_idx], random_state=42)

    sub = lambda a, i: a[i] if a is not None else None
    train_set = MultilevelFlowDataset(sub(macro_bag, train_idx), sub(macro_mask, train_idx),
                                      sub(micro_seq, train_idx), sub(micro_mask, train_idx),
                                      sub(primary_labels, train_idx), sub(secondary_labels, train_idx), sub(combined_labels, train_idx))
    val_set   = MultilevelFlowDataset(sub(macro_bag, val_idx), sub(macro_mask, val_idx),
                                      sub(micro_seq, val_idx), sub(micro_mask, val_idx),
                                      sub(primary_labels, val_idx), sub(secondary_labels, val_idx), sub(combined_labels, val_idx))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=6, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size*2, num_workers=4, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"[MODE] {mode} | 类别数 {num_classes} | 样本数 {len(labels)}")

    # 模型/优化器/调度器
    cfg = ModelConfig(); cfg.num_classes = num_classes
    model = LightweightDualBranchClassifier(cfg, label_mappings).to(device)

    class_weights = make_class_weights(labels[train_idx], num_classes).to(device)
    loss_fn = FocalLoss(alpha=class_weights, gamma=1.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=3e-4)

    def lr_lambda(ep):
        warmup = 5
        if ep < warmup: return (ep + 1) / warmup
        return 0.5 * (1 + np.cos((ep - warmup) / max(1, (args.epochs - warmup)) * np.pi))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = GradScaler()

    os.makedirs(args.checkpoints_dir, exist_ok=True)
    best_f1, best_path = 0.0, Path(args.checkpoints_dir) / f"best_{mode}.pt"
    patience, patience_limit = 0, 10

    # 训练循环（单行日志）
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, loss_fn, mode)
        val_loss, val_acc, val_f1 = validate_one_epoch(model, val_loader, device, loss_fn, mode)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        logger.info(f"Epoch {epoch:03d} | Train {tr_loss:.4f}/{tr_acc:.3f} | "
                    f"Val {val_loss:.4f}/{val_acc:.3f}/{val_f1:.3f} | LR={lr:.2e} | {elapsed:.1f}s")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_f1": best_f1,
                "label_mappings": label_mappings,
                "classification_mode": mode
            }, best_path)
            logger.info(f"✨ 新最佳 F1={best_f1:.4f} 已保存至 {best_path}")
            patience = 0
        else:
            patience += 1
            if patience >= patience_limit:
                logger.info(f"⏹️ 早停：连续 {patience_limit} 轮无提升")
                break

        if epoch % 5 == 0:
            gc.collect(); torch.cuda.empty_cache()

    logger.info(f"✅ 训练完成！最佳 F1={best_f1:.4f} | 模型路径: {best_path}")

if __name__ == "__main__":
    main()
