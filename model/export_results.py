#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_results.py
------------------------------------------------------------
导出模型推理与评估结果 (支持 primary / secondary 模式)
- 支持按协议 (VPN / TOR / QUIC) 与指定类别筛选
- 自动修复标签与 target_names 数量不匹配问题
- 输出: JSON, TXT, PNG, CSV, NPY
------------------------------------------------------------
"""

import os
import json
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, f1_score, accuracy_score
from tqdm import tqdm

from model import DualBranchFlowClassifier
from config import ModelConfig


# ==========================
# 工具函数
# ==========================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def save_text(text, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ==========================
# 主类
# ==========================
class Evaluator:
    def __init__(self, checkpoint_path, features_dir, output_dir, protocol=None, eval_classes=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.features_dir = features_dir
        self.output_dir = output_dir
        self.protocol = protocol
        self.eval_classes = eval_classes
        ensure_dir(output_dir)

    # --------------------------
    # 模型加载
    # --------------------------
    def load_model(self):
        print(f"📦 加载模型: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        self.classification_mode = checkpoint.get("classification_mode", "primary")
        self.label_mappings = checkpoint.get("label_mappings", {})
        self.config = checkpoint.get("config", ModelConfig())

        if 'labels' not in self.label_mappings.get(self.classification_mode, {}):
            n_cls = self.label_mappings[self.classification_mode].get("num_classes", 6)
            self.label_mappings[self.classification_mode]['labels'] = [f"class_{i}" for i in range(n_cls)]
            print(f"⚙️ 自动补全 labels: {self.label_mappings[self.classification_mode]['labels']}")

        self.model = DualBranchFlowClassifier(self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.model.eval()
        print(f"✅ 模型加载完成 (mode={self.classification_mode})")

    # --------------------------
    # 数据加载
    # --------------------------
    def load_features(self):
        print(f"📂 加载测试数据: {self.features_dir}")
        micro_seq = np.load(os.path.join(self.features_dir, "micro_seq.npy"))
        micro_mask = np.load(os.path.join(self.features_dir, "micro_mask.npy"))
        macro_bag = np.load(os.path.join(self.features_dir, "macro_bag.npy"))
        macro_mask = np.load(os.path.join(self.features_dir, "macro_mask.npy"))

        label_file = os.path.join(self.features_dir, f"{self.classification_mode}_labels.npy")
        if not os.path.exists(label_file):
            label_file = os.path.join(self.features_dir, "combined_labels.npy")
        labels = np.load(label_file)

        # --- 协议过滤 ---
        selected_idx = np.arange(len(labels))
        if self.protocol:
            primary_labels = np.load(os.path.join(self.features_dir, "primary_labels.npy"))
            proto_id = {"VPN": 0, "OTHER": 1, "TOR": 2, "QUIC": 3}[self.protocol]
            selected_idx = np.where(primary_labels == proto_id)[0]
            print(f"✅ 已筛选协议: {self.protocol}，样本数 {len(selected_idx)}")

        # --- 修正宏观 mask ---
        if macro_mask.ndim == 2 and macro_mask.shape[1] != 1:
            macro_mask = np.mean(macro_mask, axis=1, keepdims=True)

        # 应用协议索引
        micro_seq = micro_seq[selected_idx]
        micro_mask = micro_mask[selected_idx]
        macro_bag = macro_bag[selected_idx]
        macro_mask = macro_mask[selected_idx]
        labels = labels[selected_idx]

        # --- 类别筛选 ---
        lm = self.label_mappings.get(self.classification_mode, {})
        all_names = (
            lm.get("labels")
            or lm.get("class_names")
            or [f"class_{i}" for i in range(int(lm.get("num_classes", labels.max() + 1)))]
        )
        name_to_id = {n.upper(): i for i, n in enumerate(all_names)}

        if self.eval_classes:
            wanted = [c.upper() for c in self.eval_classes if c.upper() in name_to_id]
            target_ids = [name_to_id[c] for c in wanted]
            keep_mask = np.isin(labels, target_ids)
            micro_seq, micro_mask, macro_bag, macro_mask, labels = \
                micro_seq[keep_mask], micro_mask[keep_mask], macro_bag[keep_mask], macro_mask[keep_mask], labels[keep_mask]
            id_map = {old: new for new, old in enumerate(target_ids)}
            labels = np.array([id_map[l] for l in labels], dtype=np.int64)
            self.final_class_names = wanted
            print(f"✅ 已筛选类别: {self.final_class_names}，样本数 {len(labels)}")
        else:
            self.final_class_names = all_names

        uniq = np.unique(labels)
        print(f"🎯 唯一标签索引: {uniq.tolist()}，类名数: {len(self.final_class_names)}")

        # 转 tensor
        self.micro_seq = torch.tensor(micro_seq, dtype=torch.float32).to(self.device)
        self.micro_mask = torch.tensor(micro_mask > 0, dtype=torch.float32).to(self.device)
        self.macro_bag = torch.tensor(macro_bag, dtype=torch.float32).to(self.device)
        self.macro_mask = torch.tensor(macro_mask > 0, dtype=torch.float32).to(self.device)
        self.labels = torch.tensor(labels, dtype=torch.long).to(self.device)

        print(f"✅ 测试集加载完成: {len(self.labels)} 条样本")

    # --------------------------
    # 推理与评估
    # --------------------------
    def evaluate(self):
        print("🚀 开始模型评估...")
        preds, probs, labels = [], [], []
        with torch.no_grad():
            for i in tqdm(range(self.micro_seq.shape[0])):
                logits = self.model(
                    self.micro_seq[i:i+1],
                    self.macro_bag[i:i+1],
                    self.micro_mask[i:i+1],
                    self.macro_mask[i:i+1]
                )
                p = torch.softmax(logits, dim=-1)
                preds.append(p.argmax(dim=-1).cpu().item())
                probs.append(p.cpu().numpy()[0])
                labels.append(self.labels[i].cpu().item())

        self.y_true = np.array(labels)
        self.y_pred = np.array(preds)
        self.y_prob = np.array(probs)

        acc = accuracy_score(self.y_true, self.y_pred)
        f1 = f1_score(self.y_true, self.y_pred, average="macro")
        print(f"✅ 推理完成: Acc={acc:.4f}, F1={f1:.4f}")
        return acc, f1

    # --------------------------
    # 结果保存与可视化
    # --------------------------
    def save_results(self):
        labels = self.final_class_names
        print(f"📊 生成报告 ({len(labels)} 类)")
        cls_report = classification_report(
            self.y_true, self.y_pred,
            target_names=labels, digits=4, output_dict=True
        )
        cm = confusion_matrix(self.y_true, self.y_pred)

        # 保存预测结果
        df = pd.DataFrame({
            "true_label": [labels[i] for i in self.y_true],
            "pred_label": [labels[i] for i in self.y_pred]
        })
        for i, lbl in enumerate(labels):
            df[f"prob_{lbl}"] = self.y_prob[:, i]
        df.to_csv(os.path.join(self.output_dir, "predictions.csv"), index=False)

        # 混淆矩阵
        plt.figure(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "confusion_matrix.png"))
        plt.close()

        # ROC（仅二分类）
        if len(labels) == 2:
            fpr, tpr, _ = roc_curve(self.y_true, self.y_prob[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, lw=2, color="darkorange", label=f"AUC={roc_auc:.2f}")
            plt.plot([0, 1], [0, 1], lw=2, color="navy", linestyle="--")
            plt.legend(); plt.xlabel("FPR"); plt.ylabel("TPR")
            plt.title("ROC Curve")
            plt.savefig(os.path.join(self.output_dir, "roc_curve.png"))
            plt.close()

        # 保存 JSON + TXT 报告
        result = {
            "accuracy": float(accuracy_score(self.y_true, self.y_pred)),
            "macro_f1": float(f1_score(self.y_true, self.y_pred, average="macro")),
            "classification_report": cls_report
        }
        save_json(result, os.path.join(self.output_dir, "evaluation_results.json"))
        save_text(json.dumps(result, indent=2, ensure_ascii=False),
                  os.path.join(self.output_dir, "evaluation_report.txt"))

        print(f"✅ 评估结果已保存到 {self.output_dir}")


# ==========================
# 主入口
# ==========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--features_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--protocol", choices=["VPN", "TOR", "QUIC"], default=None)
    parser.add_argument("--eval_classes", nargs="*", default=None)
    args = parser.parse_args()

    evaluator = Evaluator(
        checkpoint_path=args.checkpoint,
        features_dir=args.features_dir,
        output_dir=args.output_dir,
        protocol=args.protocol,
        eval_classes=args.eval_classes
    )
    evaluator.load_model()
    evaluator.load_features()
    evaluator.evaluate()
    evaluator.save_results()
