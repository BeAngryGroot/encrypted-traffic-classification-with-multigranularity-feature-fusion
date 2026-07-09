# Encrypted Traffic Classification With Multi-Granularity Feature Fusion

论文方向：**基于多粒度特征融合的加密流量分类方法**。

本项目将原始 `pcap/pcapng` 流量转换为两类可训练输入：

- `packet_seq`: 突发段感知包级序列，供 Mamba 微观分支使用。
- `burst_seq`: 自适应同向突发段序列，供 Transformer 段级分支使用。

模型主线为：

```text
pcap/pcapng
-> packet CSV / flow CSV
-> adaptive same-direction burst segmentation
-> packet_seq + burst_seq
-> Mamba branch + Transformer branch
-> gated fusion
-> classifier
```

## Directory Layout

```text
data/                  pcap 解析、突发段划分、特征张量生成
model/                 Mamba 分支、Transformer 分支、融合层、训练与评估脚本
experiments/configs/   主实验和消融实验配置示例
artifacts/features/    生成的 .npy 特征文件，Git 忽略
artifacts/checkpoints/ 训练得到的模型权重，Git 忽略
artifacts/results/     评估结果、预测结果、指标文件，Git 忽略
docs/                  实验协议和项目整理说明
tests/                 小型回归测试
```

旧的 `features/`、`features_all/`、`checkpoints/`、`evaluation_results/` 已从仓库中清理。后续实验产物统一放入 `artifacts/`。

## Main Workflow

1. Convert pcap files to packet CSV files:

```powershell
python data/pcap_to_csv.py --input_dir <pcap_dir> --output_dir artifacts/csv
```

2. Build packet and burst feature tensors:

```powershell
python data/build_features.py --csv_dir artifacts/csv --output_dir artifacts/features --max_packets 64 --max_bursts 32 --alpha 1.0
```

3. Train the full model:

```powershell
python model/train_optimized.py --features_dir artifacts/features --checkpoints_dir artifacts/checkpoints --classification_mode combined --fusion_mode gated
```

4. Evaluate a checkpoint:

```powershell
python model/export_results.py --checkpoint artifacts/checkpoints/best_combined_gated.pt --features_dir artifacts/features --output_dir artifacts/results
```

## Generated Feature Schema

`data/build_features.py` writes:

- `packet_seq.npy`: `[N, max_packets, packet_feature_dim]`
- `packet_mask.npy`: `[N, max_packets]`
- `burst_seq.npy`: `[N, max_bursts, burst_feature_dim]`
- `burst_mask.npy`: `[N, max_bursts]`
- `primary_labels.npy`
- `secondary_labels.npy`
- `combined_labels.npy`
- `label_mappings.pkl`
- `sample_keys.npy`
- `feature_summary.json`

The Transformer branch should use `burst_seq.npy`, not a repeated global vector.

## Recommended Experiments

- Main comparison: MLP/statistical baseline, packet-only Mamba, burst-only Transformer, full Mamba + Transformer + gated fusion.
- Representation ablation: fixed threshold burst vs adaptive burst, remove burst context, remove burst gap/position features.
- Fusion ablation: `gated`, `concat`, `fixed`, `micro_only`, `burst_only`.
- Sensitivity: `alpha`, `max_packets`, `max_bursts`, `fusion_hidden`.

## Tests

Use the Codex bundled Python or a local Python with `numpy`, `pandas`, and `pytest`:

```powershell
python -m pytest tests -q
```
