# ISCXTor 论文实验协议

## 1. 数据源与标签

数据源为 ISCXTor2016 的原始 `pcap/pcapng`。先建立数据清单，至少记录：相对路径、Tor/Non-Tor、八类应用标签、采集工作站/日期或原始 pcap 对、文件大小与哈希。`capture_group` 优先使用同一次采集或同一对工作站/网关 pcap，不能退化成单个 flow ID。

八类应用固定为 `BROWSING, EMAIL, CHAT, AUDIO, VIDEO, FILE, VOIP, P2P`。主任务为八分类，辅助任务为 Tor/Non-Tor 二分类。

## 2. 数据处理顺序

### 2.1 先做 smoke 采样

先将少量 pcap 转 CSV；PCAP 转换会按 60 秒空闲超时和 TCP FIN/RST 拆会话：

```powershell
python data/pcap_to_csv.py --input_dir <pcap根目录> --output_dir artifacts/csv/iscxtor_full_v1 --flow_timeout 60 --workers 8
```

从生成的 flow/packet CSV 中按完整 flow 采样，不截断单个 flow：

```powershell
python data/sample_flows_by_ratio.py --input_dir artifacts/csv/iscxtor_full_v1 --output_dir artifacts/csv/iscxtor_smoke_v1 --ratio 0.02 --max_flows_per_file 50 --seed 42
```

### 2.2 构建对齐的双粒度特征

```powershell
python data/build_features.py --csv_dir artifacts/csv/iscxtor_smoke_v1 --output_dir artifacts/features/iscxtor_smoke_p32_b16_v1 --max_packets 32 --max_bursts 16 --alpha 1.0
```

正式特征：

```powershell
python data/build_features.py --csv_dir artifacts/csv/iscxtor_full_v1 --output_dir artifacts/features/iscxtor_full_p64_b32_a1_v1 --max_packets 64 --max_bursts 32 --alpha 1.0 --source_manifest <数据清单.csv>
```

特征 ID 必须包含数据范围、`max_packets`、`max_bursts`、`alpha` 和版本。包级与 burst 级特征均只使用同一个 `max_packets` 观察前缀。

## 3. 运行实验

先 dry-run 检查解析后的路径、Git commit、feature ID、split ID 与 seed：

```powershell
python experiments/run_experiment.py --config experiments/configs/main/application8_full_gated_v1.yaml --dry-run
```

正式运行：

```powershell
python experiments/run_experiment.py --config experiments/configs/main/application8_full_gated_v1.yaml
```

第一次正式运行会创建并保存 group split；后续所有 application8 模型必须复用同一个 split 文件。归一化参数只在 train token 上拟合，模型按 val Macro-F1 选择，test 集只在最终评估时使用。

```powershell
python model/export_results.py --checkpoint artifacts/runs/E10_application8_gated_v1/seed_42/best_model.pt --features_dir artifacts/features/iscxtor_full_p64_b32_a1_v1 --split_file artifacts/splits/application8_group_seed42_v1.npz --split test --output_dir artifacts/runs/E10_application8_gated_v1/seed_42/test
```

正式表格至少报告 Accuracy、Macro-F1、Weighted-F1、各类 Precision/Recall/F1、参数量、训练时间与推理时间。主结论优先使用 Macro-F1。

## 4. 实验组与消融

| 编号 | 配置 | 验证问题 |
|---|---|---|
| E10 | 八类 + gated | 完整方法主结果 |
| E11 | concat | 动态门控是否优于直接拼接 |
| E12 | fixed 0.5 | 样本自适应权重是否有效 |
| E13 | micro_only | burst Transformer 的增益 |
| E14 | burst_only | 包级 Mamba 的增益 |
| E20 | Tor/Non-Tor + gated | 辅助任务有效性 |

后续 `experiment-pipeline-v1.1` 再增加表示消融：固定阈值、去 burst 上下文、去 gap/position，以及 CNN/GRU/传统统计基线。模型消融复用同一 feature ID；表示消融必须生成新的 feature ID。

## 5. 命名与重复实验

实验 ID 使用 `E<组号>_<用途>_v<版本>`，例如 `E10_application8_gated_v1`。同类型小改使用 `v1, v2, v3`；实验管线兼容性变更使用 Git 标签 `experiment-pipeline-v1.0, v1.1, v2.0`。

最终主结果建议使用 seed `42, 123, 2026` 各运行一次。复制配置文件并同时修改 `experiment_id`、`seed` 和 `split_id`；同一 seed 下的所有 application8 对比实验必须使用同一 split ID。

每个 run 目录应包含：`resolved_config.json`、`run_metadata.json`、`normalizer.json`、`history.json`、`metrics.json`、`best_model.pt`，最终测试后再增加 `test/metrics.json` 与 `test/predictions.csv`。
