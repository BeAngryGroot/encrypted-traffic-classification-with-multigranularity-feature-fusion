# ISCXTor 论文实验协议

## 1. 数据源与标签

数据源为 ISCXTor2016 的原始 `pcap/pcapng`。先建立数据清单，至少记录：相对路径、Tor/Non-Tor、八类应用标签、采集工作站/日期或原始 pcap 对、文件大小与哈希。`capture_group` 优先使用同一次采集或同一对工作站/网关 pcap，不能退化成单个 flow ID。

八类应用固定为 `BROWSING, EMAIL, CHAT, AUDIO, VIDEO, FILE, VOIP, P2P`。主任务为八分类，辅助任务为 Tor/Non-Tor 二分类。

## 2. 数据处理顺序

### 2.1 使用已有完整会话 CSV

PCAP 转换会按 60 秒空闲超时和 TCP FIN/RST 拆父会话。转换已经完成时，不重新解析 PCAP，也不修改 `csv/full_session60_v1`。

新版入口为 `data/run_segment_feature_pipeline.py`。服务器确认文件顶部四个值：

```python
CSV_DIR = Path("/data3/wsb_workspace/study/data/Dual_data/csv/full_session60_v1")
OUTPUT_DIR = Path("/data3/wsb_workspace/study/data/Dual_data/processed/segment15_burstp95_v1_1")
RUN_MODE = "smoke"
WORKERS = 2
```

### 2.2 构建15秒片段与双粒度特征

```text
python -m data.run_segment_feature_pipeline
```

处理顺序固定为：

1. 以源 PCAP/packet CSV 为 `capture_group` 冻结 `70%/15%/15%` 划分；
2. 每条父会话切成15秒非重叠片段，尾片段保留；
3. 单包片段保留在清单中但不进入主模型；
4. 只用训练集自然 burst 时长计算全局 `D_max=P95`；
5. 按方向变化、`IAT>T_segment` 或持续时间超限生成最终 burst；
6. 超容量片段优先在 burst 边界拆分，禁止截断尾包；
7. 生成共享同一 burst 边界的 `packet_seq` 和 `burst_seq`。

smoke 对每个入选源文件只读取包数最少的 5 条完整父流；完整性优先于固定包数，不会在流内截断。成功后检查：输入包数是否等于建模包数加单包审计数、三个集合是否无采集组交集、`D_max` 来源是否为 train、特征形状是否为 `[N,64,16]` 和 `[N,32,12]`。`manifests/split_balance.csv` 与 `statistics/split_balance_summary.json` 分别记录划分前有效片段权重和生成后真实样本比例。随后把 `RUN_MODE` 改为 `"full"`，再次运行同一个 Python 文件。

本版本特征 ID 固定为 `segment15_burstp95_v1_1`。划分以源采集文件为不可拆分组，通过 5000 次确定性候选搜索尽量使总体、八类应用和 Tor/Non-Tor 的有效片段数接近 70/15/15；验证集和测试集不做重采样。旧 `build_features.py` 只用于旧版前缀截断实验，不用于本版正式数据。

## 3. 运行实验

新特征首先运行三轮 smoke：

```text
python experiments/run_experiment.py --config experiments/configs/smoke/application8_segment15_burstp95_smoke_v1_1.yaml
```

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
