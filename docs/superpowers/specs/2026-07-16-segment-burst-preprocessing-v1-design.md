# ISCXTor 流片段与自适应 Burst 预处理 v1 设计

## 1. 目标与范围

本版本只改造数据预处理，不更换论文任务、标签体系和 Mamba + Transformer + 门控融合主干。

主任务固定为八类应用分类：`BROWSING、EMAIL、CHAT、AUDIO、VIDEO、FILE、VOIP、P2P`；辅助任务固定为 `Tor/Non-Tor` 二分类。模型的预测对象是一段双向流量片段，而不是单个数据包。

现有 PCAP 到会话 CSV 的转换结果继续使用，不重新解析约 21 GB 原始 PCAP。新版管线从 `*_packets.csv` 开始，完成采集组划分、时间片段构建、训练集统计、自适应 burst 构建、无损容量拆分和模型特征生成。

## 2. 版本与输出隔离

- Git 实现分支：`feature/segment-burst-preprocessing-v1`
- 数据版本目录：`processed/segment15_burstp95_v1`
- 现有 `csv/full_session60_v1` 只读，不覆盖、不删除。
- 新版特征不覆盖旧版特征，便于用 Git 分支和数据版本目录同时回退。

## 3. 固定实验参数

首版默认参数如下：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `WINDOW_SECONDS` | 15.0 | 主实验固定时间窗口 |
| `VAL_RATIO` | 0.15 | 验证集采集组比例 |
| `TEST_RATIO` | 0.15 | 测试集采集组比例 |
| `SEED` | 42 | 固定随机种子 |
| `ALPHA` | 1.0 | 自适应 IAT 阈值系数 |
| `D_MAX_QUANTILE` | 0.95 | 训练集自然 burst 时长分位数 |
| `MIN_MODEL_PACKETS` | 2 | 进入模型的最少包数 |
| `MAX_PACKETS` | 64 | 首版每个最终样本最大包数 |
| `MAX_BURSTS` | 32 | 首版每个最终样本最大 burst 数 |

后续敏感性实验比较：

- 时间窗口：`5 / 10 / 15 / 30` 秒；
- `D_max`：`无上限 / P90 / P95 / P99`；
- 最大包数：`32 / 64 / 128`；
- 最大 burst 数：`8 / 16 / 32 / 64`；
- `alpha`：`0.5 / 1.0 / 1.5 / 2.0`。

## 4. 数据处理顺序

### 4.1 采集组级划分

先根据源 PCAP 或明确的 `capture_group` 冻结训练、验证、测试集合，再计算任何从数据分布学习出的参数。同一源 PCAP 产生的所有会话、片段和 burst 必须属于同一集合。

默认比例为 `70% / 15% / 15%`。划分后检查三个集合的采集组互不相交，并输出八类应用和 Tor/Non-Tor 分布。`full` 模式下，若某一类别在验证集或测试集中缺失，脚本必须报错停止，不允许静默继续；`smoke` 模式只做覆盖率警告，因为少量输入不保证三个集合都包含八类。

### 4.2 15 秒非重叠完整覆盖片段

每条双向父会话独立按时间排序，以首包时间 `t0` 为基准：

```text
segment_index = floor((packet_timestamp - t0) / 15)
```

时间区间采用左闭右开 `[start, end)`。每个包恰好进入一个初始片段，最后不足 15 秒的尾片段保留。不同父会话不得合并。

所有片段都写入清单。只有 `packet_count >= 2` 的片段进入主模型数据；单包片段不删除，标记为 `eligible_for_model=false`，用于统计数据保留率和敏感性分析。这个过滤规则对训练、验证、测试集合一致，并且不依赖类别标签。

### 4.3 自然 burst 与训练集 `D_max`

在每个初始片段内计算正 IAT 集合：

```text
T_segment = median(positive_IAT) + alpha * IQR(positive_IAT)
```

若片段没有正 IAT，则令 `T_segment=0`。自然 burst 在以下任一条件满足时开启：

1. 当前包方向与前一包不同；
2. 当前包 IAT 大于 `T_segment`。

只收集训练集中包数不少于 2 的自然 burst 时长，取全局 P95 作为 `D_max`。验证集和测试集不得参与该数值计算。`D_max` 写入版本化 JSON，后续所有集合复用同一个冻结值。

### 4.4 最终 burst

最终 burst 在以下任一条件满足时开启：

1. `direction_change`：方向变化；
2. `iat_gap`：`IAT > T_segment`；
3. `duration_cap`：加入当前包后，当前 burst 持续时间将超过 `D_max`。

发生 `duration_cap` 时，当前包是下一个 burst 的首包。因此相邻两个 burst 可以同方向。每个 burst 记录主要切分原因，便于审计和论文统计。

### 4.5 无损容量拆分

禁止使用 `packets[:MAX_PACKETS]` 或 `bursts[:MAX_BURSTS]` 丢弃尾部数据。

对每个 15 秒片段，按时间顺序将完整 burst 贪心装入最终模型样本：加入下一个完整 burst 会超过 `MAX_PACKETS` 或 `MAX_BURSTS` 时，在该 burst 边界结束当前样本并开启新样本。

如果单个 burst 自身超过 `MAX_PACKETS`，无法在已有 burst 边界满足容量约束。此时采用唯一例外：按连续包顺序切成不超过 `MAX_PACKETS` 的容量子 burst，切分原因记录为 `packet_capacity_cap`。该规则保证没有包丢失，同时允许论文中单独报告此例外的触发比例。

最终样本仍继承同一个父流标签，并记录 `parent_segment_id`、`subsegment_index` 和容量拆分原因。

## 5. 两个模型分支的对齐输入

同一个最终样本同时生成两种视图，并共享完全相同的包集合与 burst 边界：

- `packet_seq [T, Dp]`：包长、负载长、方向、有符号包长、IAT、TCP 标志、TTL，以及 `burst_id、pos_in_burst、burst_size、burst_bytes、is_burst_start、is_burst_end、burst_duration`；
- `burst_seq [K, Db]`：方向、包数、字节数、持续时间、长度统计、IAT 统计、与前一 burst 的间隔、首尾标记。

`packet_mask` 与 `burst_mask` 分别标记有效 token。Padding 只发生在全部切分完成之后，不参与阈值和统计量计算。

## 6. 用户入口与目录

提供一个用户直接运行的入口文件：

```text
data/run_segment_feature_pipeline.py
```

用户只需要在文件顶部修改：

```text
CSV_DIR
OUTPUT_DIR
RUN_MODE = "smoke" 或 "full"
```

`smoke` 自动选择少量但覆盖 Tor/Non-Tor 和尽可能多应用类别的 CSV；`full` 处理全部 CSV。脚本支持断点信息和清晰中文日志，不要求用户组合多条 Linux 命令。

输出目录：

```text
processed/segment15_burstp95_v1/
├── manifests/
│   ├── segment_manifest.csv
│   ├── sample_manifest.csv
│   ├── split_manifest.csv
│   └── class_summary.csv
├── statistics/
│   ├── segmentation_summary.json
│   ├── natural_burst_summary.json
│   ├── dmax_summary.json
│   └── capacity_split_summary.json
└── features/
    ├── packet_seq.npy
    ├── packet_mask.npy
    ├── burst_seq.npy
    ├── burst_mask.npy
    ├── application_labels.npy
    ├── tor_labels.npy
    ├── sample_keys.npy
    ├── group_ids.npy
    └── label_mappings.pkl
```

## 7. 失败处理与可恢复性

- 输入 CSV 缺少必要字段时，指出文件和字段后停止；
- 时间戳无法解析、标签无法识别或集合类别缺失时停止；
- 输出先写临时文件，成功后再替换正式文件，避免中断留下假成功文件；
- 完成后写入包含配置、数据计数和文件清单的成功标记；
- 相同配置和相同输入重复运行应得到一致的样本 ID、划分和特征形状。

## 8. 必须通过的验收检查

1. 所有初始片段的包数总和等于输入包数；
2. 所有最终样本的包数加上不进入模型的单包片段包数等于输入包数；
3. 同一输入包不能出现在两个片段或两个最终样本；
4. 每个样本内时间戳单调不下降；
5. 训练、验证、测试 `capture_group` 交集为空；
6. `D_max` 的来源集合只能是训练集；
7. 最终样本满足 `packet_count <= MAX_PACKETS` 且 `burst_count <= MAX_BURSTS`；
8. `packet_seq` 与 `burst_seq` 使用相同 burst 边界；
9. `full` 模式的主标签恰好为八类应用，辅助标签恰好为 Tor/Non-Tor；`smoke` 模式报告实际覆盖类别；
10. smoke 模式成功后才允许运行 full 模式。

## 9. 首版验证范围

首版只要求完成数据正确性验证和小规模模型跑通，不预设分类性能一定提升。先使用 smoke 特征运行八类应用分类的 1 至 3 个 epoch，确认 Dataset、Mamba、Transformer、门控融合、损失函数和结果导出全链路可运行。

若首版正确，再对全量数据运行基线与完整模型；若效果不理想，可通过 Git 分支和独立数据目录直接回退，不影响现有 PCAP、CSV 和旧版实验结果。
