# 论文实验管线 v1.0 设计说明

## 1. 目标

在不更换 Mamba + Transformer 双分支主干的前提下，将当前项目改造成可用于 ISCXTor2016 八类应用细粒度分类与 Tor/Non-Tor 辅助二分类的可复现实验工程。

本版本的 Git 分支为 `feature/experiment-pipeline-v1`，发布标签为 `experiment-pipeline-v1.0`。后续同类修订使用 `v1.1`、`v1.2`；发生不兼容的数据结构或训练接口升级时进入 `v2.0`。

## 2. 任务定义

### 2.1 主任务

八类应用细粒度分类：

1. `BROWSING`
2. `EMAIL`
3. `CHAT`
4. `AUDIO`
5. `VIDEO`
6. `FILE`
7. `VOIP`
8. `P2P`

主任务同时使用标签明确的 Tor 与 Non-Tor 样本，分类目标是应用行为类别。`UNKNOWN` 不进入主任务训练与评价。

### 2.2 辅助任务

Tor/Non-Tor 二分类，仅保留 `TOR` 与 `NONTOR` 样本。

### 2.3 兼容任务

保留旧项目的 `primary`、`secondary`、`combined` 入口，便于读取历史特征；新实验配置统一使用 `application8` 和 `tor_binary`。

## 3. 数据处理设计

### 3.1 标签规范化

建立唯一标签模块，统一处理路径和文件名中的同义词：

- `WEB/HTTP/HTTPS/BROWSING` -> `BROWSING`
- `MAIL/EMAIL` -> `EMAIL`
- `CHAT/IM` -> `CHAT`
- `AUDIO/MUSIC/SPOTIFY` -> `AUDIO`
- `VIDEO/YOUTUBE/VIMEO` -> `VIDEO`
- `FILE/FTP/FTPS/SFTP/TRANSFER` -> `FILE`
- `VOIP/CALL` -> `VOIP`
- `P2P/TORRENT/BITTORRENT` -> `P2P`

必须优先识别 `NONTOR/NON-TOR`，防止其中的 `TOR` 子串被误判为 Tor。

### 3.2 流与会话

正反向五元组归并为同一双向流。在基础五元组之上增加会话编号：

- 相邻包空闲时间超过 `flow_timeout` 时开启新会话；
- TCP FIN/RST 后的后续包开启新会话；
- `flow_id` 追加稳定的 `S<index>` 会话后缀。

解析器修正重复 pcap 扫描、pcapng 文件头识别和 TCP/UDP 负载长度计算。

### 3.3 三层数据规模

- `smoke`：每类 20-50 条完整 flow，用于 1-2 epoch 跑通测试；
- `pilot`：每类设置最大 flow 数或使用 5%-10% 数据，用于快速筛选实验方向；
- `full`：完整数据，用于冻结配置后的正式实验。

禁止随机截取单个包。采样以完整 flow 为单位，并保存采样清单和随机种子。

### 3.4 双粒度观察窗口

先取当前样本允许观察的前 `max_packets` 个包，再在该前缀上计算 IAT、自适应阈值和 burst。`packet_seq` 与 `burst_seq` 必须来自同一前缀和同一套 burst 边界，禁止包级分支通过 burst 聚合特征读取截断位置之后的信息。

### 3.5 数据身份与分组

特征目录额外保存：

- `group_ids.npy`：默认使用原始 pcap 相对路径作为采集组；
- `sample_manifest.csv`：样本键、flow、源文件、分组、一级标签、应用标签；
- `feature_summary.json`：表征参数和类别分布。

允许传入人工维护的 `source_manifest.csv`，覆盖自动推断的标签和 `capture_group`，用于将成对采集文件放入同一数据组。

## 4. 数据划分与归一化

固定划分为训练集、验证集、测试集。划分以 `group_ids` 为最小单位，同一采集组不能跨集合。划分结果保存为版本化 `split_v1.npz` 和摘要 JSON。

训练时只使用训练索引拟合序列特征均值与标准差。方向、协议、首尾标记等离散字段不做连续标准化。归一化参数随 run 保存，验证集和测试集只能复用训练参数。

## 5. 模型与训练约束

- 主干保持 Mamba 微观分支、Transformer burst 分支与门控融合；
- smoke 模式允许使用轻量 SSM 回退实现检查数据链路；
- formal 模式必须确认 `mamba_ssm.Mamba2` 可用，否则立即报错；
- `d_state` 必须从配置透传，不能在模型内部硬编码；
- 使用验证集 Macro-F1 选择最佳模型；
- 支持 `cross_entropy`、`weighted_cross_entropy`、`focal`，架构主对比默认 `cross_entropy`；
- 固定 Python、NumPy、PyTorch 和 DataLoader 随机种子；
- 正式测试只评价 split 文件中的测试索引。

## 6. 配置化实验管理

保持单套生产代码，通过 YAML 配置区分数据、表征、模型、融合、损失和随机种子。统一入口：

```text
python experiments/run_experiment.py --config <config.yaml>
```

每次运行创建唯一目录：

```text
artifacts/runs/<experiment_id>/seed_<seed>/
```

目录必须保存：

- `resolved_config.yaml`
- `run_metadata.json`
- `train.log`
- `history.json`
- `normalizer.json`
- `checkpoint.pt`
- `metrics.json`
- `predictions.csv`

元数据记录 Git commit、dirty 状态、数据集 ID、特征 ID、split ID、随机种子和运行命令。默认禁止覆盖已有 run。

## 7. 配置分组

项目提供以下起始配置：

- `smoke/application8_smoke_v1.yaml`
- `main/application8_full_gated_v1.yaml`
- `auxiliary/tor_binary_full_v1.yaml`
- `ablation/application8_micro_only_v1.yaml`
- `ablation/application8_burst_only_v1.yaml`
- `ablation/application8_concat_v1.yaml`
- `ablation/application8_fixed_v1.yaml`

## 8. 验证标准

### 数据层

- 八类标签同义词测试通过；
- Non-Tor 不被识别为 Tor；
- 同五元组超时或 FIN/RST 后能生成新会话；
- pcap发现列表无重复；
- packet与burst特征只使用同一包前缀；
- 分组划分交集为零；
- 归一化仅拟合训练索引。

### 工程层

- 全部非 PyTorch 单元测试通过；
- 安装 PyTorch 的环境中完成一个 batch 前向与反向；
- formal 模式在官方 Mamba 缺失时明确失败；
- dry-run 能生成唯一运行目录、解析配置并输出训练命令；
- 相同 run 未显式允许时拒绝覆盖。

## 9. 非目标

本版本不替换论文主干，不加入新的第三创新点，不在本地下载完整 ISCXTor2016 数据，不伪造模型性能结果。CNN、GRU、包级 Transformer 和传统机器学习基线可在本管线稳定后作为 `experiment-pipeline-v1.1` 增补。
