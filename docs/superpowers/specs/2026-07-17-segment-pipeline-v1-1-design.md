# ISCXTor 片段预处理 v1.1 性能与划分设计

## 目标

在不改变15秒片段、自适应 burst、Mamba + Transformer双分支和八类主任务的前提下，修复直接运行入口，缩小 smoke 规模，增加按源文件并行处理，并把正式集合划分升级为按有效初始片段数加权的采集组分层划分。

## 版本隔离

- Git分支继续使用 `feature/segment-burst-preprocessing-v1`，服务器可快进拉取。
- 新数据目录为 `processed/segment15_burstp95_v1_1`。
- 旧 `segment15_burstp95_v1` 不覆盖、不删除。

## Smoke规则

- 每个应用选择最多3个较小源文件。
- 每个源文件只选择包数最少的5条完整父流。
- 优先从同名 `_flows.csv` 读取 `packet_count` 选择父流；缺少流汇总时才扫描 packet CSV 的 `flow_id`。
- 被选中的父流完整保留，不截断其中的包。
- smoke以链路和类别覆盖为目标，不把70/15/15作为论文性能划分。

## 正式划分

1. 第一遍处理每个采集组，统计 `eligible_segment_count`，并保留该组自然 burst 时长。
2. 以采集组为不可拆分单位，对5000个确定性候选划分评分。
3. 评分同时考虑总体样本比例、八类应用各自比例和Tor/Non-Tor比例相对70/15/15的偏差。
4. 强约束为采集组不交叉、八类应用在三个集合中均存在、Tor和Non-Tor在三个集合中均存在。
5. 选定划分后，只拼接训练组保存的自然 burst 时长计算 `D_max=P95`；验证和测试组时长不得进入分位数。
6. 保存估计片段权重与最终样本两套划分审计。

类别不平衡不通过删除验证/测试数据解决。训练继续使用类别权重与Focal Loss；验证集和测试集保持采集组自然分布，主指标为Macro-F1。

## 并行策略

- 使用 `ProcessPoolExecutor`，默认 `WORKERS=2`，不是Python线程池。
- 并行单位是源CSV，避免拆散同一文件状态。
- 第一遍并行生成每个源的片段权重与自然 burst 时长。
- 第二遍并行生成每个源的连续NumPy特征批次和清单。
- 主进程按 `source_key` 固定顺序合并，保证重复运行结果一致。
- `WORKERS=1` 保留为调试模式；不默认超过4，避免多个大CSV同时占满内存和磁盘带宽。

## 入口与进度

在脚本开头显式把仓库根目录加入 `sys.path`，同时支持：

```text
python -m data.run_segment_feature_pipeline
python data/run_segment_feature_pipeline.py
cd data && python run_segment_feature_pipeline.py
```

每个阶段输出 `完成文件数/总文件数、源文件、包数、片段数、耗时`，避免长时间无反馈。

## 验收

- 直接文件运行不再报 `No module named data`。
- smoke每个源最多5条完整父流，包守恒成立。
- 加权划分确定性复现，采集组无交叉且类别覆盖完整。
- `D_max`来源仍严格为训练组。
- 单进程与双进程生成相同样本键、标签、形状和划分。
- 完整测试通过，工作区干净后推送同一远端分支。
