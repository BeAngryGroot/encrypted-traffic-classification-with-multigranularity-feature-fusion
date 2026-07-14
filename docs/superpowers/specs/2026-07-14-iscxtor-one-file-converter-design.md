# ISCXTor 单文件转换入口设计

## 目标

为不熟悉 Linux 命令的使用者提供一个可直接运行的 Python 入口。使用者只修改文件顶部的 `RAW_DIR`、`OUTPUT_DIR` 和 `RUN_MODE`，随后执行一次 Python 文件即可完成单文件试跑或全量转换。

## 范围

- 新增 `data/run_iscxtor_pipeline.py`，复用 `data/pcap_to_csv.py` 的解析核心。
- `RUN_MODE="smoke"` 自动选择最小 PCAP；`RUN_MODE="full"` 处理全部 PCAP。
- 输出保持 `Tor/NonTor/应用类别` 相对目录结构。
- 每个成功源文件写入原子完成标记；全量重跑时跳过已有标记，实现断点续跑。
- 控制台与日志同时记录进度，汇总写入 `conversion_summary.csv`。
- 正式转换前校验路径、Scapy、磁盘空间与运行模式。
- 不负责 train/val/test 划分、特征生成或模型训练。

## 配置与默认值

使用者只需修改：

```python
RAW_DIR = Path("/data3/wsb_workspace/study/data/ISCX-VPN-NonVPN/ISCX-Tor-NonTor-2017/Pcaps")
OUTPUT_DIR = Path("/data3/wsb_workspace/study/data/Dual_data")
RUN_MODE = "smoke"
```

高级默认值为：`FLOW_TIMEOUT=60.0`、`MIN_PKTS=1`、`MIN_BYTES=0`、`WORKERS=2`、`MIN_FREE_GIB=60.0`。

## 数据流

```text
顶部配置 -> 环境检查 -> PCAP发现 -> smoke/full选择 -> 跳过完成项
         -> 复用process_one转换 -> 原子完成标记 -> 汇总CSV和日志
```

## 错误处理

- 路径不存在、Scapy 缺失、模式非法时在开始前停止。
- full 模式可用空间低于阈值时停止；smoke 只告警。
- 单文件失败写入汇总并继续其他文件，不写完成标记。
- 中断后再次运行 full，只重新处理没有完成标记的文件。

## 验证

- 单元测试覆盖 smoke 最小文件选择、full 全量选择、完成标记跳过、相对输出路径和配置校验。
- 原有测试全部通过。
- 本地不读取真实 21.2 GiB 数据，只验证编排逻辑；服务器 smoke 负责真实 Scapy 转换验证。

