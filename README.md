# 基于多粒度特征融合的加密流量分类

本项目面向 ISCXTor2016：以八类应用细粒度分类为主任务，以 Tor/Non-Tor 二分类为辅助任务。模型主干保持为包级 Mamba、burst 级 Transformer 与门控融合。

## 任务定义

- 主任务 `application8`：Browsing、Email、Chat、Audio、Video、File、VoIP、P2P。
- 辅助任务 `tor_binary`：Non-Tor、Tor。
- 主任务跨 Tor/Non-Tor 识别应用类别；未知标签不会作为第九类参与训练。

## 数据与模型链路

```text
pcap/pcapng
  -> 五元组会话化 packet/flow CSV
  -> 同一个 max_packets 观察前缀
  -> packet_seq + 自适应同向 burst_seq
  -> Mamba2 + burst Transformer
  -> gated fusion
  -> application8 / tor_binary
```

自适应 burst 阈值为：`T_flow = median(IAT) + alpha * IQR(IAT)`。正式实验必须使用 `mamba_ssm.Mamba2`；轻量 SSM 回退只用于 smoke 跑通，不得报告为论文结果。

## 三阶段运行

1. smoke：每个源文件少量完整 flow，验证 PCAP、标签、特征、模型张量和保存链路。
2. pilot：每类适量完整 flow，确定显存、batch size、训练时间和参数范围。
3. full：完整数据一次生成 CSV/特征缓存，冻结 group split 后运行主实验、辅助实验和消融。

完整命令、数据清单、版本命名和服务器执行顺序见 [实验协议](docs/experiment_protocol.md)。

快速检查配置但不启动训练：

```powershell
python experiments/run_experiment.py --config experiments/configs/smoke/application8_smoke_v1.yaml --dry-run
```

运行 smoke：

```powershell
python experiments/run_experiment.py --config experiments/configs/smoke/application8_smoke_v1.yaml
```

每次实验写入 `artifacts/runs/<experiment_id>/seed_<seed>/`，已有目录默认拒绝覆盖。生成数据、特征、权重和运行结果均由 Git 忽略，配置与代码进入版本控制。

