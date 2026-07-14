# Thesis Experiment Pipeline v1.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible ISCXTor2016 experiment pipeline for eight-class application classification and auxiliary Tor/Non-Tor classification without changing the Mamba + Transformer thesis backbone.

**Architecture:** Centralize label and task semantics, make pcap-to-flow processing session-aware, build aligned packet/burst views, persist group metadata, and drive every run from an immutable YAML configuration. Training and evaluation consume a frozen group split and write to unique versioned run directories.

**Tech Stack:** Python 3.10+, NumPy, pandas, scikit-learn, PyTorch, PyYAML, Scapy, pytest, Git.

## Global Constraints

- 主任务固定为八类应用分类，辅助任务固定为 Tor/Non-Tor 二分类。
- 保留 Mamba + Transformer + gated fusion 主干。
- 核心功能、边界条件和实验约束必须使用中文注释。
- 新增生产行为必须先写失败测试，再写最小实现。
- 正式运行必须使用官方 `mamba_ssm.Mamba2`，不得静默使用回退模型。
- 所有正式结果必须绑定 Git commit、feature ID、split ID 和 seed。
- 不覆盖已有实验目录，不提交 pcap、npy、checkpoint 或运行结果。

---

## File Structure

- Create `data/label_schema.py`: ISCXTor标签推断和同义词规范化。
- Create `data/splits.py`: 分组划分、保存和加载。
- Create `data/normalization.py`: mask感知、仅训练集拟合的标准化。
- Create `experiments/configuration.py`: YAML解析、校验和run目录解析。
- Create `experiments/run_experiment.py`: dry-run和正式运行入口。
- Create `model/task_labels.py`: application8与tor_binary任务筛选和重映射。
- Modify `data/pcap_to_csv.py`: 会话化、负载长度、pcap发现去重。
- Modify `data/sample_flows_by_ratio.py`: 按完整flow采样、类别上限、清单输出。
- Modify `data/burst_features.py`: 先截断后划分burst。
- Modify `data/build_features.py`: 中央标签模块、group_ids和样本清单。
- Modify `model/mamba_branch.py`: formal模式官方Mamba保护。
- Modify `model/model.py`: 透传d_state和formal标志。
- Modify `model/train_optimized.py`: 固定split、归一化、Macro-F1、唯一run输出。
- Modify `model/export_results.py`: 仅评价指定split。
- Create/modify tests under `tests/` for every new behavior.
- Add versioned YAML examples under `experiments/configs/`.

---

### Task 1: Central ISCXTor label schema and task definitions

**Files:**
- Create: `data/label_schema.py`
- Create: `model/task_labels.py`
- Create: `tests/test_label_schema.py`
- Create: `tests/test_task_labels.py`
- Modify: `data/build_features.py`

**Interfaces:**
- Produces: `infer_labels(path: Path) -> LabelInfo`
- Produces: `select_task_labels(mode, primary, secondary, mappings) -> TaskSelection`

- [ ] **Step 1: Write failing label inference tests**

```python
def test_iscxtor_eight_application_labels():
    cases = {
        "Tor/Browsing_packets.csv": ("TOR", "BROWSING"),
        "Tor/Email_packets.csv": ("TOR", "EMAIL"),
        "Tor/P2P_packets.csv": ("TOR", "P2P"),
        "NonTor/Video-Streaming_packets.csv": ("NONTOR", "VIDEO"),
    }
    for path, expected in cases.items():
        info = infer_labels(Path(path))
        assert (info.primary, info.application) == expected
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_label_schema.py -q`
Expected: FAIL because `data.label_schema` does not exist.

- [ ] **Step 3: Implement the immutable label schema**

```python
@dataclass(frozen=True)
class LabelInfo:
    primary: str
    application: str

    @property
    def combined(self) -> str:
        return f"{self.primary}:{self.application}"
```

Implement token normalization with `NONTOR` checked before `TOR`, and exactly eight main application labels.

- [ ] **Step 4: Write and implement failing task selection tests**

```python
def test_application8_excludes_unknown_and_keeps_tor_and_nontor():
    selection = select_task_labels("application8", primary, secondary, mappings)
    assert selection.class_names == list(APPLICATION_LABELS)
    assert selection.keep_mask.tolist() == [True, True, False]
```

- [ ] **Step 5: Route feature building through the shared schema**

Remove duplicate `PRIMARY_LABELS`, `SECONDARY_LABELS`, and `infer_labels` definitions from `build_features.py`; import the shared function instead.

- [ ] **Step 6: Run focused and full tests**

Run: `python -m pytest tests/test_label_schema.py tests/test_task_labels.py tests/test_build_features.py -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```text
feat: 完善ISCXTor八类标签与任务定义
```

---

### Task 2: Reliable pcap discovery, payload length, and flow sessions

**Files:**
- Modify: `data/pcap_to_csv.py`
- Create: `tests/test_pcap_pipeline.py`

**Interfaces:**
- Produces: `discover_pcaps(root: Path) -> list[Path]`
- Produces: `FlowSessionizer.assign(base_flow_id, timestamp, tcp_flags) -> str`
- Produces: `transport_payload_length(raw, l3, l4, proto, packet_end) -> int`

- [ ] **Step 1: Write failing discovery and session tests**

```python
def test_discover_pcaps_deduplicates_overlapping_patterns(tmp_path):
    (tmp_path / "a.pcap").write_bytes(b"")
    (tmp_path / "b.pcapng").write_bytes(b"")
    assert [p.name for p in discover_pcaps(tmp_path)] == ["a.pcap", "b.pcapng"]

def test_sessionizer_splits_after_timeout_and_fin():
    s = FlowSessionizer(timeout_seconds=60)
    assert s.assign("flow", 0.0, 0) == "flow_S0"
    assert s.assign("flow", 61.0, 0) == "flow_S1"
    assert s.assign("flow", 62.0, 0x01) == "flow_S1"
    assert s.assign("flow", 62.1, 0) == "flow_S2"
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_pcap_pipeline.py -q`
Expected: FAIL because the public helpers do not exist.

- [ ] **Step 3: Implement helpers and thread parameters through workers**

Remove mutable module globals for `min_pkts/min_bytes`. Pass `flow_timeout`, `min_pkts`, and `min_bytes` explicitly to `process_one`, `parse_pcap`, and `write_csv`. Add Chinese comments explaining why session reuse and Windows process spawning matter.

- [ ] **Step 4: Correct pcapng magic and payload calculation**

Recognize pcapng magic `b"\x0a\x0d\x0d\x0a"`. For TCP, subtract the TCP data offset; for UDP, subtract the 8-byte UDP header; clamp to the captured IP packet boundary.

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_pcap_pipeline.py tests -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```text
fix: 修正pcap会话划分与解析边界
```

---

### Task 3: Aligned packet/burst views and reproducible sampling

**Files:**
- Modify: `data/burst_features.py`
- Modify: `data/sample_flows_by_ratio.py`
- Modify: `tests/test_burst_features.py`
- Create: `tests/test_sampling.py`

**Interfaces:**
- Consumes: complete flow packet records.
- Produces: packet and burst tensors derived from the same observed prefix.
- Produces: deterministic flow sample manifest.

- [ ] **Step 1: Write the future-information regression test**

```python
def test_packet_burst_context_uses_only_observed_prefix():
    packets = same_direction_packets(count=3)
    result = build_flow_features(packets, max_packets=2, max_bursts=4)
    idx = PACKET_FEATURES.index("burst_size")
    assert result.packet_seq[0, idx] == 2
```

- [ ] **Step 2: Run and verify RED**

Expected: current implementation reports burst size 3.

- [ ] **Step 3: Truncate before threshold and burst calculation**

Set `ordered = _sorted_packets(packets)[:max_packets]` before timestamps, IAT, threshold, and burst lookup are calculated. Add Chinese comments describing the shared observation window.

- [ ] **Step 4: Add deterministic sampling tests**

Test `seed`, `max_flows_per_file`, matching packet filtering, and manifest columns. Sampling must never select partial flows.

- [ ] **Step 5: Implement sampling options and manifest output**

Add `--seed`, `--max_flows_per_file`, `--min_flows_per_file`, and `sampling_manifest.csv`.

- [ ] **Step 6: Run tests and commit**

Run: `python -m pytest tests/test_burst_features.py tests/test_sampling.py -q`

Commit:

```text
feat: 对齐双粒度观察窗口并完善流采样
```

---

### Task 4: Group metadata, frozen split, and train-only normalization

**Files:**
- Modify: `data/build_features.py`
- Create: `data/splits.py`
- Create: `data/normalization.py`
- Modify: `tests/test_build_features.py`
- Create: `tests/test_splits.py`
- Create: `tests/test_normalization.py`

**Interfaces:**
- Produces: `group_ids.npy`, `sample_manifest.csv`.
- Produces: `create_group_split(labels, groups, val_ratio, test_ratio, seed)`.
- Produces: `SequenceNormalizer.fit(..., train_indices)` and `transform(...)`.

- [ ] **Step 1: Write failing group artifact test**

Extend the feature builder test to assert that every sample has a source group and manifest row.

- [ ] **Step 2: Implement group artifacts and optional source manifest override**

Default group is packet CSV relative path. If `--source_manifest` is supplied, match `source_key` and use explicit labels and `capture_group`.

- [ ] **Step 3: Write failing split invariants**

```python
def test_group_split_has_no_overlap_and_is_reproducible():
    split1 = create_group_split(labels, groups, 0.15, 0.15, seed=42)
    split2 = create_group_split(labels, groups, 0.15, 0.15, seed=42)
    assert split1 == split2
    assert not set(groups[split1.train]) & set(groups[split1.test])
```

- [ ] **Step 4: Implement and persist split files**

Save integer arrays `train_idx`, `val_idx`, `test_idx`, plus a JSON summary with group and class counts.

- [ ] **Step 5: Write normalization leakage test**

Create a test where validation values are extreme and assert fitted mean/std equal training statistics only.

- [ ] **Step 6: Implement mask-aware normalizer**

Fit only valid tokens and selected continuous columns. Preserve padding zeros after transform. Serialize means, standard deviations, and normalized feature names.

- [ ] **Step 7: Run full data tests and commit**

Commit:

```text
feat: 增加分组划分与训练集归一化
```

---

### Task 5: Config-driven runs and non-overwriting experiment records

**Files:**
- Create: `experiments/__init__.py`
- Create: `experiments/configuration.py`
- Create: `experiments/run_experiment.py`
- Create: `tests/test_experiment_configuration.py`
- Create: `tests/test_experiment_runner.py`
- Modify: `.gitignore`
- Modify: `requirements.txt`

**Interfaces:**
- Produces: `load_experiment_config(path) -> ExperimentConfig`.
- Produces: `prepare_run(config, repo_root, dry_run) -> PreparedRun`.

- [ ] **Step 1: Write failing config validation tests**

Test required `experiment_id`, valid task/fusion/loss values, safe ID characters, and rejection of an existing run directory.

- [ ] **Step 2: Implement YAML loading and validation**

Use `yaml.safe_load`; resolve relative paths against repository root; preserve the fully resolved configuration.

- [ ] **Step 3: Write failing dry-run metadata test**

Assert dry-run records command, Git commit, dirty flag, feature ID, split ID, and seed without importing PyTorch.

- [ ] **Step 4: Implement unique run preparation**

Run directory format is `artifacts/runs/<experiment_id>/seed_<seed>`. Refuse overwrite unless explicit `--resume` is provided.

- [ ] **Step 5: Update dependencies and ignores**

Preserve the user's expanded dependency constraints, add `PyYAML>=6.0,<7.0`, and ignore `.idea/`, `artifacts/runs/*`, and local data manifests while retaining placeholders/docs.

- [ ] **Step 6: Run tests and commit**

Commit:

```text
feat: 增加配置化实验入口与版本追踪
```

---

### Task 6: Formal Mamba guard, fixed splits, and test-only evaluation

**Files:**
- Modify: `model/mamba_branch.py`
- Modify: `model/model.py`
- Modify: `model/config.py`
- Modify: `model/train_optimized.py`
- Modify: `model/export_results.py`
- Create: `tests/test_training_contracts.py`

**Interfaces:**
- Consumes: resolved experiment configuration, split file, feature directory.
- Produces: unique checkpoint, history, normalizer, validation metrics, test-only predictions.

- [ ] **Step 1: Write source-contract tests that do not require PyTorch**

Use AST inspection to assert CLI/config contracts expose `split_file`, `run_dir`, `loss`, `require_official_mamba`, `test_ratio`, and use Macro-F1 as the checkpoint key.

- [ ] **Step 2: Add explicit formal Mamba behavior**

`MicroMambaBranch(..., require_official=True)` raises a clear runtime error if `Mamba2` is unavailable. Log the implementation name in checkpoints and metadata.

- [ ] **Step 3: Remove hidden configuration mismatch**

Pass `cfg.d_state` into the Mamba branch instead of hard-coded 64. Add Chinese comments around the formal/smoke distinction.

- [ ] **Step 4: Consume frozen splits and train-only normalizer**

Training must load or create group split once, build train/val/test datasets from saved indices, fit normalizer on train, and select checkpoint by validation Macro-F1.

- [ ] **Step 5: Restrict evaluation to requested split**

`export_results.py` requires `--split_file` and defaults to `--split test`; apply the same task filtering/remapping as training before indexing.

- [ ] **Step 6: Save run artifacts without overwrite**

Write `history.json`, `normalizer.json`, `metrics.json`, and `predictions.csv` inside the prepared run directory.

- [ ] **Step 7: Run available tests**

Run all tests. If PyTorch is unavailable, record model execution as not locally verified; do not claim a model forward pass.

- [ ] **Step 8: Commit**

```text
feat: 固化正式训练与测试集评估流程
```

---

### Task 7: Versioned configurations and documentation

**Files:**
- Create: `experiments/configs/smoke/application8_smoke_v1.yaml`
- Create: `experiments/configs/main/application8_full_gated_v1.yaml`
- Create: `experiments/configs/auxiliary/tor_binary_full_v1.yaml`
- Create: four fusion ablation YAML files.
- Modify: `README.md`
- Modify: `docs/experiment_protocol.md`

- [ ] **Step 1: Add configuration examples with exact paths and values**

Smoke uses `epochs: 2`, `micro_d_model: 64`, one layer per branch, and `require_official_mamba: false`. Full configs use formal Mamba, frozen split, three-seed instructions, and Macro-F1 selection.

- [ ] **Step 2: Document the three-stage workflow in Chinese**

Document pcap inventory, smoke parsing, full CSV reuse, pilot sampling, feature cache IDs, split creation, dry-run, formal run, and result directory contents.

- [ ] **Step 3: Add experiment naming table**

Use `E<group><number>_<purpose>_v<version>` and explain feature-only versus model-only experiments.

- [ ] **Step 4: Run documentation/config validation tests**

Every shipped YAML must parse and have a unique `experiment_id`.

- [ ] **Step 5: Commit**

```text
docs: 补充三级数据与实验运行说明
```

---

### Task 8: Verification, version, and GitHub publication

**Files:**
- No production changes unless verification finds a regression.

- [ ] **Step 1: Run all available tests**

Run `python -m pytest tests -q`. Expected: all executable tests pass.

- [ ] **Step 2: Run static validation**

Parse all Python files with `ast.parse`, validate all YAML files, and verify Git ignores generated artifacts.

- [ ] **Step 3: Run smoke dry-run**

Run `python experiments/run_experiment.py --config experiments/configs/smoke/application8_smoke_v1.yaml --dry-run`. Expected: resolved command and metadata without overwritten directories.

- [ ] **Step 4: Review diff and repository state**

Confirm only planned files changed and no pcap/npy/checkpoint is tracked.

- [ ] **Step 5: Create release commit and tag**

```text
chore: 发布论文实验管线experiment-pipeline-v1.0
```

Create annotated tag `experiment-pipeline-v1.0`.

- [ ] **Step 6: Push branch, tag, and fast-forward main after remote verification**

Push `feature/experiment-pipeline-v1`, verify the remote commit, then update `main` only if it is a fast-forward from the reviewed base. Push the annotated tag.

---

## Self-Review

- Spec coverage: labels, parsing, sessions, sampling, aligned views, groups, splits, normalization, YAML runs, official Mamba, evaluation, docs, versioning, and GitHub publication are mapped to Tasks 1-8.
- Placeholder scan: no TBD/TODO or unspecified implementation steps remain.
- Type consistency: label inference, task selection, group split, normalizer, and experiment configuration interfaces are defined once and consumed by later tasks.
