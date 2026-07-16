# Segment + Adaptive Burst Preprocessing v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reversible preprocessing branch that converts existing ISCXTor packet CSVs into leakage-safe 15-second flow-segment samples with adaptive burst-aligned packet/burst tensors and no silent packet truncation.

**Architecture:** Add a pure segmentation module for time windows, adaptive bursts, train-only `D_max`, and capacity packing; extend the existing feature builder to accept precomputed burst boundaries without truncation; add one user-facing pipeline script that performs group splitting, manifests, statistics, and tensor output. Existing PCAP conversion and model architecture remain unchanged.

**Tech Stack:** Python 3.10+, NumPy, pandas, pytest, existing PyTorch training pipeline.

## Global Constraints

- Main task: eight application classes; auxiliary task: Tor/Non-Tor.
- `WINDOW_SECONDS=15.0`, `ALPHA=1.0`, `D_MAX_QUANTILE=0.95`, `MIN_MODEL_PACKETS=2`.
- `MAX_PACKETS=64`, `MAX_BURSTS=32`, `VAL_RATIO=0.15`, `TEST_RATIO=0.15`, `SEED=42`.
- Split by source PCAP/capture group before fitting `D_max`; validation and test data never influence fitted parameters.
- Preserve every parsed packet in an initial segment; never use prefix truncation in the new pipeline.
- Existing CSV and old feature directories are read-only inputs.
- Core code and user configuration blocks require Chinese comments.
- User runs one file: `python data/run_segment_feature_pipeline.py`.

---

### Task 1: Pure time segmentation and burst assignment

**Files:**
- Create: `data/segment_features.py`
- Create: `tests/test_segment_features.py`

**Interfaces:**
- Produces: `time_segment_packets(packets, window_seconds) -> list[list[dict[str, Any]]]`
- Produces: `assign_bursts_with_reasons(packets, alpha, max_duration=None) -> BurstAssignment`
- Produces: `collect_mult_packet_burst_durations(packets, assignment) -> list[float]`
- `BurstAssignment` contains `burst_ids`, `split_reasons`, and `adaptive_threshold`.

- [ ] **Step 1: Write failing boundary and coverage tests**

```python
def test_time_segments_are_non_overlapping_and_keep_tail():
    packets = [{"timestamp": value, "frame_index": index} for index, value in enumerate([0.0, 14.9, 15.0, 31.0])]
    segments = time_segment_packets(packets, 15.0)
    assert [[p["frame_index"] for p in segment] for segment in segments] == [[0, 1], [2], [3]]
    assert sum(map(len, segments)) == len(packets)

def test_duration_cap_starts_same_direction_followup_burst():
    packets = [{"timestamp": t, "direction": 1.0} for t in [0.0, 0.4, 0.8, 1.2]]
    result = assign_bursts_with_reasons(packets, alpha=1.0, max_duration=1.0)
    assert result.burst_ids == [0, 0, 0, 1]
    assert result.split_reasons[3] == "duration_cap"
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/test_segment_features.py -v`

Expected: collection fails because `data.segment_features` does not exist.

- [ ] **Step 3: Implement segmentation and burst rules**

```python
@dataclass(frozen=True)
class BurstAssignment:
    burst_ids: list[int]
    split_reasons: list[str]
    adaptive_threshold: float

def time_segment_packets(packets, window_seconds):
    ordered = sorted(packets, key=lambda packet: float(packet["timestamp"]))
    if not ordered:
        return []
    start = float(ordered[0]["timestamp"])
    buckets: dict[int, list[dict[str, Any]]] = {}
    for packet in ordered:
        index = int((float(packet["timestamp"]) - start) // float(window_seconds))
        buckets.setdefault(index, []).append(dict(packet))
    return [buckets[index] for index in sorted(buckets)]
```

Implement `assign_bursts_with_reasons` with priority `direction_change`, then `iat_gap`, then `duration_cap`; the current packet opens the new burst. Compute `T_segment` from positive IAT using the existing median + alpha * IQR definition.

- [ ] **Step 4: Add tests for direction, IAT, zero-IAT and duration collection**

```python
def test_natural_burst_duration_collection_excludes_single_packet_bursts():
    packets = [
        {"timestamp": 0.0, "direction": 1.0},
        {"timestamp": 0.1, "direction": 1.0},
        {"timestamp": 0.2, "direction": -1.0},
    ]
    assignment = assign_bursts_with_reasons(packets, alpha=1.0)
    assert collect_mult_packet_burst_durations(packets, assignment) == [0.1]
```

- [ ] **Step 5: Run focused tests**

Run: `pytest tests/test_segment_features.py -v`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```text
git add data/segment_features.py tests/test_segment_features.py
git commit -m "feat: 增加流片段与自适应burst核心逻辑"
```

### Task 2: Lossless capacity packing

**Files:**
- Modify: `data/segment_features.py`
- Modify: `tests/test_segment_features.py`

**Interfaces:**
- Produces: `CapacitySample(packets, burst_ids, split_reason)`.
- Produces: `pack_by_burst_capacity(packets, assignment, max_packets, max_bursts) -> list[CapacitySample]`.

- [ ] **Step 1: Write failing capacity tests**

```python
def test_capacity_packing_preserves_every_packet_once():
    packets = [{"frame_index": i, "timestamp": i * 0.01, "direction": 1 if i < 3 else -1} for i in range(8)]
    assignment = BurstAssignment([0, 0, 0, 1, 1, 1, 2, 2], ["flow_start", "", "", "direction_change", "", "", "direction_change", ""], 1.0)
    samples = pack_by_burst_capacity(packets, assignment, max_packets=5, max_bursts=2)
    observed = [packet["frame_index"] for sample in samples for packet in sample.packets]
    assert observed == list(range(8))
    assert all(len(sample.packets) <= 5 for sample in samples)
    assert all(len(set(sample.burst_ids)) <= 2 for sample in samples)

def test_single_oversized_burst_uses_packet_capacity_cap():
    packets = [{"frame_index": i, "timestamp": i * 0.001, "direction": 1} for i in range(7)]
    assignment = BurstAssignment([0] * 7, ["flow_start"] + [""] * 6, 1.0)
    samples = pack_by_burst_capacity(packets, assignment, max_packets=3, max_bursts=2)
    assert [len(sample.packets) for sample in samples] == [3, 3, 1]
    assert any(sample.split_reason == "packet_capacity_cap" for sample in samples)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/test_segment_features.py -k capacity -v`

Expected: failure because capacity interfaces are absent.

- [ ] **Step 3: Implement whole-burst greedy packing and dense-burst fallback**

```python
@dataclass(frozen=True)
class CapacitySample:
    packets: list[dict[str, Any]]
    burst_ids: list[int]
    split_reason: str

def _renumber(ids):
    mapping = {old: new for new, old in enumerate(dict.fromkeys(ids))}
    return [mapping[value] for value in ids]
```

Group contiguous packets by burst ID, split an individual group into `max_packets` chunks only when that burst alone exceeds capacity, then greedily pack complete groups until either limit would be exceeded. Renumber burst IDs in every returned sample.

- [ ] **Step 4: Run focused tests**

Run: `pytest tests/test_segment_features.py -v`

Expected: all tests pass and packet order is unchanged.

- [ ] **Step 5: Commit**

```text
git add data/segment_features.py tests/test_segment_features.py
git commit -m "feat: 增加burst边界无损容量拆分"
```

### Task 3: Build aligned features from frozen burst IDs

**Files:**
- Modify: `data/burst_features.py`
- Modify: `tests/test_burst_features.py`

**Interfaces:**
- Extends: `build_flow_features(..., precomputed_burst_ids=None, truncate=True)`.
- New pipeline calls with `truncate=False` and capacity-safe inputs.

- [ ] **Step 1: Write failing no-truncation and alignment tests**

```python
import pytest

def test_precomputed_bursts_are_shared_by_packet_and_burst_views():
    packets = sample_packets()[:4]
    result = build_flow_features(packets, max_packets=4, max_bursts=3, precomputed_burst_ids=[0, 0, 0, 1], truncate=False)
    burst_size = PACKET_FEATURES.index("burst_size")
    burst_count = BURST_FEATURES.index("packet_count")
    assert result.packet_seq[:4, burst_size].tolist() == [3, 3, 3, 1]
    assert result.burst_seq[:2, burst_count].tolist() == [3, 1]

def test_new_pipeline_mode_rejects_overflow_instead_of_truncating():
    with pytest.raises(ValueError, match="capacity"):
        build_flow_features(sample_packets(), max_packets=2, max_bursts=4, truncate=False)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/test_burst_features.py -v`

Expected: failure due to unknown keyword arguments.

- [ ] **Step 3: Refactor feature construction**

Preserve legacy default `truncate=True` for existing callers. When `truncate=False`, raise if packet count exceeds `max_packets`. When `precomputed_burst_ids` is provided, validate equal length, monotonic contiguous IDs starting at zero, and burst count not exceeding `max_bursts`; use these IDs for both packet and burst tensors.

- [ ] **Step 4: Run backward compatibility tests**

Run: `pytest tests/test_burst_features.py tests/test_build_features.py -v`

Expected: legacy prefix test and new aligned-boundary tests all pass.

- [ ] **Step 5: Commit**

```text
git add data/burst_features.py tests/test_burst_features.py
git commit -m "feat: 支持冻结burst边界的对齐特征"
```

### Task 4: Deterministic stratified capture-group assignment

**Files:**
- Modify: `data/splits.py`
- Modify: `tests/test_splits.py`

**Interfaces:**
- Produces: `create_stratified_group_assignment(group_labels, val_ratio, test_ratio, seed, require_class_coverage=True) -> dict[str, str]`.
- Produces: `indices_from_group_assignment(groups, assignment) -> GroupSplit`.

- [ ] **Step 1: Write failing leakage and class-coverage tests**

```python
def test_stratified_group_assignment_covers_each_class_when_three_groups_exist():
    labels = {f"{label}-{index}": label for label in ["A", "B"] for index in range(5)}
    assignment = create_stratified_group_assignment(labels, 0.2, 0.2, 42, require_class_coverage=True)
    for label in ["A", "B"]:
        splits = {assignment[group] for group, value in labels.items() if value == label}
        assert splits == {"train", "val", "test"}
```

- [ ] **Step 2: Run test and verify failure**

Run: `pytest tests/test_splits.py -v`

Expected: import or attribute failure for the new function.

- [ ] **Step 3: Implement per-class deterministic allocation**

For each class, sort its unique groups, shuffle with `np.random.default_rng(seed)`, allocate at least one validation and one test group when the class has at least three groups, and leave at least one train group. Raise with the class name when `require_class_coverage=True` and coverage is mathematically impossible. When it is false, keep undersized classes in train and deterministically move groups from classes with spare groups until global train/val/test are non-empty.

- [ ] **Step 4: Preserve existing split API and run tests**

Run: `pytest tests/test_splits.py tests/test_training_contracts.py -v`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```text
git add data/splits.py tests/test_splits.py
git commit -m "feat: 增加采集组分层冻结划分"
```

### Task 5: One-file user pipeline and versioned artifacts

**Files:**
- Create: `data/run_segment_feature_pipeline.py`
- Create: `tests/test_segment_feature_pipeline.py`
- Modify: `requirements.txt` only if an existing dependency is missing; do not add new libraries for functionality available in NumPy/pandas.

**Interfaces:**
- Produces: `SegmentPipelineSettings` dataclass.
- Produces: `run_segment_pipeline(settings) -> dict[str, Any]`.
- User entry uses only top-of-file `CSV_DIR`, `OUTPUT_DIR`, and `RUN_MODE`.

- [ ] **Step 1: Write an integration test with synthetic Tor/NonTor CSV files**

```python
import numpy as np
import pandas as pd

def build_synthetic_iscxtor_csv_tree(tmp_path):
    root = tmp_path / "csv"
    for application in ["BROWSING", "EMAIL"]:
        for group_index in range(3):
            transport = "Tor" if group_index % 2 == 0 else "NonTor"
            path = root / transport / application / f"capture_{group_index}_packets.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            rows = []
            for frame, timestamp in enumerate([0.0, 0.1, 15.0, 15.1, 15.2]):
                forward = frame % 3 != 2
                rows.append({
                    "flow_id": f"flow-{group_index}",
                    "frame_index": frame,
                    "timestamp": timestamp,
                    "packet_length": 100 + frame,
                    "payload_length": 60 + frame,
                    "src_ip": "10.0.0.1" if forward else "10.0.0.2",
                    "dst_ip": "10.0.0.2" if forward else "10.0.0.1",
                    "src_port": 1234 if forward else 443,
                    "dst_port": 443 if forward else 1234,
                    "protocol": 6,
                    "ip_ttl": 64,
                    "tcp_flags": 16,
                })
            pd.DataFrame(rows).to_csv(path, index=False)
    return root

def test_segment_pipeline_writes_compatible_features_without_packet_loss(tmp_path):
    csv_root = build_synthetic_iscxtor_csv_tree(tmp_path)
    output = tmp_path / "processed"
    summary = run_segment_pipeline(SegmentPipelineSettings(csv_root, output, "smoke", max_packets=4, max_bursts=3))
    assert summary["input_packets"] == summary["modeled_packets"] + summary["ineligible_packets"]
    assert np.load(output / "features/packet_seq.npy").shape[1] == 4
    assert (output / "features/primary_labels.npy").exists()
    assert (output / "features/secondary_labels.npy").exists()
    assert (output / "features/split_seed42.npz").exists()
```

- [ ] **Step 2: Run integration test and verify failure**

Run: `pytest tests/test_segment_feature_pipeline.py -v`

Expected: module import fails.

- [ ] **Step 3: Implement discovery, labels, directions and initial manifests**

Reuse `_ensure_direction` behavior from `data/build_features.py`, `infer_labels` from `data/label_schema.py`, and the new group-assignment function. Treat each relative `*_packets.csv` path as the default capture group. Reject unknown Tor/application labels in full mode; smoke mode reports them and excludes them from model tensors. Smoke selection takes the three smallest CSV files for every application class available in the input so class-wise train/val/test assignment remains testable without reading the whole dataset.

- [ ] **Step 4: Implement two-pass train-only `D_max`**

Pass 1 builds initial 15-second segments and natural bursts, collecting durations only when `split == "train"` and burst packet count is at least two. Compute `float(np.quantile(durations, 0.95))`, error if the list is empty, and write the source split and quantile to `statistics/dmax_summary.json`.

- [ ] **Step 5: Implement final bursts, capacity samples and arrays**

Pass 2 applies frozen `D_max`, capacity packing, and `build_flow_features(..., precomputed_burst_ids=..., truncate=False)`. Use fixed label order `NONTOR, TOR` for primary and `APPLICATION_LABELS` for secondary. Write `packet_seq`, `packet_mask`, `burst_seq`, `burst_mask`, compatible label files, semantic aliases, sample/group IDs, label mappings, manifests and summary JSON via temporary files followed by atomic replacement. Write `.pipeline_success.json` only after all conservation and leakage assertions pass.

- [ ] **Step 6: Write split indices compatible with training**

Build train/val/test indices from final sample `capture_group` values, save `features/split_seed42.npz` with the existing `save_group_split`, and assert all three sets are non-empty in smoke and full modes.

- [ ] **Step 7: Add rerun and invalid-input tests**

```python
def run_fixture_pipeline(tmp_path):
    csv_root = build_synthetic_iscxtor_csv_tree(tmp_path)
    output = tmp_path / "processed"
    run_segment_pipeline(SegmentPipelineSettings(csv_root, output, "smoke", max_packets=4, max_bursts=3))
    return output

def run_pipeline_with_missing_column(tmp_path, column):
    csv_root = build_synthetic_iscxtor_csv_tree(tmp_path)
    target = next(csv_root.rglob("*_packets.csv"))
    frame = pd.read_csv(target).drop(columns=[column])
    frame.to_csv(target, index=False)
    return run_segment_pipeline(SegmentPipelineSettings(csv_root, tmp_path / "broken", "smoke", max_packets=4, max_bursts=3))

def test_pipeline_is_deterministic_on_rerun(tmp_path):
    first = run_fixture_pipeline(tmp_path)
    first_keys = np.load(first / "features/sample_keys.npy", allow_pickle=True)
    second = run_fixture_pipeline(tmp_path)
    np.testing.assert_array_equal(first_keys, np.load(second / "features/sample_keys.npy", allow_pickle=True))

def test_pipeline_rejects_missing_timestamp(tmp_path):
    with pytest.raises(ValueError, match="timestamp"):
        run_pipeline_with_missing_column(tmp_path, "timestamp")
```

- [ ] **Step 8: Run pipeline tests**

Run: `pytest tests/test_segment_feature_pipeline.py tests/test_pcap_pipeline.py -v`

Expected: all tests pass.

- [ ] **Step 9: Commit**

```text
git add data/run_segment_feature_pipeline.py tests/test_segment_feature_pipeline.py requirements.txt
git commit -m "feat: 增加ISCXTor片段特征一键入口"
```

### Task 6: Experiment configuration and user instructions

**Files:**
- Create: `experiments/configs/smoke/application8_segment15_burstp95_smoke_v1.yaml`
- Modify: `README.md`
- Modify: `docs/experiment_protocol.md`
- Modify: `docs/superpowers/specs/2026-07-16-segment-burst-preprocessing-v1-design.md`

**Interfaces:**
- Smoke config points `features_dir` and `split_file` to the new versioned output.

- [ ] **Step 1: Add shipped-config test expectation**

Extend `tests/test_shipped_configs.py` so loading the new YAML asserts `task == "application8"`, `fusion == "gated"`, `feature_id == "segment15_burstp95_v1"`, and the split file name is `split_seed42.npz`.

- [ ] **Step 2: Add the configuration**

```yaml
experiment_id: application8_segment15_burstp95_smoke_v1
task: application8
fusion: gated
loss: focal
seed: 42
feature_id: segment15_burstp95_v1
split_id: capture_group_seed42
features_dir: /data3/wsb_workspace/study/data/Dual_data/processed/segment15_burstp95_v1/features
split_file: /data3/wsb_workspace/study/data/Dual_data/processed/segment15_burstp95_v1/features/split_seed42.npz
epochs: 3
batch_size: 32
```

- [ ] **Step 3: Document the server workflow**

Document: edit the three values at the top of the new Python file; run smoke; inspect the printed packet conservation, split leakage, label coverage and tensor shapes; change `RUN_MODE` to `full`; run full; then launch the smoke experiment YAML.

- [ ] **Step 4: Run config and documentation checks**

Run: `pytest tests/test_shipped_configs.py tests/test_experiment_configuration.py -v`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```text
git add experiments/configs/smoke/application8_segment15_burstp95_smoke_v1.yaml tests/test_shipped_configs.py README.md docs/experiment_protocol.md docs/superpowers/specs/2026-07-16-segment-burst-preprocessing-v1-design.md
git commit -m "docs: 补充片段特征服务器运行流程"
```

### Task 7: Full verification and GitHub handoff

**Files:**
- No production file changes unless verification finds a defect.

**Interfaces:**
- Produces remote branch `origin/feature/segment-burst-preprocessing-v1`.

- [ ] **Step 1: Run focused data tests**

Run: `pytest tests/test_segment_features.py tests/test_segment_feature_pipeline.py tests/test_burst_features.py tests/test_splits.py -v`

Expected: all tests pass.

- [ ] **Step 2: Run full repository tests**

Run: `pytest -q`

Expected: zero failures.

- [ ] **Step 3: Run static repository checks**

Run: `git diff --check`

Expected: no whitespace errors.

- [ ] **Step 4: Verify branch and commits**

Run: `git status --short --branch`

Expected: clean `feature/segment-burst-preprocessing-v1` working tree.

- [ ] **Step 5: Push the tested branch**

```text
git push -u origin feature/segment-burst-preprocessing-v1
```

Expected: GitHub reports the new remote branch and sets its upstream.

- [ ] **Step 6: Server checkout command**

```text
git fetch origin
git switch --track origin/feature/segment-burst-preprocessing-v1
```

If a local branch with that name already exists, use `git switch feature/segment-burst-preprocessing-v1` followed by `git pull --ff-only`.
