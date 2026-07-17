# Segment Pipeline v1.2 Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace v1.1 initial-segment balancing with a deterministic, iterative 80/10/10 capture-group split based on actual final model sample counts.

**Architecture:** Keep source PCAP/CSV as the indivisible group. Add a variable-group-count optimizer and quality evaluator in `data/splits.py`; add a count-only source pass and at most three train-only-D_max refinement rounds in the segment pipeline; build full feature arrays only after the candidate passes full-mode quality gates.

**Tech Stack:** Python 3.10+, NumPy, pandas, `concurrent.futures.ProcessPoolExecutor`, pytest, existing ISCXTor feature pipeline.

## Global Constraints

- Full split is train=0.80, val=0.10, test=0.10.
- Source PCAP/CSV groups never cross splits.
- `D_max=P95` is fitted from the current/final training groups only.
- Full mode accepts overall ratios only within ±0.03 and requires each application's val/test share to be at least 0.05.
- Smoke checks coverage and leakage but does not enforce 80/10/10.
- Maximum refinement rounds is 3 and all random decisions use the configured seed.
- New output id is `segment15_burstp95_v1_2`; v1 and v1.1 outputs remain untouched.
- Validation/test are never resampled; training-loss changes are out of scope.

---

### Task 1: Variable-count weighted group optimizer and quality report

**Files:**
- Modify: `data/splits.py`
- Modify: `tests/test_splits.py`

**Interfaces:**
- Produces: `SplitQualityReport(passed, violations, overall_ratios, application_ratios, primary_ratios)`.
- Produces: `evaluate_weighted_assignment(group_labels, group_weights, group_primary, assignment, val_ratio, test_ratio, overall_tolerance, min_class_holdout) -> SplitQualityReport`.
- Produces: `create_variable_weighted_group_assignment(group_labels, group_weights, group_primary, val_ratio, test_ratio, seed, trials) -> dict[str, str]`.

- [ ] **Step 1: Write failing tests for variable group counts and deterministic quality evaluation**

```python
def test_variable_optimizer_can_use_two_validation_groups_when_one_large_group_is_bad():
    labels = {f"A-{i}": "A" for i in range(8)} | {f"B-{i}": "B" for i in range(8)}
    weights = {group: value for group, value in zip(labels, [60, 20, 10, 5, 5, 4, 3, 3] * 2)}
    primary = {group: ("TOR" if i % 2 else "NONTOR") for i, group in enumerate(labels)}
    assignment = create_variable_weighted_group_assignment(labels, weights, primary, 0.10, 0.10, 42, trials=4000)
    report = evaluate_weighted_assignment(labels, weights, primary, assignment, 0.10, 0.10, 0.03, 0.05)
    assert assignment == create_variable_weighted_group_assignment(labels, weights, primary, 0.10, 0.10, 42, trials=4000)
    assert set(assignment.values()) == {"train", "val", "test"}
    assert report.application_ratios["A"]["val"] >= 0.05
    assert report.application_ratios["A"]["test"] >= 0.05

def test_quality_report_rejects_file_like_holdout_collapse():
    report = evaluate_weighted_assignment(
        {"big": "FILE", "small-v": "FILE", "small-t": "FILE"},
        {"big": 970, "small-v": 10, "small-t": 20},
        {"big": "TOR", "small-v": "NONTOR", "small-t": "NONTOR"},
        {"big": "train", "small-v": "val", "small-t": "test"},
        0.10, 0.10, 0.03, 0.05,
    )
    assert not report.passed
    assert any("FILE" in violation for violation in report.violations)
```

- [ ] **Step 2: Run the split tests and verify RED**

Run: `python -m pytest tests/test_splits.py -q`

Expected: import failure for the two new interfaces.

- [ ] **Step 3: Implement report calculation and deterministic variable-count search**

Implementation requirements:

```python
@dataclass(frozen=True)
class SplitQualityReport:
    passed: bool
    violations: tuple[str, ...]
    overall_ratios: dict[str, float]
    application_ratios: dict[str, dict[str, float]]
    primary_ratios: dict[str, dict[str, float]]
```

Candidate generation gives every application one train/val/test group, distributes remaining groups with a seeded multinomial centered on 80/10/10, permutes group identity, rejects missing global Tor/Non-Tor coverage, then selects the lexicographically best `(hard_violation_amount, weighted_ratio_error, deterministic_assignment)` candidate. Do not force a fixed number of groups per split.

- [ ] **Step 4: Run `tests/test_splits.py` and verify GREEN**

Run: `python -m pytest tests/test_splits.py -q`

Expected: all split tests pass.

---

### Task 2: Count-only final sample profiling and iterative train-only D_max refinement

**Files:**
- Modify: `data/run_segment_feature_pipeline.py`
- Modify: `tests/test_segment_feature_pipeline.py`

**Interfaces:**
- Consumes: Task 1 optimizer and quality evaluator.
- Produces: `SourceSampleCount(source, final_sample_count, modeled_packets, elapsed_seconds)`.
- Produces: `_count_source_samples_task((source, settings, selected_flows, dmax)) -> SourceSampleCount`.
- Produces: `SplitRefinementResult(assignment, dmax, dmax_train_groups, group_sample_counts, history, converged, quality)`.
- Produces: `_refine_group_assignment(sources, profiles, settings, selected_flows) -> SplitRefinementResult`.

- [ ] **Step 1: Write a failing count-equivalence test**

```python
def test_count_only_pass_matches_built_source_sample_count(tmp_path):
    csv_root = build_synthetic_iscxtor_csv_tree(tmp_path)
    source = _discover_sources(csv_root)[0]
    settings = make_settings(csv_root, tmp_path / "out")
    count = _count_source_samples_task((source, settings, None, 0.2))
    batch = _build_source_task((source, settings, None, 0.2, "train"))
    assert count.final_sample_count == len(batch.sample_keys)
    assert count.modeled_packets == batch.modeled_packets
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `python -m pytest tests/test_segment_feature_pipeline.py::test_count_only_pass_matches_built_source_sample_count -q`

Expected: import failure for `SourceSampleCount` or `_count_source_samples_task`.

- [ ] **Step 3: Implement the count-only worker by reusing Burst and capacity rules**

The worker iterates complete initial segments, skips segments below `min_model_packets`, calls `assign_bursts_with_reasons(..., max_duration=dmax)` and `pack_by_burst_capacity(...)`, and sums sample and modeled-packet counts without calling `build_flow_features`.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run: `python -m pytest tests/test_segment_feature_pipeline.py::test_count_only_pass_matches_built_source_sample_count -q`

Expected: pass.

- [ ] **Step 5: Write failing refinement tests**

Add a deterministic unit test for refinement state plus a full-mode synthetic fixture with at least ten source groups per application, then assert:

```python
assert result.group_sample_counts == expected_final_counts
assert len(result.history) <= 3
assert set(result.dmax_train_groups) == {
    group for group, split in result.assignment.items() if split == "train"
}
assert not success_marker.exists() when full quality fails
```

Use the real synthetic CSV integration for the success-marker assertion; only the unit test may inject the count-pass callable to isolate iteration state.

- [ ] **Step 6: Run the refinement tests and verify RED**

Run: `python -m pytest tests/test_segment_feature_pipeline.py -q`

Expected: failures because refinement and full quality gating are absent.

- [ ] **Step 7: Implement at most three refinement rounds before feature construction**

Round logic:

```python
assignment = initial_assignment_from_eligible_segments
for round_index in range(1, settings.max_split_iterations + 1):
    dmax = training_only_quantile(profiles, assignment)
    counts = parallel_count_all_sources(dmax)
    new_assignment = create_variable_weighted_group_assignment(...counts...)
    quality = evaluate_weighted_assignment(...counts..., new_assignment, ...)
    record_history(round_index, dmax, assignment, new_assignment, quality)
    if new_assignment == assignment:
        converged = True
        assignment = new_assignment
        break
    assignment = new_assignment
final_dmax = training_only_quantile(profiles, assignment)
final_counts = parallel_count_all_sources(final_dmax)
final_quality = evaluate_weighted_assignment(...final_counts..., assignment, ...)
```

Full mode writes diagnostics and raises before full feature arrays when `final_quality.passed` is false. Smoke skips ratio gates but retains coverage and leakage checks.

- [ ] **Step 8: Run pipeline and parity tests and verify GREEN**

Run: `python -m pytest tests/test_segment_feature_pipeline.py -q`

Expected: all pipeline tests pass, including workers=1/2 parity.

---

### Task 3: v1.2 audit artifacts, configuration, and operator documentation

**Files:**
- Modify: `data/run_segment_feature_pipeline.py`
- Modify: `README.md`
- Modify: `docs/experiment_protocol.md`
- Rename: `experiments/configs/smoke/application8_segment15_burstp95_smoke_v1_1.yaml` to `application8_segment15_burstp95_smoke_v1_2.yaml`
- Modify: `tests/test_shipped_configs.py`
- Modify: `tests/test_segment_feature_pipeline.py`

**Interfaces:**
- Consumes: Task 2 refinement result and final batch manifests.
- Produces: `split_iteration_history.csv`, `group_weight_audit.csv`, expanded `split_balance.csv`, and expanded `split_balance_summary.json`.

- [ ] **Step 1: Add failing assertions for v1.2 paths and audit files**

```python
assert config.feature_id == "segment15_burstp95_v1_2"
assert (output / "manifests/split_iteration_history.csv").exists()
assert (output / "manifests/group_weight_audit.csv").exists()
summary = json.loads((output / "statistics/split_balance_summary.json").read_text())
assert summary["target_ratios"] == {"train": 0.8, "val": 0.1, "test": 0.1}
```

- [ ] **Step 2: Run config and pipeline tests and verify RED**

Run: `python -m pytest tests/test_shipped_configs.py tests/test_segment_feature_pipeline.py -q`

Expected: v1.1 ids and missing audit files fail.

- [ ] **Step 3: Write v1.2 artifacts atomically and update configuration/docs**

Set `VAL_RATIO=0.10`, `TEST_RATIO=0.10`, `MAX_SPLIT_ITERATIONS=3`, `OVERALL_SPLIT_TOLERANCE=0.03`, `MIN_CLASS_HOLDOUT_RATIO=0.05`, output directory `segment15_burstp95_v1_2`, experiment id suffix `_v1_2`, and split id `iterative_weighted_capture_group_80_10_10_seed42_v1_2`.

Document that full processing reuses CSV, smoke does not enforce 80/10/10, and failed quality gates leave diagnostics without a success marker.

- [ ] **Step 4: Run config and pipeline tests and verify GREEN**

Run: `python -m pytest tests/test_shipped_configs.py tests/test_segment_feature_pipeline.py -q`

Expected: all selected tests pass.

---

### Task 4: Full verification, commit, and remote handoff

**Files:** All modified files from Tasks 1-3.

- [ ] **Step 1: Run direct-entry, dry-run, and complete tests**

Run:

```text
python -c "import runpy; runpy.run_path('data/run_segment_feature_pipeline.py', run_name='not_main')"
python experiments/run_experiment.py --config experiments/configs/smoke/application8_segment15_burstp95_smoke_v1_2.yaml --dry-run
python -m pytest -q
git diff --check
```

Expected: all commands exit zero and pytest reports no failures.

- [ ] **Step 2: Review generated test audits and source diff**

Confirm synthetic output contains target 0.80/0.10/0.10, train-only `D_max`, no group overlap, packet conservation, iteration history, and no stale v1.1 paths in active v1.2 config/docs.

- [ ] **Step 3: Commit and push the existing feature branch**

```text
git add README.md data docs experiments tests
git commit -m "feat: 增加80-10-10迭代划分管线v1.2"
git push origin feature/segment-burst-preprocessing-v1
```

- [ ] **Step 4: Provide the server pull and smoke command**

Server commands remain `git pull --ff-only origin feature/segment-burst-preprocessing-v1` and `python -m data.run_segment_feature_pipeline`.
