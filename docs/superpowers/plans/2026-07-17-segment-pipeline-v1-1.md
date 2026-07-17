# Segment Pipeline v1.1 Implementation Plan

**Goal:** Make smoke genuinely small, add deterministic weighted capture-group splitting and safe two-process source parallelism, and fix direct script execution.

**Architecture:** Profile selected complete flows per source to obtain initial-segment weights and natural-burst durations, search deterministic weighted group assignments, then build source feature batches in parallel and merge them in source-key order. Keep the v1 model and feature schema unchanged while writing a new v1.1 data directory.

**Tech Stack:** Python, pandas, NumPy, concurrent.futures, pytest, existing Git experiment branch.

## Global Constraints

- Preserve capture-group isolation and train-only `D_max=P95`.
- Smoke keeps at most five complete smallest flows per selected source.
- Default `WORKERS=2`; support `WORKERS=1` with identical outputs.
- New output id is `segment15_burstp95_v1_1`; do not overwrite v1.
- Use Chinese comments and one user-facing Python entry.

---

### Task 1: Direct-entry compatibility

**Files:** Modify `data/run_segment_feature_pipeline.py`; modify `tests/test_segment_feature_pipeline.py`.

- [x] Add a subprocess test running `runpy.run_path` from the `data` directory and assert exit code zero.
- [x] Run the test and verify the existing nested `data` import fails.
- [x] Insert the repository root into `sys.path` before package imports.
- [x] Run the focused test.

### Task 2: Weighted stratified group search

**Files:** Modify `data/splits.py`; modify `tests/test_splits.py`.

- [x] Add tests for deterministic assignment, group coverage and weighted ratio improvement.
- [x] Verify imports fail for `create_weighted_group_assignment`.
- [x] Implement `create_weighted_group_assignment(group_labels, group_weights, group_primary, val_ratio, test_ratio, seed, trials)` using deterministic candidate search and normalized ratio error.
- [x] Run split tests.

### Task 3: Complete-small-flow smoke profiling

**Files:** Modify `data/run_segment_feature_pipeline.py`; modify `tests/test_segment_feature_pipeline.py`.

- [x] Add tests showing only the smallest configured flow IDs are selected and every selected flow remains complete.
- [x] Verify the current pipeline lacks complete-flow smoke selection.
- [x] Add `smoke_flows_per_file`, sibling flow-summary lookup, chunked packet filtering and `SourceProfile`.
- [x] Profile all groups once, derive weighted assignment, then concatenate durations only from assigned training groups.
- [x] Run pipeline tests.

### Task 4: Source-level process parallelism

**Files:** Modify `data/run_segment_feature_pipeline.py`; modify `tests/test_segment_feature_pipeline.py`.

- [x] Add an integration test comparing workers=1 and workers=2 sample keys, tensors and split files.
- [x] Verify failure because settings do not support workers/source batches.
- [x] Add top-level picklable profile/build worker functions and deterministic `ProcessPoolExecutor.map` orchestration with per-source progress.
- [x] Merge per-source contiguous arrays in source order and preserve packet conservation.
- [x] Run integration and full tests.

### Task 5: Versioned configuration, audit and push

**Files:** Modify `README.md`, `docs/experiment_protocol.md`, `experiments/configs/smoke/application8_segment15_burstp95_smoke_v1_1.yaml`, `tests/test_shipped_configs.py`.

- [x] Update config tests to require feature id `segment15_burstp95_v1_1` and the new path.
- [x] Update the YAML, default output path, smoke/worker instructions and split audit documentation.
- [x] Save estimated and final split balance CSV/JSON with deviations.
- [x] Run direct-entry test, workers parity test, configuration dry-run and full pytest suite.
- [x] Verify the reviewed Git diff, commit documentation, and push the existing remote feature branch.
