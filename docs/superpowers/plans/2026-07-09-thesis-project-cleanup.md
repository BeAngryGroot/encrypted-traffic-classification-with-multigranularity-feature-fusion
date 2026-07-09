# Thesis Project Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean old generated artifacts and refactor the project so experiments use adaptive burst features and real burst-token Transformer inputs.

**Architecture:** Keep the existing parser and model branches, add a focused burst feature module, route training through `packet_seq` and `burst_seq`, and move generated outputs under ignored `artifacts/` directories.

**Tech Stack:** Python, NumPy, Pandas, PyTorch, scikit-learn, pytest.

## Global Constraints

- Preserve the thesis backbone: adaptive same-direction burst representation plus Mamba + Transformer gated fusion.
- Do not claim burst itself is new; code should support adaptive segmentation and multi-granularity organization.
- Generated features, checkpoints, and results must not be tracked by Git.
- New feature tensors must include `packet_seq`, `packet_mask`, `burst_seq`, and `burst_mask`.

---

### Task 1: Clean Repository Artifacts

**Files:**
- Remove tracked generated files under `features/`, `features_all/`, `checkpoints/`, and `evaluation_results/`.
- Create: `.gitignore`
- Create: `artifacts/features/.gitkeep`
- Create: `artifacts/checkpoints/.gitkeep`
- Create: `artifacts/results/.gitkeep`

**Interfaces:**
- Produces ignored output directories for feature generation, training, and evaluation.

- [ ] Verify target paths are inside the repository.
- [ ] Remove tracked generated artifacts.
- [ ] Add `.gitignore` entries for future generated files.
- [ ] Run `git status --short` and confirm only expected deletes/adds appear.

### Task 2: Add Adaptive Burst Feature Module

**Files:**
- Create: `data/burst_features.py`
- Test: `tests/test_burst_features.py`

**Interfaces:**
- Produces `compute_adaptive_threshold(iats, alpha)`.
- Produces `assign_bursts(packets, alpha, fixed_threshold)`.
- Produces `build_flow_features(packets, max_packets, max_bursts, alpha, fixed_threshold)`.

- [ ] Write failing tests for adaptive segmentation and tensor shapes.
- [ ] Implement the feature module.
- [ ] Run `pytest tests/test_burst_features.py -q`.

### Task 3: Add Feature Builder CLI

**Files:**
- Create: `data/build_features.py`
- Modify: `data/micro_macro_features.py`

**Interfaces:**
- Consumes packet CSV files and flow CSV files.
- Produces `packet_seq.npy`, `packet_mask.npy`, `burst_seq.npy`, `burst_mask.npy`, label arrays, `label_mappings.pkl`, and `sample_keys.npy`.

- [ ] Implement CSV discovery and flow assembly.
- [ ] Build labels from filename/path conventions with explicit fallback to `UNKNOWN`.
- [ ] Save new feature tensors and transition aliases.
- [ ] Run a synthetic smoke test.

### Task 4: Update Training And Evaluation

**Files:**
- Modify: `model/train_optimized.py`
- Modify: `model/export_results.py`
- Modify: `model/fusion_head.py`
- Modify: `model/config.py`

**Interfaces:**
- Training reads true burst sequences.
- Fusion can expose gate weights for later analysis.
- Evaluation reads the same feature schema as training.

- [ ] Remove fake macro-vector repeat logic.
- [ ] Add feature schema fallback for old names only when needed.
- [ ] Add fusion mode support for `gated`, `concat`, `fixed`, `micro_only`, and `burst_only`.
- [ ] Run syntax checks.

### Task 5: Document The Workflow

**Files:**
- Modify: `README.md`
- Create: `docs/experiment_protocol.md`
- Create: `experiments/configs/full_combined.yaml`

**Interfaces:**
- Explains folder responsibilities and the end-to-end experiment flow.

- [ ] Document pcap-to-CSV, CSV-to-features, training, evaluation, and ablation flow.
- [ ] Document expected generated files and ignored directories.
- [ ] Run final smoke checks.
