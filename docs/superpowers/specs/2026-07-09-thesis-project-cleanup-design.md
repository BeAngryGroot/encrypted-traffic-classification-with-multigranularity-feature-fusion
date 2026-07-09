# Thesis Project Cleanup Design

## Goal

Clean the repository so old experimental artifacts cannot pollute future runs, then align the code with the thesis route: adaptive same-direction burst segmentation, burst-aware packet sequences, true burst token sequences, and Mamba + Transformer gated fusion.

## Chosen Approach

Use an in-place refactor of the current repository. The existing Mamba branch, Transformer branch, fusion layer, and pcap parser remain useful. The main change is to replace the old global macro-vector feature path with a real burst sequence feature path and to add clear experiment folders.

## Repository Layout

- `data/`: data parsing and feature-building code.
- `model/`: neural network modules and training/evaluation scripts.
- `experiments/`: experiment presets and lightweight orchestration.
- `artifacts/features/`: generated feature tensors, ignored by Git.
- `artifacts/checkpoints/`: trained model weights, ignored by Git.
- `artifacts/results/`: metrics and prediction outputs, ignored by Git.
- `tests/`: small regression tests for feature construction and model I/O assumptions.
- `docs/`: project workflow, experiment protocol, and implementation notes.

## Required Behavioral Changes

- Remove tracked old feature, checkpoint, and evaluation result files.
- Ignore future generated `.npy`, `.pt`, and result artifacts.
- Generate `packet_seq.npy`, `packet_mask.npy`, `burst_seq.npy`, and `burst_mask.npy`.
- Keep legacy aliases only where they help transition, but training should use the new names first.
- Do not repeat a single macro vector into fake Transformer tokens.
- Add tests for adaptive burst segmentation and tensor construction.

## Verification

- Run the feature tests with `pytest`.
- Run a small synthetic feature-building smoke test.
- Run Python syntax checks for data and model scripts.
- If PyTorch is unavailable in the current environment, report that full training was not executed.
