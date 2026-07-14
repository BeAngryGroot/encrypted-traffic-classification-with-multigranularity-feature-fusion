# ISCXTor One-File Converter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one editable Python entrypoint for reusable smoke/full ISCXTor PCAP conversion with logs and resume markers.

**Architecture:** The entrypoint imports the verified conversion core from `pcap_to_csv.py`, owns only configuration, selection, validation, progress, completion markers, and summary persistence. A `.conversion_done.json` marker is written only after both CSV files are successfully generated.

**Tech Stack:** Python 3.10+, standard library, Scapy through the existing parser, pytest.

## Global Constraints

- User edits only `RAW_DIR`, `OUTPUT_DIR`, and `RUN_MODE` for normal use.
- Core comments and user-facing messages are Chinese.
- Do not duplicate PCAP parser logic.
- Do not push GitHub.
- Production behavior follows TDD.

---

### Task 1: Selection, paths, and completion markers

**Files:**
- Create: `data/run_iscxtor_pipeline.py`
- Create: `tests/test_run_iscxtor_pipeline.py`

**Interfaces:**
- Produces: `select_pcaps(pcaps, mode) -> list[Path]`
- Produces: `output_paths(pcap, raw_root, csv_root) -> ConversionPaths`
- Produces: `is_completed(paths) -> bool`
- Produces: `write_done_marker(paths, payload) -> None`

- [ ] Write tests asserting smoke selects the smallest file, full selects all files, and invalid mode is rejected.
- [ ] Run focused tests and verify failure because the module does not exist.
- [ ] Implement selection and relative output path helpers.
- [ ] Write marker tests requiring both CSV files and a valid done marker.
- [ ] Implement atomic marker writes and rerun focused tests.

### Task 2: End-to-end orchestration contract

**Files:**
- Modify: `data/run_iscxtor_pipeline.py`
- Modify: `tests/test_run_iscxtor_pipeline.py`

**Interfaces:**
- Produces: `validate_settings(settings) -> None`
- Produces: `run_pipeline(settings) -> list[dict[str, object]]`

- [ ] Write failing tests for missing raw directory, invalid mode, and low-disk full mode.
- [ ] Implement preflight validation and Chinese logging.
- [ ] Implement sequential smoke and parallel full scheduling through `process_one`.
- [ ] Persist `conversion_summary.csv` and log file; skip successful marker files.
- [ ] Run focused tests and then all tests.
- [ ] Parse all Python files with `ast.parse` and verify Git diff contains only planned files.

## Self-Review

- Spec coverage: editable settings, smoke/full, output mirroring, logging, resume markers, disk checks, and no parser duplication are covered.
- Placeholder scan: no TODO/TBD placeholders.
- Type consistency: helper signatures are defined once and shared by tests and orchestration.
