from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from data.burst_features import BURST_FEATURES, PACKET_FEATURES, build_flow_features
    from data.label_schema import infer_labels
except ImportError:  # pragma: no cover - supports direct script execution
    from burst_features import BURST_FEATURES, PACKET_FEATURES, build_flow_features
    from label_schema import infer_labels


def _packet_csvs(csv_dir: Path) -> list[Path]:
    return sorted(csv_dir.rglob("*_packets.csv"))


def _flow_key(row: pd.Series) -> tuple[Any, Any, Any, Any, Any]:
    return (
        row.get("src_ip", ""),
        row.get("src_port", ""),
        row.get("dst_ip", ""),
        row.get("dst_port", ""),
        row.get("protocol", row.get("proto", "")),
    )


def _reverse_flow_key(key: tuple[Any, Any, Any, Any, Any]) -> tuple[Any, Any, Any, Any, Any]:
    src_ip, src_port, dst_ip, dst_port, proto = key
    return (dst_ip, dst_port, src_ip, src_port, proto)


def _ensure_direction(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("timestamp" if "timestamp" in group.columns else "packet_time").copy()
    if "direction" in group.columns:
        group["direction"] = group["direction"].astype(float)
        return group

    first_key = _flow_key(group.iloc[0])
    reverse_key = _reverse_flow_key(first_key)
    directions = []
    for _, row in group.iterrows():
        key = _flow_key(row)
        if key == first_key:
            directions.append(1.0)
        elif key == reverse_key:
            directions.append(-1.0)
        else:
            directions.append(1.0 if row.get("src_ip", "") == first_key[0] else -1.0)
    group["direction"] = directions
    return group


def _records_for_group(group: pd.DataFrame) -> list[dict[str, Any]]:
    group = _ensure_direction(group)
    return group.replace({np.nan: None}).to_dict(orient="records")


def _build_mapping(labels: list[str]) -> dict[str, Any]:
    ordered = []
    seen = set()
    for label in labels:
        if label not in seen:
            ordered.append(label)
            seen.add(label)
    label_to_id = {label: i for i, label in enumerate(ordered)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    return {
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "num_classes": len(ordered),
    }


def _encode_labels(labels: list[str], mapping: dict[str, Any]) -> np.ndarray:
    return np.asarray([mapping["label_to_id"][label] for label in labels], dtype=np.int64)


def build_features_from_csv_dir(
    csv_dir: str | Path,
    output_dir: str | Path,
    *,
    max_packets: int = 64,
    max_bursts: int = 32,
    alpha: float = 1.0,
    fixed_threshold: float | None = None,
    source_manifest: str | Path | None = None,
) -> dict[str, Any]:
    csv_dir = Path(csv_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    packet_arrays = []
    packet_masks = []
    burst_arrays = []
    burst_masks = []
    sample_keys: list[str] = []
    primary_labels: list[str] = []
    secondary_labels: list[str] = []
    combined_labels: list[str] = []
    group_ids: list[str] = []
    manifest_rows: list[dict[str, str]] = []

    source_overrides: dict[str, dict[str, Any]] = {}
    if source_manifest is not None:
        manifest_df = pd.read_csv(source_manifest)
        if "source_key" not in manifest_df.columns:
            raise ValueError("source_manifest must contain source_key")
        source_overrides = {str(row["source_key"]).replace("\\", "/"): row.to_dict() for _, row in manifest_df.iterrows()}

    packet_csvs = _packet_csvs(csv_dir)
    if not packet_csvs:
        raise FileNotFoundError(f"No *_packets.csv files found under {csv_dir}")

    for packet_csv in packet_csvs:
        df = pd.read_csv(packet_csv)
        if "flow_id" not in df.columns:
            raise ValueError(f"{packet_csv} missing required column: flow_id")
        relative_source = str(packet_csv.relative_to(csv_dir)).replace("\\", "/")
        override = source_overrides.get(relative_source, source_overrides.get(packet_csv.name, {}))
        label_info = infer_labels(packet_csv)
        primary = str(override.get("primary", label_info.primary))
        secondary = str(override.get("application", label_info.application))
        combined = f"{primary}:{secondary}"
        capture_group = str(override.get("capture_group", relative_source))
        source_name = packet_csv.stem.removesuffix("_packets")

        for flow_id, group in df.groupby("flow_id", sort=False):
            records = _records_for_group(group)
            if not records:
                continue
            result = build_flow_features(
                records,
                max_packets=max_packets,
                max_bursts=max_bursts,
                alpha=alpha,
                fixed_threshold=fixed_threshold,
            )
            packet_arrays.append(result.packet_seq)
            packet_masks.append(result.packet_mask)
            burst_arrays.append(result.burst_seq)
            burst_masks.append(result.burst_mask)
            sample_keys.append(f"{source_name}:{flow_id}")
            primary_labels.append(primary)
            secondary_labels.append(secondary)
            combined_labels.append(combined)
            group_ids.append(capture_group)
            manifest_rows.append({
                "sample_key": sample_keys[-1],
                "source_key": relative_source,
                "capture_group": capture_group,
                "primary": primary,
                "application": secondary,
                "combined": combined,
            })

    if not packet_arrays:
        raise ValueError(f"No flow samples built from {csv_dir}")

    mappings = {
        "primary": _build_mapping(primary_labels),
        "secondary": _build_mapping(secondary_labels),
        "combined": _build_mapping(combined_labels),
        "packet_features": PACKET_FEATURES,
        "burst_features": BURST_FEATURES,
    }

    packet_seq = np.stack(packet_arrays).astype(np.float32)
    packet_mask = np.stack(packet_masks).astype(np.float32)
    burst_seq = np.stack(burst_arrays).astype(np.float32)
    burst_mask = np.stack(burst_masks).astype(np.float32)
    primary_ids = _encode_labels(primary_labels, mappings["primary"])
    secondary_ids = _encode_labels(secondary_labels, mappings["secondary"])
    combined_ids = _encode_labels(combined_labels, mappings["combined"])

    np.save(output_dir / "packet_seq.npy", packet_seq)
    np.save(output_dir / "packet_mask.npy", packet_mask)
    np.save(output_dir / "burst_seq.npy", burst_seq)
    np.save(output_dir / "burst_mask.npy", burst_mask)
    np.save(output_dir / "primary_labels.npy", primary_ids)
    np.save(output_dir / "secondary_labels.npy", secondary_ids)
    np.save(output_dir / "combined_labels.npy", combined_ids)
    np.save(output_dir / "labels.npy", combined_ids)
    np.save(output_dir / "sample_keys.npy", np.asarray(sample_keys, dtype=str))
    np.save(output_dir / "group_ids.npy", np.asarray(group_ids, dtype=str))
    pd.DataFrame(manifest_rows).to_csv(output_dir / "sample_manifest.csv", index=False)
    with (output_dir / "label_mappings.pkl").open("wb") as f:
        pickle.dump(mappings, f)

    summary = {
        "num_samples": int(packet_seq.shape[0]),
        "packet_shape": list(packet_seq.shape),
        "burst_shape": list(burst_seq.shape),
        "max_packets": int(max_packets),
        "max_bursts": int(max_bursts),
        "alpha": float(alpha),
        "fixed_threshold": fixed_threshold,
        "packet_features": PACKET_FEATURES,
        "burst_features": BURST_FEATURES,
        "label_mappings": {
            key: value
            for key, value in mappings.items()
            if key in {"primary", "secondary", "combined"}
        },
    }
    (output_dir / "feature_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build packet and burst feature tensors from packet CSV files.")
    parser.add_argument("--csv_dir", required=True, help="Directory containing *_packets.csv files.")
    parser.add_argument("--output_dir", default="artifacts/features", help="Directory for generated .npy files.")
    parser.add_argument("--max_packets", type=int, default=64)
    parser.add_argument("--max_bursts", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--fixed_threshold", type=float, default=None)
    parser.add_argument("--source_manifest", default=None)
    args = parser.parse_args()

    summary = build_features_from_csv_dir(
        args.csv_dir,
        args.output_dir,
        max_packets=args.max_packets,
        max_bursts=args.max_bursts,
        alpha=args.alpha,
        fixed_threshold=args.fixed_threshold,
        source_manifest=args.source_manifest,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
