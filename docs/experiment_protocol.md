# Experiment Protocol

## Data Source

Use encrypted traffic `pcap/pcapng` files with labels recoverable from file or directory names. The current label convention is:

- Primary label: `VPN`, `TOR`, `QUIC`, or `OTHER`.
- Secondary label: `AUDIO`, `CHAT`, `FILE`, `VIDEO`, `VOIP`, or `UNKNOWN`.
- Combined label: `<primary>:<secondary>`.

If a filename does not contain a known token, the builder falls back to `OTHER:UNKNOWN`.

## Feature Building

For each flow:

1. Sort packets by timestamp.
2. Infer packet direction from the first observed endpoint pair when no explicit direction column exists.
3. Compute IAT.
4. Split adaptive same-direction bursts with:

```text
T_flow = median(IAT) + alpha * IQR(IAT)
```

5. Build `packet_seq` with burst context.
6. Build `burst_seq` as ordered burst tokens.

## Core Model

- Mamba branch consumes `packet_seq`.
- Transformer branch consumes `burst_seq`.
- Fusion projects both branches to a common hidden size.
- Fusion modes:
  - `gated`: dynamic sample-wise fusion.
  - `concat`: concatenation followed by projection.
  - `fixed`: fixed weighted fusion.
  - `micro_only`: Mamba-only ablation.
  - `burst_only`: Transformer-only ablation.

## Required Tables

Main result table:

```text
Method | Input | Accuracy | Macro-F1 | Weighted-F1 | Params | Time
```

Ablation table:

```text
Variant | Removed/Changed Component | Macro-F1 | Weighted-F1 | Delta
```

Sensitivity table:

```text
Parameter | Value | Avg Burst Count | Macro-F1 | Weighted-F1
```
