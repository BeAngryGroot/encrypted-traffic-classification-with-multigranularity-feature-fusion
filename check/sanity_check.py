#!/usr/bin/env python3
import numpy as np, pickle, sys
from pathlib import Path

p = Path(sys.argv[1])
def load(name): return np.load(p/name)

def rpt(name, arr, head=3):
    print(f"[{name}] shape={arr.shape}, dtype={arr.dtype}")
    if arr.size == 0:
        print("  (empty)"); return
    finite = np.isfinite(arr)
    print("  finite%:", finite.mean()*100)
    print("  min/max:", np.nanmin(arr), np.nanmax(arr))

micro = load("micro_seq.npy")
macro = load("macro_bag.npy")
mi_mk = load("micro_mask.npy")
ma_mk = load("macro_mask.npy")
y_p = load("primary_labels.npy")
y_s = load("secondary_labels.npy")
y_c = load("combined_labels.npy")

with open(p/"label_mappings.pkl", "rb") as f:
    lm = pickle.load(f)

print("=== arrays ===")
rpt("micro_seq", micro)
rpt("macro_bag", macro)
rpt("micro_mask", mi_mk)
rpt("macro_mask", ma_mk)

print("\n=== shapes check ===")
assert micro.ndim==3, micro.shape
N, L, D = micro.shape
assert mi_mk.shape == (N, L), mi_mk.shape
assert macro.shape[0]==N, macro.shape
assert ma_mk.shape == (N,1), ma_mk.shape

print("\n=== masks check ===")
print("micro_mask unique:", np.unique(mi_mk)[:10])
print("macro_mask unique:", np.unique(ma_mk)[:10])
assert set(np.unique(mi_mk)).issubset({0,1}), "micro_mask 非 0/1"
assert set(np.unique(ma_mk)).issubset({0,1}), "macro_mask 非 0/1"

print("\n=== labels check ===")
for mode, y, key in [
    ("primary", y_p, "primary"),
    ("secondary", y_s, "secondary"),
    ("combined", y_c, "combined"),
]:
    ncls = lm[key]["num_classes"]
    u = np.unique(y)
    print(f"{mode}: num_classes={ncls}, unique={u[:20]} (count={len(u)})")
    assert u.min()>=0 and u.max()<ncls, f"{mode} 标签越界: [{u.min()}, {u.max()}] vs num_classes={ncls}"

print("\n=== value range quick look ===")
for name, arr in [("micro_seq", micro), ("macro_bag", macro)]:
    a = arr.reshape(-1, arr.shape[-1]) if arr.ndim==3 else arr
    q = np.quantile(a, [0, 0.5, 0.9, 0.99, 1.0], axis=0)
    print(f"{name}: abs_max={np.max(np.abs(a))}")
print("\nOK.")
