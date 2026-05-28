#!/usr/bin/env python3
"""Compare two LBM probe CSV files produced by the hybrid solver.

Expected CSV columns: step,ux,uy
The script aligns rows by step and reports max/RMS differences.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def read_probe(path: Path) -> dict[int, tuple[float, float]]:
    data: dict[int, tuple[float, float]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"step", "ux", "uy"}
        if not required.issubset(reader.fieldnames or []):
            raise SystemExit(f"{path} must contain columns: step,ux,uy")
        for row in reader:
            step = int(float(row["step"]))
            data[step] = (float(row["ux"]), float(row["uy"]))
    return data


def rms(values: list[float]) -> float:
    return math.sqrt(sum(v * v for v in values) / len(values)) if values else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("probe_a", type=Path)
    ap.add_argument("probe_b", type=Path)
    ap.add_argument("--label-a", default="a")
    ap.add_argument("--label-b", default="b")
    args = ap.parse_args()

    a = read_probe(args.probe_a)
    b = read_probe(args.probe_b)
    common = sorted(set(a) & set(b))
    if not common:
        raise SystemExit("No common probe steps to compare")

    dux: list[float] = []
    duy: list[float] = []
    for step in common:
        ax, ay = a[step]
        bx, by = b[step]
        dux.append(ax - bx)
        duy.append(ay - by)

    max_abs_ux = max(abs(v) for v in dux)
    max_abs_uy = max(abs(v) for v in duy)
    rms_ux = rms(dux)
    rms_uy = rms(duy)

    print(f"Probe comparison: {args.label_a} vs {args.label_b}")
    print(f"Common samples: {len(common)}")
    print(f"Step range: {common[0]}..{common[-1]}")
    print(f"Max |Δux|: {max_abs_ux:.6e}")
    print(f"RMS  Δux : {rms_ux:.6e}")
    print(f"Max |Δuy|: {max_abs_uy:.6e}")
    print(f"RMS  Δuy : {rms_uy:.6e}")


if __name__ == "__main__":
    main()
