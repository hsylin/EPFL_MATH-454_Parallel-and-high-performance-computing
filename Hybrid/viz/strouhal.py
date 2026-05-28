#!/usr/bin/env python3
"""Estimate the Strouhal number from an LBM probe CSV.

Expected CSV columns: step,ux,uy. By default the transverse velocity `uy` is
used because vortex shedding appears as an alternating cross-flow signal.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def read_probe(path: Path, component: str) -> tuple[np.ndarray, np.ndarray]:
    steps: list[float] = []
    vals: list[float] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"step", component}
        if not required.issubset(reader.fieldnames or []):
            raise SystemExit(f"{path} must contain columns including: step,{component}")
        for row in reader:
            steps.append(float(row["step"]))
            vals.append(float(row[component]))
    if len(steps) < 8:
        raise SystemExit("Need at least 8 probe samples for FFT")
    return np.asarray(steps, dtype=float), np.asarray(vals, dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("probe_csv", type=Path)
    ap.add_argument("--ny", type=float, required=True, help="global ny; default cylinder diameter is 0.05*ny")
    ap.add_argument("--u-in", type=float, required=True)
    ap.add_argument("--diameter", type=float, default=None, help="cylinder diameter in lattice cells; overrides 0.05*ny")
    ap.add_argument("--component", choices=["ux", "uy"], default="uy")
    ap.add_argument("--discard-frac", type=float, default=0.25, help="discard initial transient fraction before FFT")
    args = ap.parse_args()

    steps, signal = read_probe(args.probe_csv, args.component)
    n0 = int(max(0, min(0.9, args.discard_frac)) * len(signal))
    steps = steps[n0:]
    signal = signal[n0:]
    if len(signal) < 8:
        raise SystemExit("Too few samples after transient discard")

    dt = float(np.median(np.diff(steps)))
    if not np.isfinite(dt) or dt <= 0:
        raise SystemExit("Invalid/non-increasing probe step spacing")

    y = signal - np.mean(signal)
    # Hann window reduces leakage when the probe does not contain an integer
    # number of shedding periods.
    y = y * np.hanning(len(y))
    freqs = np.fft.rfftfreq(len(y), d=dt)
    amp = np.abs(np.fft.rfft(y))
    if len(freqs) < 2:
        raise SystemExit("FFT failed")
    amp[0] = 0.0
    k = int(np.argmax(amp))
    f_shed = float(freqs[k])
    diameter = float(args.diameter if args.diameter is not None else 0.05 * args.ny)
    st = f_shed * diameter / float(args.u_in)

    print(f"Probe file: {args.probe_csv}")
    print(f"Component: {args.component}")
    print(f"Samples used: {len(signal)} after discarding {n0}")
    print(f"Peak shedding frequency: {f_shed:.8e} cycles/step")
    print(f"Cylinder diameter D: {diameter:.6g}")
    print(f"Strouhal number: {st:.6f}")


if __name__ == "__main__":
    main()
