#!/usr/bin/env python3
"""Estimate the Strouhal number from the probe CSV produced by the solver.

The probe records (step, ux, uy) at a fixed point a few diameters
downstream of the cylinder. For Karman vortex shedding the transverse
velocity uy oscillates at the shedding frequency f. The Strouhal number is

    St = f * D / U

where D is the cylinder diameter (in lattice cells) and U is the inlet
velocity (lattice units). At Re ~ 100 the textbook value is St ~ 0.16.

Usage:
    python3 strouhal.py probe.csv --u-in 0.05 --diameter 20

Optional flags trim the transient and choose how many leading samples to
discard before the FFT.
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv",   type=Path, help="probe CSV produced by the solver")
    ap.add_argument("--u-in",     type=float, required=True,
                    help="inlet velocity in lattice units")
    ap.add_argument("--diameter", type=float, required=True,
                    help="cylinder diameter in lattice cells")
    ap.add_argument("--skip",     type=float, default=0.5,
                    help="fraction of the signal to discard at the start "
                         "(transient). Default 0.5.")
    ap.add_argument("--plot", action="store_true",
                    help="show a plot of uy(t) and the FFT magnitude")
    args = ap.parse_args()

    if not args.csv.exists():
        sys.exit(f"File not found: {args.csv}")

    data = np.loadtxt(args.csv, delimiter=",", skiprows=1)
    if data.ndim != 2 or data.shape[1] < 3:
        sys.exit("CSV must have columns: step, ux, uy")

    step, ux, uy = data[:, 0], data[:, 1], data[:, 2]

    n0 = int(args.skip * len(step))
    if n0 >= len(step) - 16:
        sys.exit("Not enough samples after --skip; record more steps.")

    t  = step[n0:].astype(float)        # 1 lattice step apart
    s  = uy[n0:] - np.mean(uy[n0:])     # remove DC
    n  = len(s)
    dt = 1.0                            # one lattice unit per sample

    # FFT.
    F  = np.fft.rfft(s)
    f  = np.fft.rfftfreq(n, d=dt)       # cycles per lattice step
    mag = np.abs(F)
    mag[0] = 0.0                        # ignore DC

    k = int(np.argmax(mag))
    f_peak = f[k]
    if f_peak <= 0:
        sys.exit("No clear peak found in the FFT — increase steps or "
                 "decrease --skip.")

    St = f_peak * args.diameter / args.u_in
    print(f"samples used : {n}  (skipped first {n0})")
    print(f"peak freq    : {f_peak:.6e}  cycles per lattice step")
    print(f"period       : {1.0 / f_peak:.2f}  steps")
    print(f"St           : {St:.4f}    (literature ~ 0.16 at Re 100)")

    if args.plot:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 1, figsize=(8, 6))
        axs[0].plot(t, s)
        axs[0].set(xlabel="step", ylabel="uy - <uy>",
                   title="probe transverse velocity (after transient)")
        axs[1].semilogy(f, mag)
        axs[1].axvline(f_peak, color="red", ls="--", label=f"f = {f_peak:.2e}")
        axs[1].set(xlabel="frequency (1/step)", ylabel="|FFT|",
                   title="spectrum")
        axs[1].legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
