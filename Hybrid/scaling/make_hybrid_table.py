#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "results_hybrid.csv"
OUT = ROOT / "hybrid_scaling.md"


def fnum(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def fmt(x: float) -> str:
    return "—" if math.isnan(x) else f"{x:.2f}"


def median(xs: list[float]) -> float:
    good = [x for x in xs if not math.isnan(x)]
    return statistics.median(good) if good else float("nan")


def main() -> None:
    if not CSV.exists():
        raise SystemExit(f"missing {CSV}")

    raw = []
    with CSV.open(newline="") as f:
        for r in csv.DictReader(f):
            r["ranks_i"] = int(r["ranks"])
            r["mlups_f"] = fnum(r["mlups"])
            r["wall_f"] = fnum(r["wall_time_s"])
            # Prefer correctly named percent columns, but keep compatibility
            # with older files whose column names said "fraction" even though
            # the values were printed as percentages.
            r["max_halo_f"] = fnum(r.get("max_halo_overhead_percent",
                                           r.get("max_halo_overhead_fraction",
                                                 r.get("halo_overhead_fraction",
                                                       r.get("comm_fraction", "nan")))))
            r["avg_halo_f"] = fnum(r.get("avg_halo_overhead_percent",
                                           r.get("avg_halo_overhead_fraction", "nan")))
            r["max_other_f"] = fnum(r.get("max_other_time_s", "nan"))
            r["avg_other_f"] = fnum(r.get("avg_other_time_s", "nan"))
            raw.append(r)

    grouped = defaultdict(list)
    for r in raw:
        key = (
            r["ranks"], r["halo"], r["grid"], r["nx"], r["ny"], r["steps"],
            r["block_x"], r["block_y"], r["pitch_align"],
        )
        grouped[key].append(r)

    rows = []
    for key, vals in grouped.items():
        ranks, halo, grid, nx, ny, steps, block_x, block_y, pitch_align = key
        rows.append({
            "ranks": ranks,
            "ranks_i": int(ranks),
            "halo": halo,
            "grid": grid,
            "nx": nx,
            "ny": ny,
            "steps": steps,
            "block_x": block_x,
            "block_y": block_y,
            "pitch_align": pitch_align,
            "n": len(vals),
            "wall_f": median([v["wall_f"] for v in vals]),
            "mlups_f": median([v["mlups_f"] for v in vals]),
            "max_halo_f": median([v["max_halo_f"] for v in vals]),
            "avg_halo_f": median([v["avg_halo_f"] for v in vals]),
            "max_other_f": median([v["max_other_f"] for v in vals]),
            "avg_other_f": median([v["avg_other_f"] for v in vals]),
        })

    lines = ["# Hybrid MPI+CUDA scaling results", ""]
    lines.append(
        "This table is generated from `scaling/results_hybrid.csv`. "
        "For fair performance numbers, run with `every=0 probe=/dev/null probe_every=0`. "
        "If `REPEATS>1`, each row reports the median across repeats. "
        "The `Other` columns expose any residual timing outside compute and halo exchange, "
        "for example probe/output overhead if accidentally enabled."
    )
    lines.append("")

    lines.append("## All runs")
    lines.append("")
    lines.append("| Ranks/GPUs | Halo | Grid | Steps | N | Wall median (s) | MLUPS median | Speedup vs shared median 1 GPU same grid | Efficiency | Max halo overhead % | Avg halo overhead % | Max other (s) | Avg other (s) |")
    lines.append("|-----------:|:-----|:-----|------:|--:|----------------:|-------------:|------------------------------------:|-----------:|--------------------:|--------------------:|--------------:|--------------:|")

    # For ranks=1, halo=cuda and halo=staged do not exchange any halo data, so
    # they are the same logical 1-GPU baseline. Pool both modes to reduce noise.
    baseline_values = defaultdict(list)
    for r in rows:
        if r["ranks_i"] == 1 and not math.isnan(r["mlups_f"]):
            baseline_values[r["grid"]].append(r["mlups_f"])
    baseline = {key: median(vals) for key, vals in baseline_values.items()}

    for r in sorted(rows, key=lambda x: (x["grid"], x["halo"], x["ranks_i"])):
        base = baseline.get(r["grid"], float("nan"))
        speedup = r["mlups_f"] / base if base and not math.isnan(base) else float("nan")
        eff = speedup / r["ranks_i"] * 100 if not math.isnan(speedup) else float("nan")
        lines.append(
            f"| {r['ranks']} | `{r['halo']}` | {r['grid']} | {r['steps']} | {r['n']} | "
            f"{fmt(r['wall_f'])} | {fmt(r['mlups_f'])} | {fmt(speedup)} | {fmt(eff)}% | "
            f"{fmt(r['max_halo_f'])}% | {fmt(r['avg_halo_f'])}% | "
            f"{fmt(r['max_other_f'])} | {fmt(r['avg_other_f'])} |"
        )

    lines.append("")
    lines.append("## Best median run")
    lines.append("")
    valid = [r for r in rows if not math.isnan(r["mlups_f"])]
    if valid:
        b = max(valid, key=lambda x: x["mlups_f"])
        lines.append(
            f"Best median MLUPS: **{fmt(b['mlups_f'])}** with `{b['halo']}`, "
            f"ranks={b['ranks']}, grid={b['grid']}, steps={b['steps']}, N={b['n']}."
        )
        lines.append("")

    OUT.write_text("\n".join(lines))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
