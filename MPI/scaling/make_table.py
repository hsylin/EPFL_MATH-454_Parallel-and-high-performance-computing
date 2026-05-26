#!/usr/bin/env python3
"""
make_table.py — Parse scaling/logs/*.log and emit scaling/scaling.md.

Layout:

  * Summary box at the top with the "best" runs:
      - highest MLUPS
      - best strong speedup
      - best weak efficiency
  * One big sorted table of every successful run.
  * Per-mode tables for strong / weak scaling.

Important:
  * Strong scaling uses the p = 1 baseline of the same fixed global grid:
        key = (nx, ny, steps)

  * Weak scaling uses the p = 1 weak-scaling baseline:
        efficiency = T_weak(1) / T_weak(p)

Usage:
    python3 scaling/make_table.py --logs scaling/logs --out scaling/scaling.md
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

# Example lines printed by report_mpi_timers():
#
#     Wall time max       : 4.20312 s
#     Compute time max    : 3.90011 s
#     Comm time max       : 0.21234 s
#     Comm/max_total      : 5.1
#     MLUPS               : 2500.4
#
# This regex captures the metric name and numeric value.
_METRIC_RE = re.compile(
    r"^(?P<key>[A-Za-z][\w/ ]*?)\s*:\s*(?P<val>[-+0-9.eE]+)"
)

# Log filename format:
#
#     strong_p<P>_nx<NX>_ny<NY>_s<STEPS>.log
#     weak_p<P>_nx<NX>_ny<NY>_s<STEPS>.log
#
_TAG_RE = re.compile(
    r"^(?P<mode>strong|weak)_p(?P<p>\d+)_nx(?P<nx>\d+)_ny(?P<ny>\d+)_s(?P<steps>\d+)$"
)


@dataclass
class Run:
    config: str
    mode: str         # "strong" or "weak"
    p: int
    nx: int
    ny: int
    steps: int

    wall_max: float
    wall_avg: float
    compute_max: float
    compute_avg: float
    comm_max: float
    comm_avg: float

    comm_frac_max: float
    comm_frac_avg: float
    comm_over_total: float
    comm_over_compute: float

    mlups: float

    # Derived metrics
    speedup: Optional[float] = None
    efficiency: Optional[float] = None    # percentage


def _parse_log(path: Path) -> Optional[Run]:
    """Parse one log file.

    Returns None if:
      * the filename does not match the expected scaling-log pattern, or
      * the run did not finish and no MLUPS value was printed.
    """
    m = _TAG_RE.match(path.stem)
    if not m:
        return None

    metrics: dict[str, float] = {}

    for line in path.read_text(errors="replace").splitlines():
        m2 = _METRIC_RE.match(line.strip())
        if not m2:
            continue

        key = m2.group("key").strip()
        try:
            metrics[key] = float(m2.group("val"))
        except ValueError:
            pass

    # Treat a log without MLUPS as a failed or incomplete run.
    if "MLUPS" not in metrics:
        return None

    def g(key: str) -> float:
        return metrics.get(key, float("nan"))

    return Run(
        config=path.stem,
        mode=m.group("mode"),
        p=int(m.group("p")),
        nx=int(m.group("nx")),
        ny=int(m.group("ny")),
        steps=int(m.group("steps")),

        wall_max=g("Wall time max"),
        wall_avg=g("Wall time avg"),
        compute_max=g("Compute time max"),
        compute_avg=g("Compute time avg"),
        comm_max=g("Comm time max"),
        comm_avg=g("Comm time avg"),

        comm_frac_max=g("Comm fraction max"),
        comm_frac_avg=g("Comm fraction avg"),
        comm_over_total=g("Comm/max_total"),
        comm_over_compute=g("Comm/max_compute"),

        mlups=g("MLUPS"),
    )


def _attach_scaling_metrics(runs: list[Run]) -> None:
    """Fill in speedup and efficiency.

    Strong scaling:
        The global grid is fixed within each strong-scaling group.
        Therefore, every grid size must use its own p=1 baseline:

            speedup = T(1, same nx, same ny, same steps) / T(p, nx, ny, steps)
            efficiency = speedup / p * 100%

        This avoids the bug where 800x400, 1200x600, and 1600x800 all shared
        the same p=1 baseline.

    Weak scaling:
        The local work per rank is fixed, while the global problem size grows.
        Therefore, weak-scaling efficiency is:

            efficiency = T_weak(p=1) / T_weak(p) * 100%

        The "speedup" column is kept as the same ratio for convenience.
    """

    # ------------------------------------------------------------
    # Strong scaling baseline:
    # one separate p=1 baseline for each fixed global grid.
    # ------------------------------------------------------------
    strong_baseline: dict[tuple[int, int, int], float] = {}

    for r in runs:
        if r.mode == "strong" and r.p == 1 and r.wall_max > 0:
            key = (r.nx, r.ny, r.steps)
            strong_baseline[key] = r.wall_max

    for r in runs:
        if r.mode != "strong":
            continue
        if r.wall_max <= 0:
            continue

        key = (r.nx, r.ny, r.steps)
        t1 = strong_baseline.get(key)

        if t1 is None or t1 <= 0:
            # Missing p=1 baseline for this grid.
            # Leave speedup / efficiency blank instead of using the wrong baseline.
            continue

        r.speedup = t1 / r.wall_max
        r.efficiency = 100.0 * r.speedup / r.p

    # ------------------------------------------------------------
    # Weak scaling baseline:
    # one p=1 weak-scaling baseline.
    # ------------------------------------------------------------
    weak_baseline = next(
        (
            r for r in runs
            if r.mode == "weak" and r.p == 1 and r.wall_max > 0
        ),
        None,
    )

    if weak_baseline is None:
        return

    t1 = weak_baseline.wall_max

    for r in runs:
        if r.mode != "weak":
            continue
        if r.wall_max <= 0:
            continue

        r.speedup = t1 / r.wall_max
        r.efficiency = 100.0 * r.speedup


# --------------------------------------------------------------------------
# Markdown rendering
# --------------------------------------------------------------------------

def _fmt(x: Optional[float], spec: str = ".2f") -> str:
    if x is None:
        return "—"

    try:
        if x != x:        # NaN
            return "—"
    except TypeError:
        return "—"

    return format(x, spec)


_HEADER = (
    "| Rank | Config | Mode | p | Grid | Steps | Wall (s) | "
    "Compute (s) | Comm (s) | Comm % | MLUPS | Speedup | Eff % |"
)

_SEP = (
    "|-----:|:-------|:-----|--:|:-----|------:|---------:|"
    "------------:|---------:|-------:|------:|--------:|------:|"
)


def _row(rank: int, r: Run, highlight: bool = False) -> str:
    grid = f"{r.nx}×{r.ny}"

    cells = [
        str(rank),
        f"`{r.config}`",
        r.mode,
        str(r.p),
        grid,
        str(r.steps),
        _fmt(r.wall_max),
        _fmt(r.compute_max),
        _fmt(r.comm_max),
        _fmt(r.comm_over_total, ".1f"),
        _fmt(r.mlups),
        _fmt(r.speedup),
        _fmt(r.efficiency, ".1f"),
    ]

    if highlight:
        cells[0] = f"**{cells[0]}**"
        cells[10] = f"**{cells[10]}**"   # Highlight MLUPS.

    return "| " + " | ".join(cells) + " |"


def _render_table(runs: list[Run], sort_key, title: str) -> str:
    runs_sorted = sorted(runs, key=sort_key)

    out = [
        f"## {title}",
        "",
        _HEADER,
        _SEP,
    ]

    for i, r in enumerate(runs_sorted, start=1):
        out.append(_row(i, r, highlight=(i == 1)))

    out.append("")
    return "\n".join(out)


def _summary_block(runs: list[Run]) -> str:
    if not runs:
        return "_(no successful runs found)_\n"

    best_mlups = max(runs, key=lambda r: r.mlups)

    strong = [
        r for r in runs
        if r.mode == "strong" and r.speedup is not None
    ]

    weak = [
        r for r in runs
        if r.mode == "weak" and r.efficiency is not None
    ]

    lines = [
        "> ## Best results",
        ">",
        f"> **Highest MLUPS:** `{best_mlups.config}`  ",
        f"> MLUPS = **{_fmt(best_mlups.mlups)}**, "
        f"wall = {_fmt(best_mlups.wall_max)} s, "
        f"p = {best_mlups.p}, grid = {best_mlups.nx}×{best_mlups.ny}",
    ]

    if strong:
        best_strong = max(strong, key=lambda r: r.speedup or 0)
        lines += [
            ">",
            f"> **Best strong speedup:** `{best_strong.config}`  ",
            f"> speedup = **{_fmt(best_strong.speedup)}** at p = {best_strong.p}, "
            f"efficiency = {_fmt(best_strong.efficiency, '.1f')} %",
        ]

    if weak:
        best_weak = max(weak, key=lambda r: r.efficiency or 0)
        lines += [
            ">",
            f"> **Best weak efficiency:** `{best_weak.config}`  ",
            f"> efficiency = **{_fmt(best_weak.efficiency, '.1f')} %** "
            f"at p = {best_weak.p}, grid = {best_weak.nx}×{best_weak.ny}",
        ]

    return "\n".join(lines) + "\n"


def render_markdown(runs: list[Run]) -> str:
    runs = [r for r in runs if r is not None]

    if not runs:
        return "# LBM MPI Scaling\n\n_(no logs parsed)_\n"

    _attach_scaling_metrics(runs)

    parts = [
        "# LBM MPI Scaling Results",
        "",
        _summary_block(runs),
    ]

    # Full leaderboard sorted by MLUPS.
    parts.append(
        _render_table(
            runs,
            sort_key=lambda r: -r.mlups,
            title="Leaderboard (all runs, sorted by MLUPS)",
        )
    )

    strong = [r for r in runs if r.mode == "strong"]
    if strong:
        parts.append(
            _render_table(
                strong,
                sort_key=lambda r: (r.nx * r.ny, r.nx, r.ny, r.steps, r.p),
                title="Strong scaling (fixed global grid)",
            )
        )

    weak = [r for r in runs if r.mode == "weak"]
    if weak:
        parts.append(
            _render_table(
                weak,
                sort_key=lambda r: r.p,
                title="Weak scaling (fixed local rows per rank)",
            )
        )

    return "\n".join(parts).rstrip() + "\n"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent

    ap.add_argument(
        "--logs",
        type=Path,
        default=here / "logs",
        help="directory containing *.log files",
    )

    ap.add_argument(
        "--out",
        type=Path,
        default=here / "scaling.md",
        help="markdown output path",
    )

    args = ap.parse_args()

    if not args.logs.is_dir():
        print(f"ERROR: log directory not found: {args.logs}", file=sys.stderr)
        return 1

    runs: list[Run] = []
    skipped: list[Path] = []

    for log in sorted(args.logs.glob("*.log")):
        r = _parse_log(log)

        if r is None:
            skipped.append(log)
        else:
            runs.append(r)

    md = render_markdown(runs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")

    print(f"Parsed {len(runs)} successful runs from {args.logs}")

    for s in skipped:
        print(f"  (skipped incomplete log: {s.name})")

    print(f"Wrote {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())