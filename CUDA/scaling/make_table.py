#!/usr/bin/env python3
"""
make_table.py — Parse CUDA LBM sweep logs and emit a Markdown report.

Input:
  scaling/logs/*.log
  scaling/results_cuda_optimized.csv

Output:
  scaling/scaling.md

The report contains:
  * Best-result summary.
  * Full leaderboard sorted by MLUPS.
  * Best configuration per grid size.
  * Tables grouped by grid size.

Usage:
  python3 scaling/make_table.py
  python3 scaling/make_table.py --logs scaling/logs --out scaling/scaling.md
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


_TAG_RE = re.compile(
    r"^cuda_nx(?P<nx>\d+)_ny(?P<ny>\d+)_bx(?P<bx>\d+)_by(?P<by>\d+)_pa(?P<pa>\d+)_s(?P<steps>\d+)$"
)

_METRIC_RE = re.compile(
    r"^(?P<key>Wall time|MLUPS)\s*:?\s*(?P<val>[-+0-9.eE]+)"
)


@dataclass
class Run:
    config: str
    grid: str
    nx: int
    ny: int
    block_x: int
    block_y: int
    threads_per_block: int
    pitch_align: int
    steps: int
    wall_time_s: float
    mlups: float

    @property
    def cells(self) -> int:
        return self.nx * self.ny

    @property
    def block(self) -> str:
        return f"{self.block_x}x{self.block_y}"


def _to_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _parse_log(path: Path) -> Optional[Run]:
    m = _TAG_RE.match(path.stem)
    if not m:
        return None

    metrics: dict[str, float] = {}
    for line in path.read_text(errors="replace").splitlines():
        m2 = _METRIC_RE.match(line.strip())
        if not m2:
            continue
        metrics[m2.group("key")] = _to_float(m2.group("val"))

    if "MLUPS" not in metrics:
        return None

    nx = int(m.group("nx"))
    ny = int(m.group("ny"))
    bx = int(m.group("bx"))
    by = int(m.group("by"))
    pa = int(m.group("pa"))
    steps = int(m.group("steps"))

    return Run(
        config=path.stem,
        grid=f"{nx}x{ny}",
        nx=nx,
        ny=ny,
        block_x=bx,
        block_y=by,
        threads_per_block=bx * by,
        pitch_align=pa,
        steps=steps,
        wall_time_s=metrics.get("Wall time", float("nan")),
        mlups=metrics.get("MLUPS", float("nan")),
    )


def _parse_csv(path: Path) -> list[Run]:
    if not path.is_file():
        return []

    out: list[Run] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                nx = int(row["nx"])
                ny = int(row["ny"])
                bx = int(row["block_x"])
                by = int(row["block_y"])
                pa = int(row["pitch_align"])
                steps = int(row["steps"])
            except Exception:
                continue

            out.append(Run(
                config=f"cuda_nx{nx}_ny{ny}_bx{bx}_by{by}_pa{pa}_s{steps}",
                grid=row.get("grid", f"{nx}x{ny}"),
                nx=nx,
                ny=ny,
                block_x=bx,
                block_y=by,
                threads_per_block=int(row.get("threads_per_block", bx * by)),
                pitch_align=pa,
                steps=steps,
                wall_time_s=_to_float(row.get("wall_time_s", "nan")),
                mlups=_to_float(row.get("mlups", "nan")),
            ))

    return out


def _dedup(runs: list[Run]) -> list[Run]:
    best: dict[str, Run] = {}
    for r in runs:
        old = best.get(r.config)
        if old is None:
            best[r.config] = r
        elif (not math.isnan(r.mlups)) and (math.isnan(old.mlups) or r.mlups > old.mlups):
            best[r.config] = r
    return list(best.values())


def _fmt(x: float, spec: str = ".2f") -> str:
    if x is None or math.isnan(x):
        return "—"
    return format(x, spec)


_HEADER = (
    "| Rank | Config | Grid | Block | Threads/block | Pitch align | Steps | Wall (s) | MLUPS |"
)
_SEP = (
    "|-----:|:-------|:-----|:------|--------------:|------------:|------:|---------:|------:|"
)


def _row(rank: int, r: Run, highlight: bool = False) -> str:
    cells = [
        str(rank),
        f"`{r.config}`",
        f"{r.nx}x{r.ny}",
        r.block,
        str(r.threads_per_block),
        str(r.pitch_align),
        str(r.steps),
        _fmt(r.wall_time_s),
        _fmt(r.mlups),
    ]
    if highlight:
        cells[0] = f"**{cells[0]}**"
        cells[-1] = f"**{cells[-1]}**"
    return "| " + " | ".join(cells) + " |"


def _render_table(runs: list[Run], title: str, sort_key, reverse: bool = False) -> str:
    runs_sorted = sorted(runs, key=sort_key, reverse=reverse)
    out = [f"## {title}", "", _HEADER, _SEP]
    for i, r in enumerate(runs_sorted, start=1):
        out.append(_row(i, r, highlight=(i == 1)))
    out.append("")
    return "\n".join(out)


def _grid_key(g: str) -> tuple[int, int]:
    nx, ny = g.split("x")
    return int(nx), int(ny)


def _summary_block(runs: list[Run]) -> str:
    valid = [r for r in runs if not math.isnan(r.mlups)]
    if not valid:
        return "_(no successful CUDA runs found)_\n"

    best = max(valid, key=lambda r: r.mlups)
    grids = sorted(set(r.grid for r in valid), key=_grid_key)
    best_by_grid = [max([r for r in valid if r.grid == g], key=lambda r: r.mlups) for g in grids]

    lines = [
        "> ## Best CUDA results",
        ">",
        f"> **Highest MLUPS:** `{best.config}`  ",
        f"> MLUPS = **{_fmt(best.mlups)}**, wall = {_fmt(best.wall_time_s)} s, "
        f"grid = {best.grid}, block = {best.block}, pitch_align = {best.pitch_align}",
    ]

    if best_by_grid:
        lines += [">", "> **Best per grid:**  "]
        for r in best_by_grid:
            lines.append(
                f"> `{r.grid}`: **{_fmt(r.mlups)} MLUPS** "
                f"with block `{r.block}`, pitch_align = {r.pitch_align}  "
            )

    lines += [
        ">",
        "> **Interpretation hint:** CUDA LBM is usually memory-bandwidth dominated. "
        "Compare the achieved MLUPS with a rough V100 memory-bandwidth roofline.",
    ]
    return "\n".join(lines) + "\n"


def render_markdown(runs: list[Run]) -> str:
    runs = [r for r in _dedup(runs) if not math.isnan(r.mlups)]
    if not runs:
        return "# LBM CUDA Performance Results\n\n_(no successful runs parsed)_\n"

    parts = ["# LBM CUDA Performance Results", "", _summary_block(runs)]

    parts.append(
        _render_table(
            runs,
            title="Leaderboard (all CUDA runs, sorted by MLUPS)",
            sort_key=lambda r: r.mlups,
            reverse=True,
        )
    )

    best_per_grid: list[Run] = []
    for grid in sorted(set(r.grid for r in runs), key=_grid_key):
        group = [r for r in runs if r.grid == grid]
        best_per_grid.append(max(group, key=lambda r: r.mlups))

    parts.append(
        _render_table(
            best_per_grid,
            title="Best configuration per problem size",
            sort_key=lambda r: (r.nx, r.ny),
        )
    )
    parts.append(
        "_Use this table for the problem-size discussion. Small grids are more affected by "
        "kernel launch overhead, while larger grids better expose memory-bandwidth limits._\n"
    )

    for grid in sorted(set(r.grid for r in runs), key=_grid_key):
        group = [r for r in runs if r.grid == grid]
        parts.append(
            _render_table(
                group,
                title=f"Block / pitch sweep for grid {grid}",
                sort_key=lambda r: r.mlups,
                reverse=True,
            )
        )

    parts.append(
        "## Report notes\n\n"
        "- Report MLUPS for the best configuration and compare it with the serial baseline.\n"
        "- Discuss how performance changes with `block_x` / `block_y`.\n"
        "- Discuss how performance changes with `pitch_align`.\n"
        "- Discuss how performance changes with `nx` / `ny`.\n"
        "- Use a rough lower bound of `18 doubles = 144 bytes` per lattice update for the memory-bandwidth roofline estimate.\n"
    )

    return "\n".join(parts).rstrip() + "\n"


def main() -> int:
    here = Path(__file__).resolve().parent

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs", type=Path, default=here / "logs",
                    help="directory containing CUDA sweep *.log files")
    ap.add_argument("--csv", type=Path, default=here / "results_cuda_optimized.csv",
                    help="CSV file generated by run_scaling.sh")
    ap.add_argument("--out", type=Path, default=here / "scaling.md",
                    help="Markdown output path")
    args = ap.parse_args()

    runs: list[Run] = []

    if args.logs.is_dir():
        skipped: list[Path] = []
        for log in sorted(args.logs.glob("*.log")):
            r = _parse_log(log)
            if r is None:
                skipped.append(log)
            else:
                runs.append(r)
    else:
        skipped = []

    runs.extend(_parse_csv(args.csv))

    md = render_markdown(runs)
    args.out.write_text(md, encoding="utf-8")

    parsed = len(_dedup(runs))
    print(f"Parsed {parsed} CUDA runs")
    for s in skipped:
        print(f"  (skipped incomplete/non-CUDA log: {s.name})")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
