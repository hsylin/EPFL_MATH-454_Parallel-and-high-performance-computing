#!/usr/bin/env python3
import argparse
import math
import re
from pathlib import Path
from html.parser import HTMLParser

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class SimpleTableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.in_tr = False
        self.in_cell = False
        self.current_row = []
        self.current_cell = []

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.in_tr = True
            self.current_row = []
        elif tag in ("th", "td") and self.in_tr:
            self.in_cell = True
            self.current_cell = []

    def handle_data(self, data):
        if self.in_cell:
            self.current_cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("th", "td") and self.in_cell:
            text = " ".join("".join(self.current_cell).split())
            self.current_row.append(text)
            self.in_cell = False
        elif tag == "tr" and self.in_tr:
            if self.current_row:
                self.rows.append(self.current_row)
            self.in_tr = False


def to_float(x):
    if x is None:
        return math.nan
    x = str(x).replace(",", "").replace("%", "").strip()
    try:
        return float(x)
    except ValueError:
        return math.nan


def parse_html_table(path):
    html = Path(path).read_text(encoding="utf-8")
    parser = SimpleTableParser()
    parser.feed(html)

    rows = []
    header = None
    required = {"Config", "Mode", "p", "Grid", "Speedup", "Eff %"}

    for row in parser.rows:
        row_set = set(row)
        if required.issubset(row_set):
            header = row
            continue

        if header is None:
            continue

        if len(row) != len(header):
            continue

        item = dict(zip(header, row))

        if not required.issubset(item.keys()):
            continue

        if item.get("Mode") not in ("strong", "weak"):
            continue

        rows.append(item)

    # Remove duplicated rows from leaderboard + detailed tables.
    unique = {}
    for r in rows:
        unique[r["Config"]] = r

    data = list(unique.values())

    for r in data:
        r["p"] = int(to_float(r["p"]))
        r["Speedup"] = to_float(r["Speedup"])
        r["Eff %"] = to_float(r["Eff %"])
        r["MLUPS"] = to_float(r.get("MLUPS", "nan"))
        r["Grid"] = r["Grid"].replace("x", "×")

    return data


def grid_area(grid):
    m = re.match(r"(\d+)\s*[×x]\s*(\d+)", grid)
    if not m:
        return 0
    return int(m.group(1)) * int(m.group(2))


def setup_solution_style():
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "legend.title_fontsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.figsize": (4.4, 3.2),
        "savefig.dpi": 300,
    })


def set_power2_xticks(ax, p_values):
    ax.set_xscale("log", base=2)
    ax.set_xticks(p_values)
    ax.set_xticklabels([str(p) for p in p_values])


def plot_strong(data, outdir, f):
    strong = [r for r in data if r["Mode"] == "strong"]
    if not strong:
        raise RuntimeError("No strong-scaling rows found.")

    p_values = sorted({r["p"] for r in strong})
    grids = sorted({r["Grid"] for r in strong}, key=grid_area)

    fig, ax = plt.subplots()
    markers = ["o", "s", "v", "^", "D", "x"]

    for i, grid in enumerate(grids):
        g = sorted([r for r in strong if r["Grid"] == grid], key=lambda r: r["p"])
        ax.plot(
            [r["p"] for r in g],
            [r["Speedup"] for r in g],
            marker=markers[i % len(markers)],
            linewidth=1.4,
            markersize=4.5,
            label=grid,
        )

    amdahl = [1.0 / ((1.0 - f) + f / p) for p in p_values]
    ax.plot(
        p_values,
        amdahl,
        linestyle="--",
        linewidth=1.2,
        label=f"Amdahl\n($f={f:.4f}$)",
    )

    set_power2_xticks(ax, p_values)
    ax.set_yscale("log")
    ax.set_xlabel(r"number of processors $p$")
    ax.set_ylabel("speedup")
    ax.grid(True, which="both", linewidth=0.6, alpha=0.55)

    ymax = max([r["Speedup"] for r in strong] + amdahl)
    ax.set_ylim(0.8, ymax * 1.35)

    leg = ax.legend(
        title="global grid",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=2,
        frameon=True,
        fancybox=False,
    )
    leg.get_frame().set_edgecolor("black")

    fig.subplots_adjust(bottom=0.36)

    fig.savefig(outdir / "mpi_strong_scaling_amdahl_multi_size.pdf", bbox_inches="tight")
    fig.savefig(outdir / "mpi_strong_scaling_amdahl_multi_size.png", bbox_inches="tight")
    plt.close(fig)


def plot_weak(data, outdir, f):
    weak = sorted([r for r in data if r["Mode"] == "weak"], key=lambda r: r["p"])
    if not weak:
        raise RuntimeError("No weak-scaling rows found.")

    p_values = [r["p"] for r in weak]
    measured_eff = [r["Eff %"] / 100.0 for r in weak]

    alpha = 1.0 - f
    gustafson_speedup = [p - alpha * (p - 1) for p in p_values]
    gustafson_eff = [s / p for s, p in zip(gustafson_speedup, p_values)]

    fig, ax = plt.subplots()

    ax.plot(
        p_values,
        measured_eff,
        marker="o",
        linewidth=1.4,
        markersize=4.5,
        label="measured",
    )

    ax.plot(
        p_values,
        gustafson_eff,
        linestyle="--",
        linewidth=1.2,
        label=rf"Gustafson ($\alpha={alpha:.4f}$)",
    )

    set_power2_xticks(ax, p_values)
    ax.set_yscale("log")
    ax.set_xlabel(r"number of processors $p$")
    ax.set_ylabel("parallel efficiency")
    ax.grid(True, which="both", linewidth=0.6, alpha=0.55)

    ymin = max(0.01, min(measured_eff) * 0.75)
    ax.set_ylim(ymin, 1.25)

    leg = ax.legend(
        title="weak scaling",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=2,
        frameon=True,
        fancybox=False,
    )
    leg.get_frame().set_edgecolor("black")

    fig.subplots_adjust(bottom=0.36)

    fig.savefig(outdir / "mpi_weak_scaling_gustafson.pdf", bbox_inches="tight")
    fig.savefig(outdir / "mpi_weak_scaling_gustafson.png", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to MPI scaling.html")
    parser.add_argument("--outdir", default="figures")
    parser.add_argument(
        "--f",
        type=float,
        default=0.9968,
        help="Profiled parallel fraction for Amdahl/Gustafson references. "
             "Default: collide + stream + bounce_back = 0.7188 + 0.2582 + 0.0198 = 0.9968."
    )
    args = parser.parse_args()

    setup_solution_style()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = parse_html_table(args.input)

    plot_strong(data, outdir, args.f)
    plot_weak(data, outdir, args.f)

    alpha = 1.0 - args.f
    print(f"Using f = {args.f:.4f}, alpha = {alpha:.4f}")
    print(f"Amdahl S(256) = {1.0 / ((1.0 - args.f) + args.f / 256):.2f}")
    print(f"Gustafson efficiency at p=256 = {(256 - alpha * 255) / 256 * 100:.2f}%")
    print("Wrote:")
    print(outdir / "mpi_strong_scaling_amdahl_multi_size.pdf")
    print(outdir / "mpi_strong_scaling_amdahl_multi_size.png")
    print(outdir / "mpi_weak_scaling_gustafson.pdf")
    print(outdir / "mpi_weak_scaling_gustafson.png")


if __name__ == "__main__":
    main()