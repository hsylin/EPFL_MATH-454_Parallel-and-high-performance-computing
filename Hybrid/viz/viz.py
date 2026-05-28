#!/usr/bin/env python3
"""Render an animated GIF of the LBM vorticity field from an XDMF file.

Usage:
    python3 viz.py out/lbm.xdmf [--field vorticity] [--out vorticity.gif]
                               [--clim -0.05 0.05] [--cmap RdBu_r]
                               [--width 1000]

The script parses the XDMF index to find each snapshot's HDF5 file, reads
the fields directly with h5py, and writes a GIF via PyVista.

Requirements: pyvista, numpy, h5py, imageio.
Install with `pip install pyvista numpy h5py imageio`.
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np
import pyvista as pv


def _parse_xdmf(xdmf_path: Path):
    """Return list of (time_value, {field: (h5_path, dataset_name)}) dicts."""
    base_dir = xdmf_path.parent
    tree = ET.parse(xdmf_path)
    root = tree.getroot()

    # Walk: Xdmf/Domain/Grid[@CollectionType="Temporal"]/Grid[*]
    domain = root.find("Domain")
    collection = domain.find("Grid[@CollectionType='Temporal']")
    if collection is None:
        # Fallback: first Grid in Domain
        collection = domain.find("Grid")

    steps = []
    for grid in collection.findall("Grid"):
        time_el = grid.find("Time")
        t = float(time_el.get("Value")) if time_el is not None else float(len(steps))

        fields = {}
        for attr in grid.findall("Attribute"):
            name = attr.get("Name")
            di = attr.find("DataItem[@Format='HDF']")
            if di is None:
                continue
            text = di.text.strip()
            # format: "filename.h5:/dataset"
            h5_rel, _, dset = text.partition(":/")
            h5_abs = base_dir / h5_rel
            fields[name] = (h5_abs, dset)
        steps.append((t, fields))

    return steps


def _read_field(h5_path, dset_name):
    with h5py.File(h5_path, "r") as f:
        return f[dset_name][()]  # shape (ny, nx)


def _make_image_data(arr_ny_nx):
    """Wrap a (ny, nx) numpy array as a pv.ImageData (uniform grid)."""
    ny, nx = arr_ny_nx.shape
    grid = pv.ImageData(dimensions=(nx, ny, 1), spacing=(1.0, 1.0, 1.0),
                        origin=(0.0, 0.0, 0.0))
    # ImageData expects C-order flattened; point data is indexed (x,y,z).
    # arr is row-major (ny, nx) -> flatten in C order gives correct x-major
    # layout after transposing so the inner axis is x.
    grid.point_data["_arr"] = arr_ny_nx.ravel(order="C")
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xdmf", type=Path, help="Path to the .xdmf file written by the solver")
    ap.add_argument("--field", default="vorticity",
                    choices=("rho", "ux", "uy", "vorticity"))
    ap.add_argument("--out",   default=None, help="Output gif path (default <field>.gif)")
    ap.add_argument("--cmap",  default=None, help="Matplotlib colormap name")
    ap.add_argument("--clim",  type=float, nargs=2, default=None,
                    help="Manual color limits, e.g. --clim -0.05 0.05")
    ap.add_argument("--width", type=int, default=1000, help="Window width in pixels")
    args = ap.parse_args()

    if not args.xdmf.exists():
        sys.exit(f"File not found: {args.xdmf}")

    steps = _parse_xdmf(args.xdmf)
    if not steps:
        sys.exit("No time steps found in the XDMF file.")

    # Filter to steps that have the requested field.
    steps = [(t, f) for t, f in steps if args.field in f]
    if not steps:
        sys.exit(f"Field '{args.field}' not found in XDMF snapshots.")

    # Read all arrays to determine color limits.
    arrays = []
    for _t, fields in steps:
        h5_path, dset = fields[args.field]
        arrays.append(_read_field(h5_path, dset))

    if args.clim is not None:
        clim = tuple(args.clim)
    elif args.field == "vorticity":
        m = max(abs(np.nanpercentile(a, 99)) for a in arrays)
        m = max(m, 1e-9)
        clim = (-m, m)
    else:
        lo = min(np.nanpercentile(a,  1) for a in arrays)
        hi = max(np.nanpercentile(a, 99) for a in arrays)
        clim = (lo, hi)

    # Load solid mask if present.
    mask_arr = None
    if "solid" in steps[0][1]:
        h5_path, dset = steps[0][1]["solid"]
        mask_arr = _read_field(h5_path, dset).astype(float)

    cmap = args.cmap or ("RdBu_r" if args.field == "vorticity" else "viridis")
    out  = Path(args.out) if args.out else Path(f"{args.field}.gif")

    ny, nx = arrays[0].shape
    height = max(200, int(args.width * ny / nx))

    plotter = pv.Plotter(off_screen=True, window_size=(args.width, height))
    plotter.open_gif(str(out))

    solid_mesh = None
    if mask_arr is not None:
        solid_grid = _make_image_data(mask_arr)
        solid_mesh = solid_grid.threshold(value=0.5, scalars="_arr")

    for i, (t, _fields) in enumerate(steps):
        arr = arrays[i]
        grid = _make_image_data(arr)

        plotter.clear()
        plotter.add_mesh(grid, scalars="_arr", cmap=cmap, clim=clim,
                         show_scalar_bar=True,
                         scalar_bar_args={"title": args.field})
        if solid_mesh is not None and solid_mesh.n_points > 0:
            plotter.add_mesh(solid_mesh, color=(0.4, 0.4, 0.4),
                             show_scalar_bar=False)
        plotter.add_text(f"t = {t:.0f}", font_size=10, color="black")
        plotter.view_xy()
        plotter.camera.zoom("tight")
        plotter.write_frame()
        print(f"\r  frame {i + 1}/{len(steps)}", end="", flush=True)

    print()
    plotter.close()
    print(f"wrote {out} ({len(steps)} frames)")


if __name__ == "__main__":
    main()
