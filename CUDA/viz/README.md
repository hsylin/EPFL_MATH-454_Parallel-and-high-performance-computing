# Visualization helpers

Two scripts that consume the output of the solver:

- `viz.py` — read the XDMF/HDF5 snapshots and render an animated GIF of a
  chosen field (default: vorticity).
- `strouhal.py` — read the probe CSV, FFT the transverse velocity, and
  print the Strouhal number `St = f * D / U`.

## Install

```bash
pip install pyvista numpy matplotlib imageio
```

`viz.py` renders off-screen with PyVista — no display server required.

## Examples

```bash
# Run a longer simulation first so vortex shedding has time to develop.
mkdir -p out
./lbm nx=600 ny=120 re=100 u_in=0.05 steps=80000 every=200 out=out/lbm

# Animate the vorticity (default field).
python3 viz/viz.py out/lbm.xdmf

# Animate the velocity magnitude.
python3 viz/viz.py out/lbm.xdmf --field ux --cmap viridis

# Strouhal number. Cylinder diameter = 2 * cyl_r (here cyl_r defaults to ny/10).
python3 viz/strouhal.py probe.csv --u-in 0.05 --diameter 24
```

ParaView users can also open `out/lbm.xdmf` directly without running these
scripts.
