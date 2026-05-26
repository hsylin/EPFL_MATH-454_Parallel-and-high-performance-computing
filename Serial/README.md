# LBM 2D — serial starter code

A small 2D lattice Boltzmann solver (D2Q9, BGK collision) for flow past one
or two circular cylinders in a rectangular channel. This is the starting
point for the MATH-454 course project: profile it, then parallelize it.

## Build

```bash
# TODO[Davor]: confirm SCITAS module names.
module load gcc hdf5
make
```

The Makefile uses `$HDF5_ROOT` for include and library paths. If you build
locally, point it at your HDF5 installation, e.g. on macOS with Homebrew:

```bash
HDF5_ROOT=$(brew --prefix hdf5) make
```

### Local conda environment (optional)

In your own machine, if you would rather not depend on system packages, a single conda
environment can provide the C++ toolchain, HDF5, and all the Python
packages used by the visualization scripts:

```bash
conda create -n phpc-lbm -c conda-forge python=3.13 \
    cxx-compiler hdf5 h5py numpy matplotlib pyvista imageio
conda activate phpc-lbm
make
```

Activation sets `$CXX` to the conda compiler and adds the env's
`include/` and `lib/` directories to its search path automatically, so
plain `make` is enough — no `$HDF5_ROOT` needed. The `viz/` scripts then
run against the same Python.

## Run

The program takes `key=value` arguments. All defaults are reasonable for a
quick smoke test:

```bash
mkdir -p out
./lbm                                  # uses defaults (see table below)
./lbm nx=400 ny=100 re=100 steps=20000 # explicit overrides
```

Useful keys:

| key       | meaning                                                 | default      |
|-----------|---------------------------------------------------------|--------------|
| `nx`      | cells along x                                           | 800          |
| `ny`      | cells along y                                           | 400          |
| `re`      | Reynolds number based on cylinder diameter              | 100          |
| `u_in`    | inlet velocity (lattice units; keep `u_in` ≪ 1/√3)      | 0.05         |
| `steps`   | number of time steps                                    | 60000        |
| `every`   | write a snapshot every `every` steps; 0 disables output | 500          |
| `out`     | output prefix; produces `out/lbm.xdmf` + `out/lbm_*.h5` | `out/lbm`    |
| `cyl_x`   | first cylinder center, x                                | `nx/4`       |
| `cyl_y`   | first cylinder center, y                                | `ny/2`       |
| `cyl_r`   | first cylinder radius                                   | `ny/40`      |
| `cyl2_*`  | second cylinder; set `cyl2_r > 0` to enable             | disabled     |
| `probe_x` | velocity probe, x                                       | `cyl_x + 8r` |
| `probe_y` | velocity probe, y                                       | `cyl_y`      |
| `probe`   | probe CSV path                                          | `probe.csv`  |

## What the program does and writes

Each step performs collision (BGK), bounce-back at solid cells, streaming,
and inlet/outlet boundary conditions. The solver records:

- `out/lbm.xdmf` together with `out/lbm_*.h5` — open `out/lbm.xdmf` directly
  in ParaView, or feed it to the Python script in `viz/`. Each snapshot
  contains density, velocity components, and vorticity.
- `probe.csv` — `(step, ux, uy)` at a probe point a few cylinder diameters
  downstream. Used by `viz/strouhal.py` to estimate the shedding frequency.

## Performance metric

At the end, the program prints the achieved **MLUPS** (mega lattice updates
per second), the standard headline number for LBM. Use it to compare your
parallel versions against the serial baseline.

## Visualization

The two scripts in `viz/` consume the solver output. They depend on a
small set of Python packages — install them with `pip` or `conda`:

```bash
pip install pyvista numpy h5py imageio matplotlib
# or, if you created the conda env above, these are already installed.
```

| Package      | Used by                       | Why                                           |
|--------------|-------------------------------|-----------------------------------------------|
| `pyvista`    | `viz.py`                      | renders frames and writes the GIF             |
| `h5py`       | `viz.py`                      | reads field arrays directly from HDF5 files   |
| `imageio`    | `viz.py`                      | required by PyVista's `open_gif` writer       |
| `numpy`      | `viz.py`, `strouhal.py`       | array math + FFT                              |
| `matplotlib` | `strouhal.py` (with `--plot`) | optional inspection of the probe spectrum     |

Then:

```bash
python3 viz/viz.py out/lbm.xdmf            # produces vorticity.gif
python3 viz/strouhal.py probe.csv \
        --u-in 0.05 --diameter 20          # prints Strouhal number
```

For Karman shedding to develop fully and the FFT to find a clean peak, run
at least 50 000–60 000 steps with the default geometry.
