# CUDA LBM Solver

CUDA implementation of the D2Q9 BGK LBM solver for 2D flow past a cylinder.

The main lattice update is accelerated with CUDA kernels. The implementation
uses fused pull-streaming and BGK collision, double buffering, and padded
structure-of-arrays storage.

## Build

```bash
module purge
module load gcc cuda hdf5

make clean
make
```

This builds:

```bash
./lbm_cuda_opt
```

## Run

This run writes `probe.csv` and `out/lbm.xdmf`.

```bash
sbatch --time=02:00:00 \
  --export=ALL,NX=800,NY=400,STEPS=60000,EVERY=500,OUT=out/lbm,PROBE=probe.csv,PROBE_EVERY=1,BLOCK_X=32,BLOCK_Y=8,PITCH_ALIGN=32 \
  run_cuda.sbatch
```

For performance runs, disable snapshot output and probe recording:

```bash
sbatch --time=02:00:00 \
  --export=ALL,NX=1600,NY=800,STEPS=60000,EVERY=0,PROBE=/dev/null,PROBE_EVERY=0,BLOCK_X=64,BLOCK_Y=4,PITCH_ALIGN=32 \
  run_cuda.sbatch
```

## Validation

Compute the Strouhal number:

```bash
python3 viz/strouhal.py probe.csv --u-in 0.05 --diameter 20
```

Expected result:

```text
St ≈ 0.16
```

## Visualization

After the validation run finishes, generate a vorticity GIF:

```bash
python3 viz/viz.py out/lbm.xdmf --field vorticity --out vorticity.gif
```

## Notes

- CUDA runs must be executed on a GPU node.
- Use `EVERY=0` and `PROBE_EVERY=0` for timing runs.
- `block_x` and `block_y` control the CUDA block shape.
- `pitch_align` controls row pitch alignment in cells.
- Python visualization and post-processing scripts are not parallelized.
