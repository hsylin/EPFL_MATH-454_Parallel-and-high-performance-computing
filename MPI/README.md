# MPI LBM Solver

MPI implementation of the D2Q9 BGK LBM solver for 2D flow past a cylinder.

The domain is decomposed by rows. Each MPI rank owns a contiguous block of rows
and uses two ghost rows for non-blocking halo exchange.

## Build

```bash
module purge
module load gcc/13.2.0
module load openmpi/5.0.3
module load hdf5/1.14.3-mpi

make clean
make
```

This builds:

```bash
./lbm_mpi
```

## Run

This run writes `probe.csv` and `out/lbm.xdmf`.

```bash
sbatch --ntasks=256 --time=02:00:00 \
  --export=ALL,NX=800,NY=400,STEPS=60000,EVERY=500,OUT=out/lbm,PROBE=probe.csv \
  run_mpi.sbatch
```

For performance runs, disable snapshot output and probe recording:

```bash
sbatch --ntasks=256 --time=02:00:00 \
  --export=ALL,NX=800,NY=400,STEPS=60000,EVERY=0,PROBE=/dev/null \
  run_mpi.sbatch
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

After the validation job finishes, generate a vorticity GIF:

```bash
python3 viz/viz.py out/lbm.xdmf --field vorticity --out vorticity.gif
```

## Notes

- Use `EVERY=0` and `PROBE=/dev/null` for timing runs.
- The code aborts if the number of MPI ranks is larger than `ny`.
- Python visualization and post-processing scripts are not parallelized.