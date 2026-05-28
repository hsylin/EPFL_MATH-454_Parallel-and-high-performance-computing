# Hybrid MPI + CUDA LBM Solver

Intra-node hybrid MPI + CUDA  implementation of the D2Q9 BGK LBM solver for 2D flow past a cylinder.

The implementation combines MPI row-wise domain decomposition with CUDA local
updates. Each MPI rank owns one row-block subdomain and uses one GPU on the
same node. The CUDA kernel uses fused pull-streaming and BGK collision with
padded structure-of-arrays storage. Rank boundaries are exchanged through either
CUDA-aware MPI device-buffer communication or explicit host-staged halo exchange.

## Build

```bash
module purge
module load gcc cuda openmpi hdf5

make clean
make
```

This builds:

```bash
./lbm_hybrid_cuda_mpi
```

## Run

This run writes `probe_hybrid_2gpu_cuda.csv` and
`out/hybrid_2gpu_cuda.xdmf`.

```bash
sbatch --account=math-454 --gres=gpu:2 --ntasks=2 --time=02:00:00 \
  --output=hybrid_cuda_%j.out --error=hybrid_cuda_%j.err \
  --export=ALL,NX=800,NY=400,STEPS=60000,EVERY=500,OUT=out/hybrid_2gpu_cuda,PROBE=probe_hybrid_2gpu_cuda.csv,PROBE_EVERY=1,HALO=cuda,STRICT_CUDA_AWARE=1,NTASKS=2 \
  run_hybrid.sbatch
```

To run the same validation case with explicit host-staged halo exchange:

```bash
sbatch --account=math-454 --gres=gpu:2 --ntasks=2 --time=02:00:00 \
  --output=hybrid_staged_%j.out --error=hybrid_staged_%j.err \
  --export=ALL,NX=800,NY=400,STEPS=60000,EVERY=500,OUT=out/hybrid_2gpu_staged,PROBE=probe_hybrid_2gpu_staged.csv,PROBE_EVERY=1,HALO=staged,STRICT_CUDA_AWARE=0,NTASKS=2 \
  run_hybrid.sbatch
```

For performance runs, disable snapshot output and probe recording:

```bash
sbatch --account=math-454 --gres=gpu:2 --ntasks=2 --time=02:00:00 \
  --output=hybrid_perf_%j.out --error=hybrid_perf_%j.err \
  --export=ALL,NX=1600,NY=800,STEPS=60000,EVERY=0,OUT=out/hybrid_perf,PROBE=/dev/null,PROBE_EVERY=0,HALO=cuda,STRICT_CUDA_AWARE=1,NTASKS=2 \
  run_hybrid.sbatch
```


## Validation

Compute the Strouhal number for the CUDA-aware run:

```bash
python3 viz/strouhal.py probe_hybrid_2gpu_cuda.csv --u-in 0.05 --diameter 20
```

Compute the Strouhal number for the host-staged run:

```bash
python3 viz/strouhal.py probe_hybrid_2gpu_staged.csv --u-in 0.05 --diameter 20
```

Expected result:

```text
St ≈ 0.16
```

## Visualization

After the validation run finishes, generate a vorticity GIF:

```bash
python3 viz/viz.py out/hybrid_2gpu_cuda.xdmf \
  --field vorticity \
  --out vorticity.gif 
```

## Notes

- Hybrid runs must be executed on a GPU node with MPI and CUDA support.
- This implementation is intended for one node with one or two GPUs.
- Use `EVERY=0`, `PROBE=/dev/null`, and `PROBE_EVERY=0` for timing runs.
- Use `EVERY=500` and `PROBE_EVERY=1` for validation and visualization runs.
- `HALO=cuda` passes packed CUDA device buffers directly to MPI.
- `HALO=staged` explicitly copies halo buffers from device to host, exchanges them with MPI, and copies them back to the device.
- `STRICT_CUDA_AWARE=1` checks CUDA-aware MPI support before running `HALO=cuda`.
- Do not claim NVLink, GPUDirect RDMA, or zero-copy GPU-to-GPU transfer unless confirmed by `nvidia-smi topo -m` and MPI runtime logs.
- `block_x` and `block_y` control the CUDA block shape.
- `pitch_align` controls row pitch alignment in cells.
- The current implementation does not overlap halo communication with interior-row computation.
- Python visualization and post-processing scripts are not parallelized.
