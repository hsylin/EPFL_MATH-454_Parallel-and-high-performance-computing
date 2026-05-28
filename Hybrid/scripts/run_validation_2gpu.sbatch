#!/bin/bash
#SBATCH --job-name=hybrid_val
#SBATCH --partition=gpu
#SBATCH --account=math-454
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=hybrid_val_%j.out
#SBATCH --error=hybrid_val_%j.err

set -euo pipefail

module purge
# Use a CUDA-aware OpenMPI module on SCITAS if your cluster exposes one
# separately, e.g. an openmpi/*-cuda variant.
module load gcc cuda openmpi hdf5

echo "==== Loaded modules ====
"
module list 2>&1 || true

echo "==== GPU topology ===="
nvidia-smi topo -m || true

echo "==== OpenMPI CUDA-related configuration hints ===="
ompi_info --parsable --all 2>/dev/null | grep -i cuda | head -50 || true

make clean
make

mkdir -p out
VALIDATION_SUMMARY=validation_summary.txt
: > "$VALIDATION_SUMMARY"

for HALO in cuda staged; do
  echo "[validation] halo=${HALO}"
  srun --gpu-bind=none -n 2 ./lbm_hybrid_cuda_mpi \
    nx=800 ny=400 re=100 u_in=0.05 steps=60000 \
    block_x=64 block_y=4 pitch_align=32 halo=${HALO} strict_cuda_aware=1 \
    every=500 out=out/hybrid_2gpu_${HALO} \
    probe=probe_hybrid_2gpu_${HALO}.csv probe_every=1

  python3 viz/strouhal.py probe_hybrid_2gpu_${HALO}.csv --ny 400 --u-in 0.05 | tee -a "$VALIDATION_SUMMARY"
  echo "Visualization index: out/hybrid_2gpu_${HALO}.xdmf" | tee -a "$VALIDATION_SUMMARY"
done

echo "==== Probe equivalence: cuda-aware vs staged ===="
python3 viz/compare_probes.py \
  probe_hybrid_2gpu_cuda.csv probe_hybrid_2gpu_staged.csv \
  --label-a cuda-aware --label-b staged | tee -a "$VALIDATION_SUMMARY"

if [[ -f viz/quick_viz.py ]]; then
  python3 viz/quick_viz.py --prefix out/hybrid_2gpu_cuda --field vorticity \
    --png out/hybrid_vorticity_final.png --gif out/hybrid_vorticity.gif || true
fi

echo "Validation summary: ${VALIDATION_SUMMARY}"
echo "Open the XDMF files in ParaView, or run:"
echo "python3 viz/quick_viz.py --prefix out/hybrid_2gpu_cuda --field vorticity --png out/hybrid_vorticity_final.png --gif out/hybrid_vorticity.gif"
