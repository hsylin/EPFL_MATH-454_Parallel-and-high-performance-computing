#!/bin/bash
#SBATCH --job-name=hybrid_scale
#SBATCH --partition=gpu
#SBATCH --account=math-454
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=hybrid_scale_%j.out
#SBATCH --error=hybrid_scale_%j.err

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

# Final/default sweep for extension report. Override STEPS/REPEATS for quick smoke tests.
STEPS=${STEPS:-60000} \
REPEATS=${REPEATS:-3} \
RANKS=${RANKS:-1 2} \
GRIDS=${GRIDS:-800x400 1500x750 1600x800} \
HALOS=${HALOS:-cuda staged} \
STRICT_CUDA_AWARE=${STRICT_CUDA_AWARE:-1} \
SRUN_GPU_BIND=${SRUN_GPU_BIND:-none} \
bash scaling/run_hybrid_scaling.sh

python3 scaling/make_hybrid_table.py
python3 scaling/make_hybrid_html.py
