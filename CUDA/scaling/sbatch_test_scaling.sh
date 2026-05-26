#!/bin/bash
#SBATCH --job-name=lbm_cuda_test_scaling
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=scaling/logs/test_scaling_%j.out
#SBATCH --error=scaling/logs/test_scaling_%j.err

set -euo pipefail

module purge
module load gcc/11.3.0
module load cuda/11.8.0
module load hdf5/1.12.2

export OMP_NUM_THREADS=1

ROOT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT_DIR"

mkdir -p scaling/logs

echo "ROOT_DIR = $ROOT_DIR"
echo "HOSTNAME = $(hostname)"
echo "GPU:"
nvidia-smi

echo "Building CUDA solver..."
make clean
make

echo "Running small CUDA scaling test..."

LBM_BIN="$ROOT_DIR/lbm_cuda_opt" \
GRIDS="400x200" \
BLOCK_CONFIGS="32x8" \
PITCH_ALIGNS="32" \
STEPS="1000" \
EVERY="0" \
PROBE_EVERY="0" \
PROBE_PATH="/dev/null" \
bash "$ROOT_DIR/scaling/run_scaling.sh"

echo "Generating table..."
python3 "$ROOT_DIR/scaling/make_table.py" \
  --logs "$ROOT_DIR/scaling/logs" \
  --csv "$ROOT_DIR/scaling/results_cuda_optimized.csv" \
  --out "$ROOT_DIR/scaling/scaling.md"

python3 "$ROOT_DIR/scaling/make_html.py" \
  --input "$ROOT_DIR/scaling/scaling.md" \
  --output "$ROOT_DIR/scaling/scaling.html"

echo "Done."
