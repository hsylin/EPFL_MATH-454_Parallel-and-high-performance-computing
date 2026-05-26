#!/bin/bash
#SBATCH --job-name=lbm_cuda_scaling
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=scaling/logs/sbatch_%j.out
#SBATCH --error=scaling/logs/sbatch_%j.err

set -euo pipefail

# Adjust these module names if the course cluster uses different CUDA/HDF5 modules.
module purge
module load gcc/11.3.0
module load cuda/11.8.0
module load hdf5/1.12.2

export OMP_NUM_THREADS=1

ROOT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT_DIR"

mkdir -p scaling/logs

echo "ROOT_DIR = $ROOT_DIR"
echo "PWD      = $(pwd)"

echo "Loaded modules:"
module list

echo "CUDA compiler:"
which nvcc
nvcc --version

echo "Cleaning old binaries..."
rm -f lbm_cuda_opt *.o *~

echo "Building CUDA solver..."
make clean
make

echo "Running CUDA performance sweep..."
LBM_BIN="$ROOT_DIR/lbm_cuda_opt" \
GRIDS="400x200 800x400 1600x800" \
BLOCK_CONFIGS="16x16 32x4 32x8 32x16 64x4" \
PITCH_ALIGNS="1 16 32 64" \
STEPS="60000" \
EVERY="0" \
PROBE_EVERY="0" \
PROBE_PATH="/dev/null" \
    bash "$ROOT_DIR/scaling/run_scaling.sh"


echo "Running focused memory-layout sweep on non-aligned grid 1500x750..."
LBM_BIN="$ROOT_DIR/lbm_cuda_opt" \
GRIDS="1500x750" \
BLOCK_CONFIGS="64x4" \
PITCH_ALIGNS="1 16 32 64" \
STEPS="60000" \
EVERY="0" \
PROBE_EVERY="0" \
PROBE_PATH="/dev/null" \
    bash "$ROOT_DIR/scaling/run_scaling.sh"


echo "Generating CUDA markdown table..."
python3 "$ROOT_DIR/scaling/make_table.py" \
  --logs "$ROOT_DIR/scaling/logs" \
  --csv "$ROOT_DIR/scaling/results_cuda_optimized.csv" \
  --out "$ROOT_DIR/scaling/scaling.md"

echo "Generating CUDA HTML table..."
python3 "$ROOT_DIR/scaling/make_html.py" \
  --input "$ROOT_DIR/scaling/scaling.md" \
  --output "$ROOT_DIR/scaling/scaling.html"

echo "Generated files:"
ls -lh "$ROOT_DIR/scaling/results_cuda_optimized.csv" \
       "$ROOT_DIR/scaling/scaling.md" \
       "$ROOT_DIR/scaling/scaling.html"

echo "Done."
