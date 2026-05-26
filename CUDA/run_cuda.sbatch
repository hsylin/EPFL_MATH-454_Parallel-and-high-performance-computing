#!/bin/bash
#SBATCH --job-name=lbm_cuda
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=cuda_%j.out
#SBATCH --error=cuda_%j.err

set -euo pipefail

module purge
module load gcc cuda hdf5

NX=${NX:-800}
NY=${NY:-400}
RE=${RE:-100}
U_IN=${U_IN:-0.05}
STEPS=${STEPS:-60000}
EVERY=${EVERY:-500}
OUT=${OUT:-out/lbm}
PROBE=${PROBE:-probe.csv}

BLOCK_X=${BLOCK_X:-32}
BLOCK_Y=${BLOCK_Y:-8}
PITCH_ALIGN=${PITCH_ALIGN:-32}
PROBE_EVERY=${PROBE_EVERY:-1}

mkdir -p out

echo "Running LBM CUDA"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_NODELIST=${SLURM_NODELIST}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Parameters:"
echo "  nx=${NX} ny=${NY} re=${RE} u_in=${U_IN} steps=${STEPS}"
echo "  every=${EVERY} out=${OUT} probe=${PROBE}"
echo "  block_x=${BLOCK_X} block_y=${BLOCK_Y} pitch_align=${PITCH_ALIGN} probe_every=${PROBE_EVERY}"

nvidia-smi

./lbm_cuda_opt \
  nx="${NX}" ny="${NY}" re="${RE}" u_in="${U_IN}" steps="${STEPS}" \
  every="${EVERY}" out="${OUT}" probe="${PROBE}" \
  block_x="${BLOCK_X}" block_y="${BLOCK_Y}" pitch_align="${PITCH_ALIGN}" \
  probe_every="${PROBE_EVERY}"

echo "Done."
