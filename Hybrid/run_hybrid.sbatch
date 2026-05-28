#!/bin/bash
#SBATCH --job-name=lbm_hybrid
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=hybrid_%j.out
#SBATCH --error=hybrid_%j.err

set -euo pipefail

module purge
module load gcc cuda openmpi hdf5

NX=${NX:-800}
NY=${NY:-400}
RE=${RE:-100}
U_IN=${U_IN:-0.05}
STEPS=${STEPS:-60000}
EVERY=${EVERY:-500}
OUT=${OUT:-out/hybrid}
PROBE=${PROBE:-probe_hybrid.csv}

BLOCK_X=${BLOCK_X:-64}
BLOCK_Y=${BLOCK_Y:-4}
PITCH_ALIGN=${PITCH_ALIGN:-32}
PROBE_EVERY=${PROBE_EVERY:-1}

HALO=${HALO:-cuda}
STRICT_CUDA_AWARE=${STRICT_CUDA_AWARE:-1}

BIN=${BIN:-./lbm_hybrid_cuda_mpi}
NTASKS=${NTASKS:-${SLURM_NTASKS:-2}}

mkdir -p out

echo "Running LBM Hybrid MPI + CUDA"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_NODELIST=${SLURM_NODELIST}"
echo "SLURM_NTASKS=${SLURM_NTASKS:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "PWD=$(pwd)"
echo ""
echo "Parameters:"
echo "  ntasks=${NTASKS}"
echo "  nx=${NX} ny=${NY} re=${RE} u_in=${U_IN} steps=${STEPS}"
echo "  halo=${HALO} strict_cuda_aware=${STRICT_CUDA_AWARE}"
echo "  every=${EVERY} out=${OUT} probe=${PROBE}"
echo "  block_x=${BLOCK_X} block_y=${BLOCK_Y} pitch_align=${PITCH_ALIGN} probe_every=${PROBE_EVERY}"
echo ""

echo "GPU info:"
nvidia-smi
echo ""

echo "GPU topology:"
nvidia-smi topo -m || true
echo ""

echo "OpenMPI CUDA support hints:"
ompi_info --all 2>/dev/null | grep -i cuda || true
echo ""

echo "Running command:"
echo "srun --gpu-bind=none -n ${NTASKS} ${BIN} nx=${NX} ny=${NY} re=${RE} u_in=${U_IN} steps=${STEPS} block_x=${BLOCK_X} block_y=${BLOCK_Y} pitch_align=${PITCH_ALIGN} halo=${HALO} strict_cuda_aware=${STRICT_CUDA_AWARE} every=${EVERY} out=${OUT} probe=${PROBE} probe_every=${PROBE_EVERY}"
echo ""

srun --gpu-bind=none -n "${NTASKS}" "${BIN}" \
  nx="${NX}" ny="${NY}" re="${RE}" u_in="${U_IN}" steps="${STEPS}" \
  block_x="${BLOCK_X}" block_y="${BLOCK_Y}" pitch_align="${PITCH_ALIGN}" \
  halo="${HALO}" strict_cuda_aware="${STRICT_CUDA_AWARE}" \
  every="${EVERY}" out="${OUT}" probe="${PROBE}" probe_every="${PROBE_EVERY}"

echo ""
echo "Done."