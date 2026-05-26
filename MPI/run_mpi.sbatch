#!/bin/bash
#SBATCH --job-name=lbm_mpi
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --partition=academic
#SBATCH --output=mpi_%j.out
#SBATCH --error=mpi_%j.err

set -euo pipefail

module purge
module load gcc/13.2.0
module load openmpi/5.0.3
module load hdf5/1.14.3-mpi

NX=${NX:-800}
NY=${NY:-400}
RE=${RE:-100}
U_IN=${U_IN:-0.05}
STEPS=${STEPS:-60000}
EVERY=${EVERY:-500}
OUT=${OUT:-out/lbm}
PROBE=${PROBE:-probe.csv}

mkdir -p out

echo "Running LBM MPI"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_NTASKS=${SLURM_NTASKS}"
echo "SLURM_NODELIST=${SLURM_NODELIST}"
echo "Parameters:"
echo "  nx=${NX} ny=${NY} re=${RE} u_in=${U_IN} steps=${STEPS}"
echo "  every=${EVERY} out=${OUT} probe=${PROBE}"

srun ./lbm_mpi \
  nx="${NX}" ny="${NY}" re="${RE}" u_in="${U_IN}" steps="${STEPS}" \
  every="${EVERY}" out="${OUT}" probe="${PROBE}"

echo "Done."
EOF