#!/bin/bash
#SBATCH --job-name=lbm-valid
#SBATCH --partition=academic
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=validate-%j.out
#SBATCH --error=validate-%j.err

set -euo pipefail

module purge
module load gcc/13.2.0
module load openmpi/5.0.3
module load hdf5/1.14.3-mpi

cd ~/MPI

rm -f lbm lbm_mpi *.o
make clean
make

rm -rf out probe.csv
mkdir -p out

srun ./lbm_mpi \
  nx=800 ny=400 re=100 u_in=0.05 steps=60000 every=500 \
  out=out/lbm probe=probe.csv

python3 viz/strouhal.py probe.csv --u-in 0.05 --diameter 20
