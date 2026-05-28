#!/bin/bash
#SBATCH --job-name=lbm_serial_base
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --partition=academic
#SBATCH --output=serial_baseline_%j.out
#SBATCH --error=serial_baseline_%j.err

set -euo pipefail

module purge
module load gcc hdf5

echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Start time: $(date)"

mkdir -p baseline_runs

echo "== Building optimized serial baseline without -pg =="

if command -v h5c++ >/dev/null 2>&1; then
    h5c++ -O3 -DNDEBUG -std=c++11 -Wall -Wextra -pedantic \
        main.cc lbm.cc output.cc \
        -o lbm_baseline
else
    echo "ERROR: h5c++ not found. Check HDF5 module."
    module list
    exit 1
fi

CSV="baseline_runs/serial_baseline_800x400_s60000.csv"
echo "repeat,wall_s,mlups" > "$CSV"

for r in 1 2 3 4 5; do
    LOG="baseline_runs/run_${r}.log"
    echo "== Run $r/5 =="

    ./lbm_baseline \
        nx=800 ny=400 re=100 u_in=0.05 steps=60000 \
        every=0 out=baseline_runs/lbm_${r} probe=/dev/null \
        > "$LOG" 2>&1

    WALL=$(awk '/Wall time/ {print $(NF-1)}' "$LOG")
    MLUPS=$(awk '/MLUPS/ {print $NF}' "$LOG")

    if [[ -z "$WALL" || -z "$MLUPS" ]]; then
        echo "ERROR: cannot extract Wall time or MLUPS from $LOG"
        cat "$LOG"
        exit 1
    fi

    echo "${r},${WALL},${MLUPS}" | tee -a "$CSV"
done

python3 - "$CSV" <<'PY'
import csv, statistics, sys
path = sys.argv[1]
rows = list(csv.DictReader(open(path)))
mlups = [float(r["mlups"]) for r in rows]
walls = [float(r["wall_s"]) for r in rows]

print("\n== Serial baseline summary ==")
print(f"runs         : {len(mlups)}")
print(f"median MLUPS : {statistics.median(mlups):.2f}")
print(f"mean MLUPS   : {statistics.mean(mlups):.2f}")
print(f"median wall  : {statistics.median(walls):.2f} s")
print(f"CSV          : {path}")
PY

echo "End time: $(date)"

