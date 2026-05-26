#!/bin/bash
#SBATCH --job-name=lbm_scaling
#SBATCH --partition=academic
#SBATCH --ntasks=256
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --output=scaling/logs/sbatch_%j.out
#SBATCH --error=scaling/logs/sbatch_%j.err

set -euo pipefail

module purge
module load gcc/13.2.0
module load openmpi/5.0.3
module load hdf5/1.14.3-mpi

export OMP_NUM_THREADS=1

ROOT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT_DIR"

echo "ROOT_DIR = $ROOT_DIR"
echo "PWD      = $(pwd)"

echo "Loaded modules:"
module list

echo "Compiler:"
which mpicxx
mpicxx --version

mkdir -p "$ROOT_DIR/scaling/logs"

echo "Cleaning old binaries..."
rm -f lbm lbm_mpi *.o *~

echo "Building..."
make clean
make

# Remove old scaling run logs, but keep sbatch output/error logs.
rm -f "$ROOT_DIR"/scaling/logs/strong_*.log
rm -f "$ROOT_DIR"/scaling/logs/weak_*.log

LBM_BIN="$ROOT_DIR/lbm_mpi"

# MPI rank counts.
PS="${PS:-1 2 4 8 16 32 64 128 256}"

# ---------------------------------------------------------------------
# Strong scaling for several representative fixed global problem sizes.
# This directly addresses the project requirement:
# "A strong scaling study of the MPI version, against your own Amdahl
# prediction, for several representative problem sizes."
# ---------------------------------------------------------------------
STRONG_SIZES="${STRONG_SIZES:-800x400 1200x600 1600x800}"
STRONG_STEPS="${STRONG_STEPS:-60000}"

echo
echo "============================================================"
echo "Running MPI strong scaling"
echo "============================================================"
echo "PS           = $PS"
echo "STRONG_SIZES = $STRONG_SIZES"
echo "STRONG_STEPS = $STRONG_STEPS"

for size in $STRONG_SIZES; do
    NX="${size%x*}"
    NY="${size#*x}"

    echo
    echo "------------------------------------------------------------"
    echo "Strong scaling size: ${NX}x${NY}"
    echo "------------------------------------------------------------"

    for p in $PS; do
        log="$ROOT_DIR/scaling/logs/strong_p${p}_nx${NX}_ny${NY}_s${STRONG_STEPS}.log"

        echo "[strong] p=$p nx=$NX ny=$NY steps=$STRONG_STEPS"
        echo "Log: $log"

        srun -n "$p" "$LBM_BIN" \
            nx="$NX" ny="$NY" re=100 u_in=0.05 \
            steps="$STRONG_STEPS" every=0 probe=/dev/null \
            > "$log" 2>&1

        grep -E "Wall time max|Compute time max|Comm time max|Comm fraction max|MLUPS" "$log" || true
    done
done

# ---------------------------------------------------------------------
# Weak scaling.
# Keep local height fixed at WEAK_BASE_NY rows per rank.
# ---------------------------------------------------------------------
RUN_WEAK="${RUN_WEAK:-1}"
WEAK_NX="${WEAK_NX:-800}"
WEAK_BASE_NY="${WEAK_BASE_NY:-400}"
WEAK_STEPS="${WEAK_STEPS:-60000}"

if [[ "$RUN_WEAK" == "1" ]]; then
    echo
    echo "============================================================"
    echo "Running MPI weak scaling"
    echo "============================================================"
    echo "WEAK_NX      = $WEAK_NX"
    echo "WEAK_BASE_NY = $WEAK_BASE_NY"
    echo "WEAK_STEPS   = $WEAK_STEPS"

    for p in $PS; do
        NY=$((WEAK_BASE_NY * p))
        log="$ROOT_DIR/scaling/logs/weak_p${p}_nx${WEAK_NX}_ny${NY}_s${WEAK_STEPS}.log"

        echo "[weak] p=$p nx=$WEAK_NX ny=$NY steps=$WEAK_STEPS"
        echo "Log: $log"

        srun -n "$p" "$LBM_BIN" \
            nx="$WEAK_NX" ny="$NY" re=100 u_in=0.05 \
            steps="$WEAK_STEPS" every=0 probe=/dev/null \
            > "$log" 2>&1

        grep -E "Wall time max|Compute time max|Comm time max|Comm fraction max|MLUPS" "$log" || true
    done
else
    echo
    echo "Skipping weak scaling because RUN_WEAK=$RUN_WEAK"
fi

echo
echo "Generating scaling markdown table..."
python3 "$ROOT_DIR/scaling/make_table.py" \
  --logs "$ROOT_DIR/scaling/logs" \
  --out "$ROOT_DIR/scaling/scaling.md"

echo
echo "Generating HTML report if script exists..."
if [[ -f "$ROOT_DIR/scaling/make_html.py" ]]; then
    python3 "$ROOT_DIR/scaling/make_html.py"
else
    echo "No scaling/make_html.py found; skipped HTML generation."
fi

echo
echo "Generated files:"
ls -lh "$ROOT_DIR/scaling" || true

echo
echo "Useful checks:"
echo "  cat $ROOT_DIR/scaling/scaling.md"
echo "  grep -H \"MLUPS\" $ROOT_DIR/scaling/logs/*.log"
echo "  ls -lht $ROOT_DIR/scaling/logs | head"

echo
echo "Done."