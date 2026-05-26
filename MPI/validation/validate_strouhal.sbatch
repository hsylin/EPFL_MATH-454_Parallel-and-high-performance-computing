#!/bin/bash
#SBATCH --job-name=lbm_valid_st
#SBATCH --partition=academic
#SBATCH --ntasks=256
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=validation/validate_%j.out
#SBATCH --error=validation/validate_%j.err

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

echo "Building fresh binary..."
rm -f lbm lbm_mpi *.o *~
make clean
make

mkdir -p validation/probes validation/logs

OUT_MD="validation/strouhal_validation.md"

cat > "$OUT_MD" <<MD
# Strouhal Validation Results

Validation target: \`St ≈ 0.16\` at \`Re = 100\`.

This validation keeps the physical geometry fixed and only changes the number of MPI ranks. Therefore, it checks whether MPI domain decomposition and halo exchange preserve the numerical result.

| p | Grid | Steps | cyl_r | Diameter D | Probe | St | Status |
|--:|:-----|------:|------:|-----------:|:------|---:|:-------|
MD

extract_st() {
    local st_file="$1"
    awk '/St[[:space:]]*:/ {print $3}' "$st_file"
}

status_from_st() {
    local st="$1"
    python3 - <<PY
st = float("$st")
if 0.14 <= st <= 0.18:
    print("OK")
else:
    print("CHECK")
PY
}

run_validation_one() {
    local p="$1"
    local nx="$2"
    local ny="$3"
    local steps="$4"
    local cyl_r="$5"

    local cyl_x=$((nx / 4))
    local cyl_y=$((ny / 2))

    local probe_x
    probe_x=$(python3 - <<PY
print(int(round($cyl_x + 8.0 * float("$cyl_r"))))
PY
)
    local probe_y="$cyl_y"

    local diameter
    diameter=$(python3 - <<PY
print(2.0 * float("$cyl_r"))
PY
)

    local tag="fixed_p${p}_nx${nx}_ny${ny}_r${cyl_r}_s${steps}"
    local probe="validation/probes/${tag}.csv"
    local log="validation/logs/${tag}.log"
    local st_out="validation/logs/${tag}_st.txt"

    echo "[run] $tag, D=$diameter, cylinder=(${cyl_x},${cyl_y}), probe=(${probe_x},${probe_y})"

    srun -n "$p" "$ROOT_DIR/lbm_mpi" \
        nx="$nx" ny="$ny" re=100 u_in=0.05 \
        cyl_x="$cyl_x" cyl_y="$cyl_y" cyl_r="$cyl_r" \
        probe_x="$probe_x" probe_y="$probe_y" \
        steps="$steps" every=0 probe="$probe" \
        > "$log" 2>&1

    python3 "$ROOT_DIR/viz/strouhal.py" "$probe" \
        --u-in 0.05 \
        --diameter "$diameter" \
        > "$st_out"

    local st
    st=$(extract_st "$st_out")

    local status
    status=$(status_from_st "$st")

    echo "| $p | ${nx}×${ny} | $steps | $cyl_r | $diameter | (${probe_x},${probe_y}) | $st | $status |" >> "$OUT_MD"

    echo "[done] p=$p: St=$st, status=$status"
}

NX=800
NY=400
STEPS=60000
CYL_R=10

echo "Running fixed-geometry Strouhal validation..."
for p in 1 2 4 8 16 32 64 128 256; do
    run_validation_one "$p" "$NX" "$NY" "$STEPS" "$CYL_R"
done

echo
echo "Validation table written to:"
echo "$OUT_MD"
cat "$OUT_MD"

echo "Done."