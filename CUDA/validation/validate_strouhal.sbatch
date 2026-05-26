#!/bin/bash
#SBATCH --job-name=lbm_cuda_valid_st
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=validation/logs/validate_cuda_%j.out
#SBATCH --error=validation/logs/validate_cuda_%j.err




set -euo pipefail

module purge
module load gcc/11.3.0
module load cuda/11.8.0
module load hdf5/1.12.2

export OMP_NUM_THREADS=1

ROOT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT_DIR"

mkdir -p validation/probes validation/logs validation/cuda_out

echo "ROOT_DIR = $ROOT_DIR"
echo "PWD      = $(pwd)"

echo "Loaded modules:"
module list

echo "CUDA compiler:"
which nvcc
nvcc --version

echo "Building fresh CUDA binary..."
rm -f lbm_cuda lbm_cuda_opt *.o *~
make clean
make

OUT_MD="validation/cuda_strouhal_validation.md"

cat > "$OUT_MD" <<MD
# CUDA Strouhal Validation Results

Validation target: \`St ≈ 0.16\` at \`Re = 100\`.

This validation keeps the physical geometry fixed and only changes CUDA launch/memory-layout parameters. Therefore, it checks whether the CUDA implementation preserves the numerical result for the same physical problem.

| Grid | Steps | Block | Pitch align | cyl_r | Diameter D | Probe | probe_every | St | Status |
|:-----|------:|:------|------------:|------:|-----------:|:------|------------:|---:|:-------|
MD

# Fixed physical validation geometry.
NX="${NX:-800}"
NY="${NY:-400}"
CYL_R="${CYL_R:-10}"
DIAMETER="${DIAMETER:-$(python3 - <<PY
print(2.0 * float("$CYL_R"))
PY
)}"

# CUDA parameter sweep for the same fixed problem.
BLOCK_CONFIGS="${BLOCK_CONFIGS:-16x16 32x8 64x4}"
PITCH_ALIGNS="${PITCH_ALIGNS:-1 32 64}"

STEPS="${STEPS:-60000}"
RE="${RE:-100}"
U_IN="${U_IN:-0.05}"

# Keep this at 1 for safest Strouhal validation.
PROBE_EVERY="${PROBE_EVERY:-1}"

if [[ "$PROBE_EVERY" != "1" ]]; then
    echo "ERROR: PROBE_EVERY=$PROBE_EVERY, but this validation expects probe_every=1."
    echo "Set PROBE_EVERY=1 for the canonical validation."
    exit 1
fi

# Locate Strouhal script.
if [[ -f "$ROOT_DIR/strouhal.py" ]]; then
    STROUHAL_SCRIPT="$ROOT_DIR/strouhal.py"
elif [[ -f "$ROOT_DIR/viz/strouhal.py" ]]; then
    STROUHAL_SCRIPT="$ROOT_DIR/viz/strouhal.py"
else
    echo "ERROR: could not find strouhal.py or viz/strouhal.py" >&2
    exit 1
fi

# Locate CUDA binary only. Do not fall back to serial lbm.
if [[ -x "$ROOT_DIR/lbm_cuda_opt" ]]; then
    CUDA_BIN="$ROOT_DIR/lbm_cuda_opt"
elif [[ -x "$ROOT_DIR/lbm_cuda" ]]; then
    CUDA_BIN="$ROOT_DIR/lbm_cuda"
else
    echo "ERROR: could not find executable CUDA binary: lbm_cuda_opt or lbm_cuda" >&2
    exit 1
fi

echo "Using Strouhal script: $STROUHAL_SCRIPT"
echo "Using CUDA binary     : $CUDA_BIN"

extract_st() {
    local st_file="$1"
    python3 - "$st_file" <<'PY'
import re
import sys

text = open(sys.argv[1]).read()

patterns = [
    r'\bSt\b\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',
    r'\bStrouhal(?:\s+number)?\b\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',
]

for p in patterns:
    m = re.search(p, text, re.IGNORECASE)
    if m:
        print(m.group(1))
        sys.exit(0)

nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', text)
if nums:
    print(nums[-1])
else:
    raise SystemExit("ERROR: could not extract Strouhal number")
PY
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
    local block_x="$1"
    local block_y="$2"
    local pitch_align="$3"

    local cyl_x=$((NX / 4))
    local cyl_y=$((NY / 2))

    # Probe a few diameters downstream of the cylinder center.
    local probe_x
    probe_x=$(python3 - <<PY
print(int(round($cyl_x + 8.0 * float("$CYL_R"))))
PY
)
    local probe_y="$cyl_y"

    local tag="cuda_fixed_nx${NX}_ny${NY}_bx${block_x}_by${block_y}_pa${pitch_align}_r${CYL_R}_s${STEPS}"
    local probe="validation/probes/${tag}.csv"
    local log="validation/logs/${tag}.log"
    local st_out="validation/logs/${tag}_st.txt"
    local out_prefix="validation/cuda_out/${tag}"

    echo "[run] $tag, D=$DIAMETER, cylinder=(${cyl_x},${cyl_y}), probe=(${probe_x},${probe_y})"

    "$CUDA_BIN" \
        nx="$NX" ny="$NY" re="$RE" u_in="$U_IN" \
        cyl_x="$cyl_x" cyl_y="$cyl_y" cyl_r="$CYL_R" \
        probe_x="$probe_x" probe_y="$probe_y" \
        steps="$STEPS" every=0 \
        block_x="$block_x" block_y="$block_y" pitch_align="$pitch_align" \
        probe_every="$PROBE_EVERY" probe="$probe" \
        out="$out_prefix" \
        > "$log" 2>&1

    python3 "$STROUHAL_SCRIPT" "$probe" \
        --u-in "$U_IN" \
        --diameter "$DIAMETER" \
        > "$st_out"

    local st
    st=$(extract_st "$st_out")

    local status
    status=$(status_from_st "$st")

    echo "| ${NX}×${NY} | $STEPS | ${block_x}×${block_y} | $pitch_align | $CYL_R | $DIAMETER | (${probe_x},${probe_y}) | $PROBE_EVERY | $st | $status |" >> "$OUT_MD"

    echo "[done] block=${block_x}x${block_y}, pitch_align=$pitch_align: St=$st, status=$status"
}

echo "Running fixed-geometry CUDA Strouhal validation..."

for cfg in $BLOCK_CONFIGS; do
    BX="${cfg%x*}"
    BY="${cfg#*x}"

    for PA in $PITCH_ALIGNS; do
        run_validation_one "$BX" "$BY" "$PA"
    done
done

echo
echo "Validation table written to:"
echo "$OUT_MD"
cat "$OUT_MD"

echo "Done."