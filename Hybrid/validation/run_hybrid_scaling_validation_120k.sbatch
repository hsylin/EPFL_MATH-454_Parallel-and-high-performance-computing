#!/bin/bash
#SBATCH --job-name=hybrid_valid_120k
#SBATCH --partition=gpu
#SBATCH --account=math-454
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=validation/logs/validate_hybrid_120k_%j.out
#SBATCH --error=validation/logs/validate_hybrid_120k_%j.err

set -euo pipefail

module purge
module load gcc/11.3.0
module load cuda/11.8.0
module load openmpi/4.1.3-cuda
module load hdf5/1.12.2

export OMP_NUM_THREADS=1

ROOT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT_DIR"

mkdir -p validation/logs validation/probes validation/strouhal validation/hybrid_out

echo "ROOT_DIR = $ROOT_DIR"
echo "PWD      = $(pwd)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_NODELIST=${SLURM_NODELIST}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

echo
echo "Loaded modules:"
module list

echo
echo "GPU topology:"
nvidia-smi topo -m || true

echo
echo "OpenMPI CUDA-related configuration hints:"
ompi_info --parsable --all 2>/dev/null | grep -i cuda | head -80 || true

echo
echo "Building fresh hybrid binary..."
make clean
make

HYBRID_BIN="$ROOT_DIR/lbm_hybrid_cuda_mpi"
if [[ ! -x "$HYBRID_BIN" ]]; then
    echo "ERROR: could not find executable hybrid binary: $HYBRID_BIN" >&2
    exit 1
fi

if [[ -f "$ROOT_DIR/viz/strouhal.py" ]]; then
    STROUHAL_SCRIPT="$ROOT_DIR/viz/strouhal.py"
elif [[ -f "$ROOT_DIR/strouhal.py" ]]; then
    STROUHAL_SCRIPT="$ROOT_DIR/strouhal.py"
else
    echo "ERROR: could not find strouhal.py or viz/strouhal.py" >&2
    exit 1
fi

echo "Using hybrid binary   : $HYBRID_BIN"
echo "Using Strouhal script : $STROUHAL_SCRIPT"

OUT_MD="validation/hybrid_scaling_strouhal_validation_120k.md"
CONFIG_TSV="validation/scaling_validation_configs_120k.tsv"

TARGET_ST="${TARGET_ST:-0.16}"
VALIDATION_STEPS="${VALIDATION_STEPS:-120000}"

RE="${RE:-100}"
U_IN="${U_IN:-0.05}"
CYL_R="${CYL_R:-10}"
DIAMETER="${DIAMETER:-$(python3 - <<PY
print(2.0 * float("$CYL_R"))
PY
)}"

PROBE_EVERY="${PROBE_EVERY:-1}"
EVERY="${EVERY:-0}"
STRICT_CUDA_AWARE="${STRICT_CUDA_AWARE:-1}"
SRUN_GPU_BIND="${SRUN_GPU_BIND:-none}"

if [[ "$PROBE_EVERY" != "1" ]]; then
    echo "ERROR: PROBE_EVERY=$PROBE_EVERY, but Strouhal validation expects PROBE_EVERY=1."
    exit 1
fi

cat > "$OUT_MD" <<MD
# Hybrid MPI+CUDA Strouhal Validation Results, 120k Steps

Validation target: \`St ≈ 0.16\` at \`Re = 100\`.

This validation reruns the same rank/halo/grid configurations used in the hybrid scaling report, but with \`STEPS=120000\` and \`PROBE_EVERY=1\` to improve FFT frequency resolution.

| Grid | Ranks | Halo | Steps | Block | Pitch align | Probe | St | Error vs 0.16 | Status | Wall (s) | MLUPS |
|:-----|------:|:-----|------:|:------|------------:|:------|---:|--------------:|:-------|---------:|------:|
MD

echo "Generating validation config list from scaling/results_hybrid.csv"

if [[ -f "scaling/results_hybrid.csv" ]]; then
python3 - <<PY
import csv
from pathlib import Path

csv_path = Path("scaling/results_hybrid.csv")
out_path = Path("$CONFIG_TSV")
steps = "$VALIDATION_STEPS"

rows = []
seen = set()

with csv_path.open(newline="") as f:
    reader = csv.DictReader(f)
    for r in reader:
        key = (
            r["grid"],
            r["ranks"],
            r["halo"],
            steps,
            r.get("block_x", "64"),
            r.get("block_y", "4"),
            r.get("pitch_align", "32"),
        )
        if key not in seen:
            seen.add(key)
            rows.append(key)

def sort_key(row):
    grid, ranks, halo, *_ = row
    nx, ny = map(int, grid.split("x"))
    return (nx * ny, nx, ny, int(ranks), halo)

rows.sort(key=sort_key)

with out_path.open("w") as f:
    for row in rows:
        f.write("\\t".join(row) + "\\n")

print(f"Wrote {out_path} with {len(rows)} configurations")
PY
else
    echo "ERROR: scaling/results_hybrid.csv not found."
    echo "Run scaling first or place results_hybrid.csv under scaling/."
    exit 1
fi

echo
echo "Validation configurations:"
cat "$CONFIG_TSV"
echo

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
target = float("$TARGET_ST")
err = abs(st - target) / target * 100.0

if 0.14 <= st <= 0.18:
    if err <= 6.0:
        print("OK")
    else:
        print("OK-range")
else:
    print("CHECK")
PY
}

error_percent() {
    local st="$1"
    python3 - <<PY
st = float("$st")
target = float("$TARGET_ST")
print(f"{abs(st - target) / target * 100.0:.2f}%")
PY
}

extract_metric() {
    local key="$1"
    local log="$2"
    awk -F: -v key="$key" '
      $0 ~ "^" key ":" {
        value = $2
        gsub(/^[ \t]+|[ \t]+$/, "", value)
        split(value, parts, /[ \t]+/)
        print parts[1]
        found = 1
        exit 0
      }
      END { if (!found) exit 1 }
    ' "$log"
}

run_strouhal_script() {
    local probe="$1"
    local ny="$2"
    local st_out="$3"

    python3 "$STROUHAL_SCRIPT" "$probe" \
        --ny "$ny" \
        --u-in "$U_IN" \
        --diameter "$DIAMETER" \
        > "$st_out"
}

run_validation_one() {
    local grid="$1"
    local ranks="$2"
    local halo="$3"
    local steps="$4"
    local block_x="$5"
    local block_y="$6"
    local pitch_align="$7"

    local nx="${grid%x*}"
    local ny="${grid#*x}"

    local cyl_x=$((nx / 4))
    local cyl_y=$((ny / 2))

    local probe_x
    probe_x=$(python3 - <<PY
print(int(round($cyl_x + 8.0 * float("$CYL_R"))))
PY
)
    local probe_y="$cyl_y"

    local strict="$STRICT_CUDA_AWARE"
    if [[ "$halo" == "staged" ]]; then
        strict="0"
    fi

    local tag="hybrid_valid120k_r${ranks}_${halo}_nx${nx}_ny${ny}_bx${block_x}_by${block_y}_pa${pitch_align}_r${CYL_R}_s${steps}"
    local probe="validation/probes/${tag}.csv"
    local log="validation/logs/${tag}.log"
    local st_out="validation/strouhal/${tag}_st.txt"
    local out_prefix="validation/hybrid_out/${tag}"

    echo "[run] $tag"
    echo "      grid=${grid}, ranks=${ranks}, halo=${halo}, block=${block_x}x${block_y}, pitch_align=${pitch_align}"
    echo "      cylinder=(${cyl_x},${cyl_y}), D=${DIAMETER}, probe=(${probe_x},${probe_y})"

    {
        echo "Command:"
        echo "srun --gpu-bind=${SRUN_GPU_BIND} -n ${ranks} ${HYBRID_BIN} nx=${nx} ny=${ny} re=${RE} u_in=${U_IN} cyl_x=${cyl_x} cyl_y=${cyl_y} cyl_r=${CYL_R} probe_x=${probe_x} probe_y=${probe_y} steps=${steps} every=${EVERY} block_x=${block_x} block_y=${block_y} pitch_align=${pitch_align} halo=${halo} strict_cuda_aware=${strict} probe_every=${PROBE_EVERY} probe=${probe} out=${out_prefix}"
        echo "----"
    } > "$log"

    srun --gpu-bind="${SRUN_GPU_BIND}" -n "$ranks" "$HYBRID_BIN" \
        nx="$nx" ny="$ny" re="$RE" u_in="$U_IN" \
        cyl_x="$cyl_x" cyl_y="$cyl_y" cyl_r="$CYL_R" \
        probe_x="$probe_x" probe_y="$probe_y" \
        steps="$steps" every="$EVERY" \
        block_x="$block_x" block_y="$block_y" pitch_align="$pitch_align" \
        halo="$halo" strict_cuda_aware="$strict" \
        probe_every="$PROBE_EVERY" probe="$probe" \
        out="$out_prefix" \
        < /dev/null >> "$log" 2>&1

    run_strouhal_script "$probe" "$ny" "$st_out"

    local st
    st=$(extract_st "$st_out")

    local status
    status=$(status_from_st "$st")

    local err
    err=$(error_percent "$st")

    local wall
    local mlups
    wall="$(extract_metric "Wall time" "$log" || echo "NA")"
    mlups="$(extract_metric "MLUPS" "$log" || echo "NA")"

    echo "| ${nx}×${ny} | $ranks | \`$halo\` | $steps | ${block_x}×${block_y} | $pitch_align | (${probe_x},${probe_y}) | $st | $err | $status | $wall | $mlups |" >> "$OUT_MD"

    echo "[done] $tag: St=$st, error=$err, status=$status, wall=$wall, MLUPS=$mlups"
}

echo
echo "Running 120k-step hybrid MPI+CUDA Strouhal validation..."
echo

mapfile -t CONFIG_LINES < "$CONFIG_TSV"

for line in "${CONFIG_LINES[@]}"; do
    [[ -z "$line" ]] && continue
    IFS=$'\t' read -r grid ranks halo steps block_x block_y pitch_align <<< "$line"
    run_validation_one "$grid" "$ranks" "$halo" "$steps" "$block_x" "$block_y" "$pitch_align"
done

echo
echo "Validation table written to:"
echo "$OUT_MD"
cat "$OUT_MD"

echo
echo "Data rows:"
grep -c '^| [0-9]' "$OUT_MD" || true

echo
echo "Done."
