#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_scaling.sh
#
# CUDA performance experiment runner, using the same file layout as the MPI scaling scripts.
#
# This script tests:
#   - different CUDA block shapes: block_x / block_y
#   - different padded row-pitch alignments: pitch_align
#   - different global grid sizes: nx / ny
#
# Fixed settings:
#   steps = 60000 by default
#   every = 0 and probe_every = 0 by default for pure performance timing
#
# Output:
#   scaling/logs/*.log
#   scaling/results_cuda_optimized.csv
# ---------------------------------------------------------------------------

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
OUT_DIR="${SCRIPT_DIR}/out"
mkdir -p "$LOG_DIR" "$OUT_DIR"

LBM_BIN="${LBM_BIN:-${SCRIPT_DIR}/../lbm_cuda_opt}"

GRIDS="${GRIDS:-400x200 800x400 1600x800}"
BLOCK_CONFIGS="${BLOCK_CONFIGS:-16x16 32x4 32x8 32x16 64x4}"
PITCH_ALIGNS="${PITCH_ALIGNS:-1 16 32 64}"

STEPS="${STEPS:-60000}"
RE="${RE:-100}"
U_IN="${U_IN:-0.05}"

EVERY="${EVERY:-0}"
PROBE_EVERY="${PROBE_EVERY:-0}"
PROBE_PATH="${PROBE_PATH:-/dev/null}"

RESULT_CSV="${RESULT_CSV:-${SCRIPT_DIR}/results_cuda_optimized.csv}"

extract_metric() {
    local key="$1"
    local log="$2"
    awk -v key="$key" '
        $0 ~ key {
            for (i = 1; i <= NF; ++i) {
                if ($i ~ /^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$/) {
                    val = $i
                }
            }
        }
        END { if (val != "") print val; }
    ' "$log"
}

run_one() {
    local nx="$1"
    local ny="$2"
    local bx="$3"
    local by="$4"
    local pa="$5"

    local tpb=$((bx * by))
    local tag="cuda_nx${nx}_ny${ny}_bx${bx}_by${by}_pa${pa}_s${STEPS}"
    local log="${LOG_DIR}/${tag}.log"

    if [[ -s "$log" ]] && grep -q '^MLUPS' "$log"; then
        echo "[skip] $tag"
    else
        echo "[run ] $tag"

        {
            echo "Command:"
            echo "  ${LBM_BIN} nx=${nx} ny=${ny} re=${RE} u_in=${U_IN} steps=${STEPS} every=${EVERY} block_x=${bx} block_y=${by} pitch_align=${pa} probe_every=${PROBE_EVERY} out=${OUT_DIR}/${tag} probe=${PROBE_PATH}"
            echo "----"
        } > "$log"

        if ! "$LBM_BIN" \
            nx="$nx" ny="$ny" re="$RE" u_in="$U_IN" steps="$STEPS" every="$EVERY" \
            block_x="$bx" block_y="$by" pitch_align="$pa" probe_every="$PROBE_EVERY" \
            out="${OUT_DIR}/${tag}" probe="$PROBE_PATH" >> "$log" 2>&1; then
            echo "[FAIL] $tag, see $log" >&2
            return 0
        fi
    fi

    local wall
    local mlups
    wall="$(extract_metric "Wall time" "$log")"
    mlups="$(extract_metric "MLUPS" "$log")"

    if [[ -z "$wall" || -z "$mlups" ]]; then
        echo "[warn] Could not parse Wall time or MLUPS from $log" >&2
        wall="nan"
        mlups="nan"
    fi

    echo "${nx}x${ny},${nx},${ny},${bx},${by},${tpb},${pa},${STEPS},${wall},${mlups},${log}" >> "$RESULT_CSV"
}

echo "grid,nx,ny,block_x,block_y,threads_per_block,pitch_align,steps,wall_time_s,mlups,log" > "$RESULT_CSV"

echo "=== CUDA sweep: steps=${STEPS} ==="
echo "Grids        : ${GRIDS}"
echo "Blocks       : ${BLOCK_CONFIGS}"
echo "Pitch aligns : ${PITCH_ALIGNS}"
echo "Logs         : ${LOG_DIR}"
echo "CSV          : ${RESULT_CSV}"
echo

for grid in $GRIDS; do
    nx="${grid%x*}"
    ny="${grid#*x}"

    if [[ -z "$nx" || -z "$ny" || "$nx" == "$grid" ]]; then
        echo "Invalid grid format: '$grid'. Expected format like 800x400." >&2
        exit 1
    fi

    for pa in $PITCH_ALIGNS; do
        for cfg in $BLOCK_CONFIGS; do
            bx="${cfg%x*}"
            by="${cfg#*x}"

            if [[ -z "$bx" || -z "$by" || "$bx" == "$cfg" ]]; then
                echo "Invalid block config: '$cfg'. Expected format like 32x8." >&2
                exit 1
            fi

            run_one "$nx" "$ny" "$bx" "$by" "$pa"
        done
    done
done

echo
echo "Wrote $RESULT_CSV"
echo "Logs are in $LOG_DIR"
