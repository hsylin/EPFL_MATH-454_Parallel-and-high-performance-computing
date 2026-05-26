#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_scaling.sh
#
# Scaling experiment using serial-default-like problem size:
#   nx=800, ny=400, steps=60000
#
# Strong scaling:
#   fixed global grid = 800 x 400
#
# Weak scaling:
#   fixed local grid per rank = 800 x 400
#   global ny = 400 * p
#
# Output is disabled:
#   every=0, probe=/dev/null
# ---------------------------------------------------------------------------

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"

LBM_BIN="${LBM_BIN:-../lbm_mpi}"
LAUNCHER="${LAUNCHER:-srun}"

# High-p test.
STRONG_PS="${STRONG_PS:-1 2 4 8 16 32 64 128}"
WEAK_PS="${WEAK_PS:-1 2 4 8 16 32 64 128}"

# Serial-default-like values.
STRONG_NX="${STRONG_NX:-800}"
STRONG_NY="${STRONG_NY:-400}"

WEAK_NX="${WEAK_NX:-800}"
WEAK_BASE_NY="${WEAK_BASE_NY:-400}"

STEPS="${STEPS:-60000}"

launch_cmd() {
    local n="$1"
    shift

    case "$LAUNCHER" in
        srun)
            echo srun -n "$n" "$@"
            ;;
        mpirun)
            echo mpirun -np "$n" "$@"
            ;;
        *)
            echo "$LAUNCHER -n $n $*"
            ;;
    esac
}

run_one() {
    local mode="$1"
    local p="$2"
    local nx="$3"
    local ny="$4"

    local tag="${mode}_p${p}_nx${nx}_ny${ny}_s${STEPS}"
    local log="${LOG_DIR}/${tag}.log"

    if [[ -s "$log" ]] && grep -q '^MLUPS' "$log"; then
        echo "[skip ] $tag"
        return 0
    fi

    local cmd
    cmd="$(launch_cmd "$p" "$LBM_BIN" \
        nx="$nx" ny="$ny" re=100 u_in=0.05 \
        steps="$STEPS" every=0 probe=/dev/null)"

    echo "[run  ] $tag"
    echo "         $cmd" > "$log"
    echo "         ----" >> "$log"

    if ! eval "$cmd" >> "$log" 2>&1; then
        echo "[FAIL ] $tag, see $log" >&2
    fi
}

echo "=== Strong scaling: fixed grid ${STRONG_NX} x ${STRONG_NY}, steps=${STEPS} ==="
for p in $STRONG_PS; do
    run_one strong "$p" "$STRONG_NX" "$STRONG_NY"
done

echo
echo "=== Weak scaling: local grid ${WEAK_NX} x ${WEAK_BASE_NY} per rank, steps=${STEPS} ==="
for p in $WEAK_PS; do
    ny=$(( WEAK_BASE_NY * p ))
    run_one weak "$p" "$WEAK_NX" "$ny"
done

echo
echo "Logs are in $LOG_DIR"