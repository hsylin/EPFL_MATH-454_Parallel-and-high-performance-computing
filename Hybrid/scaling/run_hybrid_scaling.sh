#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
OUT_DIR="${SCRIPT_DIR}/out"
mkdir -p "$LOG_DIR" "$OUT_DIR"

BIN="${BIN:-${ROOT_DIR}/lbm_hybrid_cuda_mpi}"
RANKS="${RANKS:-1 2}"
GRIDS="${GRIDS:-800x400 1500x750 1600x800}"
HALOS="${HALOS:-cuda staged}"
REPEATS="${REPEATS:-3}"
STEPS="${STEPS:-60000}"
BLOCK_X="${BLOCK_X:-64}"
BLOCK_Y="${BLOCK_Y:-4}"
PITCH_ALIGN="${PITCH_ALIGN:-32}"
RE="${RE:-100}"
U_IN="${U_IN:-0.05}"
EVERY="${EVERY:-0}"
PROBE_EVERY="${PROBE_EVERY:-0}"
PROBE="${PROBE:-/dev/null}"
STRICT_CUDA_AWARE="${STRICT_CUDA_AWARE:-1}"
SRUN_GPU_BIND="${SRUN_GPU_BIND:-none}"
RESULT_CSV="${RESULT_CSV:-${SCRIPT_DIR}/results_hybrid.csv}"

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

run_srun() {
  local ranks="$1"
  shift
  if [[ -n "$SRUN_GPU_BIND" ]]; then
    srun --gpu-bind="$SRUN_GPU_BIND" -n "$ranks" "$@"
  else
    srun -n "$ranks" "$@"
  fi
}

echo "ranks,halo,grid,nx,ny,steps,repeat,block_x,block_y,pitch_align,wall_time_s,mlups,max_halo_overhead_percent,avg_halo_overhead_percent,max_other_time_s,avg_other_time_s,log" > "$RESULT_CSV"

for grid in $GRIDS; do
  nx="${grid%x*}"
  ny="${grid#*x}"
  for halo in $HALOS; do
    for ranks in $RANKS; do
      for rep in $(seq 1 "$REPEATS"); do
        tag="hybrid_r${ranks}_${halo}_nx${nx}_ny${ny}_s${STEPS}_rep${rep}"
        log="${LOG_DIR}/${tag}.log"
        echo "[run] $tag"
        {
          echo "Command:"
          echo "  srun --gpu-bind=${SRUN_GPU_BIND} -n ${ranks} ${BIN} nx=${nx} ny=${ny} re=${RE} u_in=${U_IN} steps=${STEPS} block_x=${BLOCK_X} block_y=${BLOCK_Y} pitch_align=${PITCH_ALIGN} halo=${halo} strict_cuda_aware=${STRICT_CUDA_AWARE} every=${EVERY} out=${OUT_DIR}/${tag} probe=${PROBE} probe_every=${PROBE_EVERY}"
          echo "----"
        } > "$log"

        if ! run_srun "$ranks" "$BIN" \
          nx="$nx" ny="$ny" re="$RE" u_in="$U_IN" steps="$STEPS" \
          block_x="$BLOCK_X" block_y="$BLOCK_Y" pitch_align="$PITCH_ALIGN" halo="$halo" \
          strict_cuda_aware="$STRICT_CUDA_AWARE" \
          every="$EVERY" out="${OUT_DIR}/${tag}" probe="$PROBE" probe_every="$PROBE_EVERY" \
          >> "$log" 2>&1; then
          echo "[FAIL] $tag, see $log" >&2
          exit 1
        fi

        wall="$(extract_metric "Wall time" "$log")" || { echo "missing Wall time in $log" >&2; exit 1; }
        mlups="$(extract_metric "MLUPS" "$log")" || { echo "missing MLUPS in $log" >&2; exit 1; }
        max_halo_frac="$(extract_metric "Max halo overhead percent" "$log")" || { echo "missing Max halo overhead percent in $log" >&2; exit 1; }
        avg_halo_frac="$(extract_metric "Avg halo overhead percent" "$log")" || { echo "missing Avg halo overhead percent in $log" >&2; exit 1; }
        max_other="$(extract_metric "Max other time" "$log")" || { echo "missing Max other time in $log" >&2; exit 1; }
        avg_other="$(extract_metric "Avg other time" "$log")" || { echo "missing Avg other time in $log" >&2; exit 1; }
        echo "${ranks},${halo},${grid},${nx},${ny},${STEPS},${rep},${BLOCK_X},${BLOCK_Y},${PITCH_ALIGN},${wall},${mlups},${max_halo_frac},${avg_halo_frac},${max_other},${avg_other},${log}" >> "$RESULT_CSV"
      done
    done
  done
done

echo "Wrote ${RESULT_CSV}"
