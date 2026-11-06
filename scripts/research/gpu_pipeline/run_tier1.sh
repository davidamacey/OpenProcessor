#!/usr/bin/env bash
# Tier-1 runs of gpu:PR-0, PR-1 and PR-2, in dependency order, on one GPU under the shared lock.
#
# Required environment (no defaults point at a private location):
#   PY            venv interpreter (see README.md)
#   LOCK          lock file every GPU job on the card agrees on
#   GPU           GPU index (a single card)
#   SET_A, SET_B  set A / set B image roots (pins are in scripts/datasets/manifests)
#   SET_P, SET_P_MANIFEST  the private sample root and its manifest (outside the repository)
#   OI_ROOT, OI_PIN        the Open Images 1,024 px sample and its pin
#   FIXTURES      directory made by exif_fixtures.py
#   DET_PLAN, PE_PLAN      read-only TensorRT plans of the detector and PE
#   OUT           raw output directory (outside the repository)
# Optional:
#   ONLY          space-separated set labels to run (default: all of the prototype's sets)
#   LEDGER        file that gets one line per GPU command: seconds held on the card
# Usage: run_tier1.sh pr0|pr1|pr2
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
pins="$repo/scripts/datasets/manifests"
: "${PY:?}" "${LOCK:?}" "${GPU:?}" "${SET_A:?}" "${SET_B:?}" "${SET_P:?}" "${SET_P_MANIFEST:?}"
: "${OI_ROOT:?}" "${OI_PIN:?}" "${FIXTURES:?}" "${DET_PLAN:?}" "${PE_PLAN:?}" "${OUT:?}"

gpu() { # every GPU command: one job at a time on the card, low CPU priority, seconds recorded
  local rc=0
  # shellcheck disable=SC2016  # the inner script expands its own variables
  CUDA_VISIBLE_DEVICES="$GPU" flock "$LOCK" bash -c '
    start=$(date +%s.%N)
    nice -n 10 "$@"
    rc=$?
    if [ -n "${LEDGER:-}" ]; then
      printf "%s %s rc=%s\n" "$(echo "$(date +%s.%N) - $start" | bc)" "$(basename "$2")" "$rc" >> "$LEDGER"
    fi
    exit "$rc"' _ "$@" || rc=$?
  return "$rc"
}

wanted() { # is this set label selected?
  [[ -z "${ONLY:-}" || " $ONLY " == *" $1 "* ]]
}

# label root pin-or-manifest-flag...
sets=(
  "A|$SET_A|--pin $pins/coco_bench_2000.json"
  "B|$SET_B|--pin $pins/commons_hires_500.json"
  "P|$SET_P|--manifest $SET_P_MANIFEST"
)

pr0() {
  for row in "${sets[@]}"; do
    IFS='|' read -r label root sel <<<"$row"
    wanted "$label" || continue
    # shellcheck disable=SC2086  # $sel is a flag and a path, split on purpose
    gpu "$PY" "$here/pr0_stage_split.py" run --root "$root" $sel --n 200 --label "$label" \
      --det-plan "$DET_PLAN" --pe-plan "$PE_PLAN" --gpu-index 0 --reps 3 --warmup 40 --check 40 --out "$OUT/pr0"
    # shellcheck disable=SC2086
    gpu "$PY" "$here/pr0_stage_split.py" profile --root "$root" $sel --n 200 --label "$label" \
      --det-plan "$DET_PLAN" --pe-plan "$PE_PLAN" --gpu-index 0 --warmup 40 --out "$OUT/pr0"
  done
}

pr1() {
  local arms=(control nvimgcodec:hybrid:1 nvimgcodec:hybrid:8 nvimgcodec:hybrid:64 dali:0:64 dali:default:64)
  local rows=("${sets[@]}" "OI|$OI_ROOT|--pin $OI_PIN")
  for row in "${rows[@]}"; do
    IFS='|' read -r label root sel <<<"$row"
    wanted "$label" || continue
    # shellcheck disable=SC2086
    gpu "$PY" "$here/pr1_decoders.py" bench --root "$root" $sel --n 200 --label "$label" \
      --arms "${arms[@]}" --gpu-index 0 --reps 3 --out "$OUT/pr1"
    # shellcheck disable=SC2086
    gpu "$PY" "$here/pr1_decoders.py" profile --root "$root" $sel --n 200 --label "$label" \
      --arms "${arms[@]}" --gpu-index 0 --out "$OUT/pr1"
  done
  gpu "$PY" "$here/pr1_decoders.py" coverage --root "$FIXTURES" --label fixtures \
    --arms control nvimgcodec:hybrid:8 nvimgcodec:gpu:8 dali:0:8 --gpu-index 0 --out "$OUT/pr1"
}

pr2() {
  for row in "${sets[@]}"; do
    IFS='|' read -r label root sel <<<"$row"
    wanted "$label" || continue
    # shellcheck disable=SC2086
    gpu "$PY" "$here/pr2_resize_ops.py" run --root "$root" $sel --n 200 --label "$label" \
      --arms K-a K-b K-b-fused K-c --det-plan "$DET_PLAN" --pe-plan "$PE_PLAN" --gpu-index 0 --reps 3 --out "$OUT/pr2"
    for pass in 1 2; do
      # shellcheck disable=SC2086
      gpu "$PY" "$here/pr2_resize_ops.py" profile --root "$root" $sel --n 200 --label "${label}_p$pass" \
        --arms K-a K-b K-b-fused K-c --det-plan "$DET_PLAN" --pe-plan "$PE_PLAN" --gpu-index 0 --out "$OUT/pr2"
    done
  done
}

"${1:?usage: run_tier1.sh pr0|pr1|pr2}"
