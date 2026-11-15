#!/bin/bash
# One exclusive hold of the native-engine comparison (plan 17 section 5, N0 "The holds"): every arm
# of a set of rows, alternating, on the card the coordinator names, under one lock, with the load
# gate, the device guard and bounded waits. Run by the coordinator, never by a lane.
#
#   scripts/bench/frame_engine_hold.sh --hold-file FILE --rows s0|h1|h2|h3 \
#       --lists DIR --out FILE.jsonl --gpu-index N --gpu-bus-id BUS [options]
#
#   --hold-file FILE   the lock every job on the card agrees on (flock -o, waits at most
#                      --lock-wait-s, then exits 75)
#   --rows NAME        s0 smoke | h1 parity | h2 rate and ceiling | h3 capacity (null models)
#   --lists DIR        the lists of frame_engine_sets.py: coco.txt, hires.txt and p.txt
#   --out FILE         one JSON line per row (op-fe-loadgen's rows), appended; the hold's
#                      evidence goes beside it as FILE.env.json, parity outputs in FILE.d/
#   --gpu-index N      the card's index in PCI bus order (nvidia-smi), the device the host
#                      processes see
#   --gpu-bus-id BUS   the bus id that card must have: nothing runs when it does not
#   --counts DIR       crop counts a Python parity run recorded (default FILE.d/counts)
#   --capture-dir DIR  the host directory mounted at /capture of the model server (s0, h1)
#   --sets "a b"       lists to run (default: coco hires p)
#   --runs N           runs per arm (default 2), alternating python native / native python
#   --warmup-s, --window-s   the loop's warm-up and window (default 5 and 20)
#   --load-max N       the 1-minute load must be under it before every row (default 16)
#   --python PY, --loadgen BIN, --*-url, --*-container   the test bed (compose.n0.yml's names)
#   --continue         keep going after a row with failures (default: stop the hold)
#   --dry-run          print the commands, run nothing
#
# Every command is `timeout --kill-after=... SECONDS command`, innermost; the whole hold runs
# under `flock -o` (the lock fd is not inherited by anything the hold starts). Both
# CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES are set for every host process.
set -euo pipefail

SCRIPT=$(readlink -f "${BASH_SOURCE[0]}")
REPO=$(cd "$(dirname "$SCRIPT")/../.." && pwd)
ORIG_ARGS=("$@")
# shellcheck source=scripts/bench/frame_engine_bed.sh
source "$(dirname "$SCRIPT")/frame_engine_bed.sh"

ROWS=""
LISTS=""
OUT=""
COUNTS=""
CAPTURE_DIR=""
SETS="coco hires p"
RUNS=2
WARMUP_S=5
WINDOW_S=20
SETUP_S=120
IDLE_S=65
CONTINUE=0
FAILED=0

while (($#)); do
  bed_option "$@"
  if ((BED_SHIFT)); then
    shift "$BED_SHIFT"
    continue
  fi
  case "$1" in
    --rows) ROWS=$2; shift 2 ;;
    --lists) LISTS=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --counts) COUNTS=$2; shift 2 ;;
    --capture-dir) CAPTURE_DIR=$2; shift 2 ;;
    --sets) SETS=$2; shift 2 ;;
    --runs) RUNS=$2; shift 2 ;;
    --warmup-s) WARMUP_S=$2; shift 2 ;;
    --window-s) WINDOW_S=$2; shift 2 ;;
    --idle-s) IDLE_S=$2; shift 2 ;;
    --continue) CONTINUE=1; shift ;;
    -h | --help) sed -n '2,/^set -euo/p' "$SCRIPT" | sed '$d'; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

bed_require HOLD_FILE ROWS LISTS OUT GPU_INDEX GPU_BUS_ID
case "$ROWS" in s0 | h1 | h2 | h3) ;; *) die "--rows is s0, h1, h2 or h3" ;; esac
OUT_DIR="$OUT.d"
COUNTS=${COUNTS:-$OUT_DIR/counts}
PLANS="$OUT_DIR/plans"
ROW_TIMEOUT=$((${WARMUP_S%.*} + ${WINDOW_S%.*} + SETUP_S))
bed_device_env

arm_order() { # alternate: odd runs python first, even runs native first (A B, B A, A B ...)
  if (($1 % 2)); then printf 'python native'; else printf 'native python'; fi
}

row_command() { # run one op-fe-loadgen row; a row with failures stops the hold unless --continue
  local status=0
  run "$ROW_TIMEOUT" "$LOADGEN" "$@" || status=$?
  if ((status == 3)); then
    FAILED=1
    ((CONTINUE)) || die "a row had failed photos (see $OUT): the hold stops"
  elif ((status != 0)); then
    die "op-fe-loadgen exited $status"
  fi
}

engine_row() { # ARM SET PLAN MODELS IN_FLIGHT RUN
  local arm=$1 set=$2 plan=$3 models=$4 n=$5 run_no=$6
  load_gate
  row_command engine --url "$(arm_url "$arm")" --list "$LISTS/$set.txt" \
    --plan-file "$PLANS/${plan}_${models}.json" --in-flight "$n" \
    --warmup-s "$WARMUP_S" --window-s "$WINDOW_S" \
    --label "$ROWS:$arm:$set:$plan:$models:$n:r$run_no" --out "$OUT" \
    --cpu-stat "$(cgroup_stat "$(arm_container "$arm")")" \
    --stats-url "$SERVER_URL" --gpu-bus-id "$GPU_BUS_ID"
}

counts_file() { # SET MODELS: the crop counts a z row replays for the ingest plan
  if [[ "$2" == real ]]; then printf '%s/%s.json' "$COUNTS" "$1"; else printf '%s/%s_%s.json' "$COUNTS" "$1" "$2"; fi
}

z_row() { # SET PLAN MODELS IN_FLIGHT
  local set=$1 plan=$2 models=$3 n=$4 counts=()
  [[ "$plan" == ingest ]] && counts=(--crop-counts "$(counts_file "$set" "$models")")
  load_gate
  row_command z --list "$LISTS/$set.txt" --plan-file "$PLANS/${plan}_${models}.json" \
    --in-flight "$n" --warmup-s "$WARMUP_S" --window-s "$WINDOW_S" \
    --label "$ROWS:z:$set:$plan:$models:$n" --out "$OUT" \
    --cpu-stat "$(cgroup_stat "$SERVER_CONTAINER")" --stats-url "$SERVER_URL" \
    --gpu-bus-id "$GPU_BUS_ID" --cuda-device 0 --server-device-id 0 "${counts[@]}"
}

parity() { # the parity tool, under timeout
  local seconds=$1
  shift
  run "$seconds" "$PY" "$REPO/scripts/bench/gpu_prep_parity.py" "$@"
}

parity_to() { # FILE SECONDS ARGS...: the parity tool with its output kept in a file
  local file=$1 seconds=$2
  shift 2
  run_to "$file" "$seconds" "$PY" "$REPO/scripts/bench/gpu_prep_parity.py" "$@"
}

make_plans() { # the plan texts, from the API's builder through the parity tool: one copy
  local plan models
  for plan in frame ingest; do
    for models in real null capture; do
      parity "$SETUP_S" plan --plan "$plan" --models "$models" --out "$PLANS/${plan}_${models}.json"
    done
  done
}

make_null_counts() { # every photo of a list gets the stand-in detector's fixed detections
  local set n crops
  if ((DRY_RUN)); then
    for set in $SETS; do show COUNTS "$(counts_file "$set" null)" "detections-per-photo"; done
    return 0
  fi
  crops=$("$PY" "$REPO/scripts/bench/frame_engine_stand_ins.py" detections)
  for set in $SETS; do
    n=$(grep -cv -e '^#' -e '^$' "$LISTS/$set.txt")
    jq -n --arg sha "$(sha256sum "$LISTS/$set.txt" | cut -d' ' -f1)" --argjson n "$n" --argjson c "$crops" \
      '{list_sha256: $sha, crops: [range(0; $n) | $c]}' >"$(counts_file "$set" null)"
  done
}

ensure_real_counts() { # the Python engine's crop counts for each list, recorded once
  local set
  for set in $SETS; do
    if [[ ! -s "$(counts_file "$set" real)" ]] || ((DRY_RUN)); then
      parity 3600 capture --url "$PYTHON_URL" --images "$LISTS/$set.txt" --plan ingest --models real \
        --concurrency 8 --out "$OUT_DIR/counts_$set.npz" --crop-counts-out "$(counts_file "$set" real)"
    fi
  done
}

write_evidence() { # what the paper's tables name: commit, image ids, plan hashes, the host
  local file="$OUT.env.json" c images="{}"
  if ((DRY_RUN)); then
    show EVIDENCE "$file"
    return 0
  fi
  for c in "$SERVER_CONTAINER" "$NATIVE_CONTAINER" "$PYTHON_CONTAINER"; do
    images=$(jq -n --argjson acc "$images" --arg c "$c" \
      --arg id "$(docker inspect -f '{{.Image}}' "$c")" \
      --arg image "$(docker inspect -f '{{.Config.Image}}' "$c")" \
      --arg started "$(docker inspect -f '{{.State.StartedAt}}' "$c")" \
      '$acc + {($c): {image_id: $id, image: $image, started_at: $started}}')
  done
  local plans
  plans=$(timeout --kill-after="${KILL_AFTER_S}s" 60 docker exec "$SERVER_CONTAINER" \
    sh -c 'sha256sum /models/*/1/model.plan /stand_in/*/1/model.plan 2>/dev/null' || true)
  jq -n --arg rows "$ROWS" --arg commit "$(git -C "$REPO" rev-parse HEAD)" \
    --arg when "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --arg host "$(hostname)" \
    --arg load "$(load1)" --arg bus "$GPU_BUS_ID" --arg index "$GPU_INDEX" \
    --arg driver "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i "$GPU_INDEX")" \
    --arg plans "$plans" --arg argv "${ORIG_ARGS[*]}" --argjson containers "$images" \
    '{hold: $rows, commit: $commit, started_at: $when, host: $host, load1_start: ($load | tonumber),
      gpu_bus_id: $bus, gpu_index: $index, driver: $driver, command: $argv, containers: $containers,
      plan_sha256: ($plans | split("\n") | map(select(length > 0)))}' >"$file"
}

preflight() {
  ((DRY_RUN)) || mkdir -p "$OUT_DIR" "$COUNTS" 2>/dev/null || true
  guard_device
  load_gate
  local arm
  for arm in native python; do
    run 30 "$LOADGEN" ready --url "$(arm_url "$arm")" || die "the $arm engine is not ready"
  done
  write_evidence
  make_plans
}

rows_h2() {
  local set plan n run_no arm
  make_null_counts
  ensure_real_counts
  for set in $SETS; do
    for plan in frame ingest; do
      for n in 1 8 32 64 128; do
        for ((run_no = 1; run_no <= RUNS; run_no++)); do
          for arm in $(arm_order "$run_no"); do
            engine_row "$arm" "$set" "$plan" real "$n" "$run_no"
          done
        done
      done
      for n in 32 64; do z_row "$set" "$plan" real "$n"; done
    done
  done
  # G1's d: the card's busy share and the rate of the native engine with the models taken away
  # (null models, plan F): decode, orientation and the resamples are all that is left
  for set in $SETS; do
    engine_row native "$set" frame null 32 1
  done
}

rows_h3() {
  local set plan n run_no arm
  make_null_counts
  for set in coco hires; do
    for plan in frame ingest; do
      for n in 32 64 128; do
        for ((run_no = 1; run_no <= RUNS; run_no++)); do
          for arm in $(arm_order "$run_no"); do
            engine_row "$arm" "$set" "$plan" null "$n" "$run_no"
          done
        done
        z_row "$set" "$plan" null "$n"
      done
    done
  done
}

set_arm() { # name the engine under test: the capture models write into that directory
  if ((DRY_RUN)); then
    show ARM "$CAPTURE_DIR/ARM" "$1"
  else
    printf '%s' "$1" >"$CAPTURE_DIR/ARM"
  fi
}

capture_for_arm() { # ARM SET: 50 frames through the capture models, into the arm's directory
  local arm=$1 set=$2
  set_arm "${arm}_$set"
  parity 600 capture --url "$(arm_url "$arm")" --images "$LISTS/$set.txt" --limit 50 --plan ingest \
    --models capture --concurrency 1 --out "$OUT_DIR/inputs_${arm}_$set.npz"
}

rows_h1() {
  [[ -n "$CAPTURE_DIR" ]] || die "--capture-dir is required for the parity hold"
  local set limit plan load status dir="$OUT_DIR/h1"
  ((DRY_RUN)) || mkdir -p "$dir" 2>/dev/null || true
  for set in $SETS; do
    case "$set" in coco) limit=200 ;; hires) limit=100 ;; *) limit=0 ;; esac
    for plan in frame ingest; do
      for load in 1 32; do
        local base="$dir/${set}_${plan}_$load"
        for arm in python python2 native; do
          parity 3600 capture --url "$(arm_url "${arm%2}")" --images "$LISTS/$set.txt" --limit "$limit" \
            --plan "$plan" --models real --concurrency "$load" --out "${base}_$arm.npz"
        done
        status=0
        parity_to "${base}_verdict.txt" 600 compare "${base}_native.npz" "${base}_python.npz" \
          --floor-a "${base}_python.npz" --floor-b "${base}_python2.npz" --in-flight "$load" || status=$?
        ((status == 0)) || FAILED=1
      done
    done
    capture_for_arm python "$set"
    capture_for_arm native "$set"
    status=0
    parity_to "$dir/${set}_inputs.json" 600 compare-inputs "$CAPTURE_DIR/python_$set" "$CAPTURE_DIR/native_$set" ||
      status=$?
    ((status == 0)) || FAILED=1
  done
}

rows_s0() { # 90 s: region registered, plan F on 3 photos, equal canvases, 65 s idle then a request
  [[ -n "$CAPTURE_DIR" ]] || die "--capture-dir is required for the smoke hold"
  local set=${SETS%% *} status=0 arm
  run 30 "$LOADGEN" regions --url "$SERVER_URL" --prefix op-fe- || die "no engine region is registered"
  for arm in python native; do
    set_arm "${arm}_s0"
    parity 120 capture --url "$(arm_url "$arm")" --images "$LISTS/$set.txt" --limit 3 --plan frame \
      --models capture --concurrency 1 --out "$OUT_DIR/s0_$arm.npz"
  done
  parity_to "$OUT_DIR/s0_inputs.json" 120 compare-inputs "$CAPTURE_DIR/python_s0" "$CAPTURE_DIR/native_s0" || status=$?
  ((status == 0)) || { FAILED=1; printf 'S0: the capture models saw different canvases (%s)\n' "$OUT_DIR/s0_inputs.json" >&2; }
  # an idle channel pinged every 30 s must not be sent away: wait, then one request
  if ((DRY_RUN)); then show IDLE "$IDLE_S"; else sleep "$IDLE_S"; fi
  parity 120 capture --url "$NATIVE_URL" --images "$LISTS/$set.txt" --limit 1 --plan frame \
    --models real --concurrency 1 --out "$OUT_DIR/s0_after_idle.npz"
  if ((!DRY_RUN)); then
    local goaway
    goaway=$(timeout --kill-after="${KILL_AFTER_S}s" 60 docker logs --since "$((IDLE_S + 60))s" "$SERVER_CONTAINER" 2>&1 |
      grep -ci -e 'too_many_pings' -e 'goaway' || true)
    printf 'S0: GOAWAY lines in the model server log: %s\n' "$goaway"
    ((goaway == 0)) || FAILED=1
  fi
}

acquire_hold
preflight
"rows_$ROWS"
if ((FAILED)); then
  printf 'frame_engine_hold: %s finished with failures (see %s)\n' "$ROWS" "$OUT_DIR" >&2
  exit 1
fi
printf 'frame_engine_hold: %s done: %s\n' "$ROWS" "$OUT"
