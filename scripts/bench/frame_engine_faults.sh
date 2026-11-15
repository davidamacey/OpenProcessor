#!/bin/bash
# The fault holds of the native-engine comparison (plan 17 bar G8, N0 part; hold H6): kill the
# engine under load and check the model server after every kill, or restart the model server under
# the engine and check that the engine registers its region again without being restarted. Run by
# the coordinator on the card it names, never by a lane.
#
#   scripts/bench/frame_engine_faults.sh --mode kill|server-restart --hold-file FILE \
#       --list FILE --plan-file FILE --out FILE.jsonl --gpu-index N --gpu-bus-id BUS [options]
#
#   --mode kill            --kills N (default 10) times: start a closed loop of --in-flight (32)
#                          photos against the engine, `docker kill -s KILL` the engine container
#                          by its exact name, start it again, and check the model server
#   --mode server-restart  `docker restart` the model server with the engine up; the engine must
#                          become ready again and hold a region without a restart of its own
#   --detector MODEL       a real model of the server to ask after (default
#                          yolov11_small_trt_end2end)
#   --mem-tolerance-mib N  how far above its baseline the card's memory may be once the engine is
#                          back (default 64)
#   --ready-wait-s N       the longest wait for an engine or the server to answer (default 90)
#
# After each kill, all of: the model server container is the same one that was running (same start
# time, restart count unchanged); its log holds no CUDA error since the kill; the engine answers
# ready and holds exactly one region of its prefix (the stale one was purged); the card's memory is
# back within the tolerance of the baseline taken with the engine idle before the first kill; a
# photo goes through the engine. One JSON line per kill (or per restart) is appended to --out.
# Every command runs under timeout --kill-after; the hold file is taken with flock -o.
set -euo pipefail

SCRIPT=$(readlink -f "${BASH_SOURCE[0]}")
# shellcheck disable=SC2034  # the sourced bed reads ORIG_ARGS (to re-execute under the lock)
ORIG_ARGS=("$@")
# shellcheck source=scripts/bench/frame_engine_bed.sh
source "$(dirname "$SCRIPT")/frame_engine_bed.sh"

MODE=""
LIST=""
PLAN_FILE=""
OUT=""
KILLS=10
IN_FLIGHT=32
DETECTOR="yolov11_small_trt_end2end"
MEM_TOLERANCE_MIB=64
READY_WAIT_S=90
SETTLE_S=8
PREFIX="op-fe-"
FAILED=0
LOOP_PID=""
BASELINE_MIB=0

while (($#)); do
  bed_option "$@"
  if ((BED_SHIFT)); then
    shift "$BED_SHIFT"
    continue
  fi
  case "$1" in
    --mode) MODE=$2; shift 2 ;;
    --list) LIST=$2; shift 2 ;;
    --plan-file) PLAN_FILE=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --kills) KILLS=$2; shift 2 ;;
    --in-flight) IN_FLIGHT=$2; shift 2 ;;
    --detector) DETECTOR=$2; shift 2 ;;
    --mem-tolerance-mib) MEM_TOLERANCE_MIB=$2; shift 2 ;;
    --ready-wait-s) READY_WAIT_S=$2; shift 2 ;;
    --settle-s) SETTLE_S=$2; shift 2 ;;
    -h | --help) sed -n '2,/^set -euo/p' "$SCRIPT" | sed '$d'; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

bed_require HOLD_FILE MODE LIST PLAN_FILE OUT GPU_INDEX GPU_BUS_ID
case "$MODE" in kill | server-restart) ;; *) die "--mode is kill or server-restart" ;; esac
bed_device_env

card_memory_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_INDEX" | tr -d ' '
}

container_state() { # NAME: the start time and restart count, to tell a restart from a survival
  docker inspect -f '{{.State.StartedAt}} {{.RestartCount}}' "$1"
}

wait_ready() { # URL MODEL: ready within READY_WAIT_S, by asking every 2 s
  local url=$1 model=$2 waited=0
  while ((waited < READY_WAIT_S)); do
    if run 10 "$LOADGEN" ready --url "$url" --model "$model" --timeout-s 5 >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
    waited=$((waited + 2))
  done
  return 1
}

one_photo() { # a photo goes through the engine: one in flight, failed 0 (exit 3 otherwise)
  run 90 "$LOADGEN" engine --url "$NATIVE_URL" --list "$LIST" --plan-file "$PLAN_FILE" --in-flight 1 \
    --warmup-s 1 --window-s 3 --no-stats --no-gpu --label "faults:next" --out "$OUT.next.jsonl"
}

region_count() {
  "$LOADGEN" regions --url "$SERVER_URL" --prefix "$PREFIX" 2>/dev/null | grep -c "^$PREFIX" || true
}

record() { # a JSON line for one kill or restart
  jq -nc --argjson n "$1" --arg what "$2" --argjson checks "$3" \
    '{faults: $what, n: $n, checks: $checks, passed: ($checks | all(.[]; .))}' >>"$OUT"
}

start_loop() { # N: the closed loop of photos in flight, in the background but bounded by timeout
  local sink=/dev/null
  ((DRY_RUN)) && sink=/dev/stdout
  run 120 "$LOADGEN" engine --url "$NATIVE_URL" --list "$LIST" --plan-file "$PLAN_FILE" \
    --in-flight "$IN_FLIGHT" --warmup-s 2 --window-s "$((SETTLE_S + 20))" --no-stats --no-gpu \
    --allow-failures --label "faults:kill$1" --out "$OUT.loops.jsonl" >"$sink" 2>>"$OUT.loops.log" &
  LOOP_PID=$!
}

kill_once() { # N
  local n=$1 before after since mem regions err ready=0 same=0 cuda=0 photo=0 within=0 one=0
  if ((DRY_RUN)); then
    start_loop "$n"
    wait "$LOOP_PID" || true
    run 30 docker kill -s KILL "$NATIVE_CONTAINER"
    run 60 docker start "$NATIVE_CONTAINER"
    return 0
  fi
  before=$(container_state "$SERVER_CONTAINER")
  since=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  start_loop "$n"
  sleep "$((SETTLE_S + 2))"
  run 30 docker kill -s KILL "$NATIVE_CONTAINER" >/dev/null
  wait "$LOOP_PID" || true
  run 60 docker start "$NATIVE_CONTAINER" >/dev/null
  wait_ready "$NATIVE_URL" frame_engine && ready=1
  after=$(container_state "$SERVER_CONTAINER")
  [[ "$before" == "$after" ]] && same=1
  err=$(timeout --kill-after="${KILL_AFTER_S}s" 60 docker logs --since "$since" "$SERVER_CONTAINER" 2>&1 |
    grep -ci -e 'cuda error' -e 'CUDA_ERROR' -e 'illegal memory' || true)
  ((err == 0)) && cuda=1
  sleep 5
  mem=$(card_memory_mib)
  regions=$(region_count)
  one_photo >/dev/null 2>&1 && photo=1
  ((mem - BASELINE_MIB <= MEM_TOLERANCE_MIB)) && within=1
  ((regions == 1)) && one=1
  local checks
  checks=$(jq -nc --argjson same "$same" --argjson cuda "$cuda" --argjson ready "$ready" \
    --argjson within "$within" --argjson regions "$one" --argjson photo "$photo" \
    '{server_same_container: ($same == 1), no_cuda_error: ($cuda == 1), engine_ready: ($ready == 1),
      memory_back_to_baseline: ($within == 1), one_engine_region: ($regions == 1),
      next_photo_served: ($photo == 1)}')
  record "$n" kill "$checks"
  jq -e 'all(.[]; .)' <<<"$checks" >/dev/null || FAILED=1
  printf 'kill %s: baseline %s MiB, now %s MiB, regions %s, server log CUDA errors %s\n' \
    "$n" "$BASELINE_MIB" "$mem" "$regions" "$err" >&2
}

restart_once() {
  local engine_before engine_after ready=0 same=0 regions=0 photo=0
  if ((DRY_RUN)); then
    run 120 docker restart --time 30 "$SERVER_CONTAINER"
    return 0
  fi
  engine_before=$(container_state "$NATIVE_CONTAINER")
  run 120 docker restart --time 30 "$SERVER_CONTAINER" >/dev/null
  wait_ready "$SERVER_URL" "$DETECTOR" || die "the model server did not come back"
  wait_ready "$NATIVE_URL" frame_engine && ready=1
  engine_after=$(container_state "$NATIVE_CONTAINER")
  [[ "$engine_before" == "$engine_after" ]] && same=1
  (($(region_count) >= 1)) && regions=1
  one_photo >/dev/null 2>&1 && photo=1
  local checks
  checks=$(jq -nc --argjson same "$same" --argjson ready "$ready" --argjson regions "$regions" --argjson photo "$photo" \
    '{engine_not_restarted: ($same == 1), engine_ready: ($ready == 1), region_registered_again: ($regions == 1),
      next_photo_served: ($photo == 1)}')
  record 1 server-restart "$checks"
  jq -e 'all(.[]; .)' <<<"$checks" >/dev/null || FAILED=1
}

acquire_hold
guard_device
load_gate
mkdir -p "$(dirname "$OUT")"
if ((!DRY_RUN)); then
  wait_ready "$NATIVE_URL" frame_engine || die "the engine is not ready before the first kill"
  wait_ready "$SERVER_URL" "$DETECTOR" || die "the model server is not ready"
  BASELINE_MIB=$(card_memory_mib)
fi
case "$MODE" in
  kill)
    for ((k = 1; k <= KILLS; k++)); do
      load_gate
      kill_once "$k"
    done
    ;;
  server-restart) restart_once ;;
esac
if ((FAILED)); then
  printf 'frame_engine_faults: a check failed (see %s)\n' "$OUT" >&2
  exit 1
fi
printf 'frame_engine_faults: %s passed (%s)\n' "$MODE" "$OUT"
