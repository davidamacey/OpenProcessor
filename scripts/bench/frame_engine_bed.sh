#!/bin/bash
# The test bed's shared parts, sourced by frame_engine_hold.sh and frame_engine_faults.sh: the
# lock, the load gate, the device guard, the bounded command, and the options both scripts take.
# One copy of each rule (plan 17 section 5, N0 "The holds"). Not run by itself.

# defaults a caller may override with its options
: "${REPO:=}"
: "${SCRIPT:=}"
: "${HOLD_FILE:=}"
: "${GPU_INDEX:=}"
: "${GPU_BUS_ID:=}"
: "${PY:=${REPO:-.}/.venv/bin/python}"
: "${LOADGEN:=${CARGO_TARGET_DIR:-${REPO:-.}/native/target}/release/op-fe-loadgen}"
: "${SERVER_URL:=127.0.0.1:4620}"
: "${NATIVE_URL:=127.0.0.1:4621}"
: "${PYTHON_URL:=127.0.0.1:4622}"
: "${SERVER_CONTAINER:=op-n0-model-server}"
: "${NATIVE_CONTAINER:=op-n0-engine-native}"
: "${PYTHON_CONTAINER:=op-n0-engine-python}"
: "${LOAD_MAX:=16}"
: "${LOAD_WAIT_S:=120}"
: "${LOCK_WAIT_S:=60}"
: "${KILL_AFTER_S:=15}"
: "${DRY_RUN:=0}"
: "${LOCKED:=0}"

die() {
  printf '%s: %s\n' "$(basename "$SCRIPT")" "$*" >&2
  exit 1
}

# bed_option "$@": when $1 is an option both scripts take, apply it and set BED_SHIFT to the number
# of words it used; otherwise BED_SHIFT is 0.
BED_SHIFT=0
# shellcheck disable=SC2034  # BED_SHIFT is read by the sourcing script
bed_option() {
  BED_SHIFT=2
  case "$1" in
    --hold-file) HOLD_FILE=$2 ;;
    --gpu-index) GPU_INDEX=$2 ;;
    --gpu-bus-id) GPU_BUS_ID=$2 ;;
    --python) PY=$2 ;;
    --loadgen) LOADGEN=$2 ;;
    --server-url) SERVER_URL=$2 ;;
    --native-url) NATIVE_URL=$2 ;;
    --python-url) PYTHON_URL=$2 ;;
    --server-container) SERVER_CONTAINER=$2 ;;
    --native-container) NATIVE_CONTAINER=$2 ;;
    --python-container) PYTHON_CONTAINER=$2 ;;
    --load-max) LOAD_MAX=$2 ;;
    --load-wait-s) LOAD_WAIT_S=$2 ;;
    --lock-wait-s) LOCK_WAIT_S=$2 ;;
    --dry-run) DRY_RUN=1; BED_SHIFT=1 ;;
    --locked) LOCKED=1; BED_SHIFT=1 ;;
    *) BED_SHIFT=0 ;;
  esac
}

bed_require() { # every name must be set: the options a script cannot run without
  local need
  for need in "$@"; do
    [[ -n "${!need}" ]] || die "--${need//_/-} is required"
  done
}

# Both halves of the device guard: PCI order makes the index the one nvidia-smi shows, and the
# visible set hides every other card from any CUDA process a script starts.
bed_device_env() {
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
  if ((DRY_RUN)); then
    printf 'ENV: CUDA_DEVICE_ORDER=%s CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_DEVICE_ORDER" "$CUDA_VISIBLE_DEVICES"
  fi
}

show() { # print a command, shell-quoted, behind a tag
  local tag=$1
  shift
  printf '%s:' "$tag"
  printf ' %q' "$@"
  printf '\n'
}

run() { # run SECONDS COMMAND...: the command under timeout, which is the innermost wrapper
  local seconds=$1
  shift
  if ((DRY_RUN)); then
    show RUN timeout "--kill-after=${KILL_AFTER_S}s" "$seconds" "$@"
    return 0
  fi
  timeout "--kill-after=${KILL_AFTER_S}s" "$seconds" "$@"
}

run_to() { # run_to FILE SECONDS COMMAND...: like run, with the command's output going to FILE
  local file=$1 seconds=$2
  shift 2
  if ((DRY_RUN)); then
    show RUN timeout "--kill-after=${KILL_AFTER_S}s" "$seconds" "$@" ">$file"
    return 0
  fi
  timeout "--kill-after=${KILL_AFTER_S}s" "$seconds" "$@" >"$file"
}

acquire_hold() { # the hold file for the whole run: re-execute under flock -o, which closes the lock fd for the command
  ((LOCKED)) && return 0
  local cmd=(flock -o -w "$LOCK_WAIT_S" -E 75 "$HOLD_FILE" "$SCRIPT" --locked "${ORIG_ARGS[@]}")
  if ((DRY_RUN)); then
    show LOCK "${cmd[@]}"
    return 0
  fi
  exec "${cmd[@]}"
}

load1() {
  local file=${FE_LOADAVG:-/proc/loadavg} first
  read -r first _ <"$file"
  printf '%s' "$first"
}

load_gate() { # the 1-minute load under the limit, waiting at most LOAD_WAIT_S for it to fall
  local waited=0 now
  while :; do
    now=$(load1)
    if awk -v l="$now" -v m="$LOAD_MAX" 'BEGIN { exit !(l < m) }'; then
      return 0
    fi
    ((waited >= LOAD_WAIT_S)) && die "the 1-minute load $now is not under $LOAD_MAX after ${waited}s"
    sleep 10
    waited=$((waited + 10))
  done
}

guard_device() { # the card at the index has the expected bus id; no process outside the bed is on it
  local idx bus seen="" pid
  while IFS=', ' read -r idx bus; do
    [[ "$idx" == "$GPU_INDEX" ]] && seen=$bus
  done < <(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader)
  [[ -n "$seen" ]] || die "nvidia-smi shows no card at index $GPU_INDEX"
  [[ "${seen^^}" == "${GPU_BUS_ID^^}" ]] ||
    die "card $GPU_INDEX is at bus id $seen, not the expected $GPU_BUS_ID: nothing runs"
  local ours=() c
  if ((!DRY_RUN)); then
    for c in "$SERVER_CONTAINER" "$NATIVE_CONTAINER" "$PYTHON_CONTAINER"; do
      ours+=("$(docker inspect -f '{{.Id}}' "$c")")
    done
  fi
  local foreign=() mine id
  while IFS=', ' read -r pid bus; do
    [[ -n "$pid" && "${bus^^}" == "${GPU_BUS_ID^^}" ]] || continue
    mine=0
    for id in "${ours[@]}"; do
      if grep -q "$id" "/proc/$pid/cgroup" 2>/dev/null; then mine=1; fi
    done
    ((mine)) || foreign+=("$pid")
  done < <(nvidia-smi --query-compute-apps=pid,gpu_bus_id --format=csv,noheader)
  ((${#foreign[@]} == 0)) || die "processes outside this test bed are on card $GPU_INDEX: ${foreign[*]}"
}

cgroup_stat() { # the cpu.stat file of a container's cgroup (cgroup v2, systemd or cgroupfs driver)
  local name=$1 id path
  if ((DRY_RUN)); then
    printf '/sys/fs/cgroup/system.slice/docker-%s.scope/cpu.stat' "$name"
    return 0
  fi
  id=$(docker inspect -f '{{.Id}}' "$name")
  for path in "/sys/fs/cgroup/system.slice/docker-$id.scope/cpu.stat" "/sys/fs/cgroup/docker/$id/cpu.stat"; do
    if [[ -r "$path" ]]; then
      printf '%s' "$path"
      return 0
    fi
  done
  die "no cgroup v2 cpu.stat for container $name"
}

engine_regions() { # INSTANCE: how many regions of that engine instance the model server holds
  # a region is named op-fe-<card>-<instance>-<process>; each engine of the bed has its own instance
  "$LOADGEN" regions --url "$SERVER_URL" --prefix op-fe- 2>/dev/null | grep -c -e "-$1-" || true
}

arm_url() {
  case "$1" in
    native) printf '%s' "$NATIVE_URL" ;;
    python) printf '%s' "$PYTHON_URL" ;;
    *) die "unknown arm $1" ;;
  esac
}

arm_container() {
  case "$1" in
    native) printf '%s' "$NATIVE_CONTAINER" ;;
    python) printf '%s' "$PYTHON_CONTAINER" ;;
    *) die "unknown arm $1" ;;
  esac
}
