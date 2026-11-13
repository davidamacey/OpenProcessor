#!/usr/bin/env bash
# Opt-in: run tests/live as N independent harness stacks in parallel (default path: one stack,
# see docker/test/README.md). Each shard i gets its own compose project op-live-s<i>, host ports
# 147<i>x and data directory, starts its stack, runs its pytest-split share, and every stack is
# torn down on exit, also on failure.
#
#   scripts/dev/live_shards.sh N [pytest args...]
#
# Needs the harness images built under VERIFY_CONTAINER_PREFIX (default op-verify), shared by all
# shards, or OP_TEST_IMAGE_TAG for the built overlay. Logs: artifacts_local/live_shards/<i>.log.
# --splitting-algorithm least_duration puts single tests on shards: a module-scoped stack change
# (test_one_vs_n, test_sse_across_replicas) runs once per shard that holds one of its tests.
set -u

N=${1:?usage: $0 N [pytest args...]}
shift
((N >= 1 && N <= 9)) || { echo "N must be 1..9 (ports 14700-14799)" >&2; exit 2; }
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUT=$ROOT/artifacts_local/live_shards
BOUND=${LIVE_SHARD_BOUND_S:-3600}
FILES=(-f docker/test/compose.yml)
[[ -n "${OP_TEST_IMAGE_TAG:-}" ]] && FILES+=(-f docker/test/compose.built.yml)
mkdir -p "$OUT"
cd "$ROOT" || exit 2

# shellcheck disable=SC2317  # called by the EXIT trap
teardown() {
  for ((i = 0; i < N; i++)); do
    timeout 300 docker compose -p "op-live-s$i" "${FILES[@]}" down -v --remove-orphans >/dev/null 2>&1
  done
}
trap teardown EXIT

# shellcheck disable=SC2317
shard_env() { # prints the environment of shard $1
  local i=$1 base=$((14700 + $1 * 10))
  echo "VERIFY_COMPOSE_PROJECT=op-live-s$i VERIFY_DATA_DIR=$OUT/data$i" \
    "VERIFY_API_PORT=$((base + 1)) VERIFY_OPENSEARCH_PORT=$((base + 2)) VERIFY_FAKE_VLM_PORT=$((base + 4))" \
    "VERIFY_REPLICA1_PORT=$((base + 5)) VERIFY_REPLICA2_PORT=$((base + 6))" \
    "VERIFY_API_URL=http://localhost:$((base + 1)) VERIFY_OPENSEARCH_URL=http://localhost:$((base + 2))" \
    "VERIFY_FAKE_VLM_URL=http://localhost:$((base + 4))" \
    "VERIFY_REPLICA1_URL=http://localhost:$((base + 5)) VERIFY_REPLICA2_URL=http://localhost:$((base + 6))"
}

run_shard() { # runs in a subshell, so the exports stay with the shard
  local i=$1
  shift
  # shellcheck disable=SC2046
  export $(shard_env "$i")
  timeout 900 docker compose -p "op-live-s$i" "${FILES[@]}" up -d --wait --wait-timeout 600 --no-build || return 1
  timeout "$BOUND" .venv/bin/python -m pytest tests/live -m live -p no:randomly --no-cov -q \
    --splits "$N" --group "$((i + 1))" --splitting-algorithm least_duration \
    --durations-path tests/live/.live_durations "$@"
}

pids=()
for ((i = 0; i < N; i++)); do
  (run_shard "$i" "$@") >"$OUT/$i.log" 2>&1 &
  pids+=($!)
done
rc=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || { echo "shard $i FAILED (see $OUT/$i.log)"; rc=1; }
done
exit $rc
