#!/usr/bin/env bash
# The local gate: the two sweep tracks (py, fe) in one command.
#
#   scripts/dev/gate.sh [py|fe|all] [OUT_DIR]      (make gate runs "all")
#
# Installs run first and alone; then the tracks run (in parallel for "all").
# Each step writes <out>/<step>.log and one line to <out>/summary_<track>.txt:
#   == <step> rc=<code> <start> <end> load=<1-min load>
# Not `set -e`: a failed step is recorded and its track goes on; the exit code
# is non-zero when any step failed.
#
# Environment:
#   GATE_OUT        output dir (default artifacts_local/gate/<utc>/, gitignored)
#   GATE_XDIST      pytest-xdist workers (default 8)
#   GATE_NICE       1 (default) runs every step under `nice -n 10 ionice -c3`
#                   and holds a new step while the 1-min load is above
#                   GATE_MAX_LOAD (default 35); set 0 on a quiet host
#   GATE_HOLD_SECS  seconds between load polls while holding (default 60)
#   GATE_ROOT, GATE_VENV, GATE_LOADAVG  override the repo root, the venv and
#                   the load source (/proc/loadavg); for tests
set -u

TRACK=${1:-all}
case "$TRACK" in py | fe | all) ;; *)
  echo "usage: $0 [py|fe|all] [out-dir]" >&2
  exit 2
  ;;
esac

ROOT=${GATE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
VENV=${GATE_VENV:-$ROOT/.venv}
OUT=${2:-${GATE_OUT:-$ROOT/artifacts_local/gate/$(date -u +%Y%m%dT%H%M%SZ)}}
LOADAVG=${GATE_LOADAVG:-/proc/loadavg}
NICE=${GATE_NICE:-1}
MAX_LOAD=${GATE_MAX_LOAD:-35}
HOLD_SECS=${GATE_HOLD_SECS:-60}
XDIST=${GATE_XDIST:-8}

mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)

load1() { cut -d' ' -f1 "$LOADAVG"; }

# step <track> <name> <dir> <cmd...>
step() {
  local track=$1 name=$2 dir=$3
  shift 3
  local sum="$OUT/summary_${track}.txt" load low=()
  load=$(load1)
  if [[ "$NICE" == 1 ]]; then
    low=(nice -n 10 ionice -c3)
    while awk -v l="$load" -v m="$MAX_LOAD" 'BEGIN{exit !(l > m)}'; do
      echo "   hold $name load=$load > $MAX_LOAD" >>"$sum"
      sleep "$HOLD_SECS"
      load=$(load1)
    done
  fi
  local start end rc
  start=$(date +%T)
  (cd "$dir" && "${low[@]}" "$@") >"$OUT/$name.log" 2>&1
  rc=$?
  end=$(date +%T)
  echo "== $name rc=$rc $start $end load=$load" >>"$sum"
}

# The -m expression is whatever pyproject addopts carries, so a later edit
# (for example adding "and not db") is followed without touching this script.
marker_expr() {
  local expr
  expr=$(awk '
    /^addopts[[:space:]]*=/ { on = 1 }
    on && want && match($0, /"[^"]*"/) { print substr($0, RSTART + 1, RLENGTH - 2); exit }
    on && /"-m"/ {
      line = $0; sub(/.*"-m"[[:space:]]*,?/, "", line)
      if (match(line, /"[^"]*"/)) { print substr(line, RSTART + 1, RLENGTH - 2); exit }
      want = 1
    }
    on && /^\]/ { exit }
  ' "$ROOT/pyproject.toml")
  echo "${expr:-not live and not families}"
}

install_track() {
  if [[ "$TRACK" != fe ]]; then
    step install install_pip "$ROOT" "$VENV/bin/pip" install -r requirements-test.txt
    step install install_pip_check "$ROOT" "$VENV/bin/pip" check
  fi
  if [[ "$TRACK" != py ]]; then
    step install install_npm_frontend "$ROOT/frontend" npm ci
    step install install_npm_docs_site "$ROOT/docs-site" npm ci
  fi
}

py_track() {
  step py precommit "$ROOT" "$VENV/bin/pre-commit" run --all-files
  step py pytest "$ROOT" "$VENV/bin/python" -m pytest tests/ -q --no-cov \
    -m "$(marker_expr)" -n "$XDIST"
  step py contracts "$ROOT" make contracts-check
  step py docs_check "$ROOT" "$VENV/bin/python" scripts/docs/check_docs_vs_code.py
}

fe_track() {
  local fe="$ROOT/frontend"
  step fe fe_check "$fe" npm run check
  step fe fe_lint "$fe" npm run lint
  step fe fe_knip "$fe" npx knip
  step fe fe_test "$fe" npm test
  step fe fe_build "$fe" npm run build
  step fe fe_e2e "$fe" npm run test:e2e
  step fe site_build "$ROOT/docs-site" npm run build
}

install_track
case "$TRACK" in
  py) py_track ;;
  fe) fe_track ;;
  all)
    py_track &
    fe_track &
    wait
    ;;
esac

cat "$OUT"/summary_*.txt
if cat "$OUT"/summary_*.txt | grep -E '^== ' | grep -qvE ' rc=0 '; then
  echo "GATE FAILED (logs in $OUT)"
  exit 1
fi
echo "GATE PASSED (logs in $OUT)"
