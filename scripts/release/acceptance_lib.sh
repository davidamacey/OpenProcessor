#!/bin/bash
# =============================================================================
# acceptance_lib.sh - state, evidence, HTTP, polling and cleanup helpers
# =============================================================================
# Sourced by acceptance_run.sh, which owns the options, WORKDIR and the
# phase runner. Not runnable on its own.

# Variables are assigned by acceptance_run.sh before this file is sourced.
# shellcheck disable=SC2154

# ---------------------------------------------------------------------------
# helpers: state, evidence, HTTP, polling
# ---------------------------------------------------------------------------
st_set() { printf '%s' "$2" > "$WORKDIR/state/$1"; }
st_get() { [[ -f "$WORKDIR/state/$1" ]] && cat "$WORKDIR/state/$1" || true; }
st_add() { printf '%s\n' "$2" >> "$WORKDIR/state/$1"; }

EV_FILE=""
ev() { printf '%s\n' "$*" >> "$EV_FILE"; }
fail() { ev "FAIL: $*"; exit 1; }
skip() { ev "SKIP: $*"; exit 77; }

HTTP_CODE=000
BODY=""
# api METHOD PATH [extra curl args...]: sets HTTP_CODE and BODY (a file).
api() {
    local method="$1" path="$2"
    shift 2
    BODY="$WORKDIR/body.$RANDOM$RANDOM"
    HTTP_CODE="$(curl -sS -o "$BODY" -w '%{http_code}' --max-time "$HTTP_TIMEOUT" \
        -X "$method" "$@" "${BASE_URL}${path}" 2>>"$WORKDIR/curl.err")" || HTTP_CODE=000
}
api_json() { api "$1" "$2" -H 'Content-Type: application/json' --data "${3:-{\}}"; }
snippet() { head -c 300 "$BODY" 2>/dev/null | tr '\n' ' '; }
# expect CODE...: fail unless HTTP_CODE is one of the listed codes
expect() {
    local c
    for c in "$@"; do [[ "$HTTP_CODE" == "$c" ]] && return 0; done
    fail "HTTP ${HTTP_CODE} (wanted $*): $(snippet)"
}
jqb() { jq -r "$1" "$BODY" 2>/dev/null; }
# poll TIMEOUT_S CMD...: run CMD every POLL_S until it returns 0.
poll() {
    local timeout="$1" deadline
    shift
    deadline=$(( $(date +%s) + timeout ))
    while true; do
        "$@" && return 0
        (( $(date +%s) >= deadline )) && return 1
        sleep "$POLL_S"
    done
}

is_selected() {
    local n="$1"
    if [[ -n "$ONLY" ]]; then
        [[ ",${ONLY}," == *",${n},"* ]] || return 1
    fi
    return 0
}
is_skipped() { [[ ",${SKIP}," == *",$1,"* ]]; }

# ---------------------------------------------------------------------------
# curation-evaluator: on demand, never part of a plain `curation up`
# ---------------------------------------------------------------------------
evaluator_ids() {
    [[ -n "$PROJECT_NAME" ]] || return 0
    docker ps -q --filter "label=com.docker.compose.project=${PROJECT_NAME}" \
        --filter "label=com.docker.compose.service=curation-evaluator"
}
# start_evaluator: succeeds when the evaluator runs; remembers that this run
# started it so stop_evaluator (and cleanup) only stops what it started.
start_evaluator() {
    [[ -z "$(evaluator_ids)" ]] || return 0
    [[ -n "$PROJECT_NAME" && -n "$INSTALL_DIR" && -x "${INSTALL_DIR}/openprocessor" ]] || return 1
    "${INSTALL_DIR}/openprocessor" curation up curation-evaluator >> "$WORKDIR/evaluator.log" 2>&1 || return 1
    st_set evaluator_started 1
    [[ -n "$(evaluator_ids)" ]]
}
stop_evaluator() {
    [[ "$(st_get evaluator_started)" == 1 ]] || return 0
    local ids
    ids="$(evaluator_ids | tr '\n' ' ')"
    # shellcheck disable=SC2086
    [[ -z "$ids" ]] || docker stop $ids >> "$WORKDIR/evaluator.log" 2>&1 || return 1
    st_set evaluator_started 0
}

# ---------------------------------------------------------------------------
# cleanup: always delete the throwaway project and the models this run promoted
# ---------------------------------------------------------------------------
# cancel_project_jobs: cancel what the project reports as running (the dry-run
# delete lists it) so the delete is not refused with project_busy. A bake-off
# has no cancel route: it needs the evaluator to drain, so make sure one runs.
cancel_project_jobs() {
    api DELETE "${PROJECT_API}?confirm=${SLUG}&dry_run=true"
    [[ "$HTTP_CODE" == 200 ]] || return 0
    local kind id jobs
    jobs="$(jq -r '.running_jobs[]? | "\(.kind) \(.id)"' "$BODY" 2>/dev/null)"
    while read -r kind id; do
        [[ -n "$kind" ]] || continue
        case "$kind" in
            train) api POST "${PROJECT_API}/train/cancel/${id}" ;;
            autolabel) api POST "${PROJECT_API}/pipeline/auto_label/cancel" ;;
            bakeoff) start_evaluator || true ;;
        esac
        ev "cleanup: cancelled/drained ${kind} ${id}"
    done <<< "$jobs"
}

cleanup_resources() {
    [[ "$(st_get cleaned)" == 1 ]] && return 0
    local rc=0 m
    local models
    models="$(st_get promoted_models)"
    if [[ -n "$models" ]]; then
        while IFS= read -r m; do
            [[ -z "$m" ]] && continue
            api DELETE "${PROJECT_API}/models/${m}?force=true"
            case "$HTTP_CODE" in 200|202|404) ;; *) rc=1 ;; esac
        done <<< "$models"
    fi
    # throwaway projects of the import and combine phases (listed before they are created)
    local extra
    while IFS= read -r extra; do
        [[ -z "$extra" ]] && continue
        _delete_extra() {
            api DELETE "${API_PREFIX}/projects/${extra}?confirm=${extra}&force=true"
            case "$HTTP_CODE" in 200|202|404) return 0 ;; esac
            return 1
        }
        poll 120 _delete_extra || rc=1
    done <<< "$(st_get extra_projects)"
    if [[ "$(st_get project_created)" == 1 ]]; then
        # project_busy (409) while a job runs: cancel it and retry until it drains
        _delete_project() {
            api DELETE "${PROJECT_API}?confirm=${SLUG}&force=true"
            case "$HTTP_CODE" in
                200|202|404) return 0 ;;
                409) cancel_project_jobs ;;
            esac
            return 1
        }
        poll "$PHASE_TIMEOUT" _delete_project || rc=1
    fi
    if [[ "$(st_get project_created)" == 1 && $rc == 0 ]]; then
        # project delete is asynchronous (status "deleting"): wait for 404
        _gone() { api GET "${PROJECT_API}"; [[ "$HTTP_CODE" == 404 ]]; }
        poll 120 _gone || rc=1
    fi
    stop_evaluator || rc=1
    ((rc == 0)) && st_set cleaned 1
    return "$rc"
}

on_exit() {
    local code=$?
    trap - EXIT INT TERM
    cleanup_resources || echo "acceptance_run.sh: WARNING: cleanup of project '${SLUG}' did not complete" >&2
    exit "$code"
}
CHILD=""
on_signal() {
    echo "acceptance_run.sh: interrupted" >&2
    if [[ -n "$CHILD" ]]; then
        pkill -TERM -P "$CHILD" 2>/dev/null || true
        kill -TERM "$CHILD" 2>/dev/null || true
    fi
    exit 130
}
trap on_exit EXIT
trap on_signal INT TERM
