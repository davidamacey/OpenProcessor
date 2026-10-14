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
# cleanup: always delete the throwaway project and the models this run promoted
# ---------------------------------------------------------------------------
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
    if [[ "$(st_get project_created)" == 1 ]]; then
        api DELETE "${PROJECT_API}?confirm=${SLUG}&force=true"
        case "$HTTP_CODE" in
            200|202|404) ;;
            *)
                # a project must be archived-or-idle to delete; one retry via archive
                api POST "${PROJECT_API}/archive"
                api DELETE "${PROJECT_API}?confirm=${SLUG}&force=true"
                case "$HTTP_CODE" in 200|202|404) ;; *) rc=1 ;; esac
                ;;
        esac
    fi
    if [[ "$(st_get project_created)" == 1 && $rc == 0 ]]; then
        # project delete is asynchronous (status "deleting"): wait for 404
        _gone() { api GET "${PROJECT_API}"; [[ "$HTTP_CODE" == 404 ]]; }
        poll 120 _gone || rc=1
    fi
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
