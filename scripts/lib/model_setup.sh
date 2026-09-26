#!/bin/bash
# =============================================================================
# model_setup.sh - Model export/load groups, retry classifier, skip logic
# =============================================================================
# Shared by setup-openprocessor.sh and the `openprocessor models` CLI, plus
# the Makefile export-* targets. One implementation, per the installer plan
# (docs/design/openprocessor_internal/one_line_installer_plan.md section 4).
#
# Every compose invocation here goes through the caller's dc() wrapper --
# this file assumes `dc` is already defined and never calls
# `docker compose` directly (tests/installer/test_static.py enforces this).

[[ -n "${_MODEL_SETUP_SH_LOADED:-}" ]] && return 0
_MODEL_SETUP_SH_LOADED=1

_MODEL_SETUP_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=colors.sh
[[ -z "${NC:-}" ]] && source "${_MODEL_SETUP_SH_DIR}/colors.sh"

# -----------------------------------------------------------------------------
# Failure classification (section 4.2)
# -----------------------------------------------------------------------------
# classify_failure LOGFILE
# Prints one of: transient | permanent | unknown
classify_failure() {
    local logfile="$1"
    [[ -f "$logfile" ]] || { echo "unknown"; return 0; }

    # Permanent, checked first: no point retrying these.
    if grep -qE 'ModuleNotFoundError' "$logfile"; then
        echo "permanent:stale_image"
        return 0
    fi
    if grep -qE '(^| )(401|403)( |$)|GatedRepoError' "$logfile"; then
        echo "permanent:gated"
        return 0
    fi
    if grep -qiE 'no space left' "$logfile"; then
        echo "permanent:disk"
        return 0
    fi
    if grep -qiE 'out of memory' "$logfile"; then
        echo "permanent:oom"
        return 0
    fi

    # Transient: known GPU-init / network races that clear on retry.
    if grep -qE \
        'CUDA initialization failure|error 100|factory function returned nullptr|CUDA error: an illegal memory access|cudaErrorDevicesUnavailable|Connection reset|Read timed out|HTTP/1\.1" 5[0-9]{2}' \
        "$logfile"; then
        echo "transient"
        return 0
    fi

    echo "unknown"
}

# permanent_hint CLASS
# Returns the targeted hint string for a permanent:* class (section 4.2).
permanent_hint() {
    case "$1" in
        permanent:stale_image)
            echo "image is stale or wrong; run 'openprocessor repair --images'" ;;
        permanent:gated)
            echo "token or licence issue: accept the model's licence and re-check the token" ;;
        permanent:disk)
            echo "no space left on the install filesystem" ;;
        permanent:oom)
            echo "out of memory on the first attempt; review the GPU plan (openprocessor gpu plan)" ;;
        *)
            echo "" ;;
    esac
}

# -----------------------------------------------------------------------------
# Retry/backoff (section 4.2)
# -----------------------------------------------------------------------------
# retry_step NAME MAX_ATTEMPTS -- CMD...
# Runs CMD, retrying on a "transient" classification with 15s/45s backoff.
# An "unknown" failure gets exactly one retry. A "permanent" failure fails
# immediately with its hint and never retries. Returns 0 on success, the
# last exit code otherwise. Writes the last attempt's log to
# "${MODEL_SETUP_LOGDIR:-.}/$NAME.log".
retry_step() {
    local name="$1" max="${2:-3}"
    shift 2
    [[ "$1" == "--" ]] && shift

    local logdir="${MODEL_SETUP_LOGDIR:-.}"
    mkdir -p "$logdir"
    local logfile="${logdir}/${name}.log"

    local attempt=1
    local backoffs=(15 45)
    local last_rc=1

    while (( attempt <= max )); do
        set +e
        "$@" >"$logfile" 2>&1
        last_rc=$?
        set -e
        if (( last_rc == 0 )); then
            log_success "step ${name}: ok (attempt ${attempt})"
            return 0
        fi

        local cls
        cls="$(classify_failure "$logfile")"

        if [[ "$cls" == permanent:* ]]; then
            log_error "step ${name}: permanent failure (${cls#permanent:}): $(permanent_hint "$cls")"
            return "$last_rc"
        fi

        if [[ "$cls" == "unknown" && $attempt -ge 2 ]]; then
            log_error "step ${name}: unknown failure, no more retries"
            return "$last_rc"
        fi

        if (( attempt >= max )); then
            log_error "step ${name}: exhausted ${max} attempts"
            return "$last_rc"
        fi

        local sleep_s="${backoffs[$((attempt - 1))]:-45}"
        log_warn "step ${name}: ${cls} failure (attempt ${attempt}/${max}), retrying in ${sleep_s}s"
        if declare -F wait_for_gpu_memory >/dev/null 2>&1; then
            wait_for_gpu_memory || true
        fi
        sleep "$sleep_s"
        attempt=$((attempt + 1))
    done

    return "$last_rc"
}

# -----------------------------------------------------------------------------
# Skip logic (section 4.2, "Idempotent skip")
# -----------------------------------------------------------------------------
# group_should_skip GROUP_OUTPUTS_GLOB TRITON_MODELS STATE_FILE TRITON_DIGEST
# GROUP_OUTPUTS_GLOB: space-separated list of expected output file paths.
# TRITON_MODELS: space-separated list of Triton model names this group loads
#   ("" for groups with no load step).
# STATE_FILE: path to state.json (may not exist).
# TRITON_DIGEST: the current triton image digest.
# Returns 0 (skip) only if every output exists, every model reports READY
# (skipped entirely when TRITON_MODELS is empty), and the state file's
# recorded triton_image_digest matches TRITON_DIGEST.
group_should_skip() {
    local outputs="$1" models="$2" state_file="$3" digest="$4"

    for f in $outputs; do
        [[ -e "$f" ]] || return 1
    done

    if [[ -n "$models" ]]; then
        declare -F triton_model_ready >/dev/null 2>&1 || return 1
        for m in $models; do
            triton_model_ready "$m" || return 1
        done
    fi

    [[ -f "$state_file" ]] || return 1
    if command -v jq >/dev/null 2>&1; then
        local recorded
        recorded="$(jq -r '.triton_image_digest // ""' "$state_file" 2>/dev/null)"
        [[ -n "$recorded" && "$recorded" == "$digest" ]]
    else
        grep -q "\"triton_image_digest\": *\"${digest}\"" "$state_file"
    fi
}

# -----------------------------------------------------------------------------
# Triton load + READY wait (section 4.2)
# -----------------------------------------------------------------------------
# triton_model_ready NAME
# Requires TRITON_HTTP_PORT in the environment (defaults to 4600) and curl.
triton_model_ready() {
    local name="$1"
    local port="${TRITON_HTTP_PORT:-4600}"
    curl -fsS "http://127.0.0.1:${port}/v2/models/${name}/ready" >/dev/null 2>&1
}

# triton_load_and_wait NAME [TIMEOUT_S]
triton_load_and_wait() {
    local name="$1" timeout="${2:-120}"
    local port="${TRITON_HTTP_PORT:-4600}"
    curl -fsS -X POST "http://127.0.0.1:${port}/v2/repository/models/${name}/load" >/dev/null 2>&1 || true

    local waited=0
    while (( waited < timeout )); do
        triton_model_ready "$name" && return 0
        sleep 2
        waited=$((waited + 2))
    done
    return 1
}
