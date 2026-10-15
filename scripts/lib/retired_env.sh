#!/bin/bash
# =============================================================================
# retired_env.sh - refuse a .env that still sets a renamed variable
# =============================================================================
# Sourced by the deploy CLI (openprocessor) and by setup-openprocessor.sh from the verified
# release. A retired name is not read any more, so leaving it in .env would look like a working
# config that silently does nothing; the compose stack's cropwright entrypoint refuses it too
# (frontend/docker-entrypoint.sh). tests/installer/test_retired_cropwright_env.py keeps the
# three lists in step.

[[ -n "${_RETIRED_ENV_SH_LOADED:-}" ]] && return 0
_RETIRED_ENV_SH_LOADED=1

# Renamed to OP_<name> (the Cropwright variables joined the OP_ prefix).
RETIRED_ENV_NAMES=(
    CROPWRIGHT_IMAGE
    CROPWRIGHT_BIND_ADDRESS
    CROPWRIGHT_PORT
    CROPWRIGHT_INGEST_MAX_REQUEST_MB
    CROPWRIGHT_DATASET_UPLOAD_MAX_MB
)

# refuse_retired_env FILE -- exit 2 naming the new variable for every retired name set in FILE
refuse_retired_env() {
    local file="$1" name found=0
    [[ -f "$file" ]] || return 0
    for name in "${RETIRED_ENV_NAMES[@]}"; do
        if awk -v k="$name" '{ i = index($0, "=") } i > 1 && substr($0, 1, i - 1) == k { f = 1 } END { exit !f }' "$file"; then
            echo "[ERROR] ${name} was renamed to OP_${name}: update ${file}" >&2
            found=1
        fi
    done
    (( found == 0 )) || exit 2
}
