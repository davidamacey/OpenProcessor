#!/bin/bash
# =============================================================================
# upgrade_060.sh - the one-step 0.6.0 -> 0.7.0 control-data upgrade and its rollback
# =============================================================================
# Sourced by setup-openprocessor.sh after the release files are installed (and by do_rollback
# before it swaps scripts/lib/), so it runs with the installer's helpers (dc, log_step, log_info,
# log_warn, log_error, require_destructive_consent) in scope. Kept out of the installer to hold
# that file to its size baseline. The code is src/services/upgrade/; the order is 02c section 2.
#
# Upgrade: stop the stack -> start postgres, valkey and opensearch -> db-migrate (Alembic) ->
# scripts.db.upgrade_from_060 (export, load, verify, delete) -> db-migrate again (its OpenSearch
# step now finds the loaded projects and stamps their indexes, RC-1) -> the caller starts the rest.
# The backup came first (backup_install). A fresh install and a same-version re-run never come here.

# upgrade_060_applies VERSION -- true for a recorded 0.x install older than 0.7, and for an
# upgrade that stopped before it finished (the installer records the new version early, so the
# marker, not the version, says it is still owed).
upgrade_060_applies() {
    [[ -f "${OP_DIR}/.install/upgrade_060.pending" || "${1#v}" =~ ^0\.[0-6]\.[0-9]+ ]]
}

# The module runs in the api image with the install's .install directory mounted at /install, as
# the calling user so the bundle is not root-owned.
_upgrade_060_module() {
    dc run --rm --no-deps -T -w /app --user "$(id -u):$(id -g)" \
        -v "${OP_DIR}/.install:/install" -e OP_INSTALL_DIR=/install api python -m "$@"
}

_upgrade_060_newest_bundle() {
    local d newest=""
    for d in "${OP_DIR}"/.install/upgrade-0.6.0-*/; do
        [[ -f "${d}MANIFEST.json" ]] && newest="${d%/}"
    done
    printf '%s' "$newest"
}

# The 0.6.0 directory variables the upgrade reads (src/services/upgrade/from_060/sources.py, from
# .env through the api container's env_file). They are retired in 0.7.0 (src/config/retired_env.py),
# so once the upgrade has read them they come out of .env, or the stack would refuse to start.
UPGRADE_060_LEGACY_ENV=(
    OP_BULK_INGEST_DIR
    OP_DATASET_IMPORTS_DIR
    OP_REPROCESS_JOBS_DIR
    OP_PROBE_JOBS_DIR
    OP_SCORES_STATE_DIR
    OP_SELECT_JOBS_DIR
    OP_VIZ_JOBS_DIR
    OP_ASSIST_JOBS_DIR
    OP_ASSIST_LOOP_JOBS_DIR
    OP_REGION_DRAIN_STATE_DIR
    OP_COMBINE_JOBS_DIR
)

# _upgrade_060_strip_legacy_env -- drop each legacy variable from .env (exact key match, atomic),
# logging each. The backup taken before the upgrade keeps the old .env.
_upgrade_060_strip_legacy_env() {
    local key tmp
    [[ -f "$ENV_FILE" ]] || return 0
    for key in "${UPGRADE_060_LEGACY_ENV[@]}"; do
        read_env_var "$ENV_FILE" "$key" >/dev/null || continue
        tmp="$(mktemp "$(dirname "$ENV_FILE")/.env.XXXXXX")" || return 1
        chmod 600 "$tmp"
        if ! awk -v k="$key" '{ i = index($0, "=") } !(i > 1 && substr($0, 1, i - 1) == k)' "$ENV_FILE" > "$tmp" \
            || ! mv -f "$tmp" "$ENV_FILE"; then
            rm -f "$tmp"
            return 1
        fi
        log_info "removed ${key} from .env: the 0.6.0 upgrade has read it and 0.7.0 no longer uses it"
    done
}

# upgrade_060_step -- the installer's call: the control data moves out of OpenSearch before anything
# starts. A stop leaves the marker, so a re-run (whose recorded version is already the new one)
# still comes back to it.
upgrade_060_step() {
    if [[ "$OP_NO_START" == 1 ]]; then
        log_error "this install is 0.6.0: the control-data upgrade needs the stack, re-run without --no-start"
        return "$EXIT_HEALTH"
    fi
    if [[ "$OP_DRY_RUN" == 1 ]]; then
        echo "DRY: 0.6.0 control-data upgrade (backup, db-migrate, upgrade_from_060, db-migrate)"
        return 0
    fi
    : > "${OP_DIR}/.install/upgrade_060.pending"
    upgrade_060_run || return 1
    _upgrade_060_strip_legacy_env || return 1
    rm -f "${OP_DIR}/.install/upgrade_060.pending"
}

# upgrade_060_run -- returns 1 (with the report path named) when the upgrade stopped; nothing of
# the 0.6.0 data is deleted in that case and the stack is not started.
upgrade_060_run() {
    local dir out
    dir="${OP_DIR}/.install/upgrade_060"
    out="${dir}/$(date -u +%Y%m%dT%H%M%SZ).json"
    mkdir -p "$dir" || return 1
    log_step "0.6.0 control-data upgrade"
    dc stop || { log_error "could not stop the 0.6.0 services"; return 1; }
    dc up -d --wait postgres valkey opensearch || { log_error "postgres, valkey and opensearch did not become healthy"; return 1; }
    dc run --rm --no-deps db-migrate || { log_error "db-migrate failed before the upgrade"; return 1; }
    if ! _upgrade_060_module scripts.db.upgrade_from_060 >"$out"; then
        log_error "the 0.6.0 upgrade stopped: ${out}"
        log_error "nothing of the 0.6.0 control data was deleted; see UPGRADE_REPORT.json in $(_upgrade_060_newest_bundle)"
        log_error "fix the cause and run the upgrade again, run ./setup-openprocessor.sh --rollback, or load again with: python -m scripts.db.upgrade_from_060 --reset"
        return 1
    fi
    dc run --rm --no-deps db-migrate || { log_error "db-migrate failed after the upgrade"; return 1; }
    log_info "0.6.0 upgrade report: ${out}"
}

# upgrade_060_rollback -- before a rollback restores the older release: rebuild the 0.6.0 control
# indexes and files from the newest bundle (OpenSearch only), verify them, and only then mark the
# PostgreSQL data abandoned (drop and recreate the database; the Valkey volume is left unused),
# so a later re-upgrade is a fresh upgrade. Nothing to do when no upgrade ran.
upgrade_060_rollback() {
    local bundle
    bundle="$(_upgrade_060_newest_bundle)"
    [[ -n "$bundle" ]] || return 0
    log_step "Restore the 0.6.0 control data from $(basename "$bundle")"
    require_destructive_consent "Restore the 0.6.0 control data and abandon everything written to PostgreSQL since the upgrade (classes, configs and jobs created after it are not carried back)"
    dc stop || return 1
    dc up -d --wait opensearch || { log_error "opensearch did not become healthy"; return 1; }
    if ! _upgrade_060_module scripts.db.rollback_to_060 --bundle "/install/$(basename "$bundle")"; then
        log_error "the restore did not match the bundle; the 0.6.0 services were not started"
        return 1
    fi
    dc up -d --wait postgres || return 1
    dc exec -T postgres psql -U op_super -d postgres -v ON_ERROR_STOP=1 \
        -c 'DROP DATABASE IF EXISTS openprocessor WITH (FORCE)' -c 'CREATE DATABASE openprocessor' || {
        log_error "could not recreate the openprocessor database"
        return 1
    }
    dc stop postgres || true
    log_info "the PostgreSQL data was abandoned; a later upgrade starts from scratch"
}
