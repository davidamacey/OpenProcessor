#!/bin/bash
# =============================================================================
# registry_migration.sh - class registry moves around an upgrade or a rollback
# =============================================================================
# Since 0.7.0 each project's class registry is an OpenSearch document; before it was
# class_registry.json. Sourced by setup-openprocessor.sh after the release files are installed
# (and by do_rollback before it swaps scripts/lib/), so it runs with the installer's helpers
# (dc, log_step, log_info, log_warn, log_error) in scope. Kept out of the installer to hold that
# file to its size baseline.

_registry_migration_module() {
    dc run --rm --no-deps -T -w /app api python -m scripts.curation.migrate_class_registry "$@"
}

# registry_migration_run -- after a healthy upgrade/repair: the dry-run report first, then the
# real migration. Both reports land under .install/registry_migration/. A failure returns 1 and
# names the report; the stack stays in its fail-closed 409 state (registry_not_migrated).
registry_migration_run() {
    local dir ts mode out
    local -a flags
    dir="${OP_DIR}/.install/registry_migration"
    ts="$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p "$dir" || return 1
    log_step "Class registry migration"
    for mode in dry-run run; do
        out="${dir}/${ts}-${mode}.json"
        flags=(--all)
        [[ "$mode" == dry-run ]] && flags+=(--dry-run)
        if ! _registry_migration_module "${flags[@]}" >"$out"; then
            log_error "class registry migration (${mode}) failed; report: ${out}"
            log_error "class routes answer 409 registry_not_migrated until it passes; fix the cause, then run: ./openprocessor curation migrate-registry --all"
            return 1
        fi
        log_info "class registry migration (${mode}): ${out}"
    done
}

# registry_migration_reverse -- before a rollback restores the older release: write every
# project's registry document back to class_registry.json, so the older version sees every class,
# including the ones added after the migration. Without a running OpenSearch it cannot, and says
# how to do it by hand.
registry_migration_reverse() {
    local dir out
    if [[ -z "$(dc ps --status running -q opensearch 2>/dev/null)" ]]; then
        log_warn "OpenSearch is not running: the class registry was NOT written back to class_registry.json"
        log_warn "start the stack and run: ./openprocessor curation migrate-registry --all --reverse  (before the older version serves classes)"
        return 0
    fi
    dir="${OP_DIR}/.install/registry_migration"
    mkdir -p "$dir" || return 1
    out="${dir}/$(date -u +%Y%m%dT%H%M%SZ)-reverse.json"
    if ! _registry_migration_module --all --reverse >"$out"; then
        log_error "writing the class registry back failed; report: ${out}"
        return 1
    fi
    log_info "class registry written back to class_registry.json: ${out}"
}
