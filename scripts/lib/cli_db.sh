#!/bin/bash
# =============================================================================
# cli_db.sh - `openprocessor db backup|restore`
# =============================================================================
# Sourced by the openprocessor CLI (dc, _env_value, log_* come from there).
# pg_dump custom format is compressed, restorable selectively and consistent without blocking
# writers: https://www.postgresql.org/docs/current/app-pgdump.html
# Recovery point: the state of the last dump. There is no point-in-time recovery in 0.7.0.

# Workers and apps that must be stopped while the database is replaced.
_DB_APP_SERVICES=(
    api jobs-worker periodic-worker open-vocab-worker jobs-beat flower cluster-worker
    curation-detection-worker curation-vlm-worker curation-auto-label-worker
    curation-assist-loop-worker
)

_db_backup_dir() {
    local dir="${1:-}"
    [[ -n "$dir" ]] || dir="$(_env_value OP_DB_BACKUP_DIR)"
    [[ -n "$dir" ]] || dir="$PROJECT_DIR/backups/postgres"
    [[ "$dir" == /* ]] || dir="$PROJECT_DIR/$dir"
    echo "$dir"
}

_db_require_postgres() {
    if [[ -z "$(dc ps -q postgres 2>/dev/null)" ]]; then
        log_error "postgres is not running; start the stack first (./openprocessor start)"
        return 1
    fi
}

_db_backup() {
    local out_arg="" dir keep file part i
    while (( $# > 0 )); do
        case "$1" in
            --out) [[ -n "${2:-}" ]] || { log_error "--out needs a directory"; return 1; }; out_arg="$2"; shift 2 ;;
            *) log_error "unknown option for db backup: $1"; return 1 ;;
        esac
    done
    keep="$(_env_value OP_DB_BACKUP_KEEP)"
    keep="${keep:-7}"
    [[ "$keep" =~ ^[1-9][0-9]*$ ]] || { log_error "OP_DB_BACKUP_KEEP must be a positive integer, got '${keep}'"; return 1; }
    _db_require_postgres || return 1
    dir="$(_db_backup_dir "$out_arg")"
    ( umask 077; mkdir -p "$dir" ) || { log_error "cannot create ${dir}"; return 1; }
    file="${dir}/openprocessor-$(date -u +%Y%m%dT%H%M%SZ).dump"
    part="${file}.part"
    # Socket connection inside the container: the image trusts local connections, so no password
    # is needed and none is placed in argv.
    if ! ( umask 077; dc exec -T postgres pg_dump -U op_owner -d openprocessor -Fc > "$part" ) || [[ ! -s "$part" ]]; then
        rm -f -- "$part"
        log_error "pg_dump failed; no backup was written"
        return 1
    fi
    mv -f -- "$part" "$file" || return 1
    local -a dumps
    mapfile -t dumps < <(printf '%s\n' "$dir"/openprocessor-*.dump | sort -r)
    for (( i = keep; i < ${#dumps[@]}; i++ )); do
        rm -f -- "${dumps[$i]}"
    done
    log_success "backup written: ${file}"
}

_db_restore() {
    local file="" yes=0 reply svc
    while (( $# > 0 )); do
        case "$1" in
            --yes) yes=1; shift ;;
            -*) log_error "unknown option for db restore: $1"; return 1 ;;
            *) [[ -z "$file" ]] || { log_error "db restore takes one file"; return 1; }; file="$1"; shift ;;
        esac
    done
    [[ -n "$file" && -s "$file" ]] || { log_error "usage: db restore <dump file> [--yes] (the file must exist)"; return 1; }
    if (( ! yes )); then
        read -r -p "Replace the database with ${file}? Everything written since that dump is lost. [y/N] " reply </dev/tty || return 1
        [[ "$reply" =~ ^[Yy]$ ]] || { log_info "nothing changed"; return 1; }
    fi
    local active
    active="$(dc config --services 2>/dev/null || true)"
    local -a stop=()
    for svc in "${_DB_APP_SERVICES[@]}"; do
        grep -qx -- "$svc" <<<"$active" && stop+=("$svc")
    done
    if (( ${#stop[@]} > 0 )); then
        dc stop "${stop[@]}" || return 1
    fi
    dc up -d --wait postgres || return 1
    # The role bootstrap first, so op_owner and op_app exist on a fresh host before pg_restore
    # grants anything; then the restore itself.
    dc run --rm --no-deps db-migrate || return 1
    dc exec -T postgres pg_restore -U op_super --clean --if-exists --no-owner --no-privileges \
        --role=op_owner -d openprocessor < "$file" || { log_error "pg_restore failed"; return 1; }
    # A dump from an older schema is upgraded, and the grants are applied again.
    dc run --rm --no-deps db-migrate || return 1
    # events.xid belongs to one cluster's transaction counter; old cursors would compare wrongly.
    dc exec -T postgres psql -U op_super -d openprocessor -v ON_ERROR_STOP=1 -c 'TRUNCATE events' || return 1
    dc up -d || return 1
    log_success "restored from ${file}"
}

# db verify: the database against OpenSearch and storage (class reconciliation, project indexes,
# storage keys). Report only; it changes nothing.
_db_verify() {
    dc run --rm --no-deps -T -w /app api python -m scripts.db.verify
}

cmd_db() {
    local sub="${1:-}"
    shift || true
    case "$sub" in
        backup) _db_backup "$@" ;;
        restore) _db_restore "$@" ;;
        verify) _db_verify "$@" ;;
        *) log_error "usage: openprocessor db backup [--out DIR] | db restore <file> [--yes] | db verify"; return 1 ;;
    esac
}
