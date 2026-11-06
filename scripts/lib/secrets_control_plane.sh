#!/bin/bash
# =============================================================================
# secrets_control_plane.sh - Postgres, Valkey and Flower secrets
# =============================================================================
# Created as the installing user, mode 600 in 700 directories, with umask 077 before compose
# runs (else Docker creates a root-owned directory). Missing or empty files are created, existing
# ones are never rotated, so the same call serves a fresh install, `upgrade` and `repair`.
# A value is written by the printf builtin and never becomes an argument of another process.
#
# Ownership rule (spike S0): the files stay owned by the installing user at mode 600.
# Postgres reads its superuser password as root and keeps CAP_DAC_OVERRIDE for it; the Valkey
# wrapper (root) copies its conf to a tmpfs because the image drops to the `valkey` user; the
# application image runs as uid 1000, so the installing user should be uid 1000 as for secrets/vlm.

[[ -n "${_SECRETS_CONTROL_PLANE_SH_LOADED:-}" ]] && return 0
_SECRETS_CONTROL_PLANE_SH_LOADED=1

# _cp_random -> 48 hex characters on stdout
_cp_random() {
    head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n'
}

# _cp_ensure_file FILE [PREFIX] -> create FILE with PREFIX + a random value unless it has content
_cp_ensure_file() {
    local file="$1" prefix="${2:-}" value
    [[ -s "$file" ]] && return 0
    value="$(_cp_random)" || return 1
    [[ -n "$value" ]] || return 1
    ( umask 077; printf '%s%s' "$prefix" "$value" > "$file" ) || return 1
    chmod 600 "$file"
}

# ensure_control_plane_secrets ROOT -> secrets/{postgres,valkey,flower} under ROOT (and the empty
# secrets/vlm directory, so Docker never creates it root-owned)
ensure_control_plane_secrets() {
    local root="${1:-.}" sdir pw
    sdir="$root/secrets"
    ( umask 077; mkdir -p "$sdir/vlm" "$sdir/postgres" "$sdir/valkey" "$sdir/flower" ) || return 1
    chmod 700 "$sdir" "$sdir/vlm" "$sdir/postgres" "$sdir/valkey" "$sdir/flower"
    local name
    for name in super_password owner_password app_password; do
        _cp_ensure_file "$sdir/postgres/$name" || return 1
    done
    _cp_ensure_file "$sdir/valkey/password" || return 1
    pw="$(<"$sdir/valkey/password")"
    # The conf is derived from the password, so a lost or stale conf is rebuilt, never the password.
    if [[ "$(<"$sdir/valkey/valkey.conf" 2>/dev/null)" != "requirepass ${pw}" ]]; then
        ( umask 077; printf 'requirepass %s\n' "$pw" > "$sdir/valkey/valkey.conf" ) || return 1
    fi
    chmod 600 "$sdir/valkey/valkey.conf"
    _cp_ensure_file "$sdir/flower/auth" "admin:" || return 1
    if [[ "$(id -u)" != 1000 ]]; then
        echo "note: the application image runs as uid 1000; secrets owned by uid $(id -u) need a matching host user" >&2
    fi
}
