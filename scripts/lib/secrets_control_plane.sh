#!/bin/bash
# =============================================================================
# secrets_control_plane.sh - Postgres, Valkey and Flower secrets
# =============================================================================
# Created as the installing user, mode 600 in 700 directories, with umask 077 before compose
# runs (else Docker creates a root-owned directory). Missing or empty files are created, existing
# ones are never rotated, so the same call serves a fresh install, `upgrade` and `repair`.
# A value is written by the printf builtin and never becomes an argument of another process.
#
# Ownership rule (spikes S0 and S1, "rule B"): the files stay owned by the installing user at mode
# 600 in 700 directories. Postgres reads its superuser password as root and keeps CAP_DAC_OVERRIDE
# for it. Valkey runs as the installing user (`user: OP_UID:OP_GID`, recorded in an existing .env
# below), because the image's own drop to uid 999 cannot read those files. The application image
# runs as uid 1000, so the installing user should be uid 1000 as for secrets/vlm.

[[ -n "${_SECRETS_CONTROL_PLANE_SH_LOADED:-}" ]] && return 0
_SECRETS_CONTROL_PLANE_SH_LOADED=1

# shellcheck source=env_generator.sh
source "$(dirname "${BASH_SOURCE[0]}")/env_generator.sh"

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
    # Compose reads these for the valkey service's user; an existing value is never overwritten.
    if [[ -f "$root/.env" ]]; then
        env_get_key "$root/.env" OP_UID >/dev/null || env_set_key "$root/.env" OP_UID "$(id -u)" || return 1
        env_get_key "$root/.env" OP_GID >/dev/null || env_set_key "$root/.env" OP_GID "$(id -g)" || return 1
    fi
    if [[ "$(id -u)" != 1000 ]]; then
        echo "note: the application image runs as uid 1000; secrets owned by uid $(id -u) need a matching host user" >&2
    fi
}
