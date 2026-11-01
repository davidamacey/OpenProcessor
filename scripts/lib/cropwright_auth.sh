#!/bin/bash
# =============================================================================
# cropwright_auth.sh - the optional basic-auth login in front of Cropwright
# =============================================================================
# Shared by setup-openprocessor.sh and `openprocessor cropwright password`.
# The login lives in secrets/cropwright/: `htpasswd` (an apr1 hash, which the
# container mounts read-only) and `password` (the plaintext, mode 600, so the
# owner can look it up). The password is never put on an argv, in .env or in a
# log; only its hash reaches the container.

CROPWRIGHT_AUTH_USER=cropwright

# cropwright_generate_password -> 24 random characters, [A-Za-z0-9]
cropwright_generate_password() {
    local pw
    command -v openssl >/dev/null 2>&1 || return 1
    pw="$(openssl rand -base64 48 | tr -d '/+=\n' | head -c 24)" || return 1
    [[ ${#pw} -eq 24 ]] || return 1
    printf '%s' "$pw"
}

# cropwright_write_login ROOT PASSWORD -- (re)write ROOT/secrets/cropwright/*.
# The hash file is 644 inside a 700 directory: the container user (uid 101)
# must read it through the bind mount, other host users cannot reach it.
cropwright_write_login() {
    local root="$1" pw="$2" dir hash
    command -v openssl >/dev/null 2>&1 || { echo "openssl is required to hash the Cropwright password" >&2; return 1; }
    [[ -n "$pw" ]] || { echo "empty Cropwright password" >&2; return 1; }
    [[ "$pw" == *$'\n'* ]] && { echo "the Cropwright password must be a single line" >&2; return 1; }
    dir="${root}/secrets/cropwright"
    ( umask 077; mkdir -p "$dir" )
    chmod 700 "${root}/secrets" "$dir"
    hash="$(printf '%s' "$pw" | openssl passwd -apr1 -stdin)" || return 1
    [[ "$hash" == '$apr1$'* ]] || { echo "openssl did not produce an apr1 hash" >&2; return 1; }
    ( umask 077; printf '%s:%s\n' "$CROPWRIGHT_AUTH_USER" "$hash" > "${dir}/htpasswd" )
    chmod 644 "${dir}/htpasswd"
    ( umask 077; printf '%s' "$pw" > "${dir}/password" )
    chmod 600 "${dir}/password"
}

# The installer-side steps below use its helpers (read_env_var, env_set, prompt_line, die,
# wait_http, ...) and globals; they live here to keep setup-openprocessor.sh within its
# size baseline.

# choose_cropwright_bind -> OP_CROPWRIGHT_BIND: the web UI serves this computer AND the local LAN
# (0.0.0.0) unless --local-only, or "no" at the prompt, keeps it on 127.0.0.1. Backend ports stay on
# OP_BIND_ADDRESS; LAN browsers reach the API through Cropwright's nginx on the docker network. A
# specific non-loopback --bind (say 10.0.0.5) narrows Cropwright to that interface too.
choose_cropwright_bind() {
    local reply prev
    prev="$(read_env_var "${OP_DIR}/.env" OP_CROPWRIGHT_BIND_ADDRESS 2>/dev/null || true)"
    if [[ "$OP_LOCAL_ONLY" == 1 ]]; then
        OP_CROPWRIGHT_BIND=127.0.0.1
    elif [[ "$OP_BIND_ADDRESS" != 0.0.0.0 ]] && ! is_loopback_ipv4 "$OP_BIND_ADDRESS"; then
        OP_CROPWRIGHT_BIND="$OP_BIND_ADDRESS"
    elif [[ -n "$prev" ]] && is_ipv4 "$prev"; then
        OP_CROPWRIGHT_BIND="$prev"
    elif [[ "$OP_UNATTENDED" != 1 ]] && tty_usable; then
        prompt_line reply "Let other computers on your LAN open the Cropwright web UI? [Y/n]: " "--local-only"
        if [[ -z "$reply" || "${reply,,}" == y* ]]; then OP_CROPWRIGHT_BIND=0.0.0.0; else OP_CROPWRIGHT_BIND=127.0.0.1; fi
    else
        OP_CROPWRIGHT_BIND=0.0.0.0
    fi
}

# choose_cropwright_auth -> OP_CROPWRIGHT_AUTH_ON, and the login under secrets/cropwright/.
# Default: on when Cropwright is published beyond loopback, off otherwise. The environment
# variable OP_CROPWRIGHT_BASIC_AUTH (true/false) overrides; failing that, a re-run keeps what
# .env already says. The generated password is stored in secrets/cropwright/password (mode
# 600) and never printed, so an unattended run cannot leak it into a log.
choose_cropwright_auth() {
    local want="${OP_CROPWRIGHT_BASIC_AUTH:-}" prev reply
    prev="$(read_env_var "$ENV_FILE" OP_CROPWRIGHT_BASIC_AUTH 2>/dev/null || true)"
    OP_CROPWRIGHT_AUTH_ON=0
    if [[ -n "$want" ]]; then
        _truthy "$want" && OP_CROPWRIGHT_AUTH_ON=1
    elif [[ -n "$prev" ]]; then
        _truthy "$prev" && OP_CROPWRIGHT_AUTH_ON=1
    elif ! is_loopback_ipv4 "$OP_CROPWRIGHT_BIND"; then
        OP_CROPWRIGHT_AUTH_ON=1
        if [[ "$OP_UNATTENDED" != 1 ]] && tty_usable; then
            prompt_line reply "Protect Cropwright with a login (HTTP basic auth)? [Y/n]: " "OP_CROPWRIGHT_BASIC_AUTH=false"
            [[ -z "$reply" || "${reply,,}" == y* ]] || OP_CROPWRIGHT_AUTH_ON=0
        fi
    fi
    if [[ "$OP_CROPWRIGHT_AUTH_ON" != 1 ]]; then
        env_set OP_CROPWRIGHT_BASIC_AUTH false
        return 0
    fi
    local dir="${OP_DIR}/secrets/cropwright" pw
    if [[ ! -s "${dir}/htpasswd" ]]; then
        pw="$(cropwright_generate_password)" || die "Cropwright basic auth needs openssl (install it, or set OP_CROPWRIGHT_BASIC_AUTH=false)" "$EXIT_USAGE"
        cropwright_write_login "$OP_DIR" "$pw" || die "could not write the Cropwright login" "$EXIT_USAGE"
        pw=""
        log_info "Cropwright login created: secrets/cropwright/password (mode 600)"
    fi
    env_set OP_CROPWRIGHT_BASIC_AUTH true
    env_set OP_CROPWRIGHT_BASIC_AUTH_FILE ./secrets/cropwright/htpasswd
}

# choose_cropwright_access -- where Cropwright is published, then whether it needs a login.
choose_cropwright_access() {
    choose_cropwright_bind
    choose_cropwright_auth
}

# verify_cropwright HOST -- health checks of the cropwright tier; appends to the caller's `fails`.
verify_cropwright() {
    local h="$1" cw_cfg=""
    wait_http "http://${h}:${OP_CROPWRIGHT_PORT}/nginx-health" 120 || fails+=("cropwright /nginx-health not 200")
    if [[ "${OP_CROPWRIGHT_AUTH_ON:-0}" == 1 ]]; then
        # The login goes to curl through a private config file, never an argv.
        cw_cfg="$(umask 077; mktemp)"
        printf 'user = "%s:%s"\n' "$CROPWRIGHT_AUTH_USER" "$(<"${OP_DIR}/secrets/cropwright/password")" > "$cw_cfg"
        [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "http://${h}:${OP_CROPWRIGHT_PORT}/" 2>/dev/null)" == 401 ]] \
            || fails+=("cropwright answered without a login although basic auth is on")
    fi
    wait_http "http://${h}:${OP_CROPWRIGHT_PORT}/" 120 "" -fsS "$cw_cfg" || fails+=("cropwright / not 200")
    wait_http "http://${h}:${OP_CROPWRIGHT_PORT}/curation/health" 60 "" -fsS "$cw_cfg" || fails+=("cropwright cannot reach the API through its proxy (network ${OP_PROJECT}_triton_net)")
    [[ -z "$cw_cfg" ]] || rm -f "$cw_cfg"
}
