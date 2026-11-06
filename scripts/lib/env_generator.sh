#!/bin/bash
# =============================================================================
# env_generator.sh - the one .env generator (ENV-4)
# =============================================================================
# Used by setup-openprocessor.sh (env_create_or_merge) and scripts/setup.sh
# (generate_env_file in config.sh). The file is created from env.template with
# umask 077 and mode 600 before any value is written; a key the file already has
# is never overwritten; a value is never an argument of another process.

[[ -n "${_ENV_GENERATOR_SH_LOADED:-}" ]] && return 0
_ENV_GENERATOR_SH_LOADED=1

# env_get_key FILE KEY -> last value of KEY (exact key match); status 1 when absent
env_get_key() {
    [[ -f "$1" ]] || return 1
    awk -v k="$2" '{ i = index($0, "=") } i > 1 && substr($0, 1, i - 1) == k { v = substr($0, i + 1); f = 1 } END { if (f) print v; exit !f }' "$1"
}

# env_set_key FILE KEY VALUE -> replace KEY (first occurrence, others dropped) or append it
env_set_key() {
    local file="$1" key="$2" value="$3" dir tmp line found=0
    [[ "$key" =~ ^[A-Z_][A-Z0-9_]*$ ]] || { echo "invalid .env key: ${key}" >&2; return 1; }
    if [[ "$value" == *$'\n'* || "$value" == *$'\r'* ]]; then
        echo "refusing a multi-line value for ${key}" >&2
        return 1
    fi
    dir="$(dirname "$file")"
    tmp="$(umask 077; mktemp "${dir}/.env.XXXXXX")" || return 1
    if ! {
        if [[ -f "$file" ]]; then
            while IFS= read -r line || [[ -n "$line" ]]; do
                if [[ "$line" == "${key}="* ]]; then
                    (( found == 0 )) && printf '%s=%s\n' "$key" "$value"
                    found=1
                else
                    printf '%s\n' "$line"
                fi
            done < "$file"
        fi
        (( found == 0 )) && printf '%s=%s\n' "$key" "$value"
        true
    } > "$tmp" || ! mv -f "$tmp" "$file"; then
        rm -f "$tmp"
        return 1
    fi
}

# env_generate TEMPLATE ENV_FILE -> create ENV_FILE from TEMPLATE (mode 600 from the first byte),
# or add the keys that are new in TEMPLATE to an existing one. Existing values are never changed.
env_generate() {
    local template="$1" env_file="$2" line key added=()
    if [[ ! -f "$env_file" ]]; then
        ( umask 077; install -m 600 "$template" "$env_file" ) || return 1
        chmod 600 "$env_file"
        echo "created ${env_file} (mode 600)"
        return 0
    fi
    chmod 600 "$env_file"
    while IFS= read -r line; do
        [[ "$line" =~ ^([A-Z_][A-Z0-9_]*)= ]] || continue
        key="${BASH_REMATCH[1]}"
        if ! env_get_key "$env_file" "$key" >/dev/null; then
            env_set_key "$env_file" "$key" "${line#*=}" || return 1
            added+=("$key")
        fi
    done < "$template"
    if (( ${#added[@]} > 0 )); then
        echo "added new .env keys from env.template: ${added[*]}"
    fi
}
