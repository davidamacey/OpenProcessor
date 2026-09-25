#!/bin/bash
# =============================================================================
# Shared port resolution (F-66)
# =============================================================================
# scripts/openprocessor.sh and scripts/setup.sh used to hardcode
# localhost:4603/4600/4607/4605 directly in curl calls. On a shared host
# running a second isolated OpenProcessor stack (remapped ports via .env,
# see env.template's "Isolation" section) those hardcoded literals read
# ANOTHER stack's health instead of the one the script was invoked for.
#
# `env_port` reads a single KEY=value out of .env without sourcing the
# whole file (.env may hold JSON-ish values in commented-out advanced
# settings that aren't safe to `source`), matching the pattern the
# Makefile uses (`-include .env` + `API_PORT ?= 4603`).

env_port() {
    local key="$1" default="$2" env_file="${PROJECT_DIR:-.}/.env"
    if [[ -f "$env_file" ]]; then
        local value
        # `|| true`: grep exits 1 on no match, which under a caller's
        # `set -e` would otherwise abort the whole script on a missing key
        # instead of falling through to the default below.
        value="$(grep -E "^${key}=" "$env_file" 2>/dev/null | tail -n1 | cut -d= -f2- || true)"
        if [[ -n "$value" ]]; then
            echo "$value"
            return 0
        fi
    fi
    echo "$default"
}
