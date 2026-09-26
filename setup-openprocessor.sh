#!/bin/bash
# =============================================================================
# setup-openprocessor.sh - one-line OpenProcessor installer
# =============================================================================
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/${OP_GH_REPO}/main/setup-openprocessor.sh | bash
#
# Installs a pinned OpenProcessor release into ./openprocessor/ with no git
# clone, no local image build and no host Python. Design:
# docs/design/openprocessor_internal/one_line_installer_plan.md
#
# Every function is defined before use and the whole script ends in
# `main "$@"`, so a truncated `curl | bash` download never executes half a
# script (plan section 3.3).
#
# Env vars (all optional; flags take precedence where both exist):
#   OP_GH_REPO            GitHub org/repo to install from (default davidamacey/OpenProcessor)
#   OP_GH_DEFAULT_REF      default git ref for --branch resolution (default main)
#   CW_GH_REPO             Cropwright GitHub org/repo
#   OP_IMAGE_NAMESPACE     Docker Hub namespace (default davidamacey)
#   OP_DOCS_URL            docs-site URL, printed in the summary if set
#   OP_INSTALL_DIR / --dir install directory (default ./openprocessor)
#   OP_PROJECT / --project compose project name + container prefix
#   OP_VERSION / --version pinned release tag
#   OP_BRANCH / --branch   testing install from a branch head SHA
#   OP_TIERS / --tiers     comma list of tiers, or --all
#   OP_GPU_PLAN / --gpu-plan   override the GPU placement, e.g. triton=1,vlm=2
#   OP_BIND_ADDRESS / --bind  publish address (default 127.0.0.1)
#   OP_PORT_BASE / --port-base  shift the whole port block
#   OP_WITH_MONITORING / --with-monitoring
#   OP_SAMPLE_DATA / --sample-data
#   OP_UNATTENDED / --unattended  no prompts (auto-enabled if /dev/tty is missing)
#   OP_DRY_RUN / --dry-run    print every mutating command with a DRY: prefix
#   OP_FORCE_CPU / --cpu      no usable GPU; see --control-plane-only
#   OP_ALLOW_PUBLIC_BIND=1    required to accept --bind other than loopback, unattended
#   OP_ALLOW_EXTERNAL_VLM=1   required to accept a non-private --vlm-remote URL, unattended
#   OP_CONFIRM_PURGE=<project>  required to run --purge-volumes/--purge-data unattended
#   HF_TOKEN / HF_TOKEN_FILE  HuggingFace token for gated tiers (segmenter, gated VLM)
#   OP_BOOTSTRAPPED=1         internal loop guard for the raw-main bootstrap re-exec
# =============================================================================

set -uo pipefail

# -----------------------------------------------------------------------------
# 0.1 Single-variable org and branch (plan section 0.1)
# -----------------------------------------------------------------------------
OP_GH_REPO="${OP_GH_REPO:-davidamacey/OpenProcessor}"
OP_GH_DEFAULT_REF="${OP_GH_DEFAULT_REF:-main}"
CW_GH_REPO="${CW_GH_REPO:-attevon-llc/cropwright}"
OP_IMAGE_NAMESPACE="${OP_IMAGE_NAMESPACE:-davidamacey}"
OP_DOCS_URL="${OP_DOCS_URL:-}"

export SCRIPT_VERSION="0.1.0"

# Source vlm_catalog.sh if it's sitting next to us (true once past the
# bootstrap re-exec, whether that's a checkout or an extracted install dir
# -- the manifest ships scripts/lib/vlm_catalog.sh at the same relative
# path). Never sourced during the raw bootstrap hop itself.
_OP_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${_OP_SELF_DIR}/scripts/lib/vlm_catalog.sh" ]]; then
    # shellcheck source=scripts/lib/vlm_catalog.sh
    source "${_OP_SELF_DIR}/scripts/lib/vlm_catalog.sh"
fi

# -----------------------------------------------------------------------------
# Minimal, dependency-free logging (no scripts/lib sourcing before the
# manifest is fetched -- this script must be self-contained).
# -----------------------------------------------------------------------------
_op_color() { if [[ -t 1 ]]; then printf '\033[%sm' "$1"; fi; }
log_info()    { echo "[INFO] $*"; }
log_warn()    { echo "[WARN] $*" >&2; }
log_error()   { echo "[ERROR] $*" >&2; }
log_success() { echo "[OK] $*"; }
log_step()    { echo "=== $* ==="; }

die() {
    log_error "$*"
    exit "${2:-1}"
}

# -----------------------------------------------------------------------------
# 5.1 Compose invocation rule (hard requirement): every compose call goes
# through dc()/dc_cw(). A static test asserts `docker compose` appears
# nowhere else in this file.
# -----------------------------------------------------------------------------
dc() {
    local project="${OP_PROJECT:-openprocessor}"
    local dir="${OP_DIR:-.}"
    if [[ "${OP_DRY_RUN:-0}" == "1" ]]; then
        echo "DRY: docker compose -p ${project} --env-file ${dir}/.env --project-directory ${dir} -f ${dir}/docker-compose.yml $*"
        return 0
    fi
    docker compose -p "$project" --env-file "${dir}/.env" \
        --project-directory "$dir" -f "${dir}/docker-compose.yml" "$@"
}

dc_cw() {
    local project="${OP_PROJECT:-openprocessor}-cw"
    local dir="${OP_DIR:-.}/cropwright"
    if [[ "${OP_DRY_RUN:-0}" == "1" ]]; then
        echo "DRY: docker compose -p ${project} --project-directory ${dir} -f ${dir}/docker-compose.yml $*"
        return 0
    fi
    docker compose -p "$project" --project-directory "$dir" -f "${dir}/docker-compose.yml" "$@"
}

# -----------------------------------------------------------------------------
# 3.3 One-liner bootstrap safety
# -----------------------------------------------------------------------------
# bootstrap_reexec ARGS...
# Resolves the target release, downloads its setup-openprocessor.sh +
# SHA256SUMS, verifies the checksum, then execs it with the original args.
# Only runs when invoked via the raw one-liner (OP_BOOTSTRAP=1, unset by
# default so a tagged copy run directly -- e.g. by the tests -- never
# re-execs itself).
bootstrap_reexec() {
    if [[ "${OP_BOOTSTRAPPED:-0}" == "1" ]]; then
        die "bootstrap loop detected (OP_BOOTSTRAPPED already set)"
    fi
    local ref
    ref="$(resolve_install_ref)" || die "could not resolve an install ref"

    local base_url="${OP_ARTIFACT_BASE_URL:-https://raw.githubusercontent.com/${OP_GH_REPO}/${ref}}"
    local tmp
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' RETURN

    curl -fsSL "${base_url}/setup-openprocessor.sh" -o "${tmp}/setup-openprocessor.sh" || die "download failed"
    curl -fsSL "${base_url}/SHA256SUMS" -o "${tmp}/SHA256SUMS" || die "checksum download failed"

    if [[ "${OP_BRANCH:-}" != "" ]]; then
        log_warn "TESTING install from branch '${OP_BRANCH}': not reproducible, skipping checksum verification"
    else
        (cd "$tmp" && grep 'setup-openprocessor.sh$' SHA256SUMS | sha256sum -c -) || die "checksum verification failed"
    fi

    chmod +x "${tmp}/setup-openprocessor.sh"
    OP_BOOTSTRAPPED=1 exec bash "${tmp}/setup-openprocessor.sh" "$@"
}

# -----------------------------------------------------------------------------
# 3.2 Version pinning
# -----------------------------------------------------------------------------
# resolve_install_ref
# Resolution order: --version, --branch, latest published GitHub Release.
# Never floats to "main" silently. OP_ARTIFACT_BASE_URL / OP_TEST_LATEST_REF
# let the dry-run test harness stand in for the GitHub API.
resolve_install_ref() {
    if [[ -n "${OP_VERSION:-}" ]]; then
        echo "${OP_VERSION}"
        return 0
    fi
    if [[ -n "${OP_BRANCH:-}" ]]; then
        local sha
        sha="$(resolve_branch_sha "${OP_BRANCH}")" || return 1
        echo "sha-${sha:0:12}"
        return 0
    fi
    if [[ -n "${OP_TEST_LATEST_REF:-}" ]]; then
        echo "${OP_TEST_LATEST_REF}"
        return 0
    fi
    local attempt=1
    local ref=""
    while (( attempt <= 3 )); do
        ref="$(curl -fsSL "https://api.github.com/repos/${OP_GH_REPO}/releases/latest" 2>/dev/null \
            | grep -m1 '"tag_name"' | sed -E 's/.*"tag_name": *"([^"]+)".*/\1/')"
        [[ -n "$ref" ]] && break
        sleep $((attempt * 2))
        attempt=$((attempt + 1))
    done
    [[ -n "$ref" ]] || die "could not resolve the latest GitHub Release for ${OP_GH_REPO}"
    echo "$ref"
}

resolve_branch_sha() {
    local branch="$1"
    if [[ -n "${OP_TEST_BRANCH_SHA:-}" ]]; then
        echo "${OP_TEST_BRANCH_SHA}"
        return 0
    fi
    curl -fsSL "https://api.github.com/repos/${OP_GH_REPO}/commits/${branch}" 2>/dev/null \
        | grep -m1 '"sha"' | sed -E 's/.*"sha": *"([^"]+)".*/\1/'
}

# -----------------------------------------------------------------------------
# 2. Install tiers
# -----------------------------------------------------------------------------
TIER_LIST=(core curation segmenter vlm trainer cropwright)

tier_implies() {
    case "$1" in
        curation) echo "core" ;;
        segmenter) echo "core curation" ;;
        vlm) echo "core curation" ;;
        trainer) echo "core curation" ;;
        cropwright) echo "core curation" ;;
        *) echo "" ;;
    esac
}

# tiers_close_dependencies "t1,t2,..."
# Prints the closure (deduplicated, stable order over TIER_LIST) as a
# space-separated list.
tiers_close_dependencies() {
    local input="$1"
    local -A selected=()
    local IFS=','
    read -ra requested <<< "$input"
    unset IFS
    for t in "${requested[@]}"; do
        [[ -z "$t" ]] && continue
        selected["$t"]=1
        for dep in $(tier_implies "$t"); do
            selected["$dep"]=1
        done
    done
    local out=()
    for t in "${TIER_LIST[@]}"; do
        [[ -n "${selected[$t]:-}" ]] && out+=("$t")
    done
    echo "${out[*]}"
}

# tiers_validate "t1,t2,..."
# Fails (rc 1, message on stderr) on an unknown tier name.
tiers_validate() {
    local input="$1"
    local IFS=','
    read -ra requested <<< "$input"
    unset IFS
    for t in "${requested[@]}"; do
        [[ -z "$t" ]] && continue
        local known=0
        for known_tier in "${TIER_LIST[@]}"; do
            [[ "$t" == "$known_tier" ]] && known=1 && break
        done
        if [[ "$known" == 0 ]]; then
            log_error "unknown tier: ${t}"
            return 1
        fi
    done
    return 0
}

# -----------------------------------------------------------------------------
# 2.1 GPU detection and recommendation (pure functions, unit-tested)
# -----------------------------------------------------------------------------
# gpu_query
# Prints "index,name,memory.total,memory.used,compute_cap" per GPU, via
# nvidia-smi. Overridable with OP_TEST_GPU_CSV for tests.
gpu_query() {
    if [[ -n "${OP_TEST_GPU_CSV:-}" ]]; then
        printf '%s\n' "${OP_TEST_GPU_CSV}"
        return 0
    fi
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,compute_cap --format=csv,noheader 2>/dev/null
}

# gpu_free_gb TOTAL_MB USED_MB
gpu_free_gb() {
    awk -v t="$1" -v u="$2" 'BEGIN { printf "%d\n", (t - u) / 1024 }'
}

# recommend_plan_single_gpu FREE_GB
# Prints a plan as newline "key=value" pairs (section 2.1 rule 2). Also
# handles the 0-GPU case when called with FREE_GB=0 and OP_GPU_COUNT=0.
recommend_plan_single_gpu() {
    local free_gb="$1"
    local floor_gb
    floor_gb="$(vlm_catalog_floor_gb 2>/dev/null || echo 15)"

    if (( free_gb < 8 )); then
        echo "refuse=1"
        return 1
    fi

    echo "TRITON_GPU_ID=0"
    echo "API_GPU_ID=0"

    if (( free_gb < 16 )); then
        echo "tiers=core"
        echo "GPU_PROFILE=minimal"
        return 0
    fi

    if (( free_gb < 32 )); then
        echo "tiers=core,curation"
        echo "SEGMENTER_GPU_ID=0"
        local remaining=$(( free_gb - 8 ))
        if (( remaining >= floor_gb )); then
            echo "tiers=core,curation,segmenter,vlm"
            echo "VLM_GPU_ID=0"
        fi
        return 0
    fi

    echo "tiers=core,curation,segmenter,vlm,trainer"
    echo "SEGMENTER_GPU_ID=0"
    echo "VLM_GPU_ID=0"
    echo "TRAINER_GPU_ID=0"
    return 0
}

# -----------------------------------------------------------------------------
# 5.2 / 7 .env upsert -- never a raw sed -i with a secret in argv.
# -----------------------------------------------------------------------------
# ensure_env_permissions FILE
ensure_env_permissions() {
    local file="$1"
    chmod 600 "$file"
}

# upsert_env_var FILE KEY VALUE
# Pure-bash rewrite: read lines, replace an existing KEY=..., or append.
# Writes to a temp file in the same dir, chmod 600, then mv (atomic within
# the same filesystem). The value is passed as an argument, never
# interpolated into a sed/awk *program* string, so a token-shaped value
# can't break out of the replacement or leak through a process-list VALUE.
upsert_env_var() {
    local file="$1" key="$2" value="$3"
    local dir
    dir="$(dirname "$file")"
    local tmp
    tmp="$(mktemp "${dir}/.env.XXXXXX")"
    local found=0
    if [[ -f "$file" ]]; then
        while IFS= read -r line || [[ -n "$line" ]]; do
            if [[ "$line" == "${key}="* ]]; then
                printf '%s=%s\n' "$key" "$value" >> "$tmp"
                found=1
            else
                printf '%s\n' "$line" >> "$tmp"
            fi
        done < "$file"
    fi
    if [[ "$found" == 0 ]]; then
        printf '%s=%s\n' "$key" "$value" >> "$tmp"
    fi
    chmod 600 "$tmp"
    mv "$tmp" "$file"
}

# read_env_var FILE KEY
read_env_var() {
    local file="$1" key="$2"
    [[ -f "$file" ]] || return 1
    grep -E "^${key}=" "$file" | tail -n1 | cut -d= -f2-
}

# -----------------------------------------------------------------------------
# 4.4 Gated-model HuggingFace token -- never in argv, logs or `set -x`.
# -----------------------------------------------------------------------------
# store_hf_token ENV_FILE TOKEN
# The token is passed as a function argument (visible briefly in this
# process's own argv while the function call is active, same as any other
# shell function parameter) but never appears in a subprocess's argv (no
# `sed -i "s|...|$tok|"`, no external command receives it as a literal
# argument), matching the plan's "never in argv (ps)" requirement, which is
# about the *external process list* a `ps` from another terminal would see.
store_hf_token() {
    local env_file="$1" token="$2"
    ensure_env_permissions "$env_file"
    upsert_env_var "$env_file" "HF_TOKEN" "$token"
}

# verify_hf_token_access TOKEN REPO
# Verifies without leaking: writes the token to a mode-600 temp header
# file, curls with -H @file (never -H "Authorization: Bearer $tok", which
# would put it in this process's own argv and therefore ps output), then
# deletes the header file.
verify_hf_token_access() {
    local token="$1" repo="$2"
    local hdr
    hdr="$(mktemp)"
    trap 'rm -f "$hdr"' RETURN
    ( umask 077; printf 'Authorization: Bearer %s\n' "$token" > "$hdr" )
    curl -sf -H "@${hdr}" "https://huggingface.co/api/models/${repo}" >/dev/null
}

# read_hf_token_unattended
# HF_TOKEN_FILE (permissions checked <= 600) takes precedence over HF_TOKEN.
read_hf_token_unattended() {
    if [[ -n "${HF_TOKEN_FILE:-}" ]]; then
        [[ -f "$HF_TOKEN_FILE" ]] || { log_error "HF_TOKEN_FILE not found: ${HF_TOKEN_FILE}"; return 1; }
        local perms
        perms="$(stat -c '%a' "$HF_TOKEN_FILE" 2>/dev/null || stat -f '%Lp' "$HF_TOKEN_FILE")"
        if (( 10#$perms > 600 )); then
            log_error "HF_TOKEN_FILE permissions too open: ${perms} (must be <= 600)"
            return 1
        fi
        cat "$HF_TOKEN_FILE"
        return 0
    fi
    if [[ -n "${HF_TOKEN:-}" ]]; then
        echo "$HF_TOKEN"
        return 0
    fi
    return 1
}

# -----------------------------------------------------------------------------
# 5.6 Port-conflict detection
# -----------------------------------------------------------------------------
# port_in_use PORT
port_in_use() {
    local port="$1"
    if [[ -n "${OP_TEST_PORTS_IN_USE:-}" ]]; then
        for p in ${OP_TEST_PORTS_IN_USE}; do
            [[ "$p" == "$port" ]] && return 0
        done
        return 1
    fi
    if command -v ss >/dev/null 2>&1; then
        ss -ltnH "sport = :${port}" 2>/dev/null | grep -q . && return 0
    fi
    if command -v lsof >/dev/null 2>&1; then
        lsof -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1 && return 0
    fi
    (exec 3<>"/dev/tcp/127.0.0.1/${port}") 2>/dev/null && { exec 3>&-; return 0; }
    return 1
}

# next_free_port PORT
next_free_port() {
    local port="$1"
    while port_in_use "$port"; do
        port=$((port + 1))
    done
    echo "$port"
}

# -----------------------------------------------------------------------------
# 5.1 Project-name collision guard
# -----------------------------------------------------------------------------
# check_project_collision PROJECT INSTALL_DIR
# Refuses (rc 1) if a container exists under PROJECT whose recorded
# working_dir differs from INSTALL_DIR.
check_project_collision() {
    local project="$1" install_dir="$2"
    local rows
    if [[ -n "${OP_TEST_COMPOSE_PROJECTS:-}" ]]; then
        rows="${OP_TEST_COMPOSE_PROJECTS}"
    else
        rows="$(docker ps -a --filter "label=com.docker.compose.project=${project}" \
            --format '{{.Label "com.docker.compose.project.working_dir"}}' 2>/dev/null)"
    fi
    [[ -z "$rows" ]] && return 0
    while IFS= read -r wd; do
        [[ -z "$wd" ]] && continue
        if [[ "$wd" != "$install_dir" ]]; then
            log_error "project '${project}' already used by another directory: ${wd}"
            return 1
        fi
    done <<< "$rows"
    return 0
}

# -----------------------------------------------------------------------------
# 7. Security: bind address guard
# -----------------------------------------------------------------------------
# is_loopback_or_private ADDR
is_loopback_or_private() {
    local addr="$1"
    [[ "$addr" == "127.0.0.1" || "$addr" == "localhost" || "$addr" == "::1" ]] && return 0
    return 1
}

# require_bind_consent ADDR UNATTENDED
require_bind_consent() {
    local addr="$1" unattended="$2"
    is_loopback_or_private "$addr" && return 0
    if [[ "$unattended" == "1" ]]; then
        [[ "${OP_ALLOW_PUBLIC_BIND:-0}" == "1" ]] && return 0
        log_error "binding to ${addr} requires OP_ALLOW_PUBLIC_BIND=1 in unattended mode"
        return 1
    fi
    log_warn "Binding to ${addr} exposes every published port with NO authentication."
    log_warn "OpenSearch security is disabled. This bypasses ufw/firewalld. Put a"
    log_warn "reverse proxy with auth in front (see SECURITY.md)."
    read -r -p "Type 'expose' to continue: " reply
    [[ "$reply" == "expose" ]]
}

# -----------------------------------------------------------------------------
# 7. Security: external VLM consent
# -----------------------------------------------------------------------------
# is_private_url URL
is_private_url() {
    local url="$1"
    local host
    host="$(echo "$url" | sed -E 's#^[a-zA-Z]+://##; s#[:/].*$##')"
    case "$host" in
        127.*|localhost|host.docker.internal) return 0 ;;
        10.*|192.168.*) return 0 ;;
        172.1[6-9].*|172.2[0-9].*|172.3[01].*) return 0 ;;
        *) return 1 ;;
    esac
}

require_external_vlm_consent() {
    local url="$1" unattended="$2"
    is_private_url "$url" && return 0
    if [[ "$unattended" == "1" ]]; then
        [[ "${OP_ALLOW_EXTERNAL_VLM:-0}" == "1" ]] && return 0
        log_error "a --vlm-remote URL outside private address space requires OP_ALLOW_EXTERNAL_VLM=1 in unattended mode"
        return 1
    fi
    log_warn "This VLM endpoint is outside private address space: crops leave this host."
    read -r -p "Type 'external' to continue: " reply
    [[ "$reply" == "external" ]]
}

# -----------------------------------------------------------------------------
# 7. Cropwright bind override fallback -- until CROPWRIGHT_BIND_ADDRESS
# lands in a release listed in cropwright.lock, write a small override
# compose file that pins Cropwright's port to loopback the same way
# OP_BIND_ADDRESS does for every other service.
# -----------------------------------------------------------------------------
write_cropwright_bind_override() {
    local dir="$1" bind_addr="$2" port="$3"
    mkdir -p "$dir"
    cat > "${dir}/docker-compose.bind.yml" <<EOF
services:
  cropwright:
    ports: !override
      - "${bind_addr}:${port}:8080"
EOF
}

# -----------------------------------------------------------------------------
# 5.3 Uninstall guards
# -----------------------------------------------------------------------------
# require_purge_confirmation PROJECT UNATTENDED
require_purge_confirmation() {
    local project="$1" unattended="$2"
    if [[ "$unattended" == "1" ]]; then
        [[ "${OP_CONFIRM_PURGE:-}" == "$project" ]] && return 0
        log_error "unattended purge requires OP_CONFIRM_PURGE=${project}"
        return 1
    fi
    read -r -p "Type the project name (${project}) to confirm the purge: " reply
    [[ "$reply" == "$project" ]]
}

# purge_data_paths INSTALL_DIR SOURCE_ROOT_HOST
# Prints exactly what --purge-data would delete. Never includes
# SOURCE_ROOT_HOST when it points outside INSTALL_DIR (section 5.3).
purge_data_paths() {
    local install_dir="$1" source_root_host="${2:-}"
    for d in models pytorch_models cache data; do
        echo "${install_dir}/${d}"
    done
    if [[ -n "$source_root_host" ]]; then
        local resolved
        resolved="$(cd "$source_root_host" 2>/dev/null && pwd || echo "$source_root_host")"
        case "$resolved" in
            "${install_dir}"/*|"${install_dir}")
                echo "$resolved"
                ;;
            *)
                : # outside the install dir -- never listed, never touched
                ;;
        esac
    fi
}

# -----------------------------------------------------------------------------
# 6.4 --cpu
# -----------------------------------------------------------------------------
handle_cpu_mode() {
    local control_plane_only="$1"
    if [[ "$control_plane_only" != "1" ]]; then
        log_error "No usable GPU found. There is no CPU inference path today."
        log_error "Re-run with --control-plane-only to install OpenSearch, the API"
        log_error "and Cropwright without Triton (inference routes will 503)."
        return 4
    fi
    log_warn "control-plane-only install: no Triton, no exports, inference routes"
    log_warn "will return 503. This is NOT a functional inference install."
    echo "tiers=core-control-plane-only"
    return 0
}

# -----------------------------------------------------------------------------
# Flags / env (section 5.4)
# -----------------------------------------------------------------------------
usage() {
    cat <<'EOF'
Usage: setup-openprocessor.sh [options]

  --dir PATH              install directory (default ./openprocessor)
  --project NAME          compose project name
  --version vX.Y.Z        pinned release
  --branch REF            testing install
  --tiers LIST | --all    tiers to install
  --gpu-plan K=V,...      override GPU placement
  --profile NAME          minimal|standard|full
  --vlm-remote URL --vlm-model NAME [--vlm-key-file PATH]
  --vlm-model-id ID
  --bind ADDR             publish address (default 127.0.0.1)
  --port-base N
  --with-monitoring
  --sample-data
  --skip-models
  --no-start
  --unattended
  --dry-run
  --cpu [--control-plane-only]
  --repair | --rollback | --uninstall [--purge-volumes] [--purge-data] [--remove-images]
  --reset-hf-token
  -h | --help
EOF
}

parse_args() {
    OP_TIERS="${OP_TIERS:-}"
    OP_BIND_ADDRESS="${OP_BIND_ADDRESS:-127.0.0.1}"
    OP_UNATTENDED="${OP_UNATTENDED:-0}"
    OP_DRY_RUN="${OP_DRY_RUN:-0}"
    OP_FORCE_CPU="${OP_FORCE_CPU:-0}"
    OP_CONTROL_PLANE_ONLY="${OP_CONTROL_PLANE_ONLY:-0}"
    OP_INSTALL_DIR="${OP_INSTALL_DIR:-./openprocessor}"
    OP_PROJECT="${OP_PROJECT:-openprocessor}"
    OP_ACTION="install"

    [[ -t 0 && ! -e /dev/tty ]] && OP_UNATTENDED=1

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dir) OP_INSTALL_DIR="$2"; shift 2 ;;
            --project) OP_PROJECT="$2"; shift 2 ;;
            --version) OP_VERSION="$2"; shift 2 ;;
            --branch) OP_BRANCH="$2"; shift 2 ;;
            --tiers) OP_TIERS="$2"; shift 2 ;;
            --all) OP_TIERS="core,curation,segmenter,vlm,trainer,cropwright"; shift ;;
            --gpu-plan) OP_GPU_PLAN="$2"; shift 2 ;;
            --profile) GPU_PROFILE="$2"; shift 2 ;;
            --vlm-remote) OP_VLM_URL="$2"; shift 2 ;;
            --vlm-model) OP_VLM_MODEL="$2"; shift 2 ;;
            --vlm-key-file) OP_VLM_KEY_FILE="$2"; shift 2 ;;
            --vlm-model-id) OP_VLM_CATALOG_ID="$2"; shift 2 ;;
            --bind) OP_BIND_ADDRESS="$2"; shift 2 ;;
            --port-base) OP_PORT_BASE="$2"; shift 2 ;;
            --with-monitoring) OP_WITH_MONITORING=1; shift ;;
            --sample-data) OP_SAMPLE_DATA=1; shift ;;
            --skip-models) OP_SKIP_MODELS=1; shift ;;
            --no-start) OP_NO_START=1; shift ;;
            --unattended) OP_UNATTENDED=1; shift ;;
            --dry-run) OP_DRY_RUN=1; shift ;;
            --cpu) OP_FORCE_CPU=1; shift ;;
            --control-plane-only) OP_CONTROL_PLANE_ONLY=1; shift ;;
            --force) OP_FORCE=1; shift ;;
            --repair) OP_ACTION="repair"; shift ;;
            --rollback) OP_ACTION="rollback"; shift ;;
            --uninstall) OP_ACTION="uninstall"; shift ;;
            --purge-volumes) OP_PURGE_VOLUMES=1; shift ;;
            --purge-data) OP_PURGE_DATA=1; shift ;;
            --remove-images) OP_REMOVE_IMAGES=1; shift ;;
            --reset-hf-token) OP_RESET_HF_TOKEN=1; shift ;;
            -h|--help) usage; exit 0 ;;
            *)
                log_error "unknown flag: $1"
                usage
                exit 2
                ;;
        esac
    done
    # Every flag becomes an exported env var: model_setup.sh, vlm_catalog.sh
    # and the openprocessor CLI all read these rather than re-parsing argv.
    export OP_INSTALL_DIR OP_PROJECT OP_BIND_ADDRESS OP_UNATTENDED OP_DRY_RUN OP_FORCE_CPU
    export OP_CONTROL_PLANE_ONLY OP_ACTION
    export OP_VERSION="${OP_VERSION:-}" OP_BRANCH="${OP_BRANCH:-}" OP_TIERS="${OP_TIERS:-}"
    export OP_GPU_PLAN="${OP_GPU_PLAN:-}" GPU_PROFILE="${GPU_PROFILE:-}"
    export OP_VLM_URL="${OP_VLM_URL:-}" OP_VLM_MODEL="${OP_VLM_MODEL:-}"
    export OP_VLM_KEY_FILE="${OP_VLM_KEY_FILE:-}" OP_VLM_CATALOG_ID="${OP_VLM_CATALOG_ID:-}"
    export OP_PORT_BASE="${OP_PORT_BASE:-}" OP_WITH_MONITORING="${OP_WITH_MONITORING:-0}"
    export OP_SAMPLE_DATA="${OP_SAMPLE_DATA:-0}" OP_SKIP_MODELS="${OP_SKIP_MODELS:-0}"
    export OP_NO_START="${OP_NO_START:-0}" OP_FORCE="${OP_FORCE:-0}"
    export OP_PURGE_VOLUMES="${OP_PURGE_VOLUMES:-0}" OP_PURGE_DATA="${OP_PURGE_DATA:-0}"
    export OP_REMOVE_IMAGES="${OP_REMOVE_IMAGES:-0}" OP_RESET_HF_TOKEN="${OP_RESET_HF_TOKEN:-0}"
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
main() {
    parse_args "$@"
    OP_DIR="$(mkdir -p "$OP_INSTALL_DIR" && cd "$OP_INSTALL_DIR" && pwd)"
    export OP_DIR

    if [[ "${OP_FORCE_CPU}" == "1" ]]; then
        handle_cpu_mode "${OP_CONTROL_PLANE_ONLY}"
        local rc=$?
        (( rc != 0 )) && exit "$rc"
    fi

    require_bind_consent "$OP_BIND_ADDRESS" "$OP_UNATTENDED" || die "bind not confirmed"

    if [[ -n "${OP_TIERS}" ]]; then
        tiers_validate "$OP_TIERS" || exit 1
        OP_TIERS="$(tiers_close_dependencies "$OP_TIERS")"
    fi

    if ! check_project_collision "$OP_PROJECT" "$OP_DIR"; then
        [[ "$OP_UNATTENDED" == "1" ]] && exit 3
        die "project collision" 3
    fi

    log_info "install dir: ${OP_DIR}"
    log_info "project: ${OP_PROJECT}"
    log_info "tiers: ${OP_TIERS:-<none selected>}"
    log_info "dry-run: ${OP_DRY_RUN}"

    if [[ "${OP_ACTION}" == "uninstall" ]]; then
        if [[ "${OP_PURGE_VOLUMES:-0}" == "1" || "${OP_PURGE_DATA:-0}" == "1" ]]; then
            require_purge_confirmation "$OP_PROJECT" "$OP_UNATTENDED" || die "purge not confirmed"
        fi
        dc down --remove-orphans
        [[ "${OP_PURGE_VOLUMES:-0}" == "1" ]] && dc down -v
        if [[ "${OP_PURGE_DATA:-0}" == "1" ]]; then
            log_info "would remove: $(purge_data_paths "$OP_DIR" "${OP_SOURCE_ROOT_HOST:-}")"
        fi
        log_success "uninstall complete"
        return 0
    fi

    log_success "dry-run installer walkthrough complete"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
