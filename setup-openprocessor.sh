#!/bin/bash
# =============================================================================
# setup-openprocessor.sh - one-line OpenProcessor installer
# =============================================================================
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/${OP_GH_REPO}/main/setup-openprocessor.sh | bash
#   curl -fsSL .../setup-openprocessor.sh | bash -s -- --tiers core,curation --unattended
#
# Installs a pinned OpenProcessor release into ./openprocessor/ with no git
# clone, no local image build and no host Python. Design:
# docs/design/openprocessor_internal/one_line_installer_plan.md
#
# The whole script is one function (__op_define) that only defines things,
# followed by a single line that calls it and then main. A truncated
# `curl | bash` download is a syntax error before anything runs (plan 3.3).
#
# When read from a pipe, this copy does one job: resolve the release,
# download that release's setup-openprocessor.sh plus SHA256SUMS, verify
# the checksum and exec the verified copy with the same arguments.
#
# Env vars (all optional; a flag wins over its env var):
#   OP_GH_REPO             GitHub org/repo to install from (default davidamacey/OpenProcessor)
#   OP_GH_DEFAULT_REF      default branch for --branch (default main)
#   CW_GH_REPO             Cropwright GitHub org/repo
#   OP_IMAGE_NAMESPACE     Docker Hub namespace (default davidamacey)
#   OP_DOCS_URL            docs-site URL, printed in the summary if set
#   OP_ARTIFACT_BASE_URL   release-asset base (https only; <base>/<ref>/<asset>)
#   OP_RAW_BASE_URL        raw-file base (https only; <base>/<ref>/<path>)
#   CW_ARTIFACT_BASE_URL / CW_RAW_BASE_URL   the same for Cropwright
#   OP_INSTALL_DIR / --dir           install directory (default ./openprocessor, or . when run inside an install dir)
#   OP_PROJECT / --project           compose project name + container prefix
#   OP_VERSION / --version           pinned release tag (vX.Y.Z)
#   OP_BRANCH / --branch             testing install from a branch head
#   OP_RELEASE_DIR / --release-dir   read the release assets from a local
#                                    directory (scripts/release/build_deploy_bundle.sh
#                                    output) instead of GitHub; still verified
#   OP_IMAGE_TAG / --image-tag       run images by tag instead of images.lock
#                                    digests (local-only test builds);
#                                    OP_IMAGE_REPO picks the repo prefix
#   OP_TIERS / --tiers               comma list of tiers, or --all
#   OP_GPU_PLAN / --gpu-plan         triton=1,segmenter=0,vlm=2,trainer=0
#   GPU_PROFILE / --profile          minimal|standard|full
#   OP_VLM_URL / --vlm-remote, OP_VLM_MODEL / --vlm-model,
#   OP_VLM_KEY_FILE / --vlm-key-file remote OpenAI-compatible VLM
#   OP_VLM_CATALOG_ID / --vlm-model-id   explicit local VLM catalog entry
#   OP_BIND_ADDRESS / --bind         publish address (default 127.0.0.1)
#   OP_PORT_BASE / --port-base       shift the whole 46xx block to N..N+12
#   OP_WITH_MONITORING / --with-monitoring, OP_SAMPLE_DATA / --sample-data
#   OP_UNATTENDED / --unattended     no prompts (auto when /dev/tty is unusable)
#   OP_DRY_RUN / --dry-run           print every mutating command, run none
#   OP_FORCE_CPU / --cpu             no GPU; see --control-plane-only
#   OP_ALLOW_PUBLIC_BIND=1           unattended consent for a non-loopback --bind
#   OP_ALLOW_EXTERNAL_VLM=1          unattended consent for a non-private --vlm-remote
#   --yes                            consent to a destructive step (uninstall, rollback,
#                                    upgrade) without a prompt; --unattended implies it.
#                                    A missing terminal alone is never consent.
#   --force-existing-dir             install into a non-empty dir this installer did not
#                                    create (its files are backed up first)
#   OP_CONFIRM_PURGE=<project>       unattended consent for --purge-volumes/--purge-data
#   OP_CONFIRM_PURGE_SECRETS=<project>  unattended consent to also delete secrets/
#   HF_TOKEN_FILE / HF_TOKEN         HuggingFace token for gated tiers (segmenter)
#   OP_HEALTH_TIMEOUT                cap (seconds) on every health wait
#
# Exit codes:
#   0 ok | 1 failure | 2 usage | 3 project or container-name collision
#   4 no usable GPU / GPU plan refused | 5 Docker unreachable
#   6 HuggingFace token missing or rejected for a gated tier
#   7 artifact, checksum or image-digest verification failed
#   8 health check or model setup failed | 9 consent not given
# =============================================================================

__op_define() {

OP_GH_REPO="${OP_GH_REPO:-davidamacey/OpenProcessor}"
OP_GH_DEFAULT_REF="${OP_GH_DEFAULT_REF:-main}"
CW_GH_REPO="${CW_GH_REPO:-attevon-llc/cropwright}"
OP_IMAGE_NAMESPACE="${OP_IMAGE_NAMESPACE:-davidamacey}"
OP_DOCS_URL="${OP_DOCS_URL:-}"

SCRIPT_VERSION="0.4.0"

EXIT_USAGE=2
EXIT_COLLISION=3
EXIT_GPU=4
EXIT_DOCKER=5
EXIT_TOKEN=6
EXIT_VERIFY=7
EXIT_HEALTH=8
EXIT_CONSENT=9

TIER_LIST=(core curation segmenter vlm trainer cropwright)
COMPOSE_MIN_OVERRIDE="2.24.4"

# -----------------------------------------------------------------------------
# Logging (self-contained: nothing is sourced before the release is verified)
# -----------------------------------------------------------------------------
log_info()    { echo "[INFO] $*"; }
log_warn()    { echo "[WARN] $*" >&2; }
log_error()   { echo "[ERROR] $*" >&2; }
log_success() { echo "[OK] $*"; }
log_step()    { echo ""; echo "=== $* ==="; }

die() {
    log_error "$1"
    exit "${2:-1}"
}

_truthy() {
    case "${1,,}" in
        1|true|yes|y|on) return 0 ;;
        *) return 1 ;;
    esac
}

# Same masking rules as scripts/lib/model_setup.sh (plan section 7).
_op_redact() {
    sed -u -E \
        -e 's/hf_[A-Za-z0-9]{10,}/hf_***REDACTED***/g' \
        -e 's/(Bearer)[[:space:]]+[^[:space:]"]+/\1 ***REDACTED***/g' \
        -e 's/(^|[^A-Za-z0-9_])([A-Z0-9_]*(_API_KEY|_TOKEN|_SECRET|_PASSWORD))=[^[:space:]]*/\1\2=***REDACTED***/g'
}

# -----------------------------------------------------------------------------
# 5.1 Compose invocation rule: every compose call goes through dc()/dc_cw().
# -----------------------------------------------------------------------------
# _dc_is_readonly ARGS... -> 0 if the compose subcommand never changes state
_dc_is_readonly() {
    while (( $# > 0 )); do
        case "$1" in
            --profile) shift 2 ;;
            -*) shift ;;
            config|ps|version|images|ls|logs|port|top) return 0 ;;
            *) return 1 ;;
        esac
    done
    return 1
}

# Shell-level COMPOSE_* variables would silently override the install's
# .env (project name drives container_name; profiles pick services), so
# dc() clears them and pins COMPOSE_PROJECT_NAME to the project.
dc() {
    local project="${OP_PROJECT:?OP_PROJECT not set}" dir="${OP_DIR:?OP_DIR not set}"
    local -a cmd=(docker compose -p "$project" --env-file "${dir}/.env"
        --project-directory "$dir" -f "${dir}/docker-compose.yml")
    if [[ -f "${dir}/docker-compose.cpu.yml" ]]; then
        cmd+=(-f "${dir}/docker-compose.cpu.yml")
    fi
    cmd+=("$@")
    if [[ "${OP_DRY_RUN:-0}" == "1" ]] && ! _dc_is_readonly "$@"; then
        echo "DRY: ${cmd[*]//"$dir"/"${OP_REAL_DIR:-$dir}"}"
        return 0
    fi
    env -u COMPOSE_PROFILES -u COMPOSE_FILE -u COMPOSE_ENV_FILES \
        COMPOSE_PROJECT_NAME="$project" "${cmd[@]}"
}

dc_cw() {
    local project="${OP_PROJECT:?OP_PROJECT not set}-cw" dir="${OP_DIR:?OP_DIR not set}/cropwright"
    local -a cmd=(docker compose -p "$project" --env-file "${dir}/.env"
        --project-directory "$dir" -f "${dir}/docker-compose.yml" "$@")
    if [[ "${OP_DRY_RUN:-0}" == "1" ]] && ! _dc_is_readonly "$@"; then
        echo "DRY: ${cmd[*]//"${OP_DIR}"/"${OP_REAL_DIR:-$OP_DIR}"}"
        return 0
    fi
    env -u COMPOSE_PROFILES -u COMPOSE_FILE -u COMPOSE_ENV_FILES -u CROPWRIGHT_BIND_ADDRESS \
        -u CROPWRIGHT_IMAGE -u CROPWRIGHT_PORT COMPOSE_PROJECT_NAME="$project" "${cmd[@]}"
}

# docker_mut ARGS... -- a state-changing plain docker call (pull/run/rmi)
docker_mut() {
    if [[ "${OP_DRY_RUN:-0}" == "1" ]]; then
        local line="docker $*"
        if [[ -n "${OP_DIR:-}" && -n "${OP_REAL_DIR:-}" ]]; then
            line="${line//"$OP_DIR"/"$OP_REAL_DIR"}"
        fi
        echo "DRY: ${line}"
        return 0
    fi
    docker "$@"
}

# -----------------------------------------------------------------------------
# Prompts: always /dev/tty, because under `curl | bash` stdin is the script.
# -----------------------------------------------------------------------------
tty_usable() {
    { : </dev/tty; } 2>/dev/null
}

# prompt_line VARNAME TEXT HINT
# Unattended or no tty: fails with HINT (the flag/env that answers it).
prompt_line() {
    local __var="$1" text="$2" hint="$3" __reply=""
    if [[ "$OP_UNATTENDED" == "1" ]] || ! tty_usable; then
        die "a prompt needs an answer but there is no terminal (unattended): ${hint}" "$EXIT_CONSENT"
    fi
    printf '%s' "$text" >/dev/tty
    IFS= read -r __reply </dev/tty || __reply=""
    printf -v "$__var" '%s' "$__reply"
}

# prompt_secret VARNAME TEXT HINT -- no echo, never logged
prompt_secret() {
    local __var="$1" text="$2" hint="$3" __reply=""
    if [[ "$OP_UNATTENDED" == "1" ]] || ! tty_usable; then
        die "a secret prompt needs an answer but there is no terminal: ${hint}" "$EXIT_TOKEN"
    fi
    printf '%s' "$text" >/dev/tty
    IFS= read -rs __reply </dev/tty || __reply=""
    printf '\n' >/dev/tty
    printf -v "$__var" '%s' "$__reply"
}

confirm_yes() {
    local reply=""
    prompt_line reply "$1 [Y/n]: " "$2"
    [[ -z "$reply" || "${reply,,}" == y || "${reply,,}" == yes ]]
}

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
validate_project_name() {
    [[ "$1" =~ ^[a-z0-9][a-z0-9_-]{0,62}$ ]]
}

validate_version() {
    [[ "$1" =~ ^v[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.]+)?$ ]]
}

validate_branch_name() {
    [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._/-]{0,199}$ && "$1" != *..* ]]
}

validate_https_base() {
    [[ "$1" =~ ^https://[A-Za-z0-9.-]+(:[0-9]+)?(/[A-Za-z0-9._~/-]*)?$ ]]
}

is_ipv4() {
    local ip="$1" o
    [[ "$ip" =~ ^(0|[1-9][0-9]{0,2})(\.(0|[1-9][0-9]{0,2})){3}$ ]] || return 1
    local IFS=.
    for o in $ip; do
        (( 10#$o <= 255 )) || return 1
    done
    return 0
}

# normalize_bind ADDR -> prints a Docker-usable host IP, or fails
normalize_bind() {
    local addr="$1"
    case "$addr" in
        localhost) echo "127.0.0.1"; return 0 ;;
        ::1|'[::1]'|::|'[::]')
            log_error "IPv6 bind addresses are not supported; use 127.0.0.1 or an IPv4 address"
            return 1 ;;
    esac
    if is_ipv4 "$addr"; then
        echo "$addr"
        return 0
    fi
    log_error "--bind must be an IPv4 address (got '${addr}')"
    return 1
}

is_loopback_ipv4() {
    is_ipv4 "$1" && [[ "$1" == 127.* ]]
}

# -----------------------------------------------------------------------------
# Downloads: https only, no protocol downgrade on redirect
# -----------------------------------------------------------------------------
_dl() {
    local url="$1" out="$2"
    # file:// only ever comes from --release-dir (never from a user URL).
    if [[ "$url" == file://* ]]; then
        [[ -n "${OP_RELEASE_DIR:-}" && "${url#file://}" == "${OP_RELEASE_DIR}"/* && -f "${url#file://}" ]] || return 1
        cp -- "${url#file://}" "$out"
        return
    fi
    curl --proto '=https' --proto-redir '=https' --tlsv1.2 -fsSL \
        --retry 2 --connect-timeout 20 -o "$out" "$url"
}

_sha256() {
    sha256sum "$1" | cut -d' ' -f1
}

# _sums_lookup SUMS_FILE PATH -> the recorded sha256 for PATH, or nothing
_sums_lookup() {
    awk -v p="$2" '($2 == p || $2 == "*" p) { print $1; exit }' "$1"
}

# verify_against_sums SUMS FILE NAME -> 0 when FILE's sha256 is NAME's entry
verify_against_sums() {
    local sums="$1" file="$2" name="$3" want got
    want="$(_sums_lookup "$sums" "$name")"
    if [[ ! "$want" =~ ^[0-9a-f]{64}$ ]]; then
        log_error "SHA256SUMS has no entry for ${name}"
        return 1
    fi
    got="$(_sha256 "$file")"
    if [[ "$got" != "$want" ]]; then
        log_error "checksum mismatch for ${name}"
        return 1
    fi
    return 0
}

_asset_url() {
    if [[ -n "${OP_RELEASE_DIR:-}" ]]; then
        printf 'file://%s/%s' "$OP_RELEASE_DIR" "$2"
        return
    fi
    printf '%s/%s/%s' "${OP_ARTIFACT_BASE_URL:-https://github.com/${OP_GH_REPO}/releases/download}" "$1" "$2"
}

_raw_url() {
    if [[ -n "${OP_RELEASE_DIR:-}" ]]; then
        printf 'file://%s/raw/%s' "$OP_RELEASE_DIR" "$2"
        return
    fi
    printf '%s/%s/%s' "${OP_RAW_BASE_URL:-https://raw.githubusercontent.com/${OP_GH_REPO}}" "$1" "$2"
}

# -----------------------------------------------------------------------------
# 3.2 Version pinning
# -----------------------------------------------------------------------------
# resolve_branch_sha BRANCH -> the full 40-hex head SHA, or fails
resolve_branch_sha() {
    local branch="$1" body sha
    body="$(curl --proto '=https' --proto-redir '=https' --tlsv1.2 -fsSL --retry 2 \
        "https://api.github.com/repos/${OP_GH_REPO}/commits/${branch}")" || return 1
    sha="$(printf '%s\n' "$body" | sed -n -E 's/^[[:space:]]*"sha":[[:space:]]*"([0-9a-f]{40})".*/\1/p' | head -n1)"
    [[ "$sha" =~ ^[0-9a-f]{40}$ ]] || return 1
    echo "$sha"
}

# resolve_install_ref -> sets RESOLVED_REF (git ref for downloads) and
# RESOLVED_MODE (release|branch). Never floats to a branch silently.
resolve_install_ref() {
    if [[ -n "${OP_VERSION:-}" ]]; then
        validate_version "$OP_VERSION" || die "--version must look like v1.2.3 (got '${OP_VERSION}')" "$EXIT_USAGE"
        RESOLVED_REF="$OP_VERSION"
        RESOLVED_MODE=release
        return 0
    fi
    if [[ -n "${OP_BRANCH:-}" ]]; then
        validate_branch_name "$OP_BRANCH" || die "invalid --branch '${OP_BRANCH}'" "$EXIT_USAGE"
        local sha
        sha="$(resolve_branch_sha "$OP_BRANCH")" || die "could not resolve the head commit of branch '${OP_BRANCH}'"
        RESOLVED_REF="$sha"
        RESOLVED_MODE=branch
        return 0
    fi
    local attempt=1 body ref=""
    while (( attempt <= 3 )); do
        if body="$(curl --proto '=https' --proto-redir '=https' --tlsv1.2 -fsSL \
                "https://api.github.com/repos/${OP_GH_REPO}/releases/latest" 2>/dev/null)"; then
            ref="$(printf '%s\n' "$body" | sed -n -E 's/.*"tag_name":[[:space:]]*"([^"]+)".*/\1/p' | head -n1)"
            [[ -n "$ref" ]] && break
        fi
        sleep $((attempt * 2))
        attempt=$((attempt + 1))
    done
    [[ -n "$ref" ]] || die "could not resolve the latest GitHub Release of ${OP_GH_REPO} (use --version)"
    validate_version "$ref" || die "latest release tag '${ref}' is not a vX.Y.Z version" "$EXIT_VERIFY"
    RESOLVED_REF="$ref"
    RESOLVED_MODE=release
}

# -----------------------------------------------------------------------------
# 3.3 One-liner bootstrap
# -----------------------------------------------------------------------------
# needs_bootstrap SELF_PATH -> 0 when this copy was read from a pipe
needs_bootstrap() {
    local self="$1"
    [[ -z "$self" || "$self" == /dev/fd/* || "$self" == /proc/self/fd/* || "$self" == /dev/stdin ]]
}

bootstrap_reexec() {
    resolve_install_ref
    local tmp
    tmp="$(mktemp -d)"
    local script="${tmp}/setup-openprocessor.sh"
    if [[ "$RESOLVED_MODE" == "branch" ]]; then
        _dl "$(_raw_url "$RESOLVED_REF" setup-openprocessor.sh)" "$script" \
            || { rm -rf "$tmp"; die "download of setup-openprocessor.sh at ${RESOLVED_REF} failed"; }
        log_warn "TESTING install from branch '${OP_BRANCH}' (${RESOLVED_REF:0:12}): not reproducible, no checksum"
    else
        if ! _dl "$(_asset_url "$RESOLVED_REF" setup-openprocessor.sh)" "$script" \
                || ! _dl "$(_asset_url "$RESOLVED_REF" SHA256SUMS)" "${tmp}/SHA256SUMS"; then
            rm -rf "$tmp"
            die "could not download the ${RESOLVED_REF} installer and its SHA256SUMS" "$EXIT_VERIFY"
        fi
        if ! verify_against_sums "${tmp}/SHA256SUMS" "$script" setup-openprocessor.sh; then
            rm -rf "$tmp"
            die "the downloaded ${RESOLVED_REF} installer failed checksum verification; nothing was run" "$EXIT_VERIFY"
        fi
    fi
    log_info "running the verified ${RESOLVED_REF} installer"
    # The child removes the temp dir on exit (exec replaces this process,
    # so no trap here could ever fire).
    OP_BOOTSTRAP_TMP="$tmp" OP_BOOTSTRAP_REF="$RESOLVED_REF" \
        OP_BOOTSTRAP_MODE="$RESOLVED_MODE" exec bash "$script" "$@"
}

# bootstrap_child_setup -- the OP_BOOTSTRAP_* hand-off is honoured only by
# the verified copy the bootstrap itself exec'd (this script's own path is
# <OP_BOOTSTRAP_TMP>/setup-openprocessor.sh, a mktemp dir we own). From
# anywhere else the variables are ignored, so the environment can neither
# point the cleanup at another path nor inject a release ref.
bootstrap_child_setup() {
    local tmp="${OP_BOOTSTRAP_TMP:-}" ref="${OP_BOOTSTRAP_REF:-}" mode="${OP_BOOTSTRAP_MODE:-}"
    unset OP_BOOTSTRAP_TMP OP_BOOTSTRAP_REF OP_BOOTSTRAP_MODE
    _OP_BOOT_TMP=""
    _OP_BOOT_REF=""
    _OP_BOOT_MODE=""
    [[ -n "$tmp" ]] || return 0
    if [[ "$tmp" =~ ^/[A-Za-z0-9._/-]*/tmp\.[A-Za-z0-9]{6,}$ && "$tmp" != *..* && -d "$tmp" && ! -L "$tmp" && -O "$tmp" \
            && "${_OP_SELF_PATH:-}" == "${tmp}/setup-openprocessor.sh" ]]; then
        _OP_BOOT_TMP="$tmp"
        if [[ "$mode" == release ]] && validate_version "$ref"; then
            _OP_BOOT_REF="$ref"; _OP_BOOT_MODE=release
        elif [[ "$mode" == branch && "$ref" =~ ^[0-9a-f]{40}$ ]]; then
            _OP_BOOT_REF="$ref"; _OP_BOOT_MODE=branch
        fi
    fi
}

_op_cleanup() {
    [[ -n "${_OP_BOOT_TMP:-}" ]] && rm -rf -- "$_OP_BOOT_TMP"
    [[ -n "${_OP_SCRATCH:-}" ]] && rm -rf -- "$_OP_SCRATCH"
    return 0
}

# -----------------------------------------------------------------------------
# 3.1 Release artifacts
# -----------------------------------------------------------------------------
# manifest_entries MANIFEST -> "path<TAB>flags" lines (comments dropped)
manifest_entries() {
    awk 'NF && $1 !~ /^#/ { print $1 "\t" (NF > 1 ? $2 : "") }' "$1"
}

# _tar_is_safe TARBALL -> no absolute paths, no "..", only files and dirs
_tar_is_safe() {
    local tb="$1" line name
    while IFS= read -r line; do
        case "${line:0:1}" in
            -|d) ;;
            *) log_error "release tarball contains a non-regular entry: ${line}"; return 1 ;;
        esac
    done < <(tar -tvzf "$tb")
    while IFS= read -r name; do
        if [[ "$name" == /* || "$name" == ".." || "$name" == ../* || "$name" == */../* || "$name" == */.. ]]; then
            log_error "release tarball contains an unsafe path: ${name}"
            return 1
        fi
    done < <(tar -tzf "$tb")
}

# fetch_release_artifacts REF MODE STAGING
# Release mode: SHA256SUMS first, then the deploy tarball (every file
# inside re-verified against SHA256SUMS, nothing extra allowed), falling
# back to raw files at the tag, each verified. Branch mode: raw files, no
# checksums (a loudly-labelled testing install).
fetch_release_artifacts() {
    local ref="$1" mode="$2" staging="$3" sums="" tb path flags f rel
    rm -rf "$staging"
    mkdir -p "$staging"

    if [[ "$mode" == "release" ]]; then
        sums="${staging}.SHA256SUMS"
        _dl "$(_asset_url "$ref" SHA256SUMS)" "$sums" \
            || die "could not download SHA256SUMS for ${ref}" "$EXIT_VERIFY"
        tb="${staging}.tar.gz"
        if _dl "$(_asset_url "$ref" "openprocessor-deploy-${ref}.tar.gz")" "$tb"; then
            verify_against_sums "$sums" "$tb" "openprocessor-deploy-${ref}.tar.gz" \
                || die "deploy tarball for ${ref} failed checksum verification; nothing was installed" "$EXIT_VERIFY"
            _tar_is_safe "$tb" || die "deploy tarball for ${ref} is unsafe; nothing was installed" "$EXIT_VERIFY"
            tar -xzf "$tb" -C "$staging" --no-same-owner --no-same-permissions
            rm -f "$tb"
            while IFS= read -r -d '' f; do
                rel="${f#"$staging"/}"
                verify_against_sums "$sums" "$f" "$rel" \
                    || die "file ${rel} in the ${ref} tarball failed verification; nothing was installed" "$EXIT_VERIFY"
            done < <(find "$staging" -type f -print0)
        else
            log_warn "deploy tarball not available for ${ref}; fetching verified raw files at the tag"
            _dl "$(_raw_url "$ref" release-manifest.txt)" "${staging}/release-manifest.txt" \
                || die "could not download release-manifest.txt at ${ref}" "$EXIT_VERIFY"
            verify_against_sums "$sums" "${staging}/release-manifest.txt" release-manifest.txt \
                || die "release-manifest.txt failed verification" "$EXIT_VERIFY"
            while IFS=$'\t' read -r path flags; do
                if [[ "$path" == *'**' ]]; then
                    local prefix="${path%%\*\*}"
                    while read -r _ rel; do
                        rel="${rel#\*}"
                        [[ "$rel" == "$prefix"* ]] || continue
                        mkdir -p "${staging}/$(dirname "$rel")"
                        _dl "$(_raw_url "$ref" "$rel")" "${staging}/${rel}" \
                            || die "download failed: ${rel}" "$EXIT_VERIFY"
                        verify_against_sums "$sums" "${staging}/${rel}" "$rel" || die "${rel} failed verification" "$EXIT_VERIFY"
                    done < "$sums"
                    continue
                fi
                [[ "$path" == "release-manifest.txt" ]] && continue
                mkdir -p "${staging}/$(dirname "$path")"
                if ! _dl "$(_raw_url "$ref" "$path")" "${staging}/${path}"; then
                    [[ "$flags" == *optional* ]] && continue
                    die "download failed: ${path}" "$EXIT_VERIFY"
                fi
                verify_against_sums "$sums" "${staging}/${path}" "$path" || die "${path} failed verification" "$EXIT_VERIFY"
            done < <(manifest_entries "${staging}/release-manifest.txt")
        fi
        rm -f "$sums"
    else
        _dl "$(_raw_url "$ref" release-manifest.txt)" "${staging}/release-manifest.txt" \
            || die "could not download release-manifest.txt at ${ref}"
        while IFS=$'\t' read -r path flags; do
            if [[ "$path" == *'**' ]]; then
                log_warn "branch install: optional '${path}' is not fetched (no checksum list to enumerate it)"
                continue
            fi
            [[ "$path" == "release-manifest.txt" ]] && continue
            mkdir -p "${staging}/$(dirname "$path")"
            if ! _dl "$(_raw_url "$ref" "$path")" "${staging}/${path}"; then
                [[ "$flags" == *optional* ]] && continue
                die "download failed: ${path}"
            fi
        done < <(manifest_entries "${staging}/release-manifest.txt")
    fi

    [[ -f "${staging}/release-manifest.txt" ]] || die "release has no release-manifest.txt" "$EXIT_VERIFY"
    while IFS=$'\t' read -r path flags; do
        [[ "$path" == *'**' || "$flags" == *optional* ]] && continue
        [[ -f "${staging}/${path}" ]] || die "release is missing ${path}" "$EXIT_VERIFY"
    done < <(manifest_entries "${staging}/release-manifest.txt")
}

# manifest_installed_files DIR -> every manifest file present under DIR
manifest_installed_files() {
    local dir="$1" path flags f
    [[ -f "${dir}/release-manifest.txt" ]] || return 0
    echo "release-manifest.txt"
    while IFS=$'\t' read -r path flags; do
        if [[ "$path" == *'**' ]]; then
            local prefix="${path%%\*\*}"
            [[ -d "${dir}/${prefix}" ]] || continue
            while IFS= read -r -d '' f; do
                echo "${f#"$dir"/}"
            done < <(find "${dir}/${prefix}" -type f -print0)
        elif [[ -f "${dir}/${path}" && "$path" != "release-manifest.txt" ]]; then
            echo "$path"
        fi
    done < <(manifest_entries "${dir}/release-manifest.txt")
}

# backup_install -> copies the current release files, .env and state into
# backups/<UTC timestamp>/ and prints that directory
backup_install() {
    local ts dest rel
    ts="$(date -u +%Y%m%dT%H%M%SZ)"
    dest="${OP_DIR}/backups/${ts}"
    mkdir -p "$dest"
    chmod 700 "${OP_DIR}/backups" "$dest"
    while IFS= read -r rel; do
        mkdir -p "${dest}/$(dirname "$rel")" && cp -p "${OP_DIR}/${rel}" "${dest}/${rel}" || return 1
    done < <(manifest_installed_files "$OP_DIR")
    for rel in .env .install/state.json .install/managed.env .install/groups.tsv; do
        if [[ -f "${OP_DIR}/${rel}" ]]; then
            mkdir -p "${dest}/$(dirname "$rel")" && cp -p "${OP_DIR}/${rel}" "${dest}/${rel}" || return 1
        fi
    done
    echo "$dest"
}

# install_staged STAGING -> copy verified files into OP_DIR (preserve-flagged
# files are never overwritten; .env is never part of a release)
install_staged() {
    local staging="$1" path flags f rel
    local -A flag_of=()
    while IFS=$'\t' read -r path flags; do
        flag_of["$path"]="$flags"
    done < <(manifest_entries "${staging}/release-manifest.txt")
    while IFS= read -r -d '' f; do
        rel="${f#"$staging"/}"
        [[ "$rel" == ".env" ]] && continue
        if [[ "${flag_of[$rel]:-}" == *preserve* && -e "${OP_DIR}/${rel}" ]]; then
            continue
        fi
        mkdir -p "${OP_DIR}/$(dirname "$rel")"
        cp -f "$f" "${OP_DIR}/${rel}"
        # Release files are not secret, and several are bind-mounted into
        # containers running as other users (Prometheus, Grafana, Loki).
        if [[ "${flag_of[$rel]:-}" == *exec* ]]; then
            chmod 755 "${OP_DIR}/${rel}"
        else
            chmod 644 "${OP_DIR}/${rel}"
        fi
    done < <(find "$staging" -type f -print0)
    rm -rf "$staging"
}

# -----------------------------------------------------------------------------
# 2. Install tiers
# -----------------------------------------------------------------------------
tier_implies() {
    case "$1" in
        curation) echo "core" ;;
        segmenter|vlm|trainer|cropwright) echo "core curation" ;;
        *) echo "" ;;
    esac
}

# tiers_close_dependencies "t1,t2" -> closure in TIER_LIST order, space-separated
tiers_close_dependencies() {
    local -A selected=()
    local -a requested out=()
    local t dep
    IFS=',' read -ra requested <<< "$1"
    for t in "${requested[@]}"; do
        [[ -z "$t" ]] && continue
        selected["$t"]=1
        for dep in $(tier_implies "$t"); do
            selected["$dep"]=1
        done
    done
    for t in "${TIER_LIST[@]}"; do
        [[ -n "${selected[$t]:-}" ]] && out+=("$t")
    done
    echo "${out[*]}"
}

tiers_validate() {
    local -a requested
    local t known
    IFS=',' read -ra requested <<< "$1"
    for t in "${requested[@]}"; do
        [[ -z "$t" ]] && continue
        for known in "${TIER_LIST[@]}"; do
            [[ "$t" == "$known" ]] && continue 2
        done
        log_error "unknown tier: ${t} (known: ${TIER_LIST[*]})"
        return 1
    done
    return 0
}

_has_tier() {
    [[ " ${SELECTED_TIERS} " == *" $1 "* ]]
}

# -----------------------------------------------------------------------------
# 2.1 GPU detection and recommendation
# -----------------------------------------------------------------------------
# gpu_query -> raw nvidia-smi rows (index, name, total MiB, used MiB, cc)
gpu_query() {
    command -v nvidia-smi >/dev/null 2>&1 || return 1
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,compute_cap \
        --format=csv,noheader,nounits 2>/dev/null
}

# gpu_normalize -> "index total_mib used_mib" per row; parses from the right
# so a GPU name containing commas cannot shift the numeric columns.
gpu_normalize() {
    awk -F',' 'NF >= 5 {
        idx = $1; total = $(NF-2); used = $(NF-1)
        gsub(/[^0-9]/, "", idx); gsub(/[^0-9]/, "", total); gsub(/[^0-9]/, "", used)
        if (idx != "" && total != "" && used != "") print idx, total, used
    }'
}

# gpu_labels -> "0=NVIDIA RTX A6000,1=..." (names with commas are squashed)
gpu_labels() {
    awk -F',' 'NF >= 5 {
        idx = $1; gsub(/[^0-9]/, "", idx)
        name = $2; for (i = 3; i <= NF - 3; i++) name = name " " $i
        gsub(/^[ \t]+|[ \t]+$/, "", name); gsub(/[^A-Za-z0-9 ._()-]/, "", name)
        printf "%s%s=%s", (n++ ? "," : ""), idx, name
    } END { print "" }'
}

# gpu_own_usage -> "index mib" per GPU: VRAM held by this project's own
# containers, so a re-run does not count its own services as "other
# processes" (the plan would otherwise refuse or shrink what is installed).
gpu_own_usage() {
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    local cid pids="" apps idx
    for cid in $(docker ps -q --filter "label=com.docker.compose.project=${OP_PROJECT}" 2>/dev/null); do
        pids+=" $(docker top "$cid" -eo pid 2>/dev/null | awk 'NR > 1 { print $1 }' | tr '\n' ' ')"
    done
    [[ -n "${pids// /}" ]] || return 0
    apps="$(nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null || true)"
    idx="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null || true)"
    awk -F', *' -v pids="$pids" -v idxmap="$idx" '
        BEGIN {
            n = split(pids, a, " "); for (i = 1; i <= n; i++) mine[a[i]] = 1
            m = split(idxmap, rows, "\n")
            for (i = 1; i <= m; i++) { split(rows[i], f, ", *"); if (f[2] != "") byuuid[f[2]] = f[1] }
        }
        ($1 in mine) && ($2 in byuuid) { sum[byuuid[$2]] += $3 }
        END { for (g in sum) print g, sum[g] }' <<< "$apps"
}

# gpu_subtract_own GPUS OWN -- GPUS rows "index total used" minus OWN rows
# "index mib" (never below 0)
gpu_subtract_own() {
    awk 'NR == FNR { own[$1] += $2; next }
         { u = $3 - ($1 in own ? own[$1] : 0); if (u < 0) u = 0; print $1, $2, u }' \
        <(printf '%s\n' "$2") <(printf '%s\n' "$1")
}

# docker_runtime_has_nvidia -> 1 yes, 0 no, 2 could not determine
docker_runtime_has_nvidia() {
    local out
    if ! out="$(docker info --format '{{json .Runtimes}}' 2>/dev/null)" || [[ -z "$out" ]]; then
        echo 2
        return 0
    fi
    if [[ "$out" == *nvidia* ]]; then
        echo 1
    else
        echo 0
    fi
}

_triton_need_gb() {
    case "$1" in
        minimal) echo 8 ;;
        full) echo 17 ;;
        *) echo 12 ;;
    esac
}

# recommend_plan GPUS TIERS [force=0|1] [vlm_id=ID] [gpu_plan=k=v,...]
#                [profile=NAME] [remote=0|1]
# GPUS: "index total_mib used_mib" lines. TIERS: requested closure
# (space-separated) or "" to recommend. Pure: reads only the VLM catalog.
# Prints key=value lines; every key at most once, "warn=" may repeat.
# Returns 1 (with refuse=) when the plan is impossible.
recommend_plan() {
    local gpus="$1" req_tiers="$2"
    shift 2
    local force=0 vlm_id="" gpu_plan="" profile_ovr="" remote=0 kv
    for kv in "$@"; do
        case "$kv" in
            force=*) force="${kv#force=}" ;;
            vlm_id=*) vlm_id="${kv#vlm_id=}" ;;
            gpu_plan=*) gpu_plan="${kv#gpu_plan=}" ;;
            profile=*) profile_ovr="${kv#profile=}" ;;
            remote=*) remote="${kv#remote=}" ;;
        esac
    done

    local -a ids=() tot=() free=() warns=()
    local -A pos=()
    local i t u n=0
    while read -r i t u; do
        [[ -z "${i:-}" ]] && continue
        ids+=("$i"); tot+=($(( (t + 512) / 1024 ))); free+=($(( (t - u) / 1024 )))
        pos["$i"]=$n
        if (( t > 0 && u * 100 / t > 10 )); then
            warns+=("GPU ${i}: $(( u * 100 / t ))% of its VRAM is already in use by other processes")
        fi
        n=$((n + 1))
    done <<< "$gpus"

    local tri="" vlm="" seg="" trn="" profile="" inst=1 single=0
    local -a usable=()
    for (( i = 0; i < n; i++ )); do
        if (( free[i] >= 8 )); then usable+=("$i"); fi
    done

    if (( n == 0 )); then
        echo "gpu_count=0"
        echo "refuse=no NVIDIA GPU detected"
        return 1
    fi
    if (( ${#usable[@]} == 0 )); then
        if [[ "$force" == 1 ]]; then
            local best=0
            for (( i = 1; i < n; i++ )); do (( free[i] > free[best] )) && best=$i; done
            usable=("$best")
            warns+=("no GPU has 8 GB free; --force installs core with GPU_PROFILE=minimal anyway")
        else
            echo "gpu_count=${n}"
            echo "refuse=no GPU has at least 8 GB of free VRAM (the core tier's floor); use --force to try anyway"
            for kv in "${warns[@]}"; do echo "warn=${kv}"; done
            return 1
        fi
    fi

    local k=${#usable[@]} a b c best
    if (( k == 1 )); then
        single=1
        tri=${usable[0]}; vlm=$tri; seg=$tri; trn=$tri
        local f=${free[tri]}
        if (( f < 16 )); then
            profile=minimal
        else
            profile=standard
            (( f >= 24 )) && inst=2
        fi
    elif (( k == 2 )); then
        a=${usable[0]}; b=${usable[1]}
        if (( free[a] > free[b] )); then c=$a; a=$b; b=$c; fi
        if (( free[a] >= 40 && free[b] >= 40 )); then
            tri=$a; seg=$a; trn=$a; vlm=$b; inst=2
        else
            tri=$a; vlm=$b; seg=$b
            (( free[b] >= 24 )) && inst=2
            if (( free[a] - $(_triton_need_gb standard) - 2 >= 16 )); then
                trn=$a
            else
                trn=$b
            fi
        fi
        if (( free[tri] >= 24 )); then profile=full
        elif (( free[tri] >= 12 )); then profile=standard
        else profile=minimal; fi
    else
        # Triton: the smallest card that is >= 12 GB (fixed, latency-bound).
        tri=""
        for i in "${usable[@]}"; do
            (( tot[i] >= 12 )) || continue
            if [[ -z "$tri" ]] || (( tot[i] < tot[tri] )); then tri=$i; fi
        done
        if [[ -z "$tri" ]]; then
            tri=${usable[0]}
            for i in "${usable[@]}"; do (( tot[i] < tot[tri] )) && tri=$i; done
        fi
        # VLM: the largest remaining card (ties: more free VRAM wins).
        vlm=""
        for i in "${usable[@]}"; do
            (( i == tri )) && continue
            if [[ -z "$vlm" ]] || (( tot[i] > tot[vlm] )) || (( tot[i] == tot[vlm] && free[i] > free[vlm] )); then
                vlm=$i
            fi
        done
        # Segmenter + trainer + evaluator: the remaining card with most free.
        seg=""
        for i in "${usable[@]}"; do
            (( i == tri || i == vlm )) && continue
            if [[ -z "$seg" ]] || (( free[i] > free[seg] )); then seg=$i; fi
        done
        trn=$seg
        (( free[seg] >= 24 )) && inst=2
        if (( free[tri] >= 24 )); then profile=full
        elif (( free[tri] >= 12 )); then profile=standard
        else profile=minimal; fi
    fi

    # Explicit --gpu-plan wins over the automatic placement.
    if [[ -n "$gpu_plan" ]]; then
        local -a pairs
        local key val
        IFS=',' read -ra pairs <<< "$gpu_plan"
        for kv in "${pairs[@]}"; do
            key="${kv%%=*}"; val="${kv#*=}"
            if [[ -z "${pos[$val]+x}" ]]; then
                echo "gpu_count=${n}"
                echo "refuse=--gpu-plan ${kv}: no GPU with index ${val}"
                return 1
            fi
            case "$key" in
                triton) tri=${pos[$val]} ;;
                segmenter) seg=${pos[$val]} ;;
                vlm) vlm=${pos[$val]} ;;
                trainer|evaluator) trn=${pos[$val]} ;;
                *)
                    echo "gpu_count=${n}"
                    echo "refuse=--gpu-plan: unknown key '${key}' (triton, segmenter, vlm, trainer)"
                    return 1 ;;
            esac
        done
        single=0
        if (( tri == vlm && vlm == seg )); then single=1; fi
    fi
    [[ -n "$profile_ovr" ]] && profile="$profile_ovr"

    local tn pe_need=2
    tn=$(_triton_need_gb "$profile")

    # _avail_on_vlm INST -> free VRAM left on the VLM card for the VLM
    local vlm_avail
    _plan_vlm_avail() {
        local s=$1 x=${free[vlm]}
        (( tri == vlm )) && x=$(( x - tn - pe_need ))
        (( seg == vlm )) && x=$(( x - 2 * s ))
        echo "$x"
    }

    # Which tiers the hardware allows (and why not).
    local -A allowed=() why=()
    local ftri=${free[tri]}
    allowed[core]=1
    if (( ftri - tn >= pe_need || tri != seg )); then allowed[curation]=1; else why[curation]="not enough VRAM for the PE encoder next to Triton"; fi
    if (( single == 1 && free[seg] < 12 )); then
        why[segmenter]="needs a card with at least 12 GB free (has ${free[seg]} GB)"
    else
        allowed[segmenter]=1
        (( single == 1 && free[seg] < 16 )) && inst=1
    fi
    if (( single == 1 && free[trn] < 16 )); then
        why[trainer]="needs at least 16 GB free for a training run"
    else
        allowed[trainer]=1
    fi
    allowed[cropwright]=1

    local wants_vlm=0
    if [[ -z "$req_tiers" ]]; then
        [[ "$remote" != 1 ]] && wants_vlm=1
    elif [[ " $req_tiers " == *" vlm "* ]]; then
        wants_vlm=1
    fi

    local vlm_pick="" vlm_status="none" line
    if (( wants_vlm == 1 )); then
        vlm_avail=$(_plan_vlm_avail "$inst")
        if [[ -n "$vlm_id" ]]; then
            local need st
            if ! need="$(vlm_catalog_field "$vlm_id" vram_gb)"; then
                echo "gpu_count=${n}"
                echo "refuse=--vlm-model-id '${vlm_id}' is not in the VLM catalog"
                return 1
            fi
            st="$(vlm_catalog_field "$vlm_id" status)"
            if (( need > vlm_avail )) && [[ "$force" != 1 ]]; then
                why[vlm]="--vlm-model-id ${vlm_id} needs ${need} GB, only ${vlm_avail} GB is available on GPU ${ids[vlm]} (use --force to try anyway)"
            else
                (( need > vlm_avail )) && warns+=("VLM ${vlm_id} needs ${need} GB but only ${vlm_avail} GB is free; forced")
                allowed[vlm]=1; vlm_pick="$vlm_id"; vlm_status="$st"
                [[ "$st" != tested ]] && warns+=("VLM ${vlm_id} is not yet verified with the prompt contract; run 'openprocessor vlm probe' after install and check the pairing warnings")
            fi
        else
            if ! line="$(pick_vlm "$vlm_avail")" && (( inst == 2 && seg == vlm )); then
                inst=1
                vlm_avail=$(_plan_vlm_avail "$inst")
                line="$(pick_vlm "$vlm_avail")" || line=""
            fi
            if [[ -n "${line:-}" ]]; then
                allowed[vlm]=1; vlm_pick="${line%%$'\t'*}"; vlm_status=tested
            else
                why[vlm]="no tested catalog entry fits the ${vlm_avail} GB left on GPU ${ids[vlm]} (tested floor $(vlm_catalog_floor_gb) GB); use --vlm-remote, or pick an unverified entry explicitly with --vlm-model-id"
            fi
        fi
    else
        vlm_avail=$(_plan_vlm_avail "$inst")
    fi

    local rec="core"
    [[ -n "${allowed[curation]:-}" ]] && rec+=",curation"
    if [[ -n "${allowed[segmenter]:-}" ]] && ! (( single == 1 && free[seg] < 16 )); then rec+=",segmenter"; fi
    if [[ -n "${allowed[vlm]:-}" && "$remote" != 1 && -z "$req_tiers" ]]; then rec+=",vlm"; fi

    local final="" tt
    if [[ -n "$req_tiers" ]]; then
        for tt in $req_tiers; do
            if [[ -z "${allowed[$tt]:-}" ]]; then
                if [[ "$force" == 1 && "$tt" != vlm ]]; then
                    warns+=("tier ${tt}: ${why[$tt]:-not supported on this hardware}; forced")
                else
                    echo "gpu_count=${n}"
                    echo "refuse=tier ${tt}: ${why[$tt]:-not supported on this hardware}"
                    for kv in "${warns[@]}"; do echo "warn=${kv}"; done
                    return 1
                fi
            fi
            final+="${final:+,}${tt}"
        done
    else
        final="$rec"
        [[ -n "${why[vlm]:-}" && "$remote" != 1 ]] && warns+=("local VLM not offered: ${why[vlm]}")
    fi

    if [[ ",$final," == *",vlm,"* ]] && (( single == 1 || tri == vlm )); then
        warns+=("the VLM shares GPU ${ids[vlm]} with Triton; it gets only the VRAM left after Triton and the segmenter")
    fi
    if [[ ",$final," == *",trainer,"* ]] && (( trn == vlm || trn == seg && single == 1 )); then
        warns+=("training shares its GPU with the VLM/segmenter: run 'openprocessor train-mode on' before a run and 'train-mode off' after")
    fi

    local util="" total_mib=$(( tot[vlm] * 1024 ))
    if [[ -n "$vlm_pick" ]]; then
        util="$(vlm_gpu_memory_utilization "$(vlm_catalog_field "$vlm_pick" vram_gb)" "${tot[vlm]}")"
    fi

    local allowed_ids
    allowed_ids="$(IFS=,; echo "${ids[*]}")"
    echo "gpu_count=${n}"
    echo "TRITON_GPU_ID=${ids[tri]}"
    echo "API_GPU_ID=${ids[tri]}"
    echo "SEGMENTER_GPU_ID=${ids[seg]}"
    echo "VLM_GPU_ID=${ids[vlm]}"
    echo "EVALUATOR_GPU_ID=${ids[trn]}"
    echo "OP_TRAIN_GPU_ORDER=${ids[trn]}"
    echo "OP_TRAIN_DEFAULT_GPUS=${ids[trn]}"
    echo "OP_GPU_ALLOWED_IDS=${allowed_ids}"
    echo "GPU_PROFILE=${profile}"
    echo "SEGMENTER_INSTANCES=${inst}"
    echo "vlm_available_gb=${vlm_avail}"
    echo "VLM_GPU_TOTAL_MIB=${total_mib}"
    echo "VLM_CATALOG_ID=${vlm_pick}"
    echo "vlm_status=${vlm_status}"
    echo "VLM_GPU_MEMORY_UTILIZATION=${util}"
    echo "recommended_tiers=${rec}"
    echo "tiers=${final}"
    for kv in "${warns[@]}"; do echo "warn=${kv}"; done
    return 0
}

# plan_get PLAN KEY -> value of KEY in recommend_plan output
plan_get() {
    printf '%s\n' "$1" | awk -v k="$2" 'index($0, k "=") == 1 { print substr($0, length(k) + 2); exit }'
}

# -----------------------------------------------------------------------------
# 5.2 / 7 .env handling: pure-bash rewrite, value never in any argv
# -----------------------------------------------------------------------------
# _env_write FILE KEY VARNAME -- writes KEY=<value of $VARNAME> atomically
_env_write() {
    local file="$1" key="$2" __name="$3"
    local value="${!__name}" dir tmp line found=0
    [[ "$key" =~ ^[A-Z_][A-Z0-9_]*$ ]] || { log_error "invalid .env key: ${key}"; return 1; }
    if [[ "$value" == *$'\n'* || "$value" == *$'\r'* ]]; then
        log_error "refusing a multi-line value for ${key}"
        return 1
    fi
    dir="$(dirname "$file")"
    tmp="$(mktemp "${dir}/.env.XXXXXX")" || return 1
    chmod 600 "$tmp"
    if ! {
        if [[ -f "$file" ]]; then
            while IFS= read -r line || [[ -n "$line" ]]; do
                if [[ "$line" == "${key}="* ]]; then
                    if (( found == 0 )); then
                        printf '%s=%s\n' "$key" "$value"
                    fi
                    found=1
                else
                    printf '%s\n' "$line"
                fi
            done < "$file"
        fi
        if (( found == 0 )); then
            printf '%s=%s\n' "$key" "$value"
        fi
    } > "$tmp" || ! mv -f "$tmp" "$file"; then
        rm -f "$tmp"
        return 1
    fi
}

# upsert_env_var FILE KEY VALUE (non-secret values only)
upsert_env_var() {
    local __val="$3"
    _env_write "$1" "$2" __val
}

# read_env_var FILE KEY -> last value of KEY (exact key match, no regex)
read_env_var() {
    [[ -f "$1" ]] || return 1
    awk -v k="$2" '{ i = index($0, "=") } i > 1 && substr($0, 1, i - 1) == k { v = substr($0, i + 1); f = 1 } END { if (f) print v; exit !f }' "$1"
}

# env_set KEY VALUE -- the installer owns this value (flags, pins, project)
env_set() {
    upsert_env_var "$ENV_FILE" "$1" "$2"
    upsert_env_var "${OP_DIR}/.install/managed.env" "$1" "$2"
}

# env_set_default KEY VALUE -- never overwrite a value the user set: the
# current value is kept unless it is unset, the template default, or the
# value this installer wrote last time.
env_set_default() {
    local key="$1" value="$2" cur tmpl managed
    cur="$(read_env_var "$ENV_FILE" "$key" || true)"
    tmpl="$(read_env_var "${OP_DIR}/env.template" "$key" || true)"
    managed="$(read_env_var "${OP_DIR}/.install/managed.env" "$key" || true)"
    if [[ -n "$cur" && "$cur" != "$tmpl" && "$cur" != "$managed" && "$cur" != "$value" ]]; then
        log_info "keeping your ${key}=${cur} (installer would set ${value})"
        return 0
    fi
    env_set "$key" "$value"
}

# env_create_or_merge -- create .env from env.template (600 before any
# value is written) or add keys that are new in env.template
env_create_or_merge() {
    local key line added=()
    if [[ ! -f "$ENV_FILE" ]]; then
        ( umask 077; install -m 600 "${OP_DIR}/env.template" "$ENV_FILE" )
        chmod 600 "$ENV_FILE"
        log_info "created ${ENV_FILE} (mode 600)"
        return 0
    fi
    chmod 600 "$ENV_FILE"
    while IFS= read -r line; do
        [[ "$line" =~ ^([A-Z_][A-Z0-9_]*)= ]] || continue
        key="${BASH_REMATCH[1]}"
        if ! read_env_var "$ENV_FILE" "$key" >/dev/null; then
            upsert_env_var "$ENV_FILE" "$key" "${line#*=}"
            added+=("$key")
        fi
    done < "${OP_DIR}/env.template"
    if (( ${#added[@]} > 0 )); then
        log_info "added new .env keys from env.template: ${added[*]}"
    fi
}

# -----------------------------------------------------------------------------
# 4.4 HuggingFace token: never in argv, logs or xtrace. The token lives in
# the global _OP_HF_TOKEN and is only ever handed around by variable name.
# -----------------------------------------------------------------------------
# check_secret_file PATH LABEL -> owned by us, no group/world bits, non-empty
check_secret_file() {
    local f="$1" label="$2" perms
    [[ -f "$f" ]] || { log_error "${label} not found: ${f}"; return 1; }
    [[ -O "$f" ]] || { log_error "${label} must be owned by you: ${f}"; return 1; }
    perms="$(stat -c '%a' "$f")" || return 1
    if (( (8#$perms & 8#077) != 0 )); then
        log_error "${label} permissions too open (${perms}); run: chmod 600 ${f}"
        return 1
    fi
    [[ -s "$f" ]] || { log_error "${label} is empty: ${f}"; return 1; }
    return 0
}

# _normalize_token VARNAME -- strip whitespace and a pasted HF_TOKEN= prefix
_normalize_token() {
    local __n="$1" __t="${!1}"
    __t="${__t#HF_TOKEN=}"
    __t="${__t//[[:space:]]/}"
    printf -v "$__n" '%s' "$__t"
}

# read_hf_token_unattended -> sets _OP_HF_TOKEN from HF_TOKEN_FILE or HF_TOKEN
read_hf_token_unattended() {
    _OP_HF_TOKEN=""
    if [[ -n "${HF_TOKEN_FILE:-}" ]]; then
        check_secret_file "$HF_TOKEN_FILE" HF_TOKEN_FILE || return 1
        _OP_HF_TOKEN="$(<"$HF_TOKEN_FILE")" || return 1
    elif [[ -n "${HF_TOKEN:-}" ]]; then
        _OP_HF_TOKEN="$HF_TOKEN"
    fi
    _normalize_token _OP_HF_TOKEN
    [[ -n "$_OP_HF_TOKEN" ]]
}

# hf_token_http_code REPO -> HTTP status for REPO with _OP_HF_TOKEN
# (000 = network error). The token goes to curl through a mode-600 header
# file (-H @file), never argv.
hf_token_http_code() {
    local repo="$1" hdr code
    hdr="$(umask 077; mktemp)"
    ( umask 077; printf 'Authorization: Bearer %s\n' "$_OP_HF_TOKEN" > "$hdr" )
    code="$(curl --proto '=https' --tlsv1.2 -sS -o /dev/null -w '%{http_code}' \
        --max-time 30 -H "@${hdr}" "https://huggingface.co/api/models/${repo}" 2>/dev/null)" || true
    rm -f "$hdr"
    [[ "$code" =~ ^[0-9]{3}$ ]] || code=000
    echo "$code"
}

# gated_repos -> HF repos the selected tiers need access to
gated_repos() {
    if _has_tier segmenter; then echo "facebook/sam3"; fi
    if _has_tier vlm && [[ -n "${VLM_PICK:-}" ]] \
            && [[ "$(vlm_catalog_field "$VLM_PICK" gated 2>/dev/null)" == "true" ]]; then
        vlm_catalog_field "$VLM_PICK" hf_repo
    fi
    return 0
}

# hf_token_available -> 0 when a token can be had without prompting
hf_token_available() {
    [[ -n "${HF_TOKEN_FILE:-}" || -n "${HF_TOKEN:-}" ]] && return 0
    [[ "$(read_env_var "$ENV_FILE" HF_TOKEN 2>/dev/null || true)" == hf_* ]]
}

# ensure_hf_token -- only when a selected tier is gated (plan 4.4)
ensure_hf_token() {
    local repos=() repo code attempt=0 existing
    mapfile -t repos < <(gated_repos)
    if (( ${#repos[@]} == 0 )); then
        log_info "no gated model in the selected tiers: no HuggingFace token needed"
        return 0
    fi
    log_step "HuggingFace token (gated: ${repos[*]})"

    _OP_HF_TOKEN=""
    existing="$(read_env_var "$ENV_FILE" HF_TOKEN || true)"
    if [[ "$OP_RESET_HF_TOKEN" != 1 && "$existing" == hf_* ]]; then
        _OP_HF_TOKEN="$existing"
    elif [[ -n "${HF_TOKEN_FILE:-}" || -n "${HF_TOKEN:-}" ]]; then
        # An exported token is used in every mode; the prompt is only for
        # when none was given.
        read_hf_token_unattended || die "HF_TOKEN_FILE/HF_TOKEN is set but unusable (see above)" "$EXIT_TOKEN"
    elif [[ "$OP_UNATTENDED" == 1 ]] || ! tty_usable; then
        die "the ${SELECTED_TIERS// /,} tiers need a HuggingFace token: set HF_TOKEN_FILE=/path (mode 600) or HF_TOKEN" "$EXIT_TOKEN"
    fi

    while :; do
        if [[ -z "$_OP_HF_TOKEN" ]]; then
            echo "A HuggingFace Read token is needed for: ${repos[*]}"
            echo "  1. Create a Read token: https://huggingface.co/settings/tokens"
            for repo in "${repos[@]}"; do echo "  2. Accept the licence at https://huggingface.co/${repo}"; done
            prompt_secret _OP_HF_TOKEN "HuggingFace token (input hidden): " "set HF_TOKEN_FILE"
            _normalize_token _OP_HF_TOKEN
            [[ -n "$_OP_HF_TOKEN" ]] || die "no token entered" "$EXIT_TOKEN"
        fi
        [[ "$_OP_HF_TOKEN" == hf_* ]] || log_warn "the token does not start with 'hf_'"
        local bad=""
        for repo in "${repos[@]}"; do
            code="$(hf_token_http_code "$repo")"
            case "$code" in
                200) ;;
                401|403) bad="$repo"; log_error "access to ${repo} denied (HTTP ${code}): accept the licence at https://huggingface.co/${repo} and check the token" ;;
                000) log_warn "could not reach huggingface.co to verify ${repo} (network error); continuing unverified" ;;
                *) log_warn "unexpected HTTP ${code} verifying ${repo}; continuing unverified" ;;
            esac
        done
        if [[ -z "$bad" ]]; then
            break
        fi
        attempt=$((attempt + 1))
        if [[ "$OP_UNATTENDED" == 1 ]] || ! tty_usable || (( attempt >= 3 )); then
            die "HuggingFace token cannot access ${bad}" "$EXIT_TOKEN"
        fi
        _OP_HF_TOKEN=""
    done

    _env_write "$ENV_FILE" HF_TOKEN _OP_HF_TOKEN
    _OP_HF_TOKEN=""
    log_success "HuggingFace token verified and stored in .env (mode 600)"
}

# -----------------------------------------------------------------------------
# 5.6 Ports
# -----------------------------------------------------------------------------
PORT_DEFAULTS=(
    "TRITON_HTTP_PORT 4600 core"
    "TRITON_GRPC_PORT 4601 core"
    "TRITON_METRICS_PORT 4602 core"
    "API_PORT 4603 base"
    "OPENSEARCH_PORT 4607 base"
    "MLFLOW_PORT 4609 trainer"
    "SEGMENTER_PORT 4611 segmenter"
    "VLM_PORT 4612 vlm"
    "PROMETHEUS_PORT 4604 monitoring"
    "GRAFANA_PORT 4605 monitoring"
    "LOKI_PORT 4606 monitoring"
    "OPENSEARCH_DASHBOARDS_PORT 4608 monitoring"
    "DCGM_PORT 4610 monitoring"
)

_host_ports_from_docker_ports() {
    grep -oE '[0-9.]*:[0-9]+->' | sed -E 's/.*:([0-9]+)->/\1/' | sort -u
}

# collect_docker_ports -> fills OWN_PORTS (this project) and DOCKER_PORTS
collect_docker_ports() {
    local out p
    OWN_PORTS=" "
    DOCKER_PORTS=" "
    out="$(docker ps --filter "label=com.docker.compose.project=${OP_PROJECT}" --format '{{.Ports}}')" \
        || die "docker ps failed while scanning ports" "$EXIT_DOCKER"
    while read -r p; do [[ -n "$p" ]] && OWN_PORTS+="${p} "; done < <(printf '%s\n' "$out" | _host_ports_from_docker_ports)
    out="$(docker ps -a --format '{{.Ports}}')" || die "docker ps failed while scanning ports" "$EXIT_DOCKER"
    while read -r p; do [[ -n "$p" ]] && DOCKER_PORTS+="${p} "; done < <(printf '%s\n' "$out" | _host_ports_from_docker_ports)
    return 0
}

# port_in_use PORT -- this project's own containers count as free
port_in_use() {
    local port="$1"
    [[ "${OWN_PORTS:- }" == *" ${port} "* ]] && return 1
    [[ "${DOCKER_PORTS:- }" == *" ${port} "* ]] && return 0
    if command -v ss >/dev/null 2>&1; then
        [[ -n "$(ss -ltnH "sport = :${port}" 2>/dev/null)" ]]
        return
    fi
    if command -v lsof >/dev/null 2>&1; then
        lsof -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1
        return
    fi
    (exec 3<>"/dev/tcp/127.0.0.1/${port}") 2>/dev/null
}

# next_free_port PORT TAKEN -> first free port >= PORT not in TAKEN
next_free_port() {
    local port="$1" taken="${2:- }"
    while (( port <= 65535 )); do
        if [[ "$taken" != *" ${port} "* ]] && ! port_in_use "$port"; then
            echo "$port"
            return 0
        fi
        port=$((port + 1))
    done
    return 1
}

# plan_ports -- pick every published port for the selected tiers, shift on
# conflict, write them to .env
plan_ports() {
    local entry key def tier want got taken=" " base="${OP_PORT_BASE:-}" reply
    collect_docker_ports
    for entry in "${PORT_DEFAULTS[@]}"; do
        read -r key def tier <<< "$entry"
        case "$tier" in
            base) ;;
            core) [[ "$OP_CONTROL_PLANE_ONLY" == 1 ]] && continue ;;
            monitoring) [[ "$OP_WITH_MONITORING" == 1 ]] || continue ;;
            *) _has_tier "$tier" || continue ;;
        esac
        if [[ -n "$base" ]]; then
            want=$(( base + def - 4600 ))
        else
            want="$(read_env_var "$ENV_FILE" "$key" || true)"
            [[ "$want" =~ ^[0-9]+$ ]] || want="$def"
        fi
        got="$(next_free_port "$want" "$taken")" || die "no free port at or above ${want} for ${key}"
        if [[ "$got" != "$want" ]]; then
            if [[ "$OP_UNATTENDED" != 1 ]] && tty_usable; then
                prompt_line reply "Port ${want} (${key}) is in use. Use ${got} instead? [Y/n]: " "--port-base"
                if [[ -n "$reply" && "${reply,,}" != y* ]]; then
                    die "port ${want} for ${key} is in use; re-run with --port-base N to move the block" "$EXIT_USAGE"
                fi
            fi
            log_warn "port ${want} (${key}) is in use: using ${got}"
            env_set "$key" "$got"
        elif [[ -n "$base" ]]; then
            env_set "$key" "$got"
        else
            env_set_default "$key" "$got"
        fi
        taken+="${got} "
        TAKEN_PORTS+="${got} "
        PORTS_SUMMARY+="${key}=${got},"
    done
}

# -----------------------------------------------------------------------------
# 5.1 Project / container-name collision guards (fail closed)
# -----------------------------------------------------------------------------
docker_reachable() {
    command -v docker >/dev/null 2>&1 || return 1
    docker info >/dev/null 2>&1
}

require_docker() {
    docker_reachable || die "cannot talk to the Docker daemon (is it running, and are you in the docker group?)" "$EXIT_DOCKER"
}

# check_project_owner PROJECT EXPECTED_DIR
# rc 0: every container of PROJECT belongs to EXPECTED_DIR (or none exist)
# rc 1: another directory owns the project; rc 2: docker could not be queried
check_project_owner() {
    local project="$1" expected="$2" out wd
    # "<name>|<working_dir>": the name keeps a row with an empty label visible.
    if ! out="$(docker ps -a --filter "label=com.docker.compose.project=${project}" \
            --format '{{.Names}}|{{.Label "com.docker.compose.project.working_dir"}}' 2>&1)"; then
        log_error "could not list containers of project '${project}': ${out}"
        return 2
    fi
    [[ -z "$out" ]] && return 0
    while IFS= read -r wd; do
        wd="${wd#*|}"
        if [[ -z "$wd" ]]; then
            log_error "a container of compose project '${project}' has no working_dir label; cannot prove who owns it"
            return 1
        fi
        if [[ "$wd" != "$expected" ]]; then
            log_error "compose project '${project}' already belongs to ${wd}, not ${expected}"
            return 1
        fi
    done <<< "$out"
    return 0
}

guard_projects() {
    local p rc
    for p in "$OP_PROJECT" "${OP_PROJECT}-cw"; do
        local expected="$OP_REAL_DIR"
        [[ "$p" == *-cw ]] && expected="${OP_REAL_DIR}/cropwright"
        if check_project_owner "$p" "$expected"; then
            continue
        else
            rc=$?
        fi
        (( rc == 2 )) && die "Docker is unreachable; refusing to continue without the collision check" "$EXIT_DOCKER"
        die "project name '${p}' is taken by another install: re-run with --project <another-name> (or OP_PROJECT=...)" "$EXIT_COLLISION"
    done
}

# compose_config_container_names JSON -> container_name values
compose_config_container_names() {
    printf '%s\n' "$1" | sed -n -E 's/^[[:space:]]*"container_name":[[:space:]]*"([^"]+)".*/\1/p'
}

# compose_config_network_names JSON -> names of the top-level networks
compose_config_network_names() {
    printf '%s\n' "$1" | awk '
        /^  "networks": \{/ { inside = 1; next }
        inside && /^  [}]/ { inside = 0 }
        inside && /^      "name": "/ { sub(/^      "name": "/, ""); sub(/".*/, ""); print }
    '
}

# _container_owner NAME -> "<name>|<project label>" for an existing container NAME
_container_owner() {
    docker ps -a --filter "name=^/${1}\$" --format '{{.Names}}|{{.Label "com.docker.compose.project"}}'
}

# assert_container_names_free NAME... -- no other project owns these names
assert_container_names_free() {
    local n owner
    for n in "$@"; do
        [[ "$n" == "${OP_PROJECT}-"* ]] \
            || die "container name '${n}' does not start with '${OP_PROJECT}-'; COMPOSE_PROJECT_NAME is not wired" "$EXIT_COLLISION"
        owner="$(_container_owner "$n")" || die "docker ps failed checking container name ${n}" "$EXIT_DOCKER"
        owner="${owner%%$'\n'*}"
        [[ -z "$owner" ]] && continue
        owner="${owner#*|}"
        if [[ -z "$owner" ]]; then
            die "a container named '${n}' already exists and was not created by Compose; remove or rename it" "$EXIT_COLLISION"
        fi
        if [[ "$owner" != "$OP_PROJECT" && "$owner" != "${OP_PROJECT}-cw" ]]; then
            die "a container named '${n}' already exists and belongs to project '${owner}'" "$EXIT_COLLISION"
        fi
    done
}

# assert_compose_config -- before any `up`: every container is prefixed with
# the project, the network is <project>_triton_net, no name is taken.
assert_compose_config() {
    local json names nets
    json="$(dc config --format json)" || die "compose config failed for ${OP_DIR}" "$EXIT_VERIFY"
    mapfile -t names < <(compose_config_container_names "$json")
    (( ${#names[@]} > 0 )) || die "compose config lists no container names" "$EXIT_VERIFY"
    assert_container_names_free "${names[@]}"
    mapfile -t nets < <(compose_config_network_names "$json")
    if [[ "${nets[*]:-}" != "${OP_PROJECT}_triton_net" ]]; then
        die "compose network is '${nets[*]:-<none>}', expected '${OP_PROJECT}_triton_net'" "$EXIT_COLLISION"
    fi
    log_success "compose config: ${#names[@]} containers, all '${OP_PROJECT}-*', network ${OP_PROJECT}_triton_net"
}

# -----------------------------------------------------------------------------
# 7. Security: bind address and external VLM consent
# -----------------------------------------------------------------------------
require_bind_consent() {
    local addr="$1" reply
    is_loopback_ipv4 "$addr" && return 0
    if [[ "$OP_UNATTENDED" == 1 ]] || ! tty_usable; then
        [[ "${OP_ALLOW_PUBLIC_BIND:-0}" == 1 ]] && return 0
        die "binding to ${addr} exposes unauthenticated services; unattended installs must also set OP_ALLOW_PUBLIC_BIND=1" "$EXIT_CONSENT"
    fi
    log_warn "Binding to ${addr} publishes every service with NO authentication."
    log_warn "OpenSearch security is disabled, and Docker-published ports bypass ufw/firewalld."
    log_warn "Put a reverse proxy with auth in front (see SECURITY.md)."
    prompt_line reply "Type 'expose' to continue: " "OP_ALLOW_PUBLIC_BIND=1"
    [[ "$reply" == "expose" ]] || die "bind address not confirmed" "$EXIT_CONSENT"
}

# url_host URL -> the real host (after any userinfo), lower-cased
url_host() {
    local url="$1" rest authority host
    [[ "$url" =~ ^[Hh][Tt][Tt][Pp][Ss]?://(.*)$ ]] || return 1
    rest="${BASH_REMATCH[1]}"
    authority="${rest%%[/?#]*}"
    authority="${authority##*@}"
    if [[ "$authority" == \[* ]]; then
        host="${authority#\[}"
        host="${host%%\]*}"
    else
        host="${authority%%:*}"
    fi
    [[ -n "$host" ]] || return 1
    echo "${host,,}"
}

# is_private_url URL -- literal private IPv4, loopback, localhost or
# host.docker.internal only. Every DNS name is external, and so is any URL
# with userinfo ("http://127.0.0.1@evil.example" connects to evil.example).
is_private_url() {
    local url="$1" rest authority host a b
    [[ "$url" =~ ^[Hh][Tt][Tt][Pp][Ss]?://(.*)$ ]] || return 1
    rest="${BASH_REMATCH[1]}"
    authority="${rest%%[/?#]*}"
    [[ "$authority" == *@* ]] && return 1
    host="$(url_host "$url")" || return 1
    case "$host" in
        localhost|host.docker.internal) return 0 ;;
    esac
    is_ipv4 "$host" || return 1
    IFS=. read -r a b _ _ <<< "$host"
    (( a == 127 || a == 10 )) && return 0
    (( a == 192 && b == 168 )) && return 0
    (( a == 172 && b >= 16 && b <= 31 )) && return 0
    return 1
}

require_external_vlm_consent() {
    local url="$1" reply
    is_private_url "$url" && return 0
    if [[ "$OP_UNATTENDED" == 1 ]] || ! tty_usable; then
        [[ "${OP_ALLOW_EXTERNAL_VLM:-0}" == 1 ]] && return 0
        die "--vlm-remote ${url} is outside private address space (crops would leave this host); unattended installs must set OP_ALLOW_EXTERNAL_VLM=1" "$EXIT_CONSENT"
    fi
    log_warn "This VLM endpoint is outside private address space: crops leave this host."
    prompt_line reply "Type 'external' to continue: " "OP_ALLOW_EXTERNAL_VLM=1"
    [[ "$reply" == "external" ]] || die "external VLM not confirmed" "$EXIT_CONSENT"
}

# -----------------------------------------------------------------------------
# 3.2 Images: images.lock digests, or an explicit tag for local-only builds
# -----------------------------------------------------------------------------
_lock_line_valid() {
    [[ "$1" =~ ^[a-z0-9_-]+=[a-z0-9][a-z0-9._/-]*(:[A-Za-z0-9._-]+)?@sha256:[0-9a-f]{64}$ && "$1" != *:latest@* ]]
}

# lock_value LOCK KEY -> image ref for KEY
lock_value() {
    awk -F= -v k="$2" '$1 == k { print substr($0, length(k) + 2); exit }' "$1"
}

# validate_images_lock LOCK -- every entry digest-pinned, none :latest
validate_images_lock() {
    local lock="$1" line bad=0
    while IFS= read -r line; do
        [[ -z "$line" || "$line" == \#* ]] && continue
        if ! _lock_line_valid "$line"; then
            if [[ "$line" =~ @sha256:0+dev[0-9]+$ ]]; then
                log_error "images.lock still has a development placeholder digest (${line%%=*}): this checkout is not a release. Install a published release (--version vX.Y.Z), or build the images locally and pass --image-tag <tag>."
            else
                log_error "images.lock entry is not a digest-pinned, non-latest image: ${line}"
            fi
            bad=1
        fi
    done < "$lock"
    return "$bad"
}

# check_lock_repos LOCK -- every key the table knows must name the repo
# scripts/lib/image_keys.sh names for it (build images under
# OP_IMAGE_NAMESPACE). Catches a release-script mix-up; the lock's own
# integrity comes from SHA256SUMS, and neither is an authenticity check.
check_lock_repos() {
    local lock="$1" line key ref repo want kind bad=0
    while IFS= read -r line; do
        [[ -z "$line" || "$line" == \#* ]] && continue
        key="${line%%=*}"
        kind="$(image_key_field "$key" kind 2>/dev/null)" || continue
        ref="${line#*=}"
        repo="${ref%%@*}"
        [[ "${repo##*/}" == *:* ]] && repo="${repo%:*}"
        if [[ "$kind" == build ]]; then
            want="${OP_IMAGE_NAMESPACE}/$(image_key_field "$key" image)"
        else
            want="$(image_key_field "$key" source)"
            [[ "${want##*/}" == *:* ]] && want="${want%:*}"
        fi
        if [[ "$repo" != "$want" ]]; then
            log_error "images.lock '${key}' names ${repo}, not ${want} (scripts/lib/image_keys.sh). This is an integrity check against a release-script mistake, not a signature check; the release is inconsistent, so nothing was pulled."
            bad=1
        fi
    done < "$lock"
    return "$bad"
}

# apply_image_pins -- lock mode: every images.lock key (scripts/lib/image_keys.sh,
# the table the release script writes from) is written to its *_IMAGE var,
# third-party images included. Tag mode: OP_IMAGE_REPO/TAG for our images,
# and any valid third-party pins the lock does carry.
apply_image_pins() {
    local lock="${OP_DIR}/images.lock" key envk ref vkey
    PINNED_IMAGES=()
    if [[ -f "$lock" ]]; then
        check_lock_repos "$lock" || die "images.lock does not match scripts/lib/image_keys.sh" "$EXIT_VERIFY"
    fi
    if [[ "$IMAGE_MODE" == lock ]]; then
        validate_images_lock "$lock" \
            || die "images.lock is not fully digest-pinned; this release cannot be installed reproducibly (for a local build use --image-tag)" "$EXIT_VERIFY"
        for key in $(image_keys build); do
            ref="$(lock_value "$lock" "$key")"
            [[ -n "$ref" ]] || die "images.lock has no '${key}' entry" "$EXIT_VERIFY"
            env_set "$(image_key_field "$key" env)" "$ref"
            PINNED_IMAGES+=("$ref")
        done
        env_set OP_IMAGE_TAG ""
    else
        log_warn "UNPINNED install: images run by tag ${OP_IMAGE_REPO}/*:${OP_IMAGE_TAG}; digests are recorded, not verified against images.lock"
        env_set OP_IMAGE_REPO "$OP_IMAGE_REPO"
        env_set OP_IMAGE_TAG "$OP_IMAGE_TAG"
        for key in $(image_keys build); do
            env_set "$(image_key_field "$key" env)" ""
        done
    fi
    for key in $(image_keys third); do
        envk="$(image_key_field "$key" env)"
        [[ "$envk" == VLM_IMAGE ]] && continue
        ref="$(lock_value "$lock" "$key" || true)"
        if _lock_line_valid "${key}=${ref}"; then
            env_set "$envk" "$ref"
            PINNED_IMAGES+=("$ref")
        elif [[ "$IMAGE_MODE" == lock ]]; then
            die "images.lock has no digest-pinned '${key}' entry (third-party images are pinned too)" "$EXIT_VERIFY"
        fi
    done
    if _has_tier vlm && [[ -n "${VLM_PICK:-}" ]]; then
        vkey="$(vlm_catalog_field "$VLM_PICK" vllm_image_key)"
        ref="$(lock_value "$lock" "$vkey" || true)"
        if _lock_line_valid "${vkey}=${ref}"; then
            env_set VLM_IMAGE "$ref"
            PINNED_IMAGES+=("$ref")
        elif [[ "$IMAGE_MODE" == lock ]]; then
            die "images.lock has no digest-pinned '${vkey}' entry for the ${VLM_PICK} VLM" "$EXIT_VERIFY"
        else
            log_warn "images.lock has no usable '${vkey}' digest; the vlm service keeps the compose default image"
        fi
    fi
}

# image_repo_digests REF -> RepoDigests of a local image (one per line)
image_repo_digests() {
    docker image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$1" 2>/dev/null
}

# pull_images -- pull every image of the active services that is missing
pull_images() {
    local images=() img missing=() out
    out="$(dc config --images)" || die "compose config --images failed"
    mapfile -t images <<< "$out"
    ACTIVE_IMAGES=("${images[@]}")
    for img in "${images[@]}"; do
        [[ -z "$img" ]] && continue
        if docker image inspect "$img" >/dev/null 2>&1; then
            continue
        fi
        missing+=("$img")
    done
    if (( ${#missing[@]} == 0 )); then
        log_info "all ${#images[@]} images are already present"
        return 0
    fi
    log_info "pulling ${#missing[@]} image(s); the Triton image alone is ~30 GB"
    for img in "${missing[@]}"; do
        if [[ "$IMAGE_MODE" == tag && "$RESOLVED_MODE" != branch && "$img" == "${OP_IMAGE_REPO}/openprocessor"*":${OP_IMAGE_TAG}" ]]; then
            die "${img} is not present locally: --image-tag runs local builds only and never pulls an unpinned image (build it, or install a release)" "$EXIT_VERIFY"
        fi
        if ! docker_mut pull "$img"; then
            if [[ "$IMAGE_MODE" == tag ]]; then
                die "image ${img} is neither local nor pullable (no images built for this tag/commit?)" "$EXIT_VERIFY"
            fi
            die "pull failed: ${img}" "$EXIT_VERIFY"
        fi
    done
}

# verify_image_digests -- lock mode: every image the active services run
# must be one images.lock pins, and its RepoDigests must contain exactly
# that digest. Tag mode: record image IDs.
verify_image_digests() {
    local ref repo digest p pinned
    IMAGE_DIGESTS=()
    if [[ "$IMAGE_MODE" == lock ]]; then
        for ref in "${ACTIVE_IMAGES[@]}"; do
            [[ -z "$ref" ]] && continue
            pinned=0
            for p in "${PINNED_IMAGES[@]}"; do [[ "$p" == "$ref" ]] && pinned=1; done
            if (( pinned == 0 )); then
                die "service image ${ref} is not pinned by images.lock" "$EXIT_VERIFY"
            fi
            repo="${ref%@*}"
            [[ "${repo##*/}" == *:* ]] && repo="${repo%:*}"
            digest="${ref##*@}"
            if [[ "$OP_DRY_RUN" == 1 ]] && ! docker image inspect "$ref" >/dev/null 2>&1; then
                echo "DRY: verify ${repo}@${digest} after pull"
                continue
            fi
            if ! image_repo_digests "$ref" | grep -qxF "${repo}@${digest}"; then
                die "image digest mismatch for ${ref}: the pulled image is not the one images.lock pins" "$EXIT_VERIFY"
            fi
            IMAGE_DIGESTS+=("${repo}@${digest}")
        done
        TRITON_DIGEST="$(lock_value "${OP_DIR}/images.lock" triton)"
        TRITON_DIGEST="${TRITON_DIGEST##*@}"
        [[ "$OP_DRY_RUN" == 1 ]] || log_success "all ${#IMAGE_DIGESTS[@]} pinned image digests verified"
    else
        local img
        TRITON_DIGEST=""
        for img in openprocessor openprocessor-triton; do
            ref="${OP_IMAGE_REPO}/${img}:${OP_IMAGE_TAG}"
            digest="$(docker image inspect --format '{{.Id}}' "$ref" 2>/dev/null || true)"
            [[ -n "$digest" ]] && IMAGE_DIGESTS+=("${ref}=${digest}")
            [[ "$img" == openprocessor-triton ]] && TRITON_DIGEST="$digest"
        done
    fi
}

# gpu_container_probe ID... -- a real container sees each planned GPU
gpu_container_probe() {
    local id img
    img="$(read_env_var "$ENV_FILE" OP_TRITON_IMAGE || true)"
    [[ -n "$img" ]] || img="${OP_IMAGE_REPO}/openprocessor-triton:${OP_IMAGE_TAG}"
    for id in "$@"; do
        if [[ "$OP_DRY_RUN" == 1 ]]; then
            echo "DRY: docker run --rm --gpus device=${id} --entrypoint nvidia-smi ${img} -L"
            continue
        fi
        if ! docker run --rm --gpus "device=${id}" --entrypoint nvidia-smi "$img" -L >/dev/null; then
            die "a container cannot see GPU ${id}: install/fix the NVIDIA Container Toolkit (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)" "$EXIT_GPU"
        fi
    done
}

# -----------------------------------------------------------------------------
# 5.2 State (.install/state.json, mode 600, atomic)
# -----------------------------------------------------------------------------
_json_str() {
    local s="$1"
    s="${s//\\/\\\\}"
    s="${s//\"/\\\"}"
    printf '"%s"' "$s"
}

state_get() {
    [[ -f "$STATE_FILE" ]] || return 1
    sed -n -E "s/^  \"$1\": \"([^\"]*)\",?\$/\\1/p" "$STATE_FILE" | head -n1
}

state_write() {
    local tmp groups="" g st secs first=1
    if [[ -f "${OP_DIR}/.install/groups.tsv" ]]; then
        while IFS=$'\t' read -r g st secs; do
            if (( first == 0 )); then groups+=","; fi
            [[ "$secs" =~ ^[0-9]+$ ]] || secs=0
            groups+=$'\n'"    $(_json_str "$g"): {\"status\": $(_json_str "$st"), \"seconds\": ${secs}}"
            first=0
        done < "${OP_DIR}/.install/groups.tsv"
    fi
    tmp="$(mktemp "${OP_DIR}/.install/state.XXXXXX")"
    chmod 600 "$tmp"
    {
        echo "{"
        echo "  \"schema\": \"1\","
        echo "  \"version\": $(_json_str "${INSTALL_REF:-}"),"
        echo "  \"mode\": $(_json_str "${RESOLVED_MODE:-}"),"
        echo "  \"image_mode\": $(_json_str "${IMAGE_MODE:-}"),"
        echo "  \"image_tag\": $(_json_str "${OP_IMAGE_TAG:-}"),"
        echo "  \"project\": $(_json_str "$OP_PROJECT"),"
        echo "  \"install_dir\": $(_json_str "$OP_REAL_DIR"),"
        echo "  \"installed_at\": $(_json_str "${INSTALLED_AT:-}"),"
        echo "  \"updated_at\": $(_json_str "$(date -u +%Y-%m-%dT%H:%M:%SZ)"),"
        echo "  \"tiers\": $(_json_str "${SELECTED_TIERS:-}"),"
        echo "  \"control_plane_only\": $(_json_str "${OP_CONTROL_PLANE_ONLY:-0}"),"
        echo "  \"with_monitoring\": $(_json_str "${OP_WITH_MONITORING:-0}"),"
        echo "  \"gpu_plan\": $(_json_str "${GPU_PLAN_SUMMARY:-}"),"
        echo "  \"ports\": $(_json_str "${PORTS_SUMMARY%,}"),"
        echo "  \"image_digests\": $(_json_str "$(IFS=' '; echo "${IMAGE_DIGESTS[*]:-}")"),"
        echo "  \"triton_image_digest\": $(_json_str "${TRITON_DIGEST:-}"),"
        echo "  \"health\": $(_json_str "${HEALTH_RESULT:-not-run}"),"
        echo "  \"groups\": {${groups}"
        echo "  }"
        echo "}"
    } > "$tmp"
    mv -f "$tmp" "$STATE_FILE"
}

# -----------------------------------------------------------------------------
# 7. Cropwright tier (separate compose project <project>-cw)
# -----------------------------------------------------------------------------
_version_ge() {
    [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1)" == "$2" ]]
}

# choose_cropwright_bind -> CROPWRIGHT_BIND (owner decision, plan 7 addendum):
# the web UI is for this computer AND the local LAN, so it binds 0.0.0.0 by
# default; --local-only, or "no" at the prompt, keeps it on 127.0.0.1. The
# backend ports stay on OP_BIND_ADDRESS (127.0.0.1); LAN browsers reach the
# API through Cropwright's nginx on the docker network. A specific
# non-loopback --bind (say 10.0.0.5) narrows Cropwright to that interface
# too: never wider than the one the user chose.
choose_cropwright_bind() {
    local reply prev
    prev="$(read_env_var "${OP_DIR}/cropwright/.env" CROPWRIGHT_BIND_ADDRESS 2>/dev/null || true)"
    if [[ "$OP_LOCAL_ONLY" == 1 ]]; then
        CROPWRIGHT_BIND=127.0.0.1
    elif [[ "$OP_BIND_ADDRESS" != 0.0.0.0 ]] && ! is_loopback_ipv4 "$OP_BIND_ADDRESS"; then
        CROPWRIGHT_BIND="$OP_BIND_ADDRESS"
    elif [[ -n "$prev" ]] && is_ipv4 "$prev"; then
        CROPWRIGHT_BIND="$prev"
    elif [[ "$OP_UNATTENDED" != 1 ]] && tty_usable; then
        prompt_line reply "Let other computers on your LAN open the Cropwright web UI? [Y/n]: " "--local-only"
        if [[ -z "$reply" || "${reply,,}" == y* ]]; then CROPWRIGHT_BIND=0.0.0.0; else CROPWRIGHT_BIND=127.0.0.1; fi
    else
        CROPWRIGHT_BIND=0.0.0.0
    fi
}

setup_cropwright() {
    local lock="${OP_DIR}/cropwright.lock" tag image sums_sha dir="${OP_DIR}/cropwright" f port cw_base
    tag="$(read_env_var "$lock" tag || true)"
    image="$(read_env_var "$lock" image || true)"
    sums_sha="$(read_env_var "$lock" sha256sums_sha256 || true)"
    [[ "$tag" =~ ^v[0-9A-Za-z._-]+$ ]] \
        || die "cropwright.lock does not name a Cropwright release tag (got '${tag}'): the cropwright tier cannot be installed from this release" "$EXIT_VERIFY"
    [[ "$sums_sha" =~ ^[0-9a-f]{64}$ ]] \
        || die "cropwright.lock has no sha256 for Cropwright's SHA256SUMS; refusing to install unverified files" "$EXIT_VERIFY"
    if [[ "$IMAGE_MODE" == lock ]] && ! _lock_line_valid "cropwright=${image}"; then
        die "cropwright.lock image is not digest-pinned: ${image}" "$EXIT_VERIFY"
    fi
    mkdir -p "$dir"
    cw_base="${CW_ARTIFACT_BASE_URL:-https://github.com/${CW_GH_REPO}/releases/download}/${tag}"
    if [[ -n "${OP_RELEASE_DIR:-}" ]]; then
        # build_deploy_bundle.sh stages these when given CW_RELEASE_DIR.
        if [[ -f "${OP_RELEASE_DIR}/cropwright/${tag}/SHA256SUMS" ]]; then
            cw_base="file://${OP_RELEASE_DIR}/cropwright/${tag}"
            log_info "Cropwright ${tag}: using the files from the release dir"
        else
            log_warn "the release dir has no cropwright/${tag}/: fetching Cropwright from the network (this install is not offline); still verified against cropwright.lock"
        fi
    fi
    # Cropwright's release assets carry the section 3 names; its SHA256SUMS is
    # itself pinned by cropwright.lock, which this release's checksums cover.
    for f in SHA256SUMS docker-compose.yml .env.example; do
        if ! _dl "${cw_base}/${f}" "${dir}/${f}.new"; then
            _dl "${CW_RAW_BASE_URL:-https://raw.githubusercontent.com/${CW_GH_REPO}}/${tag}/${f}" "${dir}/${f}.new" \
                || die "could not download Cropwright ${f} at ${tag}" "$EXIT_VERIFY"
        fi
    done
    if [[ "$(_sha256 "${dir}/SHA256SUMS.new")" != "$sums_sha" ]]; then
        _cw_discard_downloads "$dir"
        die "Cropwright SHA256SUMS does not match cropwright.lock" "$EXIT_VERIFY"
    fi
    for f in docker-compose.yml .env.example; do
        if ! verify_against_sums "${dir}/SHA256SUMS.new" "${dir}/${f}.new" "$f"; then
            _cw_discard_downloads "$dir"
            die "Cropwright ${f} failed checksum verification" "$EXIT_VERIFY"
        fi
    done
    for f in SHA256SUMS docker-compose.yml .env.example; do
        mv -f "${dir}/${f}.new" "${dir}/${f}"
        chmod 644 "${dir}/${f}"
    done

    port="$(read_env_var "${dir}/.env" CROPWRIGHT_PORT || true)"
    [[ "$port" =~ ^[0-9]+$ ]] || port=5184
    port="$(next_free_port "$port" "${TAKEN_PORTS}")" || die "no free port for Cropwright"
    CROPWRIGHT_PORT="$port"
    [[ -f "${dir}/.env" ]] || install -m 600 "${dir}/.env.example" "${dir}/.env"
    chmod 600 "${dir}/.env"
    upsert_env_var "${dir}/.env" CROPWRIGHT_PORT "$port"
    upsert_env_var "${dir}/.env" CROPWRIGHT_CONTAINER_NAME "${OP_PROJECT}-cropwright"
    upsert_env_var "${dir}/.env" OP_DOCKER_NETWORK "${OP_PROJECT}_triton_net"
    upsert_env_var "${dir}/.env" API_UPSTREAM "http://op-api:8000"
    upsert_env_var "${dir}/.env" CROPWRIGHT_BIND_ADDRESS "$CROPWRIGHT_BIND"
    if [[ "$IMAGE_MODE" == lock ]]; then
        upsert_env_var "${dir}/.env" CROPWRIGHT_IMAGE "$image"
    fi

    # Check what Compose will really do with Cropwright's own file: the
    # published host IP must be the chosen one, and the image the pinned one.
    local json hosts rendered
    json="$(dc_cw config --format json)" || die "compose config failed for Cropwright"
    hosts="$(printf '%s\n' "$json" | sed -n -E 's/^[[:space:]]*"host_ip":[[:space:]]*"([^"]*)".*/\1/p' | sort -u | tr '\n' ' ')"
    # A port without a host IP is published on every interface.
    [[ -z "$hosts" ]] && hosts="0.0.0.0 "
    if [[ "${hosts% }" != "$CROPWRIGHT_BIND" ]]; then
        die "Cropwright's compose would publish on '${hosts:-0.0.0.0}', not ${CROPWRIGHT_BIND} (its compose must honour CROPWRIGHT_BIND_ADDRESS)" "$EXIT_VERIFY"
    fi
    if [[ "$IMAGE_MODE" == lock ]]; then
        rendered="$(dc_cw config --images)" || die "compose config failed for Cropwright"
        [[ "$rendered" == "$image" ]] \
            || die "Cropwright's compose runs '${rendered}', not the pinned ${image} (it must honour CROPWRIGHT_IMAGE)" "$EXIT_VERIFY"
        PINNED_IMAGES+=("$image")
    fi
    assert_container_names_free "${OP_PROJECT}-cropwright"
    log_success "Cropwright ${tag} configured on ${CROPWRIGHT_BIND}:${port}"
}

_cw_discard_downloads() {
    local f
    for f in SHA256SUMS docker-compose.yml .env.example; do
        rm -f -- "${1:?}/${f}.new"
    done
}

# -----------------------------------------------------------------------------
# 6.1 Health
# -----------------------------------------------------------------------------
_health_host() {
    if [[ "$OP_BIND_ADDRESS" == 0.0.0.0 ]]; then echo 127.0.0.1; else echo "$OP_BIND_ADDRESS"; fi
}

# wait_http URL SECONDS [GREP_FIXED] -- until 200 (and body contains GREP)
wait_http() {
    local url="$1" secs="$2" want="${3:-}" waited=0 body
    local cap="${OP_HEALTH_TIMEOUT:-}"
    [[ -n "$cap" ]] && (( cap < secs )) && secs=$cap
    while :; do
        if body="$(curl -fsS --max-time 10 "$url" 2>/dev/null)"; then
            body="${body//[[:space:]]/}"
            if [[ -z "$want" || "$body" == *"$want"* ]]; then
                return 0
            fi
        fi
        (( waited >= secs )) && return 1
        sleep 5
        waited=$((waited + 5))
    done
}

_API_PROBE_PY='
import json, sys, urllib.request, uuid
import cv2, numpy as np
base = "http://127.0.0.1:8000"
img = np.full((320, 320, 3), 200, np.uint8)
cv2.rectangle(img, (40, 40), (280, 280), (30, 30, 30), 4)
cv2.putText(img, "OPENPROCESSOR", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
jpg = cv2.imencode(".jpg", img)[1].tobytes()
def call(path, data=None, ctype=None):
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": ctype} if ctype else {})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read() or b"{}")
def upload(path):
    b = uuid.uuid4().hex
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"p.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n").encode() + jpg + f"\r\n--{b}--\r\n".encode()
    return call(path, body, f"multipart/form-data; boundary={b}")
fails = []
for path in ("/detect", "/faces/detect", "/embed/image", "/ocr/predict"):
    try:
        upload(path)
    except Exception as e:
        fails.append(f"{path}: {e}")
try:
    r = call("/embed/text", json.dumps({"text": "a photo"}).encode(), "application/json")
    if len(r.get("embedding") or []) != 512:
        fails.append("/embed/text: not 512 dims")
except Exception as e:
    fails.append(f"/embed/text: {e}")
if sys.argv[1] == "1":
    try:
        h = call("/curation/health")
        ok = h.get("status") == "ok" or (
            h.get("status") == "degraded" and h["triton"].get("reachable") and h["opensearch"].get("reachable")
            and (h.get("vlm", {}).get("reachable") or sys.argv[2] == "0"))
        if not ok:
            fails.append("/curation/health: " + json.dumps(h)[:300])
    except Exception as e:
        fails.append(f"/curation/health: {e}")
print("\n".join(fails) if fails else "all functional probes passed")
sys.exit(1 if fails else 0)
'

# run_health -- every check in plan 6.1 for the selected tiers
run_health() {
    local h p vp m fails=() vlm_repo
    h="$(_health_host)"
    HEALTH_RESULT="not-run"
    if [[ "$OP_DRY_RUN" == 1 || "$OP_NO_START" == 1 ]]; then
        log_info "health checks skipped: nothing was started"
        return 0
    fi
    log_step "Health"
    p="$(read_env_var "$ENV_FILE" API_PORT)"
    if [[ "$OP_CONTROL_PLANE_ONLY" != 1 ]]; then
        local tp
        tp="$(read_env_var "$ENV_FILE" TRITON_HTTP_PORT)"
        if wait_http "http://${h}:${tp}/v2/health/ready" 180; then
            local models=(yolov11_small_trt_end2end scrfd_10g_bnkps arcface_w600k_r50
                mobileclip2_s2_image_encoder mobileclip2_s2_text_encoder
                paddleocr_det_trt paddleocr_rec_trt ocr_pipeline)
            _has_tier curation && models+=(pe_image_encoder pe_text_encoder)
            for m in "${models[@]}"; do
                curl -fsS --max-time 5 "http://${h}:${tp}/v2/models/${m}/ready" >/dev/null 2>&1 \
                    || fails+=("triton model ${m} not READY: ./openprocessor models install")
            done
        else
            fails+=("triton not ready: ./openprocessor logs triton-server")
        fi
    fi
    # The VLM loads for minutes on a cold start; the functional probe below
    # requires it reachable, so wait for it first.
    if _has_tier vlm; then
        vp="$(read_env_var "$ENV_FILE" VLM_PORT)"
        vlm_repo="$(read_env_var "$ENV_FILE" VLM_MODEL)"
        if wait_http "http://${h}:${vp}/health" 1200 \
                && wait_http "http://${h}:${vp}/v1/models" 60 '"id":"local-vlm"' \
                && wait_http "http://${h}:${vp}/v1/models" 5 "\"root\":\"${vlm_repo}\""; then
            :
        else
            fails+=("vlm not serving local-vlm (${vlm_repo}): ./openprocessor logs vlm")
        fi
    fi
    if ! wait_http "http://${h}:${p}/health" 240 '"status":"ready"'; then
        if [[ "$OP_CONTROL_PLANE_ONLY" == 1 ]] && wait_http "http://${h}:${p}/health" 5; then
            log_warn "API is up but not ready (expected without Triton in control-plane-only mode)"
        else
            fails+=("API /health not ready: ./openprocessor logs yolo-api")
        fi
    elif [[ "$OP_CONTROL_PLANE_ONLY" != 1 ]]; then
        local cur=0 vlm_on=0
        _has_tier curation && cur=1
        { _has_tier vlm || [[ -n "${OP_VLM_URL:-}" ]]; } && vlm_on=1
        if ! dc exec -T yolo-api python -c "$_API_PROBE_PY" "$cur" "$vlm_on"; then
            fails+=("API functional probes failed: ./openprocessor logs yolo-api")
        fi
    fi
    if _has_tier segmenter; then
        p="$(read_env_var "$ENV_FILE" SEGMENTER_PORT)"
        wait_http "http://${h}:${p}/health" 600 '"loaded":true' || fails+=("segmenter not loaded: ./openprocessor logs segmenter")
    fi
    if _has_tier trainer; then
        p="$(read_env_var "$ENV_FILE" MLFLOW_PORT)"
        wait_http "http://${h}:${p}/health" 180 || fails+=("mlflow not healthy: ./openprocessor logs curation-mlflow")
    fi
    if _has_tier cropwright; then
        wait_http "http://${h}:${CROPWRIGHT_PORT}/" 120 || fails+=("cropwright / not 200")
        wait_http "http://${h}:${CROPWRIGHT_PORT}/curation/health" 60 || fails+=("cropwright cannot reach the API through its proxy (network ${OP_PROJECT}_triton_net)")
    fi
    if [[ "$OP_WITH_MONITORING" == 1 ]]; then
        local unreadable
        unreadable="$(find "${OP_DIR}/monitoring" \( -type f ! -perm -o=r \) -o \( -type d ! -perm -o=rx \) 2>/dev/null | head -n 3)"
        [[ -z "$unreadable" ]] || fails+=("monitoring config not readable by the container users: ${unreadable//$'\n'/ }")
        p="$(read_env_var "$ENV_FILE" PROMETHEUS_PORT)"
        wait_http "http://${h}:${p}/api/v1/status/config" 120 'job_name:triton' \
            || fails+=("prometheus did not load monitoring/prometheus.yml: ./openprocessor logs prometheus")
        p="$(read_env_var "$ENV_FILE" LOKI_PORT)"
        wait_http "http://${h}:${p}/ready" 180 || fails+=("loki not ready (config not loaded?): ./openprocessor logs loki")
        p="$(read_env_var "$ENV_FILE" GRAFANA_PORT)"
        if ! wait_http "http://${h}:${p}/api/health" 120; then
            fails+=("grafana not healthy: ./openprocessor logs grafana")
        elif dc logs --no-color --tail 500 grafana 2>/dev/null | grep -qiE 'permission denied|failed to provision|failed to read'; then
            fails+=("grafana could not load its provisioning files: ./openprocessor logs grafana")
        fi
    fi
    if [[ -n "${OP_VLM_URL:-}" ]]; then
        log_warn "remote VLM registered; its endpoint probe needs the model-selection API (not in this release): check with ./openprocessor health"
    fi
    if (( ${#fails[@]} > 0 )); then
        HEALTH_RESULT="failed"
        for m in "${fails[@]}"; do log_error "health: ${m}"; done
        return 1
    fi
    HEALTH_RESULT="ok"
    log_success "all health checks passed"
}

# -----------------------------------------------------------------------------
# 5.3 Uninstall / purge
# -----------------------------------------------------------------------------
require_purge_confirmation() {
    local project="$1" what="$2" envvar="$3" reply
    if [[ "$OP_UNATTENDED" == 1 ]] || ! tty_usable; then
        [[ "${!envvar:-}" == "$project" ]] && return 0
        die "unattended ${what} requires ${envvar}=${project}" "$EXIT_CONSENT"
    fi
    prompt_line reply "Type the project name (${project}) to confirm ${what}: " "${envvar}=${project}"
    [[ "$reply" == "$project" ]] || die "${what} not confirmed" "$EXIT_CONSENT"
}

# assert_purgeable_dir DIR -- an installer-managed dir that is not / or ~ or
# a top-level system path
assert_purgeable_dir() {
    local real
    real="$(realpath -e "$1")" || die "install dir does not exist: $1"
    case "$real" in
        /|/bin|/boot|/dev|/etc|/home|/lib|/lib64|/media|/mnt|/opt|/proc|/root|/run|/sbin|/srv|/sys|/tmp|/usr|/var)
            die "refusing to purge a system directory: ${real}" ;;
    esac
    [[ "$real" == "$(realpath -m "$HOME")" ]] && die "refusing to purge your home directory"
    [[ "$real" =~ ^/[^/]+/.+ ]] || die "refusing to purge a top-level directory: ${real}"
    [[ -f "${real}/.install/state.json" ]] \
        || die "${real} has no .install/state.json: not an installer-managed directory, nothing purged"
}

# purge_data_paths INSTALL_DIR -> the real paths --purge-data deletes. Only
# models/ pytorch_models/ cache/ data/ that are real directories inside the
# install dir; a symlink or anything resolving outside is never listed.
# OP_SOURCE_ROOT_HOST is read from the install's .env, resolved against the
# install dir, and reported as kept when it points outside the purged dirs.
purge_data_paths() {
    local dir="$1" root d real src
    root="$(realpath -e "$dir")" || return 1
    for d in models pytorch_models cache data; do
        [[ -e "${root}/${d}" || -L "${root}/${d}" ]] || continue
        if [[ -L "${root}/${d}" ]]; then
            log_warn "not purging ${root}/${d}: it is a symlink" >&2
            continue
        fi
        real="$(realpath -e "${root}/${d}")" || continue
        case "$real" in
            "${root}"/*) echo "$real" ;;
            *) log_warn "not purging ${root}/${d}: resolves outside the install dir" >&2 ;;
        esac
    done
    src="$(read_env_var "${root}/.env" OP_SOURCE_ROOT_HOST 2>/dev/null || true)"
    if [[ -n "$src" ]]; then
        [[ "$src" == /* ]] || src="${root}/${src}"
        real="$(realpath -m "$src")"
        case "$real" in
            "${root}/data"|"${root}/data"/*) ;;
            *) log_info "keeping OP_SOURCE_ROOT_HOST (${real}): outside the purged directories" >&2 ;;
        esac
    fi
    return 0
}

# _tree_has_foreign_files DIR -> 0 when something under DIR is not ours
# (e.g. chowned to the container uid 1000 on a host where we are not 1000)
_tree_has_foreign_files() {
    [[ -n "$(find "$1" ! -user "$(id -u)" -print -quit 2>/dev/null)" ]]
}

safe_rm_tree() {
    local p="$1" img
    if [[ "$OP_DRY_RUN" == 1 ]]; then
        echo "DRY: rm -rf --one-file-system -- ${p}"
        return 0
    fi
    if _tree_has_foreign_files "$p"; then
        # Files owned by the container user: delete them the way they were
        # created, from a container, scoped to this one directory.
        img="$(read_env_var "$ENV_FILE" OP_API_IMAGE || true)"
        [[ -n "$img" ]] || img="$(read_env_var "$ENV_FILE" OP_IMAGE_REPO || echo "$OP_IMAGE_NAMESPACE")/openprocessor:$(read_env_var "$ENV_FILE" OP_IMAGE_TAG || true)"
        docker run --rm --user 0 --entrypoint rm -v "$(dirname "$p"):/purge" "$img" \
            -rf --one-file-system -- "/purge/$(basename "$p")" \
            || die "could not delete ${p} (it holds files owned by the container user)"
        return 0
    fi
    rm -rf --one-file-system -- "$p"
}

# require_owned_install -- the dir holds .install/state.json written by this
# installer for this dir, and its project matches .env. Everything that
# changes an existing install (uninstall, purge, rollback, repair, upgrade)
# goes through this.
require_owned_install() {
    local sp sd envp
    [[ -f "$STATE_FILE" ]] \
        || die "${OP_REAL_DIR} has no .install/state.json: not an install made by this installer; nothing was changed" "$EXIT_COLLISION"
    sp="$(state_get project || true)"
    sd="$(state_get install_dir || true)"
    envp="$(read_env_var "$ENV_FILE" COMPOSE_PROJECT_NAME || true)"
    [[ -n "$sp" && "$sd" == "$OP_REAL_DIR" ]] \
        || die "${OP_REAL_DIR}/.install/state.json records another install (${sd:-no dir}); nothing was changed" "$EXIT_COLLISION"
    [[ -z "$envp" || "$envp" == "$sp" ]] \
        || die "project mismatch: .env says '${envp}', .install/state.json says '${sp}'; nothing was changed" "$EXIT_COLLISION"
    if [[ -n "$_OP_PROJECT_FLAG" && "$_OP_PROJECT_FLAG" != "$sp" ]]; then
        die "--project ${_OP_PROJECT_FLAG} does not match this install's project '${sp}'" "$EXIT_USAGE"
    fi
    OP_PROJECT="$sp"
}

# require_destructive_consent WHAT -- a prompt on a terminal, or an explicit
# --yes/--unattended. A missing terminal on its own is never consent.
require_destructive_consent() {
    local what="$1"
    [[ "$OP_ASSUME_YES" == 1 ]] && return 0
    if tty_usable; then
        confirm_yes "${what}?" "--yes" || die "cancelled" "$EXIT_CONSENT"
        return 0
    fi
    die "${what} needs confirmation, and there is no terminal: re-run with --yes (or --unattended)" "$EXIT_CONSENT"
}

do_uninstall() {
    [[ -d "$OP_DIR" ]] || die "no install at ${OP_DIR}"
    require_owned_install
    require_docker
    guard_projects

    local purge_any=0 paths=() vols=() containers=()
    [[ "$OP_PURGE_VOLUMES" == 1 || "$OP_PURGE_DATA" == 1 ]] && purge_any=1
    if (( purge_any )); then assert_purgeable_dir "$OP_DIR"; fi

    log_step "Uninstall plan for project ${OP_PROJECT} (${OP_DIR})"
    local listing
    listing="$(docker ps -a --filter "label=com.docker.compose.project=${OP_PROJECT}" --format '{{.Names}}')" \
        || die "docker ps failed" "$EXIT_DOCKER"
    mapfile -t containers <<< "$listing"
    echo "containers to remove: ${containers[*]:-<none>}"
    if [[ "$OP_PURGE_VOLUMES" == 1 ]]; then
        listing="$(docker volume ls --filter "label=com.docker.compose.project=${OP_PROJECT}" --format '{{.Name}}')" \
            || die "docker volume ls failed" "$EXIT_DOCKER"
        mapfile -t vols <<< "$listing"
        echo "volumes to delete: ${vols[*]:-<none>}"
    fi
    if [[ "$OP_PURGE_DATA" == 1 ]]; then
        mapfile -t paths < <(purge_data_paths "$OP_DIR")
        echo "directories to delete: ${paths[*]:-<none>}"
    fi
    require_destructive_consent "Stop and remove the containers of ${OP_PROJECT}"
    if (( purge_any )); then
        require_purge_confirmation "$OP_PROJECT" "the purge" OP_CONFIRM_PURGE
    fi

    if [[ -f "${OP_DIR}/cropwright/docker-compose.yml" ]]; then
        dc_cw down --remove-orphans || die "cropwright down failed"
    fi
    if [[ "$OP_PURGE_VOLUMES" == 1 ]]; then
        dc down --remove-orphans -v || die "compose down -v failed"
    else
        dc down --remove-orphans || die "compose down failed"
    fi

    local p
    for p in "${paths[@]}"; do
        safe_rm_tree "$p"
    done
    if [[ "$OP_PURGE_DATA" == 1 && -d "${OP_DIR}/secrets" && ! -L "${OP_DIR}/secrets" ]]; then
        echo "secrets/ holds VLM API keys: it is deleted only with a separate confirmation"
        local ok=0 reply
        if [[ "$OP_UNATTENDED" == 1 ]] || ! tty_usable; then
            [[ "${OP_CONFIRM_PURGE_SECRETS:-}" == "$OP_PROJECT" ]] && ok=1
        else
            prompt_line reply "Also delete ${OP_DIR}/secrets? Type the project name to confirm: " "OP_CONFIRM_PURGE_SECRETS"
            [[ "$reply" == "$OP_PROJECT" ]] && ok=1
        fi
        if (( ok )); then safe_rm_tree "$(realpath -e "${OP_DIR}/secrets")"; else log_info "kept ${OP_DIR}/secrets"; fi
    fi
    if [[ "$OP_REMOVE_IMAGES" == 1 ]]; then
        remove_install_images
    fi
    if [[ "$OP_DRY_RUN" == 1 ]]; then
        log_info "dry-run: nothing was stopped or deleted"
    else
        log_success "uninstalled project ${OP_PROJECT}; the install dir keeps .env, the release files and anything not purged"
    fi
}

# remove_install_images -- only images pinned by this install's images.lock
# (or its tag), and only when no container at all still uses them
remove_install_images() {
    local ref users refs=()
    if [[ -f "${OP_DIR}/images.lock" ]]; then
        while IFS= read -r ref; do
            [[ -z "$ref" || "$ref" == \#* ]] && continue
            _lock_line_valid "$ref" && refs+=("${ref#*=}")
        done < "${OP_DIR}/images.lock"
    fi
    for ref in "${refs[@]}"; do
        docker image inspect "$ref" >/dev/null 2>&1 || continue
        users="$(docker ps -a --filter "ancestor=${ref}" --format '{{.Names}}')" || die "docker ps failed" "$EXIT_DOCKER"
        if [[ -n "$users" ]]; then
            log_info "keeping ${ref}: still used by ${users//$'\n'/ }"
            continue
        fi
        docker_mut rmi "$ref" || log_warn "could not remove ${ref}"
    done
}

# -----------------------------------------------------------------------------
# 5.2 Rollback
# -----------------------------------------------------------------------------
do_rollback() {
    require_owned_install
    local newest="" env_project rel current b bver
    # The newest backup of a DIFFERENT version: a same-version re-run or
    # `openprocessor upgrade` also backs up, and rolling back to that would
    # land on the version already installed. With no older version on
    # record, the newest backup (a config rollback) is used.
    current="$(state_get version || true)"
    while IFS= read -r b; do
        [[ -z "$newest" ]] && newest="$b"
        bver="$(STATE_FILE="${b}/.install/state.json" state_get version || true)"
        if [[ -n "$current" && -n "$bver" && "$bver" != "$current" ]]; then
            newest="$b"
            log_info "rolling back to the previous version ${bver} (installed: ${current}); skipping same-version backups"
            break
        fi
    done < <(find "${OP_DIR}/backups" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort -r || true)
    [[ -n "$newest" ]] || die "no backups/ entry to roll back to"
    env_project="$(read_env_var "${newest}/.env" COMPOSE_PROJECT_NAME || true)"
    [[ "$env_project" == "$OP_PROJECT" ]] || die "backup ${newest} belongs to project '${env_project:-none}', not '${OP_PROJECT}'" "$EXIT_COLLISION"
    require_docker
    guard_projects

    log_step "Rollback to $(basename "$newest")"
    echo "restores: $(cd "$newest" && find . -type f | sed 's#^\./##' | tr '\n' ' ')"
    require_destructive_consent "Restore these files and restart ${OP_PROJECT}"
    local undo
    undo="${OP_DIR}/.install/rollback-undo/$(date -u +%Y%m%dT%H%M%SZ)"
    if [[ "$OP_DRY_RUN" == 1 ]]; then
        echo "DRY: restore ${newest} into ${OP_DIR} (current files saved to ${undo})"
        return 0
    fi
    mkdir -p "$undo"
    chmod 700 "${OP_DIR}/.install/rollback-undo" "$undo"
    while IFS= read -r -d '' rel; do
        rel="${rel#"$newest"/}"
        if [[ -f "${OP_DIR}/${rel}" ]]; then
            mkdir -p "${undo}/$(dirname "$rel")"
            cp -p "${OP_DIR}/${rel}" "${undo}/${rel}"
        fi
        mkdir -p "${OP_DIR}/$(dirname "$rel")"
        cp -p "${newest}/${rel}" "${OP_DIR}/${rel}"
    done < <(find "$newest" -type f -print0)

    IMAGE_MODE=lock
    if [[ -z "$(read_env_var "$ENV_FILE" OP_TRITON_IMAGE || true)" ]]; then
        IMAGE_MODE=tag
        OP_IMAGE_REPO="$(read_env_var "$ENV_FILE" OP_IMAGE_REPO || echo "$OP_IMAGE_NAMESPACE")"
        OP_IMAGE_TAG="$(read_env_var "$ENV_FILE" OP_IMAGE_TAG || true)"
    fi
    PINNED_IMAGES=()
    local k
    # shellcheck source=scripts/lib/image_keys.sh
    source "${OP_DIR}/scripts/lib/image_keys.sh"
    for k in $(for rel in $(image_keys); do image_key_field "$rel" env; done | sort -u); do
        rel="$(read_env_var "$ENV_FILE" "$k" || true)"
        [[ -n "$rel" ]] && PINNED_IMAGES+=("$rel")
    done
    assert_compose_config
    pull_images
    if [[ "$IMAGE_MODE" == lock ]]; then
        validate_images_lock "${OP_DIR}/images.lock" || die "restored images.lock is not digest-pinned" "$EXIT_VERIFY"
    fi
    verify_image_digests
    dc up -d --remove-orphans || die "compose up failed after restoring $(basename "$newest")"
    log_success "rolled back to $(basename "$newest"); run ./openprocessor health (engines rebuild with ./openprocessor models install if the Triton image changed)"
}

# -----------------------------------------------------------------------------
# 6.2 Summary
# -----------------------------------------------------------------------------
print_summary() {
    local h="$1" api tri seg vlmp ml failed
    api="$(read_env_var "$ENV_FILE" API_PORT || true)"
    tri="$(read_env_var "$ENV_FILE" TRITON_HTTP_PORT || true)"
    log_step "Summary"
    echo "  install dir : ${OP_REAL_DIR}"
    echo "  release     : ${INSTALL_REF} (${RESOLVED_MODE}; images: ${IMAGE_MODE})"
    echo "  project     : ${OP_PROJECT}"
    echo "  tiers       : ${SELECTED_TIERS}"
    echo "  GPU plan    : ${GPU_PLAN_SUMMARY:-none}"
    echo "  health      : ${HEALTH_RESULT}"
    local heap per_gb budget
    heap="$(read_env_var "$ENV_FILE" OPENSEARCH_HEAP || true)"
    per_gb="$(read_env_var "$ENV_FILE" OP_SHARDS_PER_HEAP_GB || true)"
    [[ "$per_gb" =~ ^[0-9]+$ ]] || per_gb=20
    budget="$(opensearch_shard_budget "$heap" "$per_gb" || echo unknown)"
    echo "  OpenSearch  : heap ${heap:-unset}, soft shard budget ${budget} (${per_gb} shards per heap GB)"
    echo ""
    echo "  URLs:"
    echo "    API docs    http://${h}:${api}/docs"
    echo "    API health  http://${h}:${api}/health"
    [[ "$OP_CONTROL_PLANE_ONLY" == 1 ]] || echo "    Triton      http://${h}:${tri}/v2/health/ready"
    if _has_tier segmenter; then seg="$(read_env_var "$ENV_FILE" SEGMENTER_PORT)"; echo "    Segmenter   http://${h}:${seg}/health"; fi
    if _has_tier vlm; then vlmp="$(read_env_var "$ENV_FILE" VLM_PORT)"; echo "    VLM         http://${h}:${vlmp}/v1/models (${VLM_PICK})"; fi
    if _has_tier trainer; then ml="$(read_env_var "$ENV_FILE" MLFLOW_PORT)"; echo "    MLflow      http://${h}:${ml}"; fi
    if _has_tier cropwright; then
        echo "    Cropwright  http://${h}:${CROPWRIGHT_PORT}"
        if ! is_loopback_ipv4 "${CROPWRIGHT_BIND:-127.0.0.1}"; then
            local lan="$CROPWRIGHT_BIND"
            [[ "$lan" == 0.0.0.0 ]] && lan="$(hostname -I 2>/dev/null | awk '{ print $1 }' || true)"
            echo "    Cropwright on your LAN: http://${lan:-<this-computer-ip>}:${CROPWRIGHT_PORT}"
            log_warn "Cropwright is reachable from your LAN and has NO login: use it only on a trusted network,"
            log_warn "never port-forward it to the internet, and put a reverse proxy with authentication in front"
            log_warn "for anything wider (SECURITY.md). Re-run with --local-only to keep it on this computer."
        fi
    fi
    if [[ "$OP_WITH_MONITORING" == 1 ]]; then
        echo "    Grafana     http://${h}:$(read_env_var "$ENV_FILE" GRAFANA_PORT) (default admin/admin: change it)"
    fi
    [[ -n "$OP_DOCS_URL" ]] && echo "    Docs        ${OP_DOCS_URL}"
    echo ""
    if [[ "$OP_CONTROL_PLANE_ONLY" == 1 ]]; then
        echo "  NOTE: control-plane-only install: NOT a functional inference install (no Triton)."
    fi
    if ! is_loopback_ipv4 "$OP_BIND_ADDRESS"; then
        log_warn "SECURITY: services are published on ${OP_BIND_ADDRESS} with NO authentication. See SECURITY.md."
    elif _has_tier cropwright && ! is_loopback_ipv4 "${CROPWRIGHT_BIND:-127.0.0.1}"; then
        echo "  Security: every OpenProcessor API port is bound to ${OP_BIND_ADDRESS}; Cropwright is the exception (your LAN, above)."
        echo "            The API has no auth; use a reverse proxy before exposing it."
    else
        echo "  Security: every port is bound to ${OP_BIND_ADDRESS} only. The API has no auth; use a reverse proxy before exposing it."
    fi
    failed="$(awk -F'\t' '$2 != "ok" && $2 != "skipped" && $2 != "planned" { printf "%s ", $1 }' "${OP_DIR}/.install/groups.tsv" 2>/dev/null || true)"
    if [[ -n "$failed" ]]; then
        echo ""
        echo "  Failed model groups: ${failed}"
        for g in $failed; do echo "    re-run: ./openprocessor models install --only ${g}"; done
    fi
    echo ""
    echo "  Manage: cd ${OP_REAL_DIR} && ./openprocessor status|logs|health|stop|start"
    echo "          ./setup-openprocessor.sh --repair | --rollback | --uninstall"
    [[ "$OP_NO_START" == 1 ]] && echo "  Nothing was started (--no-start): run ./openprocessor start, then ./openprocessor models install"
    return 0
}

# -----------------------------------------------------------------------------
# Flags (section 5.4)
# -----------------------------------------------------------------------------
usage() {
    cat <<'EOF'
Usage: setup-openprocessor.sh [options]

  --dir PATH              install directory (default ./openprocessor)
  --project NAME          compose project name + container prefix ([a-z0-9][a-z0-9_-]*)
  --version vX.Y.Z        pinned release (default: latest published release)
  --branch REF            testing install from a branch head (not reproducible)
  --image-tag TAG         run images by tag (OP_IMAGE_REPO prefix) instead of images.lock digests
  --release-dir DIR       install from locally built release assets (still checksum-verified;
                          offline for cropwright only if the bundle staged it)
  --tiers LIST | --all    core,curation,segmenter,vlm,trainer,cropwright
  --gpu-plan K=V,...      override placement: triton=1,segmenter=0,vlm=2,trainer=0
  --profile NAME          minimal|standard|full (Triton instance profile)
  --vlm-remote URL --vlm-model NAME [--vlm-key-file PATH]
  --vlm-model-id ID       choose a VLM catalog entry explicitly
  --bind ADDR             publish address (default 127.0.0.1); a specific non-loopback
                          address also narrows Cropwright to that interface
  --port-base N           shift the 46xx port block to N..N+12
  --with-monitoring       add Prometheus/Grafana/Loki (default-open dashboards)
  --sample-data           fetch the public COCO sample after install
  --skip-models           do not export/load models
  --no-start              configure and pull only; start nothing
  --unattended            never prompt (automatic when there is no terminal)
  --dry-run               print every state-changing command; run none
  --force                 accept a plan the hardware check refused
  --force-existing-dir    install into a non-empty dir this installer did not create (backed up first)
  --local-only            Cropwright only on this computer (default: reachable from your LAN)
  --yes                   confirm destructive steps (uninstall, rollback, upgrade) without a prompt
  --cpu [--control-plane-only]
  --repair | --rollback | --uninstall [--purge-volumes] [--purge-data] [--remove-images]
                          (--rollback restores the newest backup of a different version)
  --reset-hf-token        ask for a new HuggingFace token
  -h | --help
EOF
}

need_arg() {
    if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
        log_error "$1 needs a value"
        usage >&2
        exit "$EXIT_USAGE"
    fi
}

parse_args() {
    OP_TIERS="${OP_TIERS:-}"
    OP_BIND_ADDRESS="${OP_BIND_ADDRESS:-127.0.0.1}"
    if [[ -z "${OP_INSTALL_DIR:-}" && -f ./.install/state.json ]]; then
        # Running from inside an install dir means "this install".
        OP_INSTALL_DIR="."
    fi
    OP_INSTALL_DIR="${OP_INSTALL_DIR:-./openprocessor}"
    OP_VERSION="${OP_VERSION:-}"
    OP_BRANCH="${OP_BRANCH:-}"
    OP_IMAGE_TAG="${OP_IMAGE_TAG:-}"
    OP_RELEASE_DIR="${OP_RELEASE_DIR:-}"
    OP_IMAGE_REPO="${OP_IMAGE_REPO:-$OP_IMAGE_NAMESPACE}"
    OP_GPU_PLAN="${OP_GPU_PLAN:-}"
    GPU_PROFILE_FLAG="${GPU_PROFILE:-}"
    OP_VLM_URL="${OP_VLM_URL:-}"
    OP_VLM_MODEL="${OP_VLM_MODEL:-}"
    OP_VLM_KEY_FILE="${OP_VLM_KEY_FILE:-}"
    OP_VLM_CATALOG_ID="${OP_VLM_CATALOG_ID:-}"
    OP_PORT_BASE="${OP_PORT_BASE:-}"
    _OP_PROJECT_FLAG="${OP_PROJECT:-}"
    OP_ACTION="install"
    local unattended="${OP_UNATTENDED:-0}" dry="${OP_DRY_RUN:-0}" cpu="${OP_FORCE_CPU:-0}"
    local mon="${OP_WITH_MONITORING:-0}" sample="${OP_SAMPLE_DATA:-0}" cpo="${OP_CONTROL_PLANE_ONLY:-0}"
    OP_SKIP_MODELS=0; OP_NO_START=0; OP_FORCE=0; OP_PURGE_VOLUMES=0; OP_PURGE_DATA=0
    OP_REMOVE_IMAGES=0; OP_RESET_HF_TOKEN=0; OP_FORCE_EXISTING_DIR=0; OP_LOCAL_ONLY=0
    local yes=0

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dir) need_arg "$@"; OP_INSTALL_DIR="$2"; shift 2 ;;
            --project) need_arg "$@"; _OP_PROJECT_FLAG="$2"; shift 2 ;;
            --version) need_arg "$@"; OP_VERSION="$2"; shift 2 ;;
            --branch) need_arg "$@"; OP_BRANCH="$2"; shift 2 ;;
            --image-tag) need_arg "$@"; OP_IMAGE_TAG="$2"; shift 2 ;;
            --release-dir) need_arg "$@"; OP_RELEASE_DIR="$2"; shift 2 ;;
            --tiers) need_arg "$@"; OP_TIERS="$2"; shift 2 ;;
            --all) OP_TIERS="$(IFS=,; echo "${TIER_LIST[*]}")"; shift ;;
            --gpu-plan) need_arg "$@"; OP_GPU_PLAN="$2"; shift 2 ;;
            --profile) need_arg "$@"; GPU_PROFILE_FLAG="$2"; shift 2 ;;
            --vlm-remote) need_arg "$@"; OP_VLM_URL="$2"; shift 2 ;;
            --vlm-model) need_arg "$@"; OP_VLM_MODEL="$2"; shift 2 ;;
            --vlm-key-file) need_arg "$@"; OP_VLM_KEY_FILE="$2"; shift 2 ;;
            --vlm-model-id) need_arg "$@"; OP_VLM_CATALOG_ID="$2"; shift 2 ;;
            --bind) need_arg "$@"; OP_BIND_ADDRESS="$2"; shift 2 ;;
            --port-base) need_arg "$@"; OP_PORT_BASE="$2"; shift 2 ;;
            --with-monitoring) mon=1; shift ;;
            --sample-data) sample=1; shift ;;
            --skip-models) OP_SKIP_MODELS=1; shift ;;
            --no-start) OP_NO_START=1; shift ;;
            --unattended) unattended=1; shift ;;
            --dry-run) dry=1; shift ;;
            --cpu) cpu=1; shift ;;
            --control-plane-only) cpo=1; shift ;;
            --force) OP_FORCE=1; shift ;;
            --force-existing-dir) OP_FORCE_EXISTING_DIR=1; shift ;;
            --local-only) OP_LOCAL_ONLY=1; shift ;;
            --yes|-y) yes=1; shift ;;
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
                usage >&2
                exit "$EXIT_USAGE"
                ;;
        esac
    done

    OP_UNATTENDED=0; _truthy "$unattended" && OP_UNATTENDED=1
    # Consent to destructive steps only ever comes from an explicit flag or
    # env var, never from the terminal simply being absent.
    OP_ASSUME_YES=0
    if [[ "$yes" == 1 || "$OP_UNATTENDED" == 1 ]]; then OP_ASSUME_YES=1; fi
    OP_DRY_RUN=0; _truthy "$dry" && OP_DRY_RUN=1
    OP_FORCE_CPU=0; _truthy "$cpu" && OP_FORCE_CPU=1
    OP_WITH_MONITORING=0; _truthy "$mon" && OP_WITH_MONITORING=1
    OP_SAMPLE_DATA=0; _truthy "$sample" && OP_SAMPLE_DATA=1
    OP_CONTROL_PLANE_ONLY=0; _truthy "$cpo" && OP_CONTROL_PLANE_ONLY=1
    if [[ "$OP_CONTROL_PLANE_ONLY" == 1 && "$OP_WITH_MONITORING" == 1 ]]; then
        die "--with-monitoring is not available with --control-plane-only (only OpenSearch and the API run)" "$EXIT_USAGE"
    fi
    if ! tty_usable; then OP_UNATTENDED=1; fi

    if [[ -n "$_OP_PROJECT_FLAG" ]] && ! validate_project_name "$_OP_PROJECT_FLAG"; then
        die "--project must match [a-z0-9][a-z0-9_-]* (got '${_OP_PROJECT_FLAG}')" "$EXIT_USAGE"
    fi
    if [[ -n "$OP_PORT_BASE" ]] && { [[ ! "$OP_PORT_BASE" =~ ^[0-9]+$ ]] || (( OP_PORT_BASE < 1024 || OP_PORT_BASE > 65500 )); }; then
        die "--port-base must be a number in 1024..65500" "$EXIT_USAGE"
    fi
    if [[ -n "$GPU_PROFILE_FLAG" && ! "$GPU_PROFILE_FLAG" =~ ^(minimal|standard|full)$ ]]; then
        die "--profile must be minimal, standard or full" "$EXIT_USAGE"
    fi
    if [[ -n "$OP_IMAGE_TAG" && ! "$OP_IMAGE_TAG" =~ ^[A-Za-z0-9_][A-Za-z0-9._-]{0,127}$ ]]; then
        die "--image-tag is not a valid Docker tag" "$EXIT_USAGE"
    fi
    if [[ "$OP_IMAGE_TAG" == latest ]]; then
        die "--image-tag latest is refused: pin a version or a local build tag" "$EXIT_USAGE"
    fi
    [[ "$OP_IMAGE_REPO" =~ ^[a-z0-9][a-z0-9._/-]*$ ]] || die "OP_IMAGE_REPO is not a valid image repository" "$EXIT_USAGE"
    if [[ -n "$OP_VERSION" && -n "$OP_BRANCH" ]]; then
        die "use either --version or --branch" "$EXIT_USAGE"
    fi
    if [[ -n "$OP_RELEASE_DIR" ]]; then
        [[ -n "$OP_VERSION" && -z "$OP_BRANCH" ]] || die "--release-dir needs --version (the version the assets were built for)" "$EXIT_USAGE"
        OP_RELEASE_DIR="$(cd -P "$OP_RELEASE_DIR" 2>/dev/null && pwd)" || die "--release-dir not found" "$EXIT_USAGE"
        [[ -f "${OP_RELEASE_DIR}/SHA256SUMS" ]] || die "--release-dir has no SHA256SUMS (build it with scripts/release/build_deploy_bundle.sh)" "$EXIT_USAGE"
    fi
    if [[ -n "$OP_VLM_URL" && -z "$OP_VLM_MODEL" ]]; then
        die "--vlm-remote needs --vlm-model NAME" "$EXIT_USAGE"
    fi
    if [[ -n "$OP_VLM_URL" ]] && { [[ "$OP_VLM_URL" == *[[:space:]]* ]] || ! url_host "$OP_VLM_URL" >/dev/null; }; then
        die "--vlm-remote must be an http(s) URL (got '${OP_VLM_URL}')" "$EXIT_USAGE"
    fi
    if [[ -n "$OP_VLM_MODEL" && ! "$OP_VLM_MODEL" =~ ^[A-Za-z0-9][A-Za-z0-9._:/@-]{0,199}$ ]]; then
        die "--vlm-model has unsupported characters" "$EXIT_USAGE"
    fi
    local b
    b="$(normalize_bind "$OP_BIND_ADDRESS")" || exit "$EXIT_USAGE"
    OP_BIND_ADDRESS="$b"
    local base
    for base in "${OP_ARTIFACT_BASE_URL:-}" "${OP_RAW_BASE_URL:-}" "${CW_ARTIFACT_BASE_URL:-}" "${CW_RAW_BASE_URL:-}"; do
        [[ -z "$base" ]] && continue
        validate_https_base "$base" || die "artifact base URLs must be https:// (got '${base}')" "$EXIT_USAGE"
    done
    [[ "$OP_GH_REPO" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || die "OP_GH_REPO must be owner/repo" "$EXIT_USAGE"
    [[ "$CW_GH_REPO" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || die "CW_GH_REPO must be owner/repo" "$EXIT_USAGE"
    if [[ "$OP_ACTION" != uninstall ]] && (( OP_PURGE_VOLUMES || OP_PURGE_DATA || OP_REMOVE_IMAGES )); then
        die "--purge-volumes, --purge-data and --remove-images only go with --uninstall" "$EXIT_USAGE"
    fi
}

# -----------------------------------------------------------------------------
# Install / repair
# -----------------------------------------------------------------------------
# classify_install_dir -> DIR_STATE = fresh | owned | forced
# owned: .install/state.json written by this installer for this very dir.
# A non-empty dir without it (a git checkout, another tool's dir) is refused
# unless --force-existing-dir, which backs its files up first.
classify_install_dir() {
    local sd
    if [[ ! -e "$OP_REAL_DIR" ]]; then
        DIR_STATE=fresh
        return 0
    fi
    [[ -d "$OP_REAL_DIR" ]] || die "${OP_REAL_DIR} exists and is not a directory" "$EXIT_USAGE"
    if [[ -f "$STATE_FILE" ]]; then
        sd="$(state_get install_dir || true)"
        [[ "$sd" == "$OP_REAL_DIR" ]] \
            || die "${OP_REAL_DIR}/.install/state.json records another install (${sd:-no dir}); refusing" "$EXIT_COLLISION"
        DIR_STATE=owned
        return 0
    fi
    if [[ -z "$(find "$OP_REAL_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        DIR_STATE=fresh
        return 0
    fi
    if [[ "$OP_FORCE_EXISTING_DIR" == 1 ]]; then
        log_warn "${OP_REAL_DIR} is not empty and was not created by this installer; adopting it (--force-existing-dir)"
        DIR_STATE=forced
        return 0
    fi
    local what="not empty"
    [[ -e "${OP_REAL_DIR}/.git" ]] && what="a git checkout"
    die "${OP_REAL_DIR} is ${what} and was not created by this installer; nothing was changed. Use another --dir, or --force-existing-dir to adopt it (its files are backed up first)" "$EXIT_COLLISION"
}

# backup_foreign_dir -- before adopting a dir we did not create, save every
# file we could overwrite (everything but data/model/cache trees).
backup_foreign_dir() {
    local ts dest
    ts="$(date -u +%Y%m%dT%H%M%SZ)"
    dest="${OP_REAL_DIR}/backups"
    mkdir -p "$dest"
    chmod 700 "$dest"
    tar -C "$OP_REAL_DIR" --exclude=./models --exclude=./pytorch_models --exclude=./data --exclude=./cache \
        --exclude=./backups -czf "${dest}/${ts}-pre-install.tar.gz" . \
        || die "could not back up ${OP_REAL_DIR}; nothing was changed"
    chmod 600 "${dest}/${ts}-pre-install.tar.gz"
    log_info "backed up the existing files to ${dest}/${ts}-pre-install.tar.gz"
}

# dry_run_report -- what a real run would change in an existing dir
dry_run_report() {
    [[ "$OP_DRY_RUN" == 1 && "$DIR_STATE" != fresh ]] || return 0
    local rel changed=0 key
    log_step "Dry-run: changes a real run would make in ${OP_REAL_DIR}"
    while IFS= read -r rel; do
        if [[ ! -f "${OP_REAL_DIR}/${rel}" ]]; then
            echo "  would add    ${rel}"; changed=1
        elif ! cmp -s "${OP_DIR}/${rel}" "${OP_REAL_DIR}/${rel}"; then
            echo "  would change ${rel}"; changed=1
        fi
    done < <(manifest_installed_files "$OP_DIR")
    while IFS= read -r key; do
        [[ -z "$key" ]] && continue
        if [[ "$(read_env_var "${OP_DIR}/.env" "$key" || true)" != "$(read_env_var "${OP_REAL_DIR}/.env" "$key" || true)" ]]; then
            echo "  would set    .env ${key}"; changed=1
        fi
    done < <(awk -F= '/^[A-Z_][A-Z0-9_]*=/ { print $1 }' "${OP_DIR}/.env" 2>/dev/null | sort -u)
    (( changed )) || echo "  no file or .env changes"
}

# select_tiers_interactive REC -> the validated tier list; an invalid answer
# re-prompts, and three invalid answers exit 2 (never a silent fallback).
select_tiers_interactive() {
    local rec="$1" reply attempt
    echo "Recommended tiers for this machine: ${rec}" >&2
    echo "Available: ${TIER_LIST[*]} (dependencies are added automatically)" >&2
    for attempt in 1 2 3; do
        prompt_line reply "Tiers to install [${rec}]: " "--tiers LIST"
        [[ -z "$reply" ]] && reply="$rec"
        reply="${reply// /}"
        if tiers_validate "$reply"; then
            echo "$reply"
            return 0
        fi
        (( attempt < 3 )) && echo "Try again (comma-separated, from: ${TIER_LIST[*]})." >&2
    done
    die "no valid tier list after 3 tries" "$EXIT_USAGE"
}

print_gpu_plan() {
    local plan="$1" key
    for key in TRITON_GPU_ID SEGMENTER_GPU_ID VLM_GPU_ID OP_TRAIN_GPU_ORDER GPU_PROFILE SEGMENTER_INSTANCES VLM_CATALOG_ID vlm_available_gb; do
        printf '  %-22s %s\n' "$key" "$(plan_get "$plan" "$key")"
    done
    while IFS= read -r key; do
        log_warn "${key#warn=}"
    done < <(printf '%s\n' "$plan" | grep '^warn=' || true)
}

do_install() {
    local mode="$1" existing=0 plan rec tiers_arg=""
    if [[ -f "$STATE_FILE" ]]; then
        require_owned_install
        [[ -n "$(state_get version || true)" ]] && existing=1
    fi
    if [[ "$mode" == repair ]]; then
        (( existing )) || die "--repair needs an existing install (no ${STATE_FILE})"
        FORCED_REF="$(state_get version)"
        FORCED_MODE="$(state_get mode)"
        [[ -n "$FORCED_REF" && "$FORCED_MODE" =~ ^(release|branch)$ ]] || die "state.json has no usable version to repair"
        if [[ "$FORCED_MODE" == release ]]; then
            validate_version "$FORCED_REF" || die "state.json version '${FORCED_REF}' is not a release tag"
        else
            [[ "$FORCED_REF" =~ ^[0-9a-f]{40}$ ]] || die "state.json branch ref is not a commit SHA"
            OP_BRANCH="${OP_BRANCH:-${FORCED_REF:0:12}}"
        fi
        if [[ "$(state_get image_mode)" == tag && -z "$OP_IMAGE_TAG" ]]; then
            OP_IMAGE_TAG="$(state_get image_tag)"
        fi
    fi
    if (( existing )); then
        # A re-run or upgrade keeps what is installed unless told otherwise.
        if [[ -z "$OP_TIERS" ]]; then
            OP_TIERS="$(state_get tiers)"
            OP_TIERS="${OP_TIERS// /,}"
            log_info "keeping the installed tiers: ${OP_TIERS} (pass --tiers to change them)"
        fi
        [[ "$(state_get control_plane_only)" == 1 ]] && OP_CONTROL_PLANE_ONLY=1
        if [[ "$(state_get with_monitoring)" == 1 && "$OP_CONTROL_PLANE_ONLY" != 1 ]]; then
            OP_WITH_MONITORING=1
        fi
        INSTALLED_AT="$(state_get installed_at || true)"
    elif [[ "$DIR_STATE" == forced ]]; then
        local envp
        envp="$(read_env_var "$ENV_FILE" COMPOSE_PROJECT_NAME || true)"
        if [[ -n "$envp" && -z "$_OP_PROJECT_FLAG" ]]; then
            OP_PROJECT="$envp"
        fi
    fi
    [[ -n "$INSTALLED_AT" ]] || INSTALLED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    # --- preflight: docker, collisions (fail closed) ----------------------
    log_step "Preflight"
    require_docker
    guard_projects
    if [[ ! -f "$STATE_FILE" && "$OP_DRY_RUN" != 1 ]]; then
        # Claim the dir now that the project name is known to be free, so an
        # interrupted install can be resumed (and only resumed) by this installer.
        state_write
    fi
    local rt
    rt="$(docker_runtime_has_nvidia)"
    case "$rt" in
        1) log_info "Docker has the nvidia runtime registered" ;;
        0) log_warn "Docker lists no nvidia runtime; relying on the --gpus container probe" ;;
        *) log_warn "could not determine Docker's runtimes; relying on the --gpus container probe" ;;
    esac

    # --- release artifacts --------------------------------------------------
    log_step "Release"
    if [[ -n "$FORCED_REF" ]]; then
        RESOLVED_REF="$FORCED_REF"
        RESOLVED_MODE="$FORCED_MODE"
    elif [[ -n "${_OP_BOOT_REF:-}" && -z "$OP_VERSION" && -z "$OP_BRANCH" ]]; then
        RESOLVED_REF="$_OP_BOOT_REF"
        RESOLVED_MODE="$_OP_BOOT_MODE"
    else
        resolve_install_ref
    fi
    INSTALL_REF="$RESOLVED_REF"
    if [[ "$RESOLVED_MODE" == branch ]]; then
        log_warn "================================================================"
        log_warn " TESTING install from branch ${OP_BRANCH} at ${RESOLVED_REF:0:12}"
        log_warn " Not reproducible: files are not checksum-verified and images"
        log_warn " run by the sha-${RESOLVED_REF:0:12} tag."
        log_warn "================================================================"
        [[ -n "$OP_IMAGE_TAG" ]] || OP_IMAGE_TAG="sha-${RESOLVED_REF:0:12}"
    fi
    IMAGE_MODE=lock
    [[ -n "$OP_IMAGE_TAG" ]] && IMAGE_MODE=tag
    local prev
    prev="$(state_get version || true)"
    if (( existing )) && [[ -n "$prev" && "$prev" != "$INSTALL_REF" && "$mode" != repair ]]; then
        log_info "upgrade: ${prev} -> ${INSTALL_REF}"
        require_destructive_consent "Upgrade ${OP_PROJECT} from ${prev} to ${INSTALL_REF} (files are backed up first)"
    fi
    fetch_release_artifacts "$INSTALL_REF" "$RESOLVED_MODE" "${OP_DIR}/.install/staging"
    log_success "release ${INSTALL_REF} downloaded and verified"
    if (( existing )); then
        local bdir
        bdir="$(backup_install)" || die "could not back up the current install; nothing was changed"
        log_info "backed up the current install to ${bdir}"
    fi
    install_staged "${OP_DIR}/.install/staging"

    # Only now, from verified files, is any library sourced.
    # shellcheck source=scripts/lib/vlm_catalog.sh
    source "${OP_DIR}/scripts/lib/vlm_catalog.sh"
    # shellcheck source=scripts/lib/model_setup.sh
    source "${OP_DIR}/scripts/lib/model_setup.sh"
    # shellcheck source=scripts/lib/image_keys.sh
    source "${OP_DIR}/scripts/lib/image_keys.sh"
    # shellcheck source=scripts/lib/opensearch_heap.sh
    source "${OP_DIR}/scripts/lib/opensearch_heap.sh"

    env_create_or_merge
    env_set COMPOSE_PROJECT_NAME "$OP_PROJECT"
    if [[ "$OP_CONTROL_PLANE_ONLY" != 1 ]]; then
        # Without a primary ingest detector /ingest answers 503 on a fresh install.
        env_set_default OP_INGEST_PRIMARY_DETECTOR_MODEL yolov11_small_trt_end2end
    fi

    # A re-run keeps the installed catalog VLM unless --vlm-model-id says
    # otherwise: re-planning must never silently swap the model.
    if (( existing )) && [[ -z "$OP_VLM_CATALOG_ID" && -z "$OP_VLM_URL" && ",${OP_TIERS}," == *",vlm,"* ]]; then
        local kept_vlm
        kept_vlm="$(read_env_var "$ENV_FILE" VLM_CATALOG_ID || true)"
        if [[ -n "$kept_vlm" && -n "$(vlm_catalog_field "$kept_vlm" hf_repo 2>/dev/null || true)" ]]; then
            OP_VLM_CATALOG_ID="$kept_vlm"
            log_info "keeping the installed VLM: ${kept_vlm} (pass --vlm-model-id to change it)"
        fi
    fi

    # Likewise the installed GPU placement: re-planning from today's free VRAM
    # would move services (and recreate their containers) on an unchanged re-run.
    if (( existing )) && [[ -z "$OP_GPU_PLAN" && "$OP_CONTROL_PLANE_ONLY" != 1 ]]; then
        local kept_plan="" kv_id
        kv_id="$(read_env_var "$ENV_FILE" TRITON_GPU_ID || true)"
        [[ -n "$kv_id" ]] && kept_plan+="triton=${kv_id},"
        kv_id="$(read_env_var "$ENV_FILE" SEGMENTER_GPU_ID || true)"
        [[ -n "$kv_id" && ",${OP_TIERS}," == *",segmenter,"* ]] && kept_plan+="segmenter=${kv_id},"
        kv_id="$(read_env_var "$ENV_FILE" VLM_GPU_ID || true)"
        [[ -n "$kv_id" && ",${OP_TIERS}," == *",vlm,"* && -z "$OP_VLM_URL" ]] && kept_plan+="vlm=${kv_id},"
        kv_id="$(read_env_var "$ENV_FILE" OP_TRAIN_GPU_ORDER || true)"
        kv_id="${kv_id%%,*}"
        [[ -n "$kv_id" && ",${OP_TIERS}," == *",trainer,"* ]] && kept_plan+="trainer=${kv_id},"
        if [[ -n "$kept_plan" ]]; then
            OP_GPU_PLAN="${kept_plan%,}"
            log_info "keeping the installed GPU placement: ${OP_GPU_PLAN} (pass --gpu-plan to change it)"
        fi
    fi

    # --- consent ------------------------------------------------------------
    require_bind_consent "$OP_BIND_ADDRESS"
    env_set OP_BIND_ADDRESS "$OP_BIND_ADDRESS"
    if [[ -n "$OP_VLM_URL" ]]; then
        require_external_vlm_consent "$OP_VLM_URL"
    fi

    # --- GPUs and tiers -------------------------------------------------------
    log_step "GPUs"
    local raw_gpus gpus=""
    raw_gpus="$(gpu_query || true)"
    gpus="$(printf '%s\n' "$raw_gpus" | gpu_normalize)"
    if (( existing )) && [[ -n "$gpus" ]]; then
        gpus="$(gpu_subtract_own "$gpus" "$(gpu_own_usage)")"
    fi
    if [[ "$OP_FORCE_CPU" == 1 || -z "$gpus" ]]; then
        if [[ "$OP_CONTROL_PLANE_ONLY" != 1 ]]; then
            log_error "No usable NVIDIA GPU. There is no CPU inference path today."
            log_error "Re-run with --cpu --control-plane-only to install OpenSearch, the API"
            log_error "and Cropwright without Triton (inference routes will return 503)."
            exit "$EXIT_GPU"
        fi
    fi
    if [[ "$OP_CONTROL_PLANE_ONLY" == 1 ]]; then
        log_warn "control-plane-only install: no Triton, no exports, inference routes return 503."
        log_warn "This is NOT a functional inference install."
        local t
        for t in segmenter vlm trainer; do
            [[ ",${OP_TIERS}," == *",${t},"* ]] && die "--control-plane-only cannot install the ${t} tier" "$EXIT_USAGE"
        done
        SELECTED_TIERS="core"
        if [[ ",${OP_TIERS}," == *",cropwright,"* ]]; then SELECTED_TIERS="core cropwright"; fi
        # yolo-api reserves a GPU in the base compose; without one it could
        # never start, so the control-plane install drops that reservation.
        local cver
        cver="$(dc version --short 2>/dev/null || true)"
        _version_ge "${cver#v}" "$COMPOSE_MIN_OVERRIDE" \
            || die "Docker Compose ${COMPOSE_MIN_OVERRIDE}+ is needed for --control-plane-only (found '${cver:-unknown}')"
        ( umask 022; printf 'services:\n  yolo-api:\n    deploy: !reset {}\n' > "${OP_DIR}/docker-compose.cpu.yml" )
        GPU_PLAN_SUMMARY="none (control-plane-only)"
    else
        if [[ -n "$OP_TIERS" ]]; then
            tiers_validate "$OP_TIERS" || exit "$EXIT_USAGE"
            tiers_arg="$(tiers_close_dependencies "$OP_TIERS")"
        fi
        if [[ -n "$OP_VLM_URL" && " $tiers_arg " == *" vlm "* ]]; then
            die "choose either the local vlm tier or --vlm-remote, not both" "$EXIT_USAGE"
        fi
        local remote=0
        [[ -n "$OP_VLM_URL" ]] && remote=1
        if ! plan="$(recommend_plan "$gpus" "$tiers_arg" "force=${OP_FORCE}" "vlm_id=${OP_VLM_CATALOG_ID}" \
                "gpu_plan=${OP_GPU_PLAN}" "profile=${GPU_PROFILE_FLAG}" "remote=${remote}")"; then
            print_gpu_plan "$plan"
            die "GPU plan refused: $(plan_get "$plan" refuse)" "$EXIT_GPU"
        fi
        print_gpu_plan "$plan"
        local busy line
        while read -r line; do
            [[ "$line" =~ ^[0-9]+$ ]] || continue
            busy="$(nvidia-smi -i "$line" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)"
            (( busy > 0 )) && log_info "GPU ${line}: ${busy} other process(es) hold VRAM"
        done < <(printf '%s\n' "$plan" | sed -n -E 's/^warn=GPU ([0-9]+):.*in use.*/\1/p')
        rec="$(plan_get "$plan" recommended_tiers)"
        if [[ -z "$tiers_arg" && ",${rec}," == *",segmenter,"* ]] && ! hf_token_available \
                && { [[ "$OP_UNATTENDED" == 1 ]] || ! tty_usable; }; then
            rec="${rec//,segmenter/}"
            log_warn "the segmenter tier needs a HuggingFace token (SAM 3 is gated) and none was given: not selected (set HF_TOKEN_FILE and add --tiers ...,segmenter)"
        fi
        if [[ -z "$tiers_arg" ]]; then
            local chosen="$rec"
            if [[ "$OP_UNATTENDED" != 1 ]] && tty_usable; then
                chosen="$(select_tiers_interactive "$rec")"
            fi
            tiers_arg="$(tiers_close_dependencies "$chosen")"
            if ! plan="$(recommend_plan "$gpus" "$tiers_arg" "force=${OP_FORCE}" "vlm_id=${OP_VLM_CATALOG_ID}" \
                    "gpu_plan=${OP_GPU_PLAN}" "profile=${GPU_PROFILE_FLAG}" "remote=${remote}")"; then
                die "GPU plan refused: $(plan_get "$plan" refuse)" "$EXIT_GPU"
            fi
        fi
        SELECTED_TIERS="$(tiers_close_dependencies "$(plan_get "$plan" tiers)")"
        VLM_PICK="$(plan_get "$plan" VLM_CATALOG_ID)"
        local key val
        for key in TRITON_GPU_ID API_GPU_ID SEGMENTER_GPU_ID VLM_GPU_ID EVALUATOR_GPU_ID \
                OP_TRAIN_GPU_ORDER OP_TRAIN_DEFAULT_GPUS OP_GPU_ALLOWED_IDS GPU_PROFILE SEGMENTER_INSTANCES; do
            val="$(plan_get "$plan" "$key")"
            if [[ "$key" == SEGMENTER_* && " $(tiers_close_dependencies "$(plan_get "$plan" tiers)") " != *" segmenter "* \
                    && -n "$(read_env_var "$ENV_FILE" "$key" || true)" ]]; then
                continue
            fi
            if [[ -n "$OP_GPU_PLAN" || ( "$key" == GPU_PROFILE && -n "$GPU_PROFILE_FLAG" ) ]]; then
                env_set "$key" "$val"
            else
                env_set_default "$key" "$val"
            fi
        done
        env_set OP_GPU_LABELS "$(printf '%s\n' "$raw_gpus" | gpu_labels)"
        GPU_PROFILE="$(read_env_var "$ENV_FILE" GPU_PROFILE)"
        GPU_PLAN_SUMMARY="triton=$(read_env_var "$ENV_FILE" TRITON_GPU_ID),segmenter=$(read_env_var "$ENV_FILE" SEGMENTER_GPU_ID),vlm=$(read_env_var "$ENV_FILE" VLM_GPU_ID),trainer=$(read_env_var "$ENV_FILE" OP_TRAIN_GPU_ORDER),profile=${GPU_PROFILE}"
        if _has_tier vlm; then
            [[ -n "$VLM_PICK" ]] || die "the vlm tier was selected but no catalog entry was chosen" "$EXIT_GPU"
            local maxi ctx
            maxi="$(vlm_catalog_field "$VLM_PICK" max_images)"
            ctx="$(vlm_catalog_field "$VLM_PICK" served_context)"
            env_set VLM_CATALOG_ID "$VLM_PICK"
            env_set VLM_MODEL "$(vlm_catalog_field "$VLM_PICK" hf_repo)"
            env_set VLM_SERVED_MODEL_NAME local-vlm
            env_set OP_VLM_MODEL local-vlm
            env_set OP_VLM_URL "http://vlm:8000/v1"
            env_set OP_LOCAL_VLM_ENDPOINT env
            env_set VLM_MAX_MODEL_LEN "$ctx"
            env_set VLM_LIMIT_MM_IMAGES "$maxi"
            env_set OP_VLM_MAX_IMAGES_PER_CALL "$maxi"
            env_set VLM_GPU_MEMORY_UTILIZATION "$(plan_get "$plan" VLM_GPU_MEMORY_UTILIZATION)"
            env_set VLM_GPU_TOTAL_MIB "$(plan_get "$plan" VLM_GPU_TOTAL_MIB)"
            log_info "local VLM: ${VLM_PICK} ($(read_env_var "$ENV_FILE" VLM_MODEL), status $(plan_get "$plan" vlm_status))"
        fi
    fi
    log_info "tiers: ${SELECTED_TIERS}"

    if [[ -n "$OP_VLM_URL" ]]; then
        env_set OP_VLM_URL "$OP_VLM_URL"
        env_set OP_VLM_MODEL "$OP_VLM_MODEL"
        if [[ -n "$OP_VLM_KEY_FILE" ]]; then
            check_secret_file "$OP_VLM_KEY_FILE" "--vlm-key-file" || exit "$EXIT_USAGE"
            ( umask 077; mkdir -p "${OP_DIR}/secrets/vlm" )
            chmod 700 "${OP_DIR}/secrets" "${OP_DIR}/secrets/vlm"
            local _vk
            _vk="$(<"$OP_VLM_KEY_FILE")"
            ( umask 077; printf '%s' "$_vk" > "${OP_DIR}/secrets/vlm/env" )
            _vk=""
            chmod 600 "${OP_DIR}/secrets/vlm/env"
            log_info "remote VLM key stored in secrets/vlm/env (mode 600)"
        fi
    fi

    # --- .env: profiles, heap, sources ----------------------------------------
    local profiles=() t
    for t in $SELECTED_TIERS; do
        case "$t" in
            curation|segmenter|vlm) profiles+=("$t") ;;
            trainer) profiles+=(training) ;;
        esac
    done
    [[ "$OP_WITH_MONITORING" == 1 ]] && profiles+=(monitoring)
    env_set COMPOSE_PROFILES "$(IFS=,; echo "${profiles[*]:-}")"
    local heap
    heap="$(opensearch_heap_for_host)" || die "could not read host memory from ${OP_MEMINFO_PATH:-/proc/meminfo} to size the OpenSearch heap"
    env_set_default OPENSEARCH_HEAP "$heap"
    env_set_default OP_SOURCE_ROOT_HOST ./data

    ensure_hf_token
    plan_ports
    apply_image_pins
    if _has_tier cropwright; then
        choose_cropwright_bind
        setup_cropwright
    fi

    local d
    for d in models pytorch_models data data/samples cache cache/huggingface cache/vllm; do
        mkdir -p "${OP_DIR}/${d}"
    done
    assert_compose_config
    state_write

    # --- disk -----------------------------------------------------------------
    local need=58 free_kib root
    _has_tier curation && need=$((need + 19))
    _has_tier segmenter && need=$((need + 11))
    _has_tier vlm && need=$((need + 34))
    _has_tier trainer && need=$((need + 29))
    need=$((need + 20))
    free_kib="$(df -Pk "$OP_DIR" 2>/dev/null | awk 'NR==2 { print $4 }')" || free_kib=""
    root="$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || true)"
    if [[ "$free_kib" =~ ^[0-9]+$ ]]; then
        if (( free_kib / 1024 / 1024 < need )) && [[ "$OP_FORCE" != 1 ]]; then
            die "need ~${need} GB free on ${OP_DIR}'s filesystem, have $(( free_kib / 1024 / 1024 )) GB (use --force to try anyway)"
        fi
    else
        log_warn "could not check free disk space on ${OP_DIR}"
    fi
    if [[ -n "$root" ]] && free_kib="$(df -Pk "$root" 2>/dev/null | awk 'NR==2 { print $4 }')" && [[ "$free_kib" =~ ^[0-9]+$ ]]; then
        (( free_kib / 1024 / 1024 < need )) && log_warn "Docker's data root ${root} has $(( free_kib / 1024 / 1024 )) GB free; images need ~${need} GB"
    else
        log_warn "could not check free disk space on Docker's data root"
    fi

    # --- images ---------------------------------------------------------------
    log_step "Images"
    pull_images
    verify_image_digests
    state_write
    if [[ "$OP_CONTROL_PLANE_ONLY" != 1 ]]; then
        local ids=()
        mapfile -t ids < <(for key in TRITON_GPU_ID SEGMENTER_GPU_ID VLM_GPU_ID OP_TRAIN_GPU_ORDER; do
            read_env_var "$ENV_FILE" "$key"; done | sort -u)
        gpu_container_probe "${ids[@]}"
    fi

    # --- host dir ownership (containers run as uid 1000) ----------------------
    if [[ "$(id -u)" != 1000 ]]; then
        local api_img
        api_img="$(read_env_var "$ENV_FILE" OP_API_IMAGE || true)"
        [[ -n "$api_img" ]] || api_img="${OP_IMAGE_REPO}/openprocessor:${OP_IMAGE_TAG}"
        docker_mut run --rm --user 0 --entrypoint chown -v "${OP_DIR}:/w" "$api_img" \
            -R 1000:1000 /w/models /w/pytorch_models /w/data /w/cache \
            || die "could not hand models/ data/ cache/ to the container user (uid 1000)"
    fi

    if [[ "$OP_NO_START" == 1 ]]; then
        log_info "--no-start: configuration and images are ready; nothing was started"
        HEALTH_RESULT="not-started"
        state_write
        print_summary "$(_health_host)"
        return 0
    fi

    # --- model setup (Triton first, then the export groups) -------------------
    TRITON_HTTP_PORT="$(read_env_var "$ENV_FILE" TRITON_HTTP_PORT || echo 4600)"
    # These four are read by scripts/lib/model_setup.sh (same shell).
    # shellcheck disable=SC2034
    API_PORT="$(read_env_var "$ENV_FILE" API_PORT || echo 4603)"
    OP_HEALTH_HOST="$(_health_host)"
    # shellcheck disable=SC2034
    MODEL_SETUP_TRITON_DIGEST="$TRITON_DIGEST"
    # shellcheck disable=SC2034
    MODEL_SETUP_LOGDIR="${OP_DIR}/.install/logs"
    local group_rc=0
    if [[ "$OP_CONTROL_PLANE_ONLY" == 1 ]]; then
        dc up -d opensearch || die "compose up opensearch failed"
        dc up -d --no-deps yolo-api || die "compose up yolo-api failed"
    elif [[ "$OP_SKIP_MODELS" == 1 ]]; then
        log_info "--skip-models: engines are not exported; run ./openprocessor models install later"
    else
        log_step "Model setup"
        dc up -d triton-server || die "could not start Triton"
        if [[ "$OP_DRY_RUN" != 1 ]] && ! wait_http "http://${OP_HEALTH_HOST}:${TRITON_HTTP_PORT}/v2/health/live" 180; then
            die "Triton did not come up: ./openprocessor logs triton-server" "$EXIT_HEALTH"
        fi
        if model_setup_run_groups "$SELECTED_TIERS"; then
            group_rc=0
        else
            group_rc=1
        fi
        state_write
    fi

    # --- up + health ----------------------------------------------------------
    log_step "Start"
    if [[ "$OP_CONTROL_PLANE_ONLY" != 1 ]]; then
        dc up -d --remove-orphans || die "compose up failed"
    fi
    if _has_tier cropwright; then
        dc_cw up -d --no-build --remove-orphans || die "cropwright up failed"
    fi
    local health_rc=0
    run_health || health_rc=1
    if [[ "$OP_SAMPLE_DATA" == 1 ]]; then
        log_step "Sample data (public COCO subset)"
        if [[ "$OP_DRY_RUN" == 1 || "$health_rc" == 0 ]]; then
            model_setup_sample_coco || log_warn "sample fetch failed: re-run ./openprocessor sample coco"
        else
            log_warn "sample data skipped: the stack is not healthy"
        fi
    fi
    state_write
    print_summary "$(_health_host)"
    if [[ "$OP_DRY_RUN" == 1 ]]; then
        dry_run_report
        log_info "dry-run finished: nothing was written, pulled, started or removed"
        return 0
    fi
    if (( health_rc != 0 || group_rc != 0 )); then
        log_error "install finished with failures (see above); fix and run ./setup-openprocessor.sh --repair"
        exit "$EXIT_HEALTH"
    fi
    log_success "OpenProcessor ${INSTALL_REF} is installed and healthy"
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
main() {
    # Nothing below may ever be traced: the HF token passes through here.
    { set +x; } 2>/dev/null
    set -euo pipefail
    shopt -s inherit_errexit
    unset BASH_XTRACEFD
    # Secrets get explicit 600/700 modes; release files and data dirs stay
    # readable because containers running as other users bind-mount them.
    umask 022
    _OP_SCRATCH=""
    bootstrap_child_setup
    trap _op_cleanup EXIT

    parse_args "$@"
    if needs_bootstrap "${_OP_SELF_PATH:-}"; then
        bootstrap_reexec "$@"
    fi

    if (( EUID == 0 )); then
        log_warn "running as root: the install dir will be root-owned; prefer a user in the docker group"
    fi

    OP_PROJECT="${_OP_PROJECT_FLAG:-openprocessor}"
    OP_REAL_DIR="$(realpath -m -- "$OP_INSTALL_DIR")" || die "bad install dir: ${OP_INSTALL_DIR}"
    [[ "$OP_REAL_DIR" != / ]] || die "refusing to use '/' as the install dir"
    SELECTED_TIERS=""
    PORTS_SUMMARY=""
    TAKEN_PORTS=" "
    FORCED_REF=""
    FORCED_MODE=""
    INSTALLED_AT=""
    HEALTH_RESULT="not-run"
    TRITON_DIGEST=""
    IMAGE_DIGESTS=()
    PINNED_IMAGES=()
    CROPWRIGHT_PORT=""
    CROPWRIGHT_BIND=""
    ACTIVE_IMAGES=()
    VLM_PICK=""
    RESOLVED_MODE=""
    INSTALL_REF=""
    IMAGE_MODE=""
    GPU_PLAN_SUMMARY=""
    DIR_STATE=""

    if [[ "$OP_ACTION" == uninstall || "$OP_ACTION" == rollback ]]; then
        [[ -d "$OP_REAL_DIR" ]] || die "no install at ${OP_REAL_DIR}"
        OP_DIR="$OP_REAL_DIR"
        ENV_FILE="${OP_DIR}/.env"
        STATE_FILE="${OP_DIR}/.install/state.json"
    else
        # Decide ownership before anything is created or written.
        STATE_FILE="${OP_REAL_DIR}/.install/state.json"
        ENV_FILE="${OP_REAL_DIR}/.env"
        classify_install_dir
        if [[ "$OP_DRY_RUN" == 1 ]]; then
            # A dry run writes nothing under the install dir (or anywhere
            # else that outlives it): it works on a private scratch copy.
            _OP_SCRATCH="$(mktemp -d)"
            if [[ "$DIR_STATE" != fresh ]]; then
                tar -C "$OP_REAL_DIR" --exclude=./models --exclude=./pytorch_models --exclude=./data \
                    --exclude=./cache --exclude=./backups --exclude=./.git -cf - . | tar -C "$_OP_SCRATCH" -xf -
            fi
            OP_DIR="$_OP_SCRATCH"
        else
            mkdir -p "$OP_REAL_DIR" || die "cannot create ${OP_REAL_DIR}"
            OP_DIR="$OP_REAL_DIR"
            if [[ "$DIR_STATE" == forced ]]; then
                backup_foreign_dir
            fi
        fi
        ENV_FILE="${OP_DIR}/.env"
        STATE_FILE="${OP_DIR}/.install/state.json"
        mkdir -p "${OP_DIR}/.install"
        chmod 700 "${OP_DIR}/.install"
        if [[ "$OP_DRY_RUN" != 1 ]]; then
            : >> "${OP_DIR}/.install/install.log"
            chmod 600 "${OP_DIR}/.install/install.log"
            exec > >(_op_redact | tee -a "${OP_DIR}/.install/install.log") \
                2> >(_op_redact | tee -a "${OP_DIR}/.install/install.log" >&2)
        fi
    fi
    [[ "$OP_DRY_RUN" == 1 ]] && log_info "DRY-RUN: nothing is written, pulled, started or removed; state-changing commands are printed"
    log_info "setup-openprocessor ${SCRIPT_VERSION}: action ${OP_ACTION}, dir ${OP_REAL_DIR}"

    case "$OP_ACTION" in
        install|repair) do_install "$OP_ACTION" ;;
        uninstall) do_uninstall ;;
        rollback) do_rollback ;;
    esac
}

}

__op_define && { _OP_SELF_PATH="${BASH_SOURCE[0]:-}"; [[ "${OP_SOURCE_ONLY:-0}" == 1 && -n "$_OP_SELF_PATH" && "$_OP_SELF_PATH" != "$0" ]] || main "$@"; }
