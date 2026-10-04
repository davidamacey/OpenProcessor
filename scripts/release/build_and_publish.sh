#!/bin/bash
# Local release script for OpenProcessor's published images.
#
# Owner decision (docs/design/openprocessor_internal/one_line_installer_plan.md
# §11.1 item 2): no CI image builds. This script, run by hand on this host
# (or via `make release`), builds every published image with the Docker
# build cache, gates on a Trivy CRITICAL scan, and -- only with --push --
# pushes digest-pinned tags and writes images.lock (plus images.lock.sha256)
# for the installer to verify against. images.lock pins every image an
# install can pull: the five built here and the third-party ones (vLLM,
# OpenSearch, MLflow, monitoring), whose upstream tags are resolved to
# digests at release time. Key names come from scripts/lib/image_keys.sh,
# the same table the installer reads.
#
# It never touches release-manifest.txt: that is the installer's file list,
# read by scripts/release/build_deploy_bundle.sh.
#
# Modeled on Cropwright's local multi-arch release script
# (scripts/release.sh and scripts/release/*.sh in that repo), trimmed to
# this project's single-arch (linux/amd64, GPU) image set.
#
# Usage:
#   scripts/release/build_and_publish.sh --dry-run
#   scripts/release/build_and_publish.sh --push
#   scripts/release/build_and_publish.sh --push --only api,triton
#   scripts/release/build_and_publish.sh --dry-run --allow-dirty   (local dev/test only)
#
# Exit codes: 0 ok · 1 a gate failed (build/scan/push) · 2 misuse · 3 precondition unmet

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'
log()  { echo -e "${BLUE}[release]${NC} $*" >&2; }
ok()   { echo -e "${GREEN}[release] OK${NC} $*" >&2; }
warn() { echo -e "${YELLOW}[release] !${NC} $*" >&2; }
err()  { echo -e "${RED}${BOLD}[release] X${NC} $*" >&2; }

EXIT_GATE=1
EXIT_MISUSE=2
EXIT_PRECONDITION=3

# ── configuration ────────────────────────────────────────────────────────────

OP_IMAGE_NAMESPACE="${OP_IMAGE_NAMESPACE:-davidamacey}"
DOCKER_BIN="${DOCKER_BIN:-docker}"
TRIVY_BIN="${TRIVY_BIN:-trivy}"
TRIVY_TIMEOUT="${TRIVY_TIMEOUT:-30m}"
ALLOWLIST_FILE="${TRIVY_ALLOWLIST_FILE:-$SCRIPT_DIR/trivy-allowlist.txt}"
LOCK_FILE="${IMAGES_LOCK_FILE:-$REPO_ROOT/images.lock}"
LOCK_SUMS_FILE="${IMAGES_LOCK_SUMS_FILE:-${LOCK_FILE}.sha256}"
if [[ "$(basename "$LOCK_SUMS_FILE")" == release-manifest.txt || "$(basename "$LOCK_FILE")" == release-manifest.txt ]]; then
    err "release-manifest.txt is the installer's file list; the release script never writes it"
    exit "$EXIT_MISUSE"
fi

# shellcheck source=../lib/image_keys.sh
source "$REPO_ROOT/scripts/lib/image_keys.sh"

# service_key -> "dockerfile|build_context|image_name", from the shared table.
# image_name is the repo-local part; the pushed tag is
# "${OP_IMAGE_NAMESPACE}/${image_name}:${VERSION}".
declare -A IMAGE_SPECS=()
for _key in $(image_keys build); do
    # The segmenter's Dockerfile COPYs from its own directory (compose builds it
    # with context docker/segmenter); every other image builds from the repo root.
    _ctx="."
    [[ "$_key" == segmenter ]] && _ctx="docker/segmenter"
    IMAGE_SPECS[$_key]="$(image_key_field "$_key" dockerfile)|${_ctx}|$(image_key_field "$_key" image)"
done
ALL_SERVICES="$(image_keys build | tr '\n' ' ')"
THIRD_PARTY_KEYS="$(image_keys third | tr '\n' ' ')"

# ── args ──────────────────────────────────────────────────────────────────

MODE=""            # "dry-run" or "push"
ONLY=""
VERSION_OVERRIDE=""
ALLOW_DIRTY=false
LOCAL_TAG_SUFFIX=""

usage() {
    cat >&2 <<'EOF'
Usage: build_and_publish.sh (--dry-run|--push) [--only svc1,svc2] [--version vX.Y.Z]
                             [--namespace NAME] [--allow-dirty] [--local-tag-suffix SUF]

  --dry-run           build + scan every selected image, never push
  --push              build + scan + push, resolve third-party digests, write images.lock
  --only LIST         comma-separated subset of: api,triton,evaluator,segmenter,trainer
  --version vX.Y.Z    override the VERSION file (must equal it unless --allow-dirty)
  --namespace NAME    override OP_IMAGE_NAMESPACE (default: davidamacey)
  --allow-dirty       skip the clean-worktree / VERSION-tag gate (local testing only)
  --local-tag-suffix  append -SUF to every local tag, for a throwaway proof build
EOF
}

[[ $# -eq 0 ]] && { usage; exit "$EXIT_MISUSE"; }

while (( $# > 0 )); do
    case "$1" in
        --dry-run) MODE="dry-run"; shift ;;
        --push) MODE="push"; shift ;;
        --only) ONLY="$2"; shift 2 ;;
        --version) VERSION_OVERRIDE="$2"; shift 2 ;;
        --namespace) OP_IMAGE_NAMESPACE="$2"; shift 2 ;;
        --allow-dirty) ALLOW_DIRTY=true; shift ;;
        --local-tag-suffix) LOCAL_TAG_SUFFIX="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) err "unknown option: $1"; usage; exit "$EXIT_MISUSE" ;;
    esac
done

[[ -n "$MODE" ]] || { err "one of --dry-run or --push is required"; exit "$EXIT_MISUSE"; }

if [[ -n "$ONLY" ]]; then
    IFS=',' read -r -a SERVICES <<< "$ONLY"
    for svc in "${SERVICES[@]}"; do
        [[ -n "${IMAGE_SPECS[$svc]:-}" ]] || { err "unknown service in --only: $svc"; exit "$EXIT_MISUSE"; }
    done
else
    read -r -a SERVICES <<< "$ALL_SERVICES"
fi

# ── version / preflight ─────────────────────────────────────────────────────

VERSION_FILE="$REPO_ROOT/VERSION"
[[ -f "$VERSION_FILE" ]] || { err "no VERSION file at $VERSION_FILE"; exit "$EXIT_PRECONDITION"; }
FILE_VERSION="$(tr -d '[:space:]' < "$VERSION_FILE")"

if [[ -n "$VERSION_OVERRIDE" ]]; then
    VERSION="${VERSION_OVERRIDE#v}"
    if [[ "$VERSION" != "$FILE_VERSION" && "$ALLOW_DIRTY" != true ]]; then
        err "--version $VERSION_OVERRIDE does not match VERSION file ($FILE_VERSION)"
        exit "$EXIT_GATE"
    fi
else
    VERSION="$FILE_VERSION"
fi

if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    err "VERSION '$VERSION' is not a plain X.Y.Z semver"
    exit "$EXIT_GATE"
fi

if [[ "$MODE" == "push" && "$ALLOW_DIRTY" != true ]]; then
    if [[ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]]; then
        err "worktree is dirty -- a pushed release must be reproducible from a committed tree"
        exit "$EXIT_GATE"
    fi
    tag="v$VERSION"
    if git -C "$REPO_ROOT" rev-parse "$tag" >/dev/null 2>&1; then
        head_sha="$(git -C "$REPO_ROOT" rev-parse HEAD)"
        tag_sha="$(git -C "$REPO_ROOT" rev-parse "${tag}^{commit}")"
        if [[ "$head_sha" != "$tag_sha" ]]; then
            err "tag $tag exists but does not point at HEAD ($head_sha != $tag_sha)"
            exit "$EXIT_GATE"
        fi
    fi
fi

REVISION="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"

log "version=$VERSION namespace=$OP_IMAGE_NAMESPACE mode=$MODE services=${SERVICES[*]}"

command -v "$DOCKER_BIN" >/dev/null 2>&1 || { err "$DOCKER_BIN not found"; exit "$EXIT_PRECONDITION"; }

HAVE_TRIVY=true
command -v "$TRIVY_BIN" >/dev/null 2>&1 || { warn "$TRIVY_BIN not found -- scan gate skipped (install trivy to enforce it)"; HAVE_TRIVY=false; }

# ── trivy allowlist -> ignorefile ───────────────────────────────────────────
# ALLOWLIST_FILE format: one CVE per line, "CVE-XXXX-XXXXX # reason". Blank
# lines and lines starting with # are comments. A CVE with no reason after
# '#' is rejected -- every accepted CVE must record why.

build_trivy_ignorefile() {
    local ignorefile="$1"
    : > "$ignorefile"
    [[ -f "$ALLOWLIST_FILE" ]] || return 0
    local lineno=0
    while IFS= read -r line || [[ -n "$line" ]]; do
        lineno=$((lineno + 1))
        [[ -z "${line// /}" ]] && continue
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        local cve="${line%%#*}"
        local reason="${line#*#}"
        cve="$(echo "$cve" | tr -d '[:space:]')"
        reason="$(echo "$reason" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
        [[ "$cve" == "$line" ]] && reason=""
        if [[ -z "$reason" ]]; then
            err "$ALLOWLIST_FILE:$lineno: '$cve' has no reason -- every allowlisted CVE needs '# reason'"
            return 1
        fi
        echo "$cve" >> "$ignorefile"
    done < "$ALLOWLIST_FILE"
    return 0
}

# ── build ────────────────────────────────────────────────────────────────────

build_image() {
    local svc="$1"
    IFS='|' read -r dockerfile context image_name <<< "${IMAGE_SPECS[$svc]}"
    local repo="${OP_IMAGE_NAMESPACE}/${image_name}"
    local version_tag="${repo}:${VERSION}${LOCAL_TAG_SUFFIX}"
    local latest_tag="${repo}:latest${LOCAL_TAG_SUFFIX}"

    log "building $svc -> $version_tag (cache on, $dockerfile)"
    "$DOCKER_BIN" build \
        --file "$dockerfile" \
        --tag "$version_tag" \
        --tag "$latest_tag" \
        --build-arg "OP_BUILD_SHA=${REVISION}" \
        --label "org.opencontainers.image.revision=${REVISION}" \
        --label "org.opencontainers.image.version=${VERSION}" \
        "$context" || return 1
    ok "built $version_tag"
    echo "$version_tag"
}

# ── scan ─────────────────────────────────────────────────────────────────────

scan_image() {
    local svc="$1" tag="$2"
    if [[ "$HAVE_TRIVY" != true ]]; then
        warn "skipping Trivy scan for $svc ($tag): trivy not installed"
        return 0
    fi
    local ignorefile
    ignorefile="$(mktemp)"
    trap 'rm -f "$ignorefile"' RETURN
    build_trivy_ignorefile "$ignorefile" || return 1

    # --exit-code 10 is reserved for "findings": any other non-zero exit
    # (timeout, DB download failure, ...) is a scanner error, not a finding.
    # Vuln scanner only: the secret scanner walks the baked model seed.
    log "scanning $svc ($tag) -- fail on CRITICAL (timeout $TRIVY_TIMEOUT)"
    local rc=0
    "$TRIVY_BIN" image --scanners vuln --severity CRITICAL --exit-code 10 \
        --timeout "$TRIVY_TIMEOUT" --ignorefile "$ignorefile" "$tag" || rc=$?
    case "$rc" in
        0) ok "$tag -- no un-allowlisted CRITICAL findings"; return 0 ;;
        10) err "$tag has a CRITICAL finding not in $ALLOWLIST_FILE"; return 1 ;;
        *) err "$tag: Trivy scan ERROR (exit $rc, e.g. timeout or DB failure) -- the image was NOT assessed; raise TRIVY_TIMEOUT or fix the scanner"; return 1 ;;
    esac
}

# ── push + digest capture ───────────────────────────────────────────────────

push_image() {
    local svc="$1" tag="$2"
    IFS='|' read -r _ _ image_name <<< "${IMAGE_SPECS[$svc]}"
    local repo="${OP_IMAGE_NAMESPACE}/${image_name}"
    local latest_tag="${repo}:latest"

    log "pushing $tag"
    "$DOCKER_BIN" push "$tag" >&2
    log "pushing $latest_tag (manual-pull convenience; installer never uses it)"
    "$DOCKER_BIN" push "$latest_tag" >&2

    local digest
    digest="$("$DOCKER_BIN" image inspect "$tag" --format '{{index .RepoDigests 0}}' 2>/dev/null)"
    [[ -n "$digest" ]] || { err "could not read a pushed digest for $tag"; return 1; }
    # RepoDigests is "repo@sha256:...": strip the repo, we already know it.
    echo "${digest##*@}"
}

# ── third-party digests ─────────────────────────────────────────────────────

# resolve_third_party KEY -> sha256 digest of the key's upstream image
resolve_third_party() {
    local key="$1" src digest
    src="$(image_key_field "$key" source)"
    log "resolving $key ($src)"
    "$DOCKER_BIN" pull "$src" >&2 || { err "could not pull $src"; return 1; }
    digest="$("$DOCKER_BIN" image inspect "$src" --format '{{index .RepoDigests 0}}' 2>/dev/null)"
    [[ "$digest" == *@sha256:* ]] || { err "no repo digest for $src"; return 1; }
    echo "${digest##*@}"
}

# ── lock writers ─────────────────────────────────────────────────────────────

write_images_lock() {
    local -n digests_ref="$1"
    local key
    : > "$LOCK_FILE"
    for svc in "${SERVICES[@]}"; do
        IFS='|' read -r _ _ image_name <<< "${IMAGE_SPECS[$svc]}"
        echo "${svc}=${OP_IMAGE_NAMESPACE}/${image_name}:${VERSION}@${digests_ref[$svc]}" >> "$LOCK_FILE"
    done
    for key in $THIRD_PARTY_KEYS; do
        echo "${key}=$(image_key_field "$key" source)@${digests_ref[$key]}" >> "$LOCK_FILE"
    done
    sort -o "$LOCK_FILE" "$LOCK_FILE"
    ok "wrote $LOCK_FILE"
}

write_lock_sums() {
    local sha
    sha="$(sha256sum "$LOCK_FILE" | awk '{print $1}')"
    printf '%s  %s\n' "$sha" "$(basename "$LOCK_FILE")" > "$LOCK_SUMS_FILE"
    ok "wrote $LOCK_SUMS_FILE"
}

# ── main ─────────────────────────────────────────────────────────────────────

declare -A BUILT_TAGS=()
declare -A DIGESTS=()
FAILED=false

for svc in "${SERVICES[@]}"; do
    tag="$(build_image "$svc")" || { FAILED=true; continue; }
    BUILT_TAGS["$svc"]="$tag"
done

$FAILED && { err "one or more builds failed"; exit "$EXIT_GATE"; }

for svc in "${SERVICES[@]}"; do
    scan_image "$svc" "${BUILT_TAGS[$svc]}" || FAILED=true
done

$FAILED && { err "one or more images failed the Trivy gate"; exit "$EXIT_GATE"; }

if [[ "$MODE" == "dry-run" ]]; then
    ok "dry-run complete: built + scanned ${#SERVICES[@]} image(s), nothing pushed"
    exit 0
fi

for svc in "${SERVICES[@]}"; do
    digest="$(push_image "$svc" "${BUILT_TAGS[$svc]}")" || { FAILED=true; continue; }
    # shellcheck disable=SC2034  # read via the write_images_lock nameref
    DIGESTS["$svc"]="$digest"
done

$FAILED && { err "one or more pushes failed"; exit "$EXIT_GATE"; }

for key in $THIRD_PARTY_KEYS; do
    digest="$(resolve_third_party "$key")" || { FAILED=true; continue; }
    # shellcheck disable=SC2034  # read via the write_images_lock nameref
    DIGESTS["$key"]="$digest"
done

$FAILED && { err "one or more third-party digests could not be resolved"; exit "$EXIT_GATE"; }

write_images_lock DIGESTS
write_lock_sums

ok "release complete: ${#SERVICES[@]} image(s) pushed as v${VERSION} (+ latest); images.lock pins them and $(wc -w <<< "$THIRD_PARTY_KEYS") third-party images"
