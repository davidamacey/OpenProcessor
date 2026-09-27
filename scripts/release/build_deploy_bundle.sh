#!/bin/bash
# =============================================================================
# build_deploy_bundle.sh - build the installer's release assets
# =============================================================================
# Usage: scripts/release/build_deploy_bundle.sh vX.Y.Z [SRC_DIR] [OUT_DIR]
#
# Produces, in OUT_DIR (default <SRC_DIR>/dist/release-vX.Y.Z):
#   openprocessor-deploy-vX.Y.Z.tar.gz  every file release-manifest.txt lists
#   SHA256SUMS                           sha256 of the tarball and of every file
#   setup-openprocessor.sh               the tagged installer (bootstrap target)
#   release-manifest.txt
#   cropwright/<tag>/                    only with CW_RELEASE_DIR (see below)
# Upload the first four as GitHub Release assets of the tag (installer plan 3.1).
#
# CW_RELEASE_DIR=<dir holding Cropwright's release SHA256SUMS,
# docker-compose.yml and .env.example> stages those files into
# OUT_DIR/cropwright/<tag>/ (tag from cropwright.lock), after checking them
# against cropwright.lock, so a --release-dir install of the cropwright tier
# needs no network. Without it, that tier is fetched from Cropwright's
# GitHub release at install time (still verified against cropwright.lock).
#
# SHA256SUMS is an integrity check against a corrupted or truncated
# download, fetched from the same origin as the files it covers; it is not
# an authenticity signature (see the README's paranoid-install section).
#
# Refuses an images.lock that is not fully digest-pinned unless
# ALLOW_UNPINNED_LOCK=1 (used only by the test fixtures).
# =============================================================================
set -euo pipefail

ref="${1:?usage: build_deploy_bundle.sh vX.Y.Z [SRC_DIR] [OUT_DIR]}"
src="$(cd "${2:-$(dirname "${BASH_SOURCE[0]}")/../..}" && pwd)"
out="${3:-${src}/dist/release-${ref}}"

[[ "$ref" =~ ^v[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.]+)?$ ]] || { echo "ref must look like v1.2.3" >&2; exit 2; }
[[ -f "${src}/release-manifest.txt" ]] || { echo "no release-manifest.txt in ${src}" >&2; exit 1; }

if [[ "${ALLOW_UNPINNED_LOCK:-0}" != 1 ]]; then
    while IFS= read -r line; do
        [[ -z "$line" || "$line" == \#* ]] && continue
        if [[ ! "$line" =~ ^[a-z0-9_-]+=[a-z0-9][a-z0-9._/-]*(:[A-Za-z0-9._-]+)?@sha256:[0-9a-f]{64}$ || "$line" == *:latest@* ]]; then
            echo "images.lock is not fully digest-pinned: ${line}" >&2
            exit 1
        fi
    done < "${src}/images.lock"
fi

mkdir -p "$out"
list="$(mktemp)"
trap 'rm -f "$list"' EXIT

while read -r path flags; do
    [[ -z "$path" || "$path" == \#* ]] && continue
    if [[ "$path" == *'**' ]]; then
        prefix="${path%%\*\*}"
        if [[ -d "${src}/${prefix}" ]]; then
            (cd "$src" && find "${prefix%/}" -type f -print) >> "$list"
        fi
        continue
    fi
    if [[ -f "${src}/${path}" ]]; then
        echo "$path" >> "$list"
    elif [[ "${flags:-}" != *optional* ]]; then
        echo "manifest lists a missing file: ${path}" >&2
        exit 1
    fi
done < "${src}/release-manifest.txt"
sort -u -o "$list" "$list"

tarball="openprocessor-deploy-${ref}.tar.gz"
tar -czf "${out}/${tarball}" --owner=0 --group=0 --numeric-owner --sort=name \
    --mtime='@0' -C "$src" -T "$list"

(
    cd "$src"
    while IFS= read -r f; do
        sha256sum "$f"
    done < "$list"
) > "${out}/SHA256SUMS"
(cd "$out" && sha256sum "$tarball") >> "${out}/SHA256SUMS"

cp "${src}/setup-openprocessor.sh" "${src}/release-manifest.txt" "$out/"

if [[ -n "${CW_RELEASE_DIR:-}" ]]; then
    cw_lock="${src}/cropwright.lock"
    cw_tag="$(sed -n 's/^tag=//p' "$cw_lock" | head -n1)"
    cw_sums="$(sed -n 's/^sha256sums_sha256=//p' "$cw_lock" | head -n1)"
    [[ "$cw_tag" =~ ^v[0-9A-Za-z._-]+$ && "$cw_sums" =~ ^[0-9a-f]{64}$ ]] \
        || { echo "cropwright.lock names no tag/sha256sums_sha256" >&2; exit 1; }
    [[ "$(sha256sum "${CW_RELEASE_DIR}/SHA256SUMS" | cut -d' ' -f1)" == "$cw_sums" ]] \
        || { echo "${CW_RELEASE_DIR}/SHA256SUMS does not match cropwright.lock" >&2; exit 1; }
    (cd "$CW_RELEASE_DIR" && for f in docker-compose.yml .env.example; do
        grep -E "^[0-9a-f]{64}  ${f//./\\.}\$" SHA256SUMS | sha256sum -c --quiet - >/dev/null
    done) || { echo "Cropwright files in ${CW_RELEASE_DIR} fail their SHA256SUMS (cropwright.lock)" >&2; exit 1; }
    mkdir -p "${out}/cropwright/${cw_tag}"
    for f in SHA256SUMS docker-compose.yml .env.example; do
        cp "${CW_RELEASE_DIR}/${f}" "${out}/cropwright/${cw_tag}/${f}"
    done
    echo "staged Cropwright ${cw_tag} into ${out}/cropwright/${cw_tag}"
fi
echo "release assets for ${ref} written to ${out}"
