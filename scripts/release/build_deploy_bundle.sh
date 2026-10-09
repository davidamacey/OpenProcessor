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
# Upload these as GitHub Release assets of the tag (installer plan 3.1).
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

echo "release assets for ${ref} written to ${out}"
