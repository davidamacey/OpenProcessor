#!/bin/bash
# =============================================================================
# verify_release_assets.sh - prove the release asset set installs, offline
# =============================================================================
# Usage: scripts/release/verify_release_assets.sh [vX.Y.Z]   (default: v$(cat VERSION))
#
# Builds the release assets into a temp dir with build_deploy_bundle.sh, then
# runs setup-openprocessor.sh --dry-run against that LOCAL dir (--release-dir):
#   1. the untouched assets must pass checksum verification;
#   2. a tampered deploy tarball must be refused (exit 7, nothing installed);
#   3. a tampered SHA256SUMS entry for the installer must be refused.
# Publishes nothing and installs nothing: --dry-run writes/pulls/starts nothing,
# and the install dir and HOME are throwaway temp dirs. Docker is only queried
# read-only. images.lock in a development checkout holds placeholder digests, so
# the bundle is built with ALLOW_UNPINNED_LOCK=1 and the installer runs with
# --image-tag; the digest-pinning path itself is covered by tests/installer/.
# The installer stops at "image not present locally" after verification
# succeeds -- that is expected here and is not a verification failure.
# =============================================================================
set -euo pipefail

src="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ref="${1:-v$(cat "${src}/VERSION")}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

ALLOW_UNPINNED_LOCK=1 "${src}/scripts/release/build_deploy_bundle.sh" "$ref" "$src" "${tmp}/rel" >/dev/null

install_dry_run() {
    local assets="$1"
    mkdir -p "${tmp}/home"
    HOME="${tmp}/home" bash "${assets}/setup-openprocessor.sh" --dry-run --unattended \
        --release-dir "$assets" --version "$ref" --image-tag "${ref#v}" \
        --dir "${tmp}/inst" --project "opverify-$$" --skip-models --no-start 2>&1
}

fail() { echo "FAIL: $*" >&2; exit 1; }

out="$(install_dry_run "${tmp}/rel" || true)"
grep -q "release ${ref} downloaded and verified" <<<"$out" \
    || fail "untouched assets were not verified: $(tail -n 5 <<<"$out")"
echo "ok: untouched assets pass checksum verification"

cp -r "${tmp}/rel" "${tmp}/bad_tar"
printf 'x' >> "${tmp}/bad_tar/openprocessor-deploy-${ref}.tar.gz"
rc=0
out="$(install_dry_run "${tmp}/bad_tar")" || rc=$?
[[ "$rc" == 7 ]] || fail "tampered tarball exited ${rc}, want 7"
grep -q "verified" <<<"$out" && fail "tampered tarball was reported verified"
echo "ok: tampered tarball refused (exit 7)"

cp -r "${tmp}/rel" "${tmp}/bad_script"
printf '\n# tampered\n' >> "${tmp}/bad_script/setup-openprocessor.sh"
rc=0
out="$(install_dry_run "${tmp}/bad_script")" || rc=$?
[[ "$rc" == 7 ]] || fail "tampered installer exited ${rc}, want 7"
echo "ok: tampered installer refused (exit 7)"
