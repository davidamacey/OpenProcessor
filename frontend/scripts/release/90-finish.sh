#!/bin/bash
# Publish the GitHub Release, notes taken from the matching CHANGELOG.md
# section. Immediate, never a draft, per the owner's standing direction.
#
# Exit: 0 released · 1 refused or failed

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 2

VERSION="${1:?90-finish.sh needs a version}"
RED='\033[0;31m'; GREEN='\033[0;32m'; NC='\033[0m'

command -v gh >/dev/null 2>&1 || { echo -e "${RED}gh CLI required${NC}" >&2; exit 1; }

if ! git rev-parse "v$VERSION" >/dev/null 2>&1; then
    echo -e "${RED}v$VERSION is not tagged — run: ./scripts/release.sh tag $VERSION --yes${NC}" >&2
    exit 1
fi

notes_file="$(mktemp)"
trap 'rm -f "$notes_file"' EXIT
awk -v ver="$VERSION" '
    BEGIN { found=0 }
    /^## \[/ {
        if (found) exit
        if (index($0, "[" ver "]")) { found=1; next }
        next
    }
    found { print }
' CHANGELOG.md > "$notes_file"

if [[ ! -s "$notes_file" ]]; then
    echo -e "${RED}no ## [$VERSION] section in CHANGELOG.md — nothing to publish as release notes${NC}" >&2
    exit 1
fi

# OpenProcessor's one-line installer fetches these from the release assets
# at the pinned tag and checks them against SHA256SUMS, so they come from
# the tagged commit, never the working tree.
asset_dir="$(mktemp -d)"
trap 'rm -f "$notes_file"; rm -rf "$asset_dir"' EXIT
for f in docker-compose.yml .env.example; do
    if ! git show "v$VERSION:$f" > "$asset_dir/$f"; then
        echo -e "${RED}$f is missing at v$VERSION${NC}" >&2
        exit 1
    fi
done
(cd "$asset_dir" && sha256sum docker-compose.yml .env.example > SHA256SUMS)

if gh release create "v$VERSION" --title "v$VERSION" --notes-file "$notes_file" \
    "$asset_dir/docker-compose.yml" "$asset_dir/.env.example" "$asset_dir/SHA256SUMS"; then
    echo -e "${GREEN}published GitHub release v$VERSION${NC}" >&2
    exit 0
fi
echo -e "${RED}gh release create failed${NC}" >&2
exit 1
