#!/bin/bash
# Preconditions for a release: clean worktree, version consistency, the tag is
# free, the multi-arch builder is reachable with both platforms, and (only
# needed once we reach publish) a Docker Hub login is present.
#
# Exit: 0 all clear · 1 a gate failed · 3 builder/login precondition unmet

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 2

VERSION="${1:?10-preflight.sh needs a version}"
BUILDER="${CROPWRIGHT_BUILDER:-cropwright-multiarch}"
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
status=0

pass() { echo -e "${GREEN}PASS${NC}  $*" >&2; }
fail() { echo -e "${RED}FAIL${NC}  $*" >&2; status=1; }
warn() { echo -e "${YELLOW}WARN${NC}  $*" >&2; }

if [[ -n "$(git status --porcelain)" ]]; then
    fail "worktree is dirty — a release must be reproducible from a committed tree"
else
    pass "clean worktree"
fi

pkg_version="$(node -p "require('./package.json').version" 2>/dev/null || echo "")"
if [[ "$pkg_version" == "$VERSION" ]]; then
    pass "package.json version matches ($pkg_version)"
else
    fail "package.json is $pkg_version, expected $VERSION"
fi

if grep -qE "^## \[${VERSION//./\\.}\]" CHANGELOG.md 2>/dev/null; then
    pass "CHANGELOG.md has a ## [$VERSION] section"
else
    fail "CHANGELOG.md has no ## [$VERSION] section — move the Unreleased entries under one"
fi

if git rev-parse "v$VERSION" >/dev/null 2>&1; then
    fail "tag v$VERSION already exists"
else
    pass "tag v$VERSION is free"
fi

if ! docker buildx inspect "$BUILDER" >/dev/null 2>&1; then
    fail "buildx builder '$BUILDER' does not exist"
    echo "  ./scripts/setup-remote-builder.sh setup" >&2
elif [[ "$(docker buildx inspect "$BUILDER" 2>/dev/null | grep -ci 'error')" -gt 0 ]]; then
    fail "'$BUILDER' has a node reporting an error"
else
    platforms="$(docker buildx inspect "$BUILDER" 2>/dev/null | grep -A1 'Platforms:' | tr ',' '\n')"
    have_amd64=false; have_arm64=false
    grep -q 'linux/amd64' <<<"$platforms" && have_amd64=true
    grep -q 'linux/arm64' <<<"$platforms" && have_arm64=true
    if $have_amd64 && $have_arm64; then
        pass "'$BUILDER' serves both linux/amd64 and linux/arm64"
    else
        fail "'$BUILDER' does not report both linux/amd64 and linux/arm64"
    fi
fi

# Needed only by the publish stage, but flagged here so a release doesn't
# discover a missing login three stages in.
if docker system info --format '{{json .}}' 2>/dev/null | grep -q '"IndexServerAddress"'; then
    : # informational only; `docker login` state isn't reliably introspectable
fi
if [[ -f "$HOME/.docker/config.json" ]] && grep -q 'https://index.docker.io' "$HOME/.docker/config.json" 2>/dev/null; then
    pass "docker login present for Docker Hub"
else
    warn "no Docker Hub login found in ~/.docker/config.json — required before 'publish'"
fi

if [[ "$status" -ne 0 ]]; then
    echo -e "${RED}preflight found gate failures — see FAIL lines above${NC}" >&2
    exit 1
fi
exit 0
