#!/bin/bash
# Create and push the annotated release tag. FIRST STAGE THAT LEAVES THIS
# MACHINE (release.sh already required an explicit confirmation before
# calling this). Annotated, never lightweight — tags are release artifacts.
#
# Exit: 0 tagged and pushed · 1 a gate failed

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 2

VERSION="${1:?70-tag.sh needs a version}"
RED='\033[0;31m'; GREEN='\033[0;32m'; NC='\033[0m'

if [[ -n "$(git status --porcelain)" ]]; then
    echo -e "${RED}worktree is dirty — a tag must be reproducible${NC}" >&2
    exit 1
fi

if git rev-parse "v$VERSION" >/dev/null 2>&1; then
    echo -e "${RED}v$VERSION already exists${NC}" >&2
    echo "  git tag -d v$VERSION && git push origin :refs/tags/v$VERSION   # if it was never released" >&2
    exit 1
fi

git tag -a "v$VERSION" -m "Release v$VERSION"
if ! git push origin "v$VERSION"; then
    echo -e "${RED}push failed — deleting the local tag so this stage is retryable${NC}" >&2
    git tag -d "v$VERSION" >/dev/null 2>&1 || true
    exit 1
fi

echo -e "${GREEN}tagged and pushed v$VERSION${NC}" >&2
exit 0
