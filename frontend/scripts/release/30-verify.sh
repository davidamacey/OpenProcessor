#!/bin/bash
# Full local test gate, mirroring CI: install, lint, type-check, unit tests,
# build, stubbed e2e, and (private checkouts only) the OSS leak scan.
#
# scripts/oss-export/ is private-only and excluded from the public export, so
# this stage skips the leak-scan step cleanly (not a failure) when it's absent.
#
# Exit: 0 all green · 1 any check failed

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 2

: "${1:?30-verify.sh needs a version}"
RED='\033[0;31m'; GREEN='\033[0;32m'; BLUE='\033[0;34m'; NC='\033[0m'
status=0

step() {
    local name="$1"; shift
    echo -e "${BLUE}verify: ${name}${NC}" >&2
    if "$@"; then
        echo -e "${GREEN}PASS${NC}  $name" >&2
    else
        echo -e "${RED}FAIL${NC}  $name" >&2
        status=1
    fi
}

step "npm ci"      npm ci
step "lint"        npm run lint
step "check"       npm run check
step "unit tests"  npx vitest run
step "build"       npm run build
step "e2e (stubbed)" npm run test:e2e

if [[ -x scripts/oss-export/leak-scan.sh ]]; then
    step "leak scan" bash scripts/oss-export/leak-scan.sh --tree .
else
    echo -e "${BLUE}verify: leak scan${NC} — scripts/oss-export/leak-scan.sh absent (public export), skipping" >&2
fi

exit "$status"
