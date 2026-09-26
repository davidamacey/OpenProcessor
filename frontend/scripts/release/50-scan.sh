#!/bin/bash
# Trivy scan of both architecture legs built by 40-build.sh. Fails on any
# CRITICAL/HIGH finding with an available fix. Trivy inspects image content
# directly, so this needs no emulation for the arm64 leg — verified directly:
# scanning a throwaway arm64 build of this Dockerfile completed normally on
# this amd64 host.
#
# Exit: 0 both legs clean · 1 a finding (or a leg is missing) · 3 docker/trivy unusable

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.." || exit 2

VERSION="${1:?50-scan.sh needs a version}"
IMAGE="${CROPWRIGHT_IMAGE:-davidamacey/cropwright}"
RED='\033[0;31m'; GREEN='\033[0;32m'; BLUE='\033[0;34m'; NC='\033[0m'

command -v docker >/dev/null 2>&1 || { echo "docker not found" >&2; exit 3; }

status=0
for arch in amd64 arm64; do
    tag="${IMAGE}:${VERSION}-${arch}"
    if ! docker image inspect "$tag" >/dev/null 2>&1; then
        echo -e "${RED}FAIL${NC}  ${tag} not built locally — run: ./scripts/release.sh build ${VERSION}" >&2
        status=1
        continue
    fi
    echo -e "${BLUE}scanning ${tag}${NC}" >&2
    if docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
        -v trivy-cache:/root/.cache/ aquasec/trivy image \
        --severity CRITICAL,HIGH --ignore-unfixed --exit-code 1 "$tag"; then
        echo -e "${GREEN}PASS${NC}  ${tag} — no CRITICAL/HIGH with a fix" >&2
    else
        echo -e "${RED}FAIL${NC}  ${tag} has a CRITICAL/HIGH finding with a fix available" >&2
        status=1
    fi
done

exit "$status"
