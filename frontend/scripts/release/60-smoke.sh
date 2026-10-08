#!/bin/bash
# Run scripts/release-smoke.sh against both architecture legs built by
# 40-build.sh.
#
# The arm64 leg cannot run on this amd64 host without QEMU (slow, and not
# representative of what a real arm64 host will see), so — mirroring
# that project's own convention of checking a non-host arch leg over its
# remote builder's docker context rather than installing binfmt — this loads
# the arm64 image into the `remote-arm64` docker context (the same Mac Studio
# node the multi-arch buildx builder already uses) via `docker save | docker
# --context remote-arm64 load`, then re-runs the smoke script with
# DOCKER_CONTEXT set so every `docker` call inside it targets that engine
# natively. Verified directly: a throwaway arm64 image round-tripped through
# this exact save/load and reported the correct architecture on the remote
# engine.
#
# Exit: 0 both legs pass · 1 a leg failed · 3 remote context unavailable

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 2

VERSION="${1:?60-smoke.sh needs a version}"
IMAGE="${CROPWRIGHT_IMAGE:-davidamacey/cropwright}"
REMOTE_CTX="${CROPWRIGHT_REMOTE_ARM64_CONTEXT:-remote-arm64}"
RED='\033[0;31m'; GREEN='\033[0;32m'; BLUE='\033[0;34m'; YELLOW='\033[1;33m'; NC='\033[0m'

status=0

amd64_tag="${IMAGE}:${VERSION}-amd64"
if docker image inspect "$amd64_tag" >/dev/null 2>&1; then
    echo -e "${BLUE}smoke: ${amd64_tag} (local)${NC}" >&2
    if ./scripts/release-smoke.sh "$amd64_tag"; then
        echo -e "${GREEN}PASS${NC}  ${amd64_tag}" >&2
    else
        echo -e "${RED}FAIL${NC}  ${amd64_tag}" >&2
        status=1
    fi
else
    echo -e "${RED}FAIL${NC}  ${amd64_tag} not built — run: ./scripts/release.sh build ${VERSION}" >&2
    status=1
fi

arm64_tag="${IMAGE}:${VERSION}-arm64"
if ! docker image inspect "$arm64_tag" >/dev/null 2>&1; then
    echo -e "${RED}FAIL${NC}  ${arm64_tag} not built — run: ./scripts/release.sh build ${VERSION}" >&2
    exit 1
fi
if ! docker context inspect "$REMOTE_CTX" >/dev/null 2>&1; then
    echo -e "${YELLOW}SKIP${NC}  no '${REMOTE_CTX}' docker context — arm64 unverified" >&2
    echo "  docker context create ${REMOTE_CTX} --docker host=ssh://user@<arm64-host>" >&2
    exit 3
fi

echo -e "${BLUE}smoke: ${arm64_tag} (loading into ${REMOTE_CTX})${NC}" >&2
if docker save "$arm64_tag" | docker --context "$REMOTE_CTX" load >/dev/null; then
    if DOCKER_CONTEXT="$REMOTE_CTX" ./scripts/release-smoke.sh "$arm64_tag"; then
        echo -e "${GREEN}PASS${NC}  ${arm64_tag} (over ${REMOTE_CTX})" >&2
    else
        echo -e "${RED}FAIL${NC}  ${arm64_tag} (over ${REMOTE_CTX})" >&2
        status=1
    fi
    docker --context "$REMOTE_CTX" rmi "$arm64_tag" >/dev/null 2>&1 || true
else
    echo -e "${RED}FAIL${NC}  could not load ${arm64_tag} into ${REMOTE_CTX}" >&2
    status=1
fi

exit "$status"
