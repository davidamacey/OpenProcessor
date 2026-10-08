#!/bin/bash
# Multi-arch build and push of :X.Y.Z, :X.Y and :latest to Docker Hub in one
# buildx invocation (a single push can carry every tag at once, so there is
# no separate "move :latest later" step — this is one small static image,
# not a multi-GB backend image, so the risk that motivates a two-step
# "publish the version tag, promote :latest by digest later" split doesn't
# apply here).
#
# Exit: 0 published · 1 failed · 3 builder unreachable

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 2

VERSION="${1:?80-publish.sh needs a version}"
BUILDER="${CROPWRIGHT_BUILDER:-cropwright-multiarch}"
IMAGE="${CROPWRIGHT_IMAGE:-davidamacey/cropwright}"
REVISION="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
MINOR="${VERSION%.*}"
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

if ! docker buildx inspect "$BUILDER" >/dev/null 2>&1; then
    echo -e "${RED}buildx builder '$BUILDER' missing${NC}" >&2
    echo "  ./scripts/setup-remote-builder.sh setup" >&2
    exit 3
fi
if [[ "$(docker buildx inspect "$BUILDER" 2>/dev/null | grep -ci 'error')" -gt 0 ]]; then
    echo -e "${RED}'$BUILDER' has a node reporting an error${NC}" >&2
    exit 3
fi

echo -e "${YELLOW}PUBLISHING ${VERSION} to Docker Hub as :${VERSION}, :${MINOR} and :latest${NC}" >&2
if docker buildx build --builder "$BUILDER" --platform linux/amd64,linux/arm64 \
    --build-arg "VERSION=${VERSION}" --build-arg "REVISION=${REVISION}" \
    --sbom=true --provenance=mode=min \
    -t "${IMAGE}:${VERSION}" -t "${IMAGE}:${MINOR}" -t "${IMAGE}:latest" \
    --push .; then
    echo -e "${GREEN}published ${IMAGE}:${VERSION} / :${MINOR} / :latest (amd64+arm64)${NC}" >&2
    exit 0
fi
echo -e "${RED}publish failed${NC}" >&2
exit 1
