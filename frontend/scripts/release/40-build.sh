#!/bin/bash
# Build both release architectures LOCALLY (loaded into the local docker
# engine). Publishes nothing.
#
# arm64 builds on the remote Mac Studio node of the multi-arch
# buildx builder (native compilation, not QEMU) and buildx transparently
# streams the result back for --load — verified directly on this host: a
# throwaway arm64 build+load of this same Dockerfile took ~6s.
#
# One `--load` per platform because `--load` cannot export a multi-arch
# manifest; each platform gets its own local tag,
# cropwright:<version>-<arch>, so scan/smoke can address either
# unambiguously regardless of build order.
#
# Exit: 0 both legs built · 1 a build failed

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 2

VERSION="${1:?40-build.sh needs a version}"
BUILDER="${CROPWRIGHT_BUILDER:-cropwright-multiarch}"
IMAGE="${CROPWRIGHT_IMAGE:-davidamacey/cropwright}"
REVISION="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
RED='\033[0;31m'; GREEN='\033[0;32m'; BLUE='\033[0;34m'; NC='\033[0m'

status=0
for platform in linux/amd64 linux/arm64; do
    arch="${platform#linux/}"
    tag="${IMAGE}:${VERSION}-${arch}"
    echo -e "${BLUE}building ${tag} (${platform}, builder=${BUILDER})${NC}" >&2
    if docker buildx build --builder "$BUILDER" --platform "$platform" \
        --build-arg "VERSION=${VERSION}" --build-arg "REVISION=${REVISION}" \
        -t "$tag" --load .; then
        actual="$(docker image inspect "$tag" --format '{{.Architecture}}' 2>/dev/null)"
        if [[ "$actual" == "$arch" ]]; then
            echo -e "${GREEN}PASS${NC}  ${tag} (${actual})" >&2
        else
            echo -e "${RED}FAIL${NC}  ${tag} reports '${actual:-<unreadable>}', expected ${arch}" >&2
            status=1
        fi
    else
        echo -e "${RED}FAIL${NC}  build failed for ${platform}" >&2
        status=1
    fi
done

exit "$status"
