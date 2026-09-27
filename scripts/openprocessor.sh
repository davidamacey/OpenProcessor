#!/bin/bash
# Deploy-mode CLI body moved to the repo-root `openprocessor` script
# (installer plan section 5.5, section 8). Kept here as a shim so existing
# docs/muscle-memory (`./scripts/openprocessor.sh ...`) keep working from a
# checkout.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/openprocessor" "$@"
