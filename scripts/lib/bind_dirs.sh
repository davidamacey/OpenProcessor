#!/bin/bash
# =============================================================================
# Bind-mount source directories (F-29/F-71 follow-up)
# =============================================================================
# Docker creates a missing bind-mount source root-owned on the first `up`,
# after which the host user cannot write into it. `openprocessor vlm key set`
# (secrets/vlm) and `make download-test-images` (test_images) both failed
# that way. Create every one of them as the invoking user before compose runs.
# secrets/ holds API keys, so it is created private (mode 700).

ensure_bind_mount_dirs() {
    local root="${1:-.}"
    ( umask 077; mkdir -p "$root/secrets/vlm" )
    mkdir -p "$root/data/source" "$root/cache/huggingface" "$root/cache/vllm"
    # test_images is only mounted by the checkout (dev) overlay.
    if [[ -f "$root/src/main.py" ]]; then
        mkdir -p "$root/test_images"
    fi
}
