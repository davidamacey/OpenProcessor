#!/bin/bash
# =============================================================================
# vlm_catalog.sh - VLM catalog reader + pick_vlm (installer plan section 2.2)
# =============================================================================
# Pure bash + awk, shared by setup-openprocessor.sh and the `openprocessor
# vlm` CLI. Reads examples/vlm/catalog.tsv (columns:
# id  hf_repo  licence  vram_gb  served_context  max_context  max_images
# status  rank  gated  vllm_image_key).
#
# NOTE: examples/vlm/catalog.tsv is owned by the any-domain plan's W9
# (cutover/model-selection). W9 has not merged as of this branch, so this
# file ships a standalone copy of the section 2.2 table so pick_vlm and the
# installer/CLI are testable now. Whichever of W9 / this branch merges
# second must reconcile the two copies (they must describe the same rows).

[[ -n "${_VLM_CATALOG_SH_LOADED:-}" ]] && return 0
_VLM_CATALOG_SH_LOADED=1

_VLM_CATALOG_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLM_CATALOG_DEFAULT_PATH="$(cd "${_VLM_CATALOG_SH_DIR}/../.." && pwd)/examples/vlm/catalog.tsv"

# vlm_catalog_rows [CATALOG_PATH]
# Prints every data row (no header), tab-separated, unmodified.
vlm_catalog_rows() {
    local path="${1:-$VLM_CATALOG_DEFAULT_PATH}"
    tail -n +2 "$path"
}

# vlm_catalog_get ID [CATALOG_PATH]
# Prints the one row matching ID, or nothing (rc 1) if not found.
vlm_catalog_get() {
    local id="$1" path="${2:-$VLM_CATALOG_DEFAULT_PATH}"
    awk -F'\t' -v id="$id" 'NR>1 && $1==id { print; found=1 } END { exit !found }' "$path"
}

# pick_vlm AVAILABLE_GB [CATALOG_PATH]
# Highest-rank row with vram_gb <= AVAILABLE_GB. Among rows that fit,
# "tested" rows are preferred over any "to_verify" row (regardless of
# rank), matching section 2.2. Never auto-picks a row whose status is
# anything other than "tested" or "to_verify" (owner answer 11.1 #6:
# unverified entries are never auto-picked -- there is currently no third
# status in the shipped catalog, but a future "unverified" row must not
# be returned here).
# Prints "ID<TAB>STATUS" on success, nothing and rc 1 if nothing fits.
pick_vlm() {
    local available_gb="$1" path="${2:-$VLM_CATALOG_DEFAULT_PATH}"
    awk -F'\t' -v avail="$available_gb" '
        NR==1 { next }
        ($8 == "tested" || $8 == "to_verify") && $4 <= avail {
            # Encode: tested beats to_verify outright; within the same
            # tier, higher rank wins.
            tier = ($8 == "tested") ? 1 : 0
            key = tier * 100000 + $9
            if (key > best_key) {
                best_key = key
                best_id = $1
                best_status = $8
            }
        }
        END {
            if (best_id != "") {
                print best_id "\t" best_status
            } else {
                exit 1
            }
        }
    ' "$path"
}

# vlm_catalog_floor_gb [CATALOG_PATH]
# The smallest vram_gb among tested/to_verify rows -- the refusal floor
# for the local vlm tier (section 2.1 rule 2, 11.1 #4).
vlm_catalog_floor_gb() {
    local path="${1:-$VLM_CATALOG_DEFAULT_PATH}"
    awk -F'\t' '
        NR==1 { next }
        ($8 == "tested" || $8 == "to_verify") {
            if (min == "" || $4 < min) { min = $4 }
        }
        END { print min }
    ' "$path"
}

# vlm_catalog_field ID FIELD_NAME [CATALOG_PATH]
# FIELD_NAME one of: hf_repo licence vram_gb served_context max_context
# max_images status rank gated vllm_image_key
vlm_catalog_field() {
    local id="$1" field="$2" path="${3:-$VLM_CATALOG_DEFAULT_PATH}"
    local col
    case "$field" in
        hf_repo) col=2 ;;
        licence) col=3 ;;
        vram_gb) col=4 ;;
        served_context) col=5 ;;
        max_context) col=6 ;;
        max_images) col=7 ;;
        status) col=8 ;;
        rank) col=9 ;;
        gated) col=10 ;;
        vllm_image_key) col=11 ;;
        *) return 1 ;;
    esac
    awk -F'\t' -v id="$id" -v col="$col" 'NR>1 && $1==id { print $col; found=1 } END { exit !found }' "$path"
}

# vlm_gpu_memory_utilization VRAM_GB CARD_TOTAL_GB
# clamp((vram_gb - 3) / card_total_gb, 0.2, 0.9), section 2.2.
vlm_gpu_memory_utilization() {
    local vram_gb="$1" card_total_gb="$2"
    awk -v v="$vram_gb" -v c="$card_total_gb" 'BEGIN {
        u = (v - 3) / c
        if (u < 0.2) u = 0.2
        if (u > 0.9) u = 0.9
        printf "%.2f\n", u
    }'
}
