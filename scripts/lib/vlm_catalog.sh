#!/bin/bash
# =============================================================================
# vlm_catalog.sh - VLM catalog reader + pick_vlm (installer plan section 2.2)
# =============================================================================
# Pure bash + awk, shared by setup-openprocessor.sh and the `openprocessor
# vlm` CLI. Reads examples/vlm/catalog.tsv (columns:
# id  hf_repo  licence  vram_gb  served_context  max_context  max_images
# status  rank  gated  vllm_image_key).
#
# Status column: "tested" rows are auto-pickable; every other status
# (today "to_verify") is only ever selected explicitly (owner answer 11.1 #6).
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
# Highest-rank row with status "tested" and vram_gb <= AVAILABLE_GB.
# Owner answer 11.1 #6: unverified entries ("to_verify" in this catalog,
# or any status other than "tested") are NEVER auto-picked; the user can
# still choose one explicitly with --vlm-model-id (installer) or
# `openprocessor vlm use <id>` (CLI), which print a warning.
# Prints "ID<TAB>STATUS" on success, nothing and rc 1 if nothing fits.
pick_vlm() {
    local available_gb="$1" path="${2:-$VLM_CATALOG_DEFAULT_PATH}"
    awk -F'\t' -v avail="$available_gb" '
        NR==1 { next }
        $8 == "tested" && ($4 + 0) <= (avail + 0) {
            if (best_id == "" || ($9 + 0) > best_rank) {
                best_rank = $9 + 0
                best_id = $1
            }
        }
        END {
            if (best_id != "") {
                print best_id "\ttested"
            } else {
                exit 1
            }
        }
    ' "$path"
}

# pick_vlm_candidates AVAILABLE_GB [CATALOG_PATH]
# Every row that fits, best first (tested before to_verify, then rank), as
# "ID<TAB>VRAM_GB<TAB>LICENCE<TAB>STATUS". For the interactive list only:
# nothing here is ever selected without the user choosing it.
pick_vlm_candidates() {
    local available_gb="$1" path="${2:-$VLM_CATALOG_DEFAULT_PATH}"
    awk -F'\t' -v avail="$available_gb" '
        NR==1 { next }
        ($4 + 0) <= (avail + 0) {
            printf "%d\t%d\t%s\t%s\t%s\t%s\n", ($8 == "tested") ? 1 : 0, $9, $1, $4, $3, $8
        }
    ' "$path" | sort -t$'\t' -k1,1nr -k2,2nr | cut -f3-
}

# vlm_catalog_floor_gb [CATALOG_PATH]
# The smallest vram_gb among TESTED rows: the refusal floor for an
# automatically offered local vlm tier (section 2.1 rule 2, 11.1 #4 and
# #6). Unverified rows do not lower it, since they are never auto-picked.
vlm_catalog_floor_gb() {
    local path="${1:-$VLM_CATALOG_DEFAULT_PATH}"
    awk -F'\t' '
        NR==1 { next }
        $8 == "tested" {
            if (min == "" || ($4 + 0) < min) { min = $4 + 0 }
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
