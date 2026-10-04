#!/bin/bash
# =============================================================================
# opensearch_heap.sh - OpenSearch JVM heap sized from host RAM
# =============================================================================
# The one place the heap rule lives (projects plan 2.3, installer plan
# 11.1 #8). Sourced by setup-openprocessor.sh and scripts/lib/config.sh.
#
# heap = clamp(floor(RAM_GiB / 2), 2, 30) GB: the OpenSearch guidance of
# about half of the memory for heap (the rest is page cache), never below a
# 2 GB working minimum, and capped at 30 GB to stay under the ~31 GB
# compressed-pointer limit. The shard budget (heap GB x 20) follows the heap.
#
# OP_MEMINFO_PATH overrides /proc/meminfo (tests).
# =============================================================================

# opensearch_heap_for_host -> heap for this host, e.g. "2g"
opensearch_heap_for_host() {
    local kib ram_gib heap
    kib="$(awk '/^MemTotal:/ { print $2; exit }' "${OP_MEMINFO_PATH:-/proc/meminfo}" 2>/dev/null)"
    [[ "$kib" =~ ^[0-9]+$ ]] || return 1
    ram_gib=$(( kib / 1024 / 1024 ))
    heap=$(( ram_gib / 2 ))
    (( heap < 2 )) && heap=2
    (( heap > 30 )) && heap=30
    echo "${heap}g"
}

# opensearch_shard_budget HEAP [PER_GB] -> soft shard budget for a heap such
# as "2g" or "2048m": whole heap GB * PER_GB (default OP_SHARDS_PER_HEAP_GB,
# else 20). Not a cap: creating past it only warns.
opensearch_shard_budget() {
    local heap="$1" per_gb="${2:-${OP_SHARDS_PER_HEAP_GB:-20}}" gb
    [[ "$per_gb" =~ ^[0-9]+$ ]] || per_gb=20
    if [[ "$heap" =~ ^([0-9]+)[gG]$ ]]; then
        gb=$(( 10#${BASH_REMATCH[1]} ))
    elif [[ "$heap" =~ ^([0-9]+)[mM]$ ]]; then
        gb=$(( 10#${BASH_REMATCH[1]} / 1024 ))
    else
        return 1
    fi
    echo $(( gb * per_gb ))
}
