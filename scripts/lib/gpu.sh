#!/bin/bash
# =============================================================================
# gpu.sh - GPU detection and profile selection for OpenProcessor
# =============================================================================

# Avoid redefining if already sourced
[[ -n "${_GPU_SH_LOADED:-}" ]] && return 0
_GPU_SH_LOADED=1

# Source colors if not already loaded
_GPU_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=colors.sh
[[ -z "${NC:-}" ]] && source "${_GPU_SH_DIR}/colors.sh"

# =============================================================================
# GPU Detection
# =============================================================================

# Detect NVIDIA GPU information
# Sets: GPU_NAME, GPU_VRAM_MB, GPU_COMPUTE_CAP, GPU_COUNT
detect_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not found. Please install NVIDIA drivers."
        return 1
    fi

    # Get GPU count
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [[ -z "$GPU_COUNT" ]] || [[ "$GPU_COUNT" -eq 0 ]]; then
        log_error "No NVIDIA GPUs detected."
        return 1
    fi

    # Get GPU info for first GPU (or specified GPU_ID)
    local gpu_id="${GPU_ID:-0}"

    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$gpu_id" 2>/dev/null)
    GPU_VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null)

    # Get compute capability
    local compute_cap
    compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i "$gpu_id" 2>/dev/null || echo "")
    if [[ -z "$compute_cap" ]]; then
        # Fallback: check via deviceQuery if available
        compute_cap="unknown"
    fi
    GPU_COMPUTE_CAP="$compute_cap"

    # Convert to GB for display
    GPU_VRAM_GB=$((GPU_VRAM_MB / 1024))

    export GPU_NAME GPU_VRAM_MB GPU_VRAM_GB GPU_COMPUTE_CAP GPU_COUNT

    log_debug "Detected GPU: $GPU_NAME ($GPU_VRAM_GB GB, compute $GPU_COMPUTE_CAP)"
    return 0
}

# List all available GPUs
list_gpus() {
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not found."
        return 1
    fi

    echo -e "${CYAN}Available GPUs:${NC}"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | while IFS=',' read -r idx name mem; do
        echo "  [$idx] $name ($mem)"
    done
}

# Check if selected GPU has sufficient VRAM
check_gpu_memory() {
    local required_mb="${1:-6000}"

    detect_gpu || return 1

    if [[ "$GPU_VRAM_MB" -lt "$required_mb" ]]; then
        log_warn "GPU has ${GPU_VRAM_GB}GB VRAM, minimum ${required_mb}MB recommended."
        return 1
    fi

    return 0
}

# =============================================================================
# Profile Selection
# =============================================================================

# Profiles based on VRAM
# minimal:  <10GB  (6-8GB GPUs like RTX 3060, RTX 4060)
# standard: 10-40GB (12-24GB GPUs like RTX 3080, RTX 4090)
# full:     >40GB  (A6000, A100, etc.)

# Auto-select profile based on detected GPU VRAM
select_profile() {
    detect_gpu || return 1

    if [[ "$GPU_VRAM_MB" -lt 10000 ]]; then
        SELECTED_PROFILE="minimal"
    elif [[ "$GPU_VRAM_MB" -lt 40000 ]]; then
        SELECTED_PROFILE="standard"
    else
        SELECTED_PROFILE="full"
    fi

    export SELECTED_PROFILE
    log_debug "Auto-selected profile: $SELECTED_PROFILE (based on ${GPU_VRAM_GB}GB VRAM)"
}

# Get profile description
get_profile_description() {
    local profile="${1:-$SELECTED_PROFILE}"

    case "$profile" in
        minimal)
            echo "6-8GB VRAM (RTX 3060, RTX 4060) - Core models only"
            ;;
        standard)
            echo "12-24GB VRAM (RTX 3080, RTX 4090) - All models"
            ;;
        full)
            echo "48GB+ VRAM (A6000, A100) - All models, max parallelism"
            ;;
        *)
            echo "Unknown profile"
            ;;
    esac
}

# 1 when the profile file turns the GPU ingest front end on (it needs about 8.5 GB more on the
# Triton card), else 0. A missing key is 0: never grow a card's footprint by omission.
profile_ingest_gpu_frontend() {
    if grep -Eq '"ingest_gpu_frontend"[[:space:]]*:[[:space:]]*true' "$1"; then echo 1; else echo 0; fi
}

# Load profile configuration from JSON
load_profile() {
    local profile="${1:-$SELECTED_PROFILE}"
    local project_dir="${PROJECT_DIR:-$(dirname "$(dirname "$(dirname "${BASH_SOURCE[0]}")")")}"
    local profile_file="$project_dir/config_templates/profiles/${profile}.json"

    if [[ ! -f "$profile_file" ]]; then
        log_error "Profile not found: $profile_file"
        return 1
    fi

    # Parse JSON and export variables
    # Using simple grep/sed since jq may not be installed
    PROFILE_NAME=$(grep -o '"name"[[:space:]]*:[[:space:]]*"[^"]*"' "$profile_file" | cut -d'"' -f4)
    PROFILE_INSTANCE_COUNT=$(grep -o '"instance_count"[[:space:]]*:[[:space:]]*[0-9]*' "$profile_file" | grep -o '[0-9]*$')
    PROFILE_MAX_BATCH=$(grep -o '"max_batch_size"[[:space:]]*:[[:space:]]*[0-9]*' "$profile_file" | grep -o '[0-9]*$')
    PROFILE_SHM_SIZE=$(grep -o '"shm_size"[[:space:]]*:[[:space:]]*"[^"]*"' "$profile_file" | cut -d'"' -f4)
    PROFILE_WORKERS=$(grep -o '"workers"[[:space:]]*:[[:space:]]*[0-9]*' "$profile_file" | grep -o '[0-9]*$')
    PROFILE_OCR_PIPELINE_INSTANCES=$(grep -o '"ocr_pipeline_instances"[[:space:]]*:[[:space:]]*[0-9]*' "$profile_file" | grep -o '[0-9]*$')

    PROFILE_INGEST_GPU_FRONTEND=$(profile_ingest_gpu_frontend "$profile_file")

    export PROFILE_NAME PROFILE_INSTANCE_COUNT PROFILE_MAX_BATCH PROFILE_SHM_SIZE PROFILE_WORKERS PROFILE_OCR_PIPELINE_INSTANCES PROFILE_INGEST_GPU_FRONTEND

    log_debug "Loaded profile: $PROFILE_NAME (instances=$PROFILE_INSTANCE_COUNT, batch=$PROFILE_MAX_BATCH)"
    return 0
}

# =============================================================================
# CUDA Validation
# =============================================================================

# Check CUDA driver version compatibility
validate_cuda() {
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not found."
        return 1
    fi

    local driver_version
    driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)

    # Extract major version
    local major_version
    major_version=$(echo "$driver_version" | cut -d'.' -f1)

    # Triton 26.06 (CUDA 13.x) requires driver >= 570
    local min_driver=570

    if [[ "$major_version" -lt "$min_driver" ]]; then
        log_error "NVIDIA driver $driver_version is too old."
        log_error "Minimum required: $min_driver.x"
        log_info "Update drivers: https://www.nvidia.com/drivers"
        return 1
    fi

    log_debug "CUDA driver version: $driver_version (OK)"
    return 0
}

# Check Docker with NVIDIA runtime
validate_docker_nvidia() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker not found. Please install Docker."
        return 1
    fi

    # Check if nvidia runtime is available
    if ! docker info 2>/dev/null | grep -q "nvidia"; then
        log_warn "NVIDIA Docker runtime not detected."
        log_info "Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"

        # Try a test container
        if docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi &>/dev/null; then
            log_success "Docker GPU access verified via --gpus flag."
            return 0
        else
            log_error "Cannot access GPU from Docker containers."
            return 1
        fi
    fi

    log_debug "Docker NVIDIA runtime available."
    return 0
}

# =============================================================================
# GPU Memory Check
# =============================================================================

# Check current GPU memory usage
check_gpu_memory_usage() {
    local gpu_id="${1:-0}"

    if ! command -v nvidia-smi &> /dev/null; then
        return 1
    fi

    local used free total
    read -r used free total < <(nvidia-smi --query-gpu=memory.used,memory.free,memory.total \
        --format=csv,noheader,nounits -i "$gpu_id" | tr ',' ' ')

    local used_gb=$((used / 1024))
    local free_gb=$((free / 1024))
    local total_gb=$((total / 1024))
    local percent_used=$((used * 100 / total))

    echo "GPU $gpu_id: ${used_gb}GB / ${total_gb}GB used (${percent_used}%), ${free_gb}GB free"
}

# Wait for GPU memory to be available
wait_for_gpu_memory() {
    local required_mb="${1:-4000}"
    local timeout="${2:-60}"
    local gpu_id="${GPU_ID:-0}"

    log_info "Waiting for ${required_mb}MB free GPU memory..."

    local elapsed=0
    while [[ $elapsed -lt $timeout ]]; do
        local free_mb
        free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null)

        if [[ "$free_mb" -ge "$required_mb" ]]; then
            log_success "GPU memory available: ${free_mb}MB"
            return 0
        fi

        sleep 2
        elapsed=$((elapsed + 2))
    done

    log_error "Timeout waiting for GPU memory. Only ${free_mb}MB free."
    return 1
}

# =============================================================================
# PE encoder placement (TRITON_GPU_ID may list several GPUs)
# =============================================================================

# gpu_pe_placement IDS PE_MIB TABLE [TRITON_RUNNING] [PRIMARY_MIB]
#   IDS     TRITON_GPU_ID as written in .env: "1" or "0,2" (host indexes)
#   PE_MIB  GPU memory of one PE encoder instance
#   TABLE   one GPU per line: index|name|total_mib|used_mib|compute_cap
#   TRITON_RUNNING=1: this stack's Triton already holds its memory, so a
#           shortfall of free memory is a warning, not a refusal
#   PRIMARY_MIB: memory the primary card needs for every Triton model (the
#           profile's budget); a 12 GB card is a PE-only secondary, not a primary
# Prints, one per line: primary=ID, pe_gpus=ID,ID, gpu=ID role=... lines,
# warn=... and problem=.... Returns 1 when there is any problem. Fails closed:
# a listed GPU the host does not have, a repeated id, cards of different
# compute capability (one TensorRT plan serves them all) or too little free
# memory all refuse.
gpu_pe_placement() {
    local ids_csv="$1" pe_mib="$2" table="$3" running="${4:-0}" primary_mib="${5:-0}"
    local -A name=() total=() used=() cc=()
    local -a have=() want=() problems=() warns=() listed=()
    local -A kinds=()
    local idx n t u c id seen=" " primary="" line free ccs="" need what

    while IFS='|' read -r idx n t u c; do
        [[ -z "$idx" ]] && continue
        have+=("$idx"); name[$idx]="$n"; total[$idx]="$t"; used[$idx]="$u"; cc[$idx]="$c"
    done <<< "$table"

    if [[ -z "$ids_csv" ]]; then
        problems+=("TRITON_GPU_ID is empty")
    else
        IFS=',' read -ra want <<< "$ids_csv"
    fi
    for id in "${want[@]}"; do
        id="${id// /}"
        if [[ ! "$id" =~ ^[0-9]+$ ]]; then
            problems+=("TRITON_GPU_ID entry '${id}' is not a GPU index")
        elif [[ "$seen" == *" ${id} "* ]]; then
            problems+=("GPU ${id} is listed twice in TRITON_GPU_ID")
        elif [[ -z "${total[$id]+x}" ]]; then
            problems+=("GPU ${id} is not present on this host (it has: $(IFS=,; echo "${have[*]:-none}"))")
            seen+="${id} "
        else
            seen+="${id} "
            listed+=("$id")
            if [[ -z "$primary" ]] || (( id < primary )); then primary=$id; fi
        fi
    done

    for id in "${listed[@]}"; do
        free=$(( total[$id] - used[$id] ))
        need=$pe_mib
        what="the PE encoder needs about ${pe_mib} MiB"
        if [[ "$id" == "$primary" ]] && (( primary_mib > pe_mib )); then
            need=$primary_mib
            what="the primary card runs every Triton model and needs about ${primary_mib} MiB (put a larger card first, i.e. give it the lowest id)"
        fi
        if (( free < need )); then
            line="GPU ${id} has ${free} MiB free; ${what}"
            if [[ "$running" == 1 ]]; then
                warns+=("${line} (Triton is running: its own memory is already counted as used)")
            else
                problems+=("$line")
            fi
        fi
        [[ " ${ccs} " == *" ${cc[$id]} "* ]] || ccs+="${cc[$id]} "
        kinds[${name[$id]}]=1
    done
    if (( $(wc -w <<< "$ccs") > 1 )); then
        problems+=("the listed GPUs have different compute capabilities (${ccs% }): one TensorRT plan cannot serve them all")
    fi
    if (( ${#kinds[@]} > 1 )); then
        warns+=("the listed GPUs are different models; the PE plan is built on one model, TensorRT warns when another loads it and the cards run at their own speed")
    fi

    if [[ -n "$primary" ]]; then
        echo "primary=${primary}"
        echo "pe_gpus=$(printf '%s\n' "${listed[@]}" | sort -n | paste -sd, -)"
        for id in $(printf '%s\n' "${listed[@]}" | sort -n); do
            local role=pe
            [[ "$id" == "$primary" ]] && role=primary
            echo "gpu=${id} role=${role} name=${name[$id]} total_mib=${total[$id]} free_mib=$(( total[$id] - used[$id] )) compute_cap=${cc[$id]}"
        done
    fi
    for line in "${warns[@]}"; do echo "warn=${line}"; done
    for line in "${problems[@]}"; do echo "problem=${line}"; done
    (( ${#problems[@]} == 0 ))
}

# =============================================================================
# Print GPU Summary
# =============================================================================

print_gpu_summary() {
    detect_gpu || return 1

    echo ""
    echo -e "${BOLD}GPU Configuration${NC}"
    echo "-------------------------------------------"
    echo -e "  GPU:              ${GREEN}$GPU_NAME${NC}"
    echo -e "  VRAM:             ${GPU_VRAM_GB}GB"
    echo -e "  Compute Cap:      $GPU_COMPUTE_CAP"
    echo -e "  GPU Count:        $GPU_COUNT"

    if [[ -n "${SELECTED_PROFILE:-}" ]]; then
        echo -e "  Selected Profile: ${CYAN}$SELECTED_PROFILE${NC}"
        echo -e "  Profile Info:     $(get_profile_description "$SELECTED_PROFILE")"
    fi

    echo "-------------------------------------------"
    check_gpu_memory_usage "${GPU_ID:-0}"
    echo ""
}
