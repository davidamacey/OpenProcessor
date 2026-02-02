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
    PROFILE_HEAP=$(grep -o '"opensearch_heap"[[:space:]]*:[[:space:]]*"[^"]*"' "$profile_file" | cut -d'"' -f4)
    PROFILE_WORKERS=$(grep -o '"workers"[[:space:]]*:[[:space:]]*[0-9]*' "$profile_file" | grep -o '[0-9]*$')

    export PROFILE_NAME PROFILE_INSTANCE_COUNT PROFILE_MAX_BATCH PROFILE_SHM_SIZE PROFILE_HEAP PROFILE_WORKERS

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

    # Triton 25.10 requires driver >= 535
    local min_driver=535

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
