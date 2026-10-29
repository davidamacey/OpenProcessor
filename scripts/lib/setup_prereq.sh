#!/bin/bash
# =============================================================================
# setup_prereq.sh - prerequisite, GPU, directory and model-download steps
# =============================================================================
# Sourced by scripts/setup.sh (checkout only), which supplies dc(), log_*(),
# the option flags, PROJECT_DIR and SCRIPT_DIR.

[[ -n "${_SETUP_PREREQ_SH_LOADED:-}" ]] && return 0
_SETUP_PREREQ_SH_LOADED=1

# =============================================================================
# Prerequisites Check
# =============================================================================

check_prerequisites() {
    print_section "Checking Prerequisites"

    local errors=0

    # Check Docker
    if command -v docker &> /dev/null; then
        local docker_version
        docker_version=$(docker --version | grep -oP '\d+\.\d+\.\d+' | head -1)
        log_success "Docker: $docker_version"
    else
        log_error "Docker not found"
        log_info "Install: https://docs.docker.com/engine/install/"
        errors=$((errors + 1))
    fi

    # Check Docker Compose
    if docker compose version &> /dev/null; then
        local compose_version
        compose_version=$(docker compose version --short 2>/dev/null || echo "v2+")
        log_success "Docker Compose: $compose_version"
    else
        log_error "Docker Compose not found"
        errors=$((errors + 1))
    fi

    # Check NVIDIA drivers
    if command -v nvidia-smi &> /dev/null; then
        local driver_version
        driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
        log_success "NVIDIA Driver: $driver_version"
    else
        log_error "nvidia-smi not found - NVIDIA drivers not installed"
        log_info "Install: https://www.nvidia.com/drivers"
        errors=$((errors + 1))
    fi

    # Check Docker GPU access (via nvidia-container-toolkit)
    if docker info 2>/dev/null | grep -qi "nvidia"; then
        log_success "Docker GPU access: OK (nvidia runtime detected)"
    else
        log_warn "NVIDIA container runtime not detected in docker info"
        log_info "This may be OK if nvidia-container-toolkit is installed"
        log_info "Install if needed: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    fi

    # Check curl
    if command -v curl &> /dev/null; then
        log_success "curl: available"
    else
        log_error "curl not found"
        errors=$((errors + 1))
    fi

    # Check git
    if command -v git &> /dev/null; then
        log_success "git: available"
    else
        log_warn "git not found (optional, needed for some features)"
    fi

    if [[ $errors -gt 0 ]]; then
        echo ""
        log_error "$errors prerequisite(s) missing. Please install them and try again."
        return 1
    fi

    echo ""
    log_success "All prerequisites met!"
    return 0
}

# =============================================================================
# GPU Detection and Profile Selection
# =============================================================================

detect_and_select_gpu() {
    print_section "GPU Detection"

    # Detect GPU
    if ! detect_gpu; then
        log_error "Failed to detect GPU"
        return 1
    fi

    # List available GPUs if multiple
    if [[ "$GPU_COUNT" -gt 1 ]]; then
        echo ""
        list_gpus
        echo ""

        if [[ -n "$SELECTED_GPU" ]]; then
            GPU_ID="$SELECTED_GPU"
        elif [[ "$AUTO_YES" == "true" ]]; then
            GPU_ID=0
        else
            read -r -p "Which GPU to use? [0]: " GPU_ID
            GPU_ID="${GPU_ID:-0}"
        fi
    else
        GPU_ID="${SELECTED_GPU:-0}"
    fi

    export GPU_ID

    # Re-detect for selected GPU
    detect_gpu

    # Select profile
    if [[ -n "$FORCE_PROFILE" ]]; then
        SELECTED_PROFILE="$FORCE_PROFILE"
    else
        select_profile
    fi

    # Confirm with user
    print_gpu_summary

    if [[ "$AUTO_YES" != "true" ]]; then
        local profile_desc
        profile_desc=$(get_profile_description "$SELECTED_PROFILE")
        echo -e "Selected profile: ${CYAN}$SELECTED_PROFILE${NC} - $profile_desc"
        echo ""

        if ! confirm "Use this configuration?" "Y"; then
            echo ""
            echo "Available profiles:"
            echo "  minimal  - 6-8GB VRAM, core models only"
            echo "  standard - 12-24GB VRAM, all models"
            echo "  full     - 40GB+ VRAM, maximum parallelism"
            echo ""
            read -r -p "Enter profile name: " SELECTED_PROFILE
        fi
    fi

    log_success "Using profile: $SELECTED_PROFILE on GPU $GPU_ID"
    return 0
}

# =============================================================================
# Directory Setup
# =============================================================================

create_directories() {
    print_section "Setting Up Directories"

    local dirs=(
        "$PROJECT_DIR/pytorch_models"
        "$PROJECT_DIR/pytorch_models/paddleocr"
        "$PROJECT_DIR/pytorch_models/mobileclip2_s2"
        "$PROJECT_DIR/cache/huggingface"
        "$PROJECT_DIR/test_results"
    )

    for dir in "${dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            log_info "Created: $dir"
        fi
    done

    # Create .gitkeep for cache directory
    touch "$PROJECT_DIR/cache/huggingface/.gitkeep"

    log_success "Directories ready"
}

# =============================================================================
# Model Download
# =============================================================================

download_models_step() {
    if [[ "$SKIP_DOWNLOAD" == "true" ]]; then
        log_info "Skipping model download (--skip-download)"
        return 0
    fi

    print_section "Downloading Models"

    # Check what's already downloaded
    check_downloaded_models

    # Determine what to download based on profile
    local download_size
    download_size=$(estimate_download_size "$SELECTED_PROFILE")

    echo ""
    log_info "Estimated download size: ~${download_size}MB"

    if [[ "$AUTO_YES" != "true" ]]; then
        if ! confirm "Download models?" "Y"; then
            log_info "Skipping download"
            return 0
        fi
    fi

    # Download based on profile
    if [[ "$SELECTED_PROFILE" == "minimal" ]]; then
        download_essential_models
    else
        download_all_models
    fi
}
