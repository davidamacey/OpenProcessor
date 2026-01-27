#!/bin/bash
# =============================================================================
# download.sh - Model download utilities for triton-api
# =============================================================================
# Downloads models from public sources (no authentication required):
# - GitHub releases (Ultralytics)
# - HuggingFace public models (SCRFD, ArcFace, MobileCLIP, PaddleOCR)
# =============================================================================

# Avoid redefining if already sourced
[[ -n "${_DOWNLOAD_SH_LOADED:-}" ]] && return 0
_DOWNLOAD_SH_LOADED=1

# Source colors if not already loaded
_DOWNLOAD_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=colors.sh
[[ -z "${NC:-}" ]] && source "${_DOWNLOAD_SH_DIR}/colors.sh"

# =============================================================================
# Configuration
# =============================================================================

# Project directory (set by caller or auto-detect)
PROJECT_DIR="${PROJECT_DIR:-$(dirname "$(dirname "$(dirname "${BASH_SOURCE[0]}")")")}"
PYTORCH_MODELS_DIR="${PROJECT_DIR}/pytorch_models"

# Model sources (all public, no auth required)
# Use -gA to ensure global scope when sourced from within a function
declare -gA MODEL_SOURCES=(
    # YOLO models from Ultralytics GitHub releases
    ["yolo11s.pt"]="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt"
    ["yolo11n.pt"]="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"

    # SCRFD-10G face detection with 5-point landmarks (from InsightFace via HuggingFace)
    ["scrfd_10g_bnkps.onnx"]="https://huggingface.co/LPDoctor/insightface/resolve/main/scrfd_10g_bnkps.onnx"

    # ArcFace from FaceFusion (public HuggingFace)
    ["arcface_w600k_r50.onnx"]="https://huggingface.co/facefusion/models-3.0.0/resolve/main/arcface_w600k_r50.onnx"

    # MobileCLIP from Apple (via HuggingFace)
    ["mobileclip_s2.pt"]="https://huggingface.co/apple/mobileclip-s2/resolve/main/mobileclip_s2.pt"
)

# Model checksums for validation (sha256)
declare -gA MODEL_CHECKSUMS=(
    # Add checksums as needed - empty means skip validation
    ["yolo11s.pt"]=""
    ["yolo11n.pt"]=""
    ["scrfd_10g_bnkps.onnx"]=""
    ["arcface_w600k_r50.onnx"]="f1f79dc3b0b79a69f94799af1fffebff09fbd78fd96a275fd8f0cbbea23270d1"
)

# =============================================================================
# Download Functions
# =============================================================================

# Download a file with retry and progress
# Usage: download_with_retry URL OUTPUT_PATH [MAX_RETRIES]
download_with_retry() {
    local url="$1"
    local output="$2"
    local max_retries="${3:-3}"
    local retry_delay=5

    local attempt=1
    while [[ $attempt -le $max_retries ]]; do
        log_debug "Download attempt $attempt/$max_retries: $url"

        # Create output directory
        mkdir -p "$(dirname "$output")"

        # Download with curl (shows progress bar)
        if curl -L --fail --progress-bar \
            --connect-timeout 30 \
            --max-time 600 \
            -o "$output" \
            "$url" 2>&1; then

            # Verify file exists and has content
            if [[ -f "$output" ]] && [[ -s "$output" ]]; then
                log_debug "Download successful: $output"
                return 0
            else
                log_warn "Download created empty file, retrying..."
                rm -f "$output"
            fi
        else
            log_warn "Download failed (attempt $attempt/$max_retries)"
        fi

        # Exponential backoff
        if [[ $attempt -lt $max_retries ]]; then
            log_info "Retrying in ${retry_delay}s..."
            sleep $retry_delay
            retry_delay=$((retry_delay * 2))
        fi

        attempt=$((attempt + 1))
    done

    log_error "Failed to download after $max_retries attempts: $url"
    return 1
}

# Validate file checksum
# Usage: validate_checksum FILE EXPECTED_SHA256
validate_checksum() {
    local file="$1"
    local expected="$2"

    # Skip if no checksum provided
    if [[ -z "$expected" ]]; then
        log_debug "No checksum provided, skipping validation for: $file"
        return 0
    fi

    if [[ ! -f "$file" ]]; then
        log_error "File not found for checksum validation: $file"
        return 1
    fi

    local actual
    actual=$(sha256sum "$file" | cut -d' ' -f1)

    if [[ "$actual" != "$expected" ]]; then
        log_error "Checksum mismatch for $file"
        log_error "  Expected: $expected"
        log_error "  Actual:   $actual"
        return 1
    fi

    log_debug "Checksum valid: $file"
    return 0
}

# Download a model by name
# Usage: download_model MODEL_NAME [OUTPUT_DIR]
download_model() {
    local model_name="$1"
    local output_dir="${2:-$PYTORCH_MODELS_DIR}"

    # Handle special subdirectories for certain model types
    local output_path
    if [[ "$model_name" == "mobileclip_s2.pt" ]]; then
        # MobileCLIP goes in mobileclip2_s2 subdirectory with renamed file
        # Export script expects: mobileclip2_s2/mobileclip2_s2.pt
        mkdir -p "$output_dir/mobileclip2_s2"
        output_path="$output_dir/mobileclip2_s2/mobileclip2_s2.pt"
    else
        output_path="$output_dir/$model_name"
    fi

    # Check if already downloaded
    if [[ -f "$output_path" ]] && [[ -s "$output_path" ]]; then
        local size
        size=$(du -h "$output_path" | cut -f1)
        log_info "Already downloaded: $model_name ($size)"
        return 0
    fi

    # Get source URL
    local url="${MODEL_SOURCES[$model_name]}"
    if [[ -z "$url" ]]; then
        log_error "Unknown model: $model_name"
        log_info "Available models: ${!MODEL_SOURCES[*]}"
        return 1
    fi

    log_info "Downloading: $model_name"
    log_debug "  URL: $url"
    log_debug "  Output: $output_path"

    if download_with_retry "$url" "$output_path"; then
        # Validate checksum if available
        local checksum="${MODEL_CHECKSUMS[$model_name]:-}"
        if ! validate_checksum "$output_path" "$checksum"; then
            log_warn "Checksum validation failed, file may be corrupt"
            # Don't delete - let user decide
        fi

        local size
        size=$(du -h "$output_path" | cut -f1)
        log_success "Downloaded: $model_name ($size)"
        return 0
    else
        return 1
    fi
}

# =============================================================================
# Batch Download Functions
# =============================================================================

# Download essential models (minimum for API to work)
download_essential_models() {
    log_step "Downloading essential models..."

    local models=(
        "yolo11s.pt"
        "scrfd_10g_bnkps.onnx"
        "arcface_w600k_r50.onnx"
        "mobileclip_s2.pt"
    )

    local failed=0
    for model in "${models[@]}"; do
        if ! download_model "$model"; then
            failed=$((failed + 1))
        fi
    done

    if [[ $failed -gt 0 ]]; then
        log_error "Failed to download $failed model(s)"
        return 1
    fi

    log_success "Essential models downloaded"
    return 0
}

# Download all available models
download_all_models() {
    log_step "Downloading all models..."

    local failed=0
    for model in "${!MODEL_SOURCES[@]}"; do
        if ! download_model "$model"; then
            failed=$((failed + 1))
        fi
    done

    if [[ $failed -gt 0 ]]; then
        log_warn "Failed to download $failed model(s)"
        return 1
    fi

    log_success "All models downloaded"
    return 0
}

# Download models based on profile
download_profile_models() {
    local profile="${1:-standard}"

    case "$profile" in
        minimal)
            download_essential_models
            ;;
        standard|full)
            download_all_models
            ;;
        *)
            log_error "Unknown profile: $profile"
            return 1
            ;;
    esac
}

# =============================================================================
# PaddleOCR Download (Special handling)
# =============================================================================

# Download PaddleOCR models via Python script
download_paddleocr_models() {
    log_step "Downloading PaddleOCR models..."

    # Check if container is running
    if ! docker ps --format '{{.Names}}' | grep -q "^yolo-api$"; then
        log_error "yolo-api container not running"
        log_info "Start with: docker compose up -d yolo-api"
        return 1
    fi

    # Run download script in container
    if docker compose exec yolo-api python /app/export/download_paddleocr.py; then
        log_success "PaddleOCR models downloaded"
        return 0
    else
        log_error "PaddleOCR download failed"
        return 1
    fi
}

# =============================================================================
# Status Functions
# =============================================================================

# Check which models are already downloaded
check_downloaded_models() {
    echo ""
    echo -e "${BOLD}Downloaded Models${NC}"
    echo "-------------------------------------------"

    for model in "${!MODEL_SOURCES[@]}"; do
        local path
        # Handle special subdirectories
        if [[ "$model" == "mobileclip_s2.pt" ]]; then
            path="$PYTORCH_MODELS_DIR/mobileclip2_s2/mobileclip2_s2.pt"
        else
            path="$PYTORCH_MODELS_DIR/$model"
        fi

        if [[ -f "$path" ]] && [[ -s "$path" ]]; then
            local size
            size=$(du -h "$path" | cut -f1)
            echo -e "  ${GREEN}[OK]${NC} $model ($size)"
        else
            echo -e "  ${RED}[--]${NC} $model"
        fi
    done

    echo "-------------------------------------------"
    echo ""
}

# Get total size of models to download
estimate_download_size() {
    local profile="${1:-all}"
    local total_mb=0

    case "$profile" in
        minimal)
            # Essential models only (~250MB)
            total_mb=250
            ;;
        standard|full|all)
            # All models (~500MB)
            total_mb=500
            ;;
    esac

    echo "$total_mb"
}

# =============================================================================
# MobileCLIP Reference Repository (for tokenizer)
# =============================================================================

# Clone MobileCLIP reference repo for tokenizer support
clone_mobileclip_repo() {
    local repo_dir="$PROJECT_DIR/reference_repos/ml-mobileclip"

    if [[ -d "$repo_dir/.git" ]]; then
        log_info "MobileCLIP repo already cloned"
        return 0
    fi

    log_step "Cloning MobileCLIP reference repo..."

    mkdir -p "$(dirname "$repo_dir")"

    if git clone --depth 1 \
        https://github.com/apple/ml-mobileclip.git \
        "$repo_dir" 2>/dev/null; then
        log_success "MobileCLIP repo cloned"
        return 0
    else
        log_warn "Failed to clone MobileCLIP repo (tokenizer may need manual setup)"
        return 1
    fi
}
