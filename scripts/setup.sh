#!/bin/bash
# =============================================================================
# triton-api Setup Script
# =============================================================================
# One-command setup for triton-api visual AI system
#
# Usage:
#   ./scripts/setup.sh              # Interactive setup
#   ./scripts/setup.sh --profile=standard --gpu=0 --yes
#   curl -fsSL <repo>/setup.sh | bash
#
# This script will:
#   1. Check prerequisites (Docker, NVIDIA drivers, etc.)
#   2. Detect GPU and select appropriate profile
#   3. Download required models
#   4. Export models to TensorRT
#   5. Generate configuration files
#   6. Start services
#   7. Run smoke tests
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
export PROJECT_DIR

# Source library functions
source "${SCRIPT_DIR}/lib/colors.sh"
source "${SCRIPT_DIR}/lib/gpu.sh"
source "${SCRIPT_DIR}/lib/download.sh"
source "${SCRIPT_DIR}/lib/export.sh"
source "${SCRIPT_DIR}/lib/config.sh"

# Default options
AUTO_YES=false
SKIP_EXPORT=false
SKIP_DOWNLOAD=false
SKIP_START=false
SELECTED_GPU=""
FORCE_PROFILE=""

# =============================================================================
# Argument Parsing
# =============================================================================

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --yes|-y)
                AUTO_YES=true
                shift
                ;;
            --profile=*)
                FORCE_PROFILE="${1#*=}"
                shift
                ;;
            --gpu=*)
                SELECTED_GPU="${1#*=}"
                shift
                ;;
            --skip-export)
                SKIP_EXPORT=true
                shift
                ;;
            --skip-download)
                SKIP_DOWNLOAD=true
                shift
                ;;
            --skip-start)
                SKIP_START=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

show_help() {
    cat << EOF
triton-api Setup Script

One-Line Setup:
  git clone https://github.com/your-org/triton-api.git && cd triton-api && ./scripts/setup.sh

Usage: ./scripts/setup.sh [OPTIONS]

Options:
  --yes, -y           Skip confirmation prompts (auto-accept defaults)
  --profile=NAME      Force GPU profile (minimal, standard, full)
  --gpu=ID            Use specific GPU ID (default: 0)
  --skip-export       Skip TensorRT export (use existing models)
  --skip-download     Skip model download (use existing)
  --skip-start        Don't start services after setup
  --help, -h          Show this help

Examples:
  ./scripts/setup.sh                          # Interactive setup
  ./scripts/setup.sh --yes                    # Auto-accept defaults
  ./scripts/setup.sh --profile=minimal --yes  # Minimal profile, non-interactive
  ./scripts/setup.sh --skip-export            # Skip long TensorRT export
  ./scripts/setup.sh --profile=standard --gpu=1 --yes  # Use GPU 1

GPU Profiles:
  minimal   6-8GB VRAM (RTX 3060, RTX 4060) - core models only
  standard  12-24GB VRAM (RTX 3080, RTX 4090) - all models, balanced
  full      48GB+ VRAM (A6000, A100) - maximum parallelism
EOF
}

# =============================================================================
# Banner
# =============================================================================

print_banner() {
    echo ""
    echo -e "${CYAN}${BOLD}"
    cat << 'EOF'
  _        _ _                              _
 | |_ _ __(_) |_ ___  _ __         __ _ _ __(_)
 | __| '__| | __/ _ \| '_ \ _____ / _` | '_ \| |
 | |_| |  | | || (_) | | | |_____| (_| | |_) | |
  \__|_|  |_|\__\___/|_| |_|      \__,_| .__/|_|
                                       |_|
EOF
    echo -e "${NC}"
    echo -e "  ${DIM}Visual AI API - Powered by NVIDIA Triton${NC}"
    echo -e "  ${DIM}https://github.com/your-org/triton-api${NC}"
    echo ""
}

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
            echo "  full     - 48GB+ VRAM, maximum parallelism"
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
        "$PROJECT_DIR/pytorch_models/yolo11_face"
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

# =============================================================================
# TensorRT Export
# =============================================================================

export_models_step() {
    if [[ "$SKIP_EXPORT" == "true" ]]; then
        log_info "Skipping TensorRT export (--skip-export)"
        return 0
    fi

    print_section "TensorRT Model Export"

    # Check what's already exported
    check_exported_models

    # Check if all required models are already exported
    if all_models_exported; then
        log_success "All required models already exported!"

        if [[ "$AUTO_YES" != "true" ]]; then
            if ! confirm "Re-export models?" "N"; then
                return 0
            fi
        else
            return 0
        fi
    fi

    echo ""
    log_warn "TensorRT export can take 45-60 minutes for all models."
    log_info "The GPU will be heavily utilized during this process."
    echo ""

    if [[ "$AUTO_YES" != "true" ]]; then
        if ! confirm "Start TensorRT export?" "Y"; then
            log_info "Skipping export. Run manually with: ./scripts/triton-api.sh export all"
            return 0
        fi
    fi

    # Start yolo-api container for exports that use Python TRT API
    # (YOLO, ArcFace, MobileCLIP exports run inside yolo-api container)
    # Use --no-deps to avoid starting triton-api which would crash without model files
    log_info "Starting yolo-api container for model exports..."
    docker compose up -d --no-deps yolo-api

    # Wait for yolo-api container to be running
    log_info "Waiting for yolo-api container..."
    local yolo_ready=false
    for i in {1..30}; do
        if docker ps --format '{{.Names}}' | grep -q "^yolo-api$"; then
            yolo_ready=true
            break
        fi
        sleep 2
    done

    if [[ "$yolo_ready" != "true" ]]; then
        log_error "yolo-api container failed to start"
        return 1
    fi
    log_success "yolo-api container ready"

    # PaddleOCR TRT export uses 'docker compose run' to create a temporary
    # triton-api container for trtexec, avoiding the chicken-and-egg problem
    # where triton-api crashes because model.plan files don't exist yet.

    # Export based on profile
    if [[ "$SELECTED_PROFILE" == "minimal" ]]; then
        export_essential_models
    else
        export_all_models
    fi
}

# =============================================================================
# Configuration Generation
# =============================================================================

generate_configs_step() {
    print_section "Generating Configuration"

    # Generate Triton model configs
    generate_all_configs "$SELECTED_PROFILE" "$GPU_ID"

    # Generate .env file
    generate_env_file "$SELECTED_PROFILE" "$GPU_ID"

    # Generate docker-compose.override.yml
    generate_compose_override "$SELECTED_PROFILE" "$GPU_ID"

    log_success "Configuration generated"
}

# =============================================================================
# Service Start
# =============================================================================

start_services_step() {
    if [[ "$SKIP_START" == "true" ]]; then
        log_info "Skipping service start (--skip-start)"
        return 0
    fi

    print_section "Starting Services"

    if [[ "$AUTO_YES" != "true" ]]; then
        if ! confirm "Start triton-api services?" "Y"; then
            log_info "Skipping. Start manually with: docker compose up -d"
            return 0
        fi
    fi

    # Stop any running containers first (including yolo-api from export step)
    log_info "Stopping existing containers..."
    docker compose stop 2>/dev/null || true
    docker compose rm -f 2>/dev/null || true

    # Start all services (now that model.plan files exist, triton-api will load them)
    log_info "Starting all services..."
    docker compose up -d

    # Wait for services to be ready
    wait_for_services
}

wait_for_services() {
    print_section "Waiting for Services"

    # Wait for Triton HTTP endpoint
    log_info "Waiting for Triton HTTP endpoint..."
    local triton_http=false
    for i in {1..60}; do
        if curl -s localhost:4600/v2/health/ready > /dev/null 2>&1; then
            triton_http=true
            break
        fi
        printf "\r  Waiting for Triton... %ds " "$((i*2))"
        sleep 2
    done
    echo ""

    if [[ "$triton_http" != "true" ]]; then
        log_error "Triton HTTP endpoint not responding after 120s"
        return 1
    fi
    log_success "Triton HTTP endpoint ready"

    # Wait for required models to load
    log_info "Waiting for TensorRT models to load..."
    local required_models=("yolov11_small_trt_end2end" "yolo11_face_small_trt_end2end" "arcface_w600k_r50" "mobileclip2_s2_image_encoder" "mobileclip2_s2_text_encoder" "paddleocr_det_trt" "paddleocr_rec_trt")
    local all_loaded=false

    for i in {1..90}; do  # Wait up to 3 minutes for models
        local loaded=0
        local total=${#required_models[@]}

        for model in "${required_models[@]}"; do
            # Triton returns HTTP 200 with empty body when model is ready
            if curl -s -o /dev/null -w "%{http_code}" "localhost:4600/v2/models/${model}/ready" 2>/dev/null | grep -q "200"; then
                loaded=$((loaded + 1))
            fi
        done

        printf "\r  Models loaded: %d/%d " "$loaded" "$total"

        if [[ $loaded -eq $total ]]; then
            all_loaded=true
            break
        fi
        sleep 2
    done
    echo ""

    if [[ "$all_loaded" == "true" ]]; then
        log_success "All TensorRT models loaded"
    else
        log_warn "Some models may not be loaded yet - continuing anyway"
        log_info "Loaded models:"
        for model in "${required_models[@]}"; do
            if curl -s -o /dev/null -w "%{http_code}" "localhost:4600/v2/models/${model}/ready" 2>/dev/null | grep -q "200"; then
                log_success "  $model"
            else
                log_warn "  $model (not ready)"
            fi
        done
    fi

    # Wait for API
    log_info "Waiting for API service..."
    local api_ready=false
    for i in {1..30}; do
        if curl -s localhost:4603/health 2>/dev/null | grep -q "healthy"; then
            api_ready=true
            break
        fi
        printf "\r  Waiting for API... %ds " "$((i*2))"
        sleep 2
    done
    echo ""

    if [[ "$api_ready" == "true" ]]; then
        log_success "API is ready"
    else
        log_warn "API not fully ready yet"
    fi

    # Wait for OpenSearch
    log_info "Waiting for OpenSearch..."
    local opensearch_ready=false
    for i in {1..30}; do
        if curl -s localhost:4607 > /dev/null 2>&1; then
            opensearch_ready=true
            break
        fi
        printf "\r  Waiting for OpenSearch... %ds " "$((i*2))"
        sleep 2
    done
    echo ""

    if [[ "$opensearch_ready" == "true" ]]; then
        log_success "OpenSearch is ready"
    else
        log_warn "OpenSearch not ready yet"
    fi
}

# =============================================================================
# Smoke Test
# =============================================================================

run_smoke_test() {
    print_section "Running Smoke Tests"

    local passed=0
    local failed=0
    local test_image="$PROJECT_DIR/test_images/bus.jpg"
    local face_image="$PROJECT_DIR/test_images/zidane.jpg"

    # Check if test images exist
    if [[ ! -f "$test_image" ]]; then
        log_warn "Test image not found: $test_image"
        log_info "Skipping image-based smoke tests"
        return 0
    fi

    # Test 1: Health endpoint
    log_info "Testing /health endpoint..."
    if curl -s localhost:4603/health | grep -q "healthy"; then
        log_success "[1/6] Health check - PASS"
        passed=$((passed + 1))
    else
        log_error "[1/6] Health check - FAIL"
        failed=$((failed + 1))
    fi

    # Test 2: Object detection
    log_info "Testing object detection..."
    local detect_result
    detect_result=$(curl -s -X POST localhost:4603/detect -F "image=@$test_image" 2>/dev/null)
    if echo "$detect_result" | grep -q "detections"; then
        local det_count
        det_count=$(echo "$detect_result" | grep -o '"class_name"' | wc -l)
        log_success "[2/6] Object detection - PASS ($det_count objects detected)"
        passed=$((passed + 1))
    else
        log_error "[2/6] Object detection - FAIL"
        echo "  Response: $(echo "$detect_result" | head -c 200)"
        failed=$((failed + 1))
    fi

    # Test 3: Face detection
    if [[ -f "$face_image" ]]; then
        log_info "Testing face detection..."
        local face_result
        face_result=$(curl -s -X POST localhost:4603/faces/recognize -F "image=@$face_image" 2>/dev/null)
        if echo "$face_result" | grep -q "faces"; then
            local face_count
            face_count=$(echo "$face_result" | grep -o '"box"' | wc -l)
            log_success "[3/6] Face detection - PASS ($face_count faces detected)"
            passed=$((passed + 1))
        else
            log_error "[3/6] Face detection - FAIL"
            echo "  Response: $(echo "$face_result" | head -c 200)"
            failed=$((failed + 1))
        fi
    else
        log_info "[3/6] Face detection - SKIPPED (no test image)"
    fi

    # Test 4: CLIP embedding
    log_info "Testing CLIP embedding..."
    local embed_result
    embed_result=$(curl -s -X POST localhost:4603/embed/image -F "image=@$test_image" 2>/dev/null)
    if echo "$embed_result" | grep -q "embedding"; then
        log_success "[4/6] CLIP embedding - PASS"
        passed=$((passed + 1))
    else
        log_error "[4/6] CLIP embedding - FAIL"
        echo "  Response: $(echo "$embed_result" | head -c 200)"
        failed=$((failed + 1))
    fi

    # Test 5: Full analysis
    log_info "Testing full analysis..."
    local analyze_result
    analyze_result=$(curl -s -X POST localhost:4603/analyze -F "image=@$test_image" 2>/dev/null)
    if echo "$analyze_result" | grep -q "status"; then
        log_success "[5/6] Full analysis - PASS"
        passed=$((passed + 1))
    else
        log_error "[5/6] Full analysis - FAIL"
        echo "  Response: $(echo "$analyze_result" | head -c 200)"
        failed=$((failed + 1))
    fi

    # Test 6: Triton model count
    log_info "Checking Triton models..."
    local model_count
    model_count=$(curl -s localhost:4600/v2/models 2>/dev/null | grep -c '"name"' || echo "0")
    if [[ "$model_count" -ge 5 ]]; then
        log_success "[6/6] Triton models - PASS ($model_count models loaded)"
        passed=$((passed + 1))
    else
        log_warn "[6/6] Triton models - WARN (only $model_count models)"
    fi

    # Summary
    echo ""
    print_header "Smoke Test Results"
    local total=$((passed + failed))
    if [[ $failed -eq 0 ]]; then
        echo -e "${GREEN}All tests passed: $passed/$total${NC}"
    else
        echo -e "${YELLOW}Tests: $passed passed, $failed failed${NC}"
        echo ""
        log_info "Some tests failed. This may be because models are still loading."
        log_info "Wait a minute and run: make test-all"
    fi
}

# =============================================================================
# Summary
# =============================================================================

print_success_summary() {
    print_header "Setup Complete!"

    echo -e "${GREEN}triton-api is ready to use!${NC}"
    echo ""
    echo "Services running:"
    echo "  API:         http://localhost:4603"
    echo "  API Docs:    http://localhost:4603/docs"
    echo "  Triton:      http://localhost:4600"
    echo "  Grafana:     http://localhost:4605 (admin/admin)"
    echo "  OpenSearch:  http://localhost:4607"
    echo ""
    echo "Quick test:"
    echo "  curl http://localhost:4603/health"
    echo ""
    echo "Documentation:"
    echo "  README.md        - API overview and examples"
    echo "  INSTALLATION.md  - Detailed installation guide"
    echo "  CLAUDE.md        - Full technical reference"
    echo ""
    echo "Management:"
    echo "  ./scripts/triton-api.sh status   - Check service status"
    echo "  ./scripts/triton-api.sh logs     - View logs"
    echo "  ./scripts/triton-api.sh restart  - Restart services"
    echo ""
}

# =============================================================================
# Main
# =============================================================================

main() {
    parse_args "$@"

    print_banner

    # Step 1: Prerequisites
    check_prerequisites || exit 1

    # Step 2: GPU detection and profile selection
    detect_and_select_gpu || exit 1

    # Step 3: Create directories
    create_directories

    # Step 4: Download models
    download_models_step

    # Step 5: Export to TensorRT
    export_models_step

    # Step 6: Generate configuration
    generate_configs_step

    # Step 7: Start services
    start_services_step

    # Step 8: Smoke tests
    if [[ "$SKIP_START" != "true" ]]; then
        run_smoke_test
    fi

    # Step 9: Success summary
    print_success_summary
}

main "$@"
