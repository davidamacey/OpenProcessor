#!/bin/bash
# =============================================================================
# OpenProcessor Setup Script
# =============================================================================
# One-command setup for OpenProcessor visual AI system
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

# Option flags and ports are read by the sourced lib/setup_*.sh step functions.
# shellcheck disable=SC2034
set -eo pipefail

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
source "${SCRIPT_DIR}/lib/ports.sh"
source "${SCRIPT_DIR}/lib/setup_prereq.sh"
source "${SCRIPT_DIR}/lib/setup_services.sh"

# Installer plan §1: docker-compose.yml is deploy-safe (no `build:`, no
# source mounts) as of this split; every `build:` block and source mount
# now lives in docker-compose.dev.yml. This script only ever runs from a
# checkout (it lives in scripts/), so it always needs that overlay --
# detect it the same way the Makefile's COMPOSE var does, by checking for
# src/main.py next to the compose file.
DC_FILES=(-f "$PROJECT_DIR/docker-compose.yml")
if [[ -f "$PROJECT_DIR/src/main.py" ]]; then
    DC_FILES+=(-f "$PROJECT_DIR/docker-compose.dev.yml")
fi
dc() { docker compose "${DC_FILES[@]}" "$@"; }

# env_port() (scripts/lib/ports.sh) reads a single KEY=value
# out of .env without sourcing the whole file. Smoke tests below hit
# whatever ports THIS deployment's .env actually configured, not the
# hardcoded defaults -- a remapped API_PORT/TRITON_HTTP_PORT/OPENSEARCH_PORT
# (e.g. a second isolated stack) must still smoke-test correctly.
API_PORT="$(env_port API_PORT 4603)"
TRITON_HTTP_PORT="$(env_port TRITON_HTTP_PORT 4600)"
OPENSEARCH_PORT="$(env_port OPENSEARCH_PORT 4607)"
GRAFANA_PORT="$(env_port GRAFANA_PORT 4605)"

# Default options
AUTO_YES=false
SKIP_EXPORT=false
SKIP_DOWNLOAD=false
SKIP_START=false
SELECTED_GPU=""
FORCE_PROFILE=""
# G-04: never clobber an existing .env on a shared host without an
# explicit --force -- a re-run (or a second operator on the same
# checkout) must not silently overwrite someone else's port/GPU choices.
FORCE_ENV=false
# G-04: print the curation-subsystem next steps at the end of setup
# instead of leaving it undiscoverable. Runs the core pipeline setup
# either way; this only controls whether the extra printout happens.
SHOW_CURATION_NEXT_STEPS=false

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
            --force)
                FORCE_ENV=true
                shift
                ;;
            --curation)
                SHOW_CURATION_NEXT_STEPS=true
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
OpenProcessor Setup Script

One-Line Setup:
  git clone https://github.com/davidamacey/OpenProcessor.git && cd OpenProcessor && ./scripts/setup.sh

Usage: ./scripts/setup.sh [OPTIONS]

Options:
  --yes, -y           Skip confirmation prompts (auto-accept defaults)
  --profile=NAME      Force GPU profile (minimal, standard, full)
  --gpu=ID            Use specific GPU ID (default: 0)
  --skip-export       Skip TensorRT export (use existing models)
  --skip-download     Skip model download (use existing)
  --skip-start        Don't start services after setup
  --force             Overwrite an existing .env (default: never touch one)
  --curation          Print the curation/labeling subsystem next steps at the end
  --help, -h          Show this help

Examples:
  ./scripts/setup.sh                          # Interactive setup
  ./scripts/setup.sh --yes                    # Auto-accept defaults
  ./scripts/setup.sh --profile=minimal --yes  # Minimal profile, non-interactive
  ./scripts/setup.sh --skip-export            # Skip long TensorRT export
  ./scripts/setup.sh --profile=standard --gpu=1 --yes  # Use GPU 1
  ./scripts/setup.sh --curation --yes         # Also print curation setup steps

Shared-host note: this script only ever stops/removes containers in ITS
OWN compose project (the directory it runs from, or COMPOSE_PROJECT_NAME
if set) and never touches an existing .env unless you pass --force.
Running a second isolated stack on the same host? Clone to a second
directory (or set COMPOSE_PROJECT_NAME) and re-map any port this .env
already claims -- see env.template's "Isolation" section.

GPU Profiles:
  minimal   6-8GB VRAM (RTX 3060, RTX 4060) - core models only
  standard  12-24GB VRAM (RTX 3080, RTX 4090) - all models, balanced
  full      40GB+ VRAM (A6000, A100) - maximum parallelism
EOF
}

# =============================================================================
# Banner
# =============================================================================

print_banner() {
    echo ""
    echo -e "${CYAN}${BOLD}"
    cat << 'EOF'
   ___                   ____
  / _ \ _ __   ___ _ __ |  _ \ _ __ ___   ___ ___  ___ ___  ___  _ __
 | | | | '_ \ / _ \ '_ \| |_) | '__/ _ \ / __/ _ \/ __/ __|/ _ \| '__|
 | |_| | |_) |  __/ | | |  __/| | | (_) | (_|  __/\__ \__ \ (_) | |
  \___/| .__/ \___|_| |_|_|   |_|  \___/ \___\___||___/___/\___/|_|
       |_|
EOF
    echo -e "${NC}"
    echo -e "  ${DIM}Visual AI API - Powered by NVIDIA Triton${NC}"
    echo -e "  ${DIM}https://github.com/davidamacey/OpenProcessor${NC}"
    echo ""
}

# =============================================================================
# Docker Image Build
# =============================================================================

build_docker_images() {
    if [[ "$SKIP_EXPORT" == "true" ]] && [[ "$SKIP_START" == "true" ]]; then
        log_info "Skipping Docker build (export and start both skipped)"
        return 0
    fi

    print_section "Pulling/Building Docker Images"

    # Try pulling pre-built images from Docker Hub first (fast path).
    # Fall back to building from Dockerfiles if pull fails (e.g., first release,
    # no internet, or user wants a custom build).
    log_step "Pulling pre-built images from Docker Hub..."
    if dc pull api triton-server 2>/dev/null; then
        log_success "Pre-built images pulled from Docker Hub"
        return 0
    else
        log_warn "Could not pull pre-built images from Docker Hub"
        log_info "Building from source instead (first run pulls base images ~15GB)..."
        echo ""
    fi

    # Build from Dockerfiles
    log_step "Building api image..."
    if ! dc build api; then
        log_error "Failed to build api image"
        log_info "Check your internet connection and disk space (need ~5GB free)"
        return 1
    fi
    log_success "api image ready"

    log_step "Building triton-server image..."
    if ! dc build triton-server; then
        log_error "Failed to build triton-server image"
        log_info "Check your internet connection and disk space (need ~20GB free)"
        return 1
    fi
    log_success "triton-server image ready"

    log_success "Docker images ready"
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
            log_info "Skipping export. Run manually with: ./scripts/openprocessor.sh export all"
            return 0
        fi
    fi

    # All exports use 'docker compose run --rm --no-deps -T' to create temporary
    # containers. This avoids the chicken-and-egg problem where triton-server
    # crashes because model.plan files don't exist yet, and api crash-loops
    # because it can't connect to triton-server.

    # Export based on profile
    # Use || true to prevent set -e from aborting setup on partial failures.
    # Individual export functions log failures and export_all_models reports a summary.
    # Setup should continue to start services with whatever models exported successfully.
    local export_rc=0
    if [[ "$SELECTED_PROFILE" == "minimal" ]]; then
        export_essential_models || export_rc=$?
    else
        export_all_models || export_rc=$?
    fi

    if [[ $export_rc -ne 0 ]]; then
        log_warn "Some model exports failed (exit code $export_rc)"
        log_info "Setup will continue with successfully exported models."
        log_info "Re-run failed exports later: ./scripts/openprocessor.sh export all"
    fi
}

# =============================================================================
# Configuration Generation
# =============================================================================

generate_configs_step() {
    print_section "Generating Configuration"

    # Generate Triton model configs
    generate_all_configs "$SELECTED_PROFILE" "$GPU_ID"

    # Generate .env file (G-04: never overwrites an existing one unless
    # --force was passed)
    generate_env_file "$SELECTED_PROFILE" "$GPU_ID" "$FORCE_ENV"

    log_success "Configuration generated"
}

# =============================================================================
# Main
# =============================================================================

main() {
    parse_args "$@"

    print_banner

    local setup_start=$SECONDS

    # Step 1: Prerequisites
    check_prerequisites || exit 1

    # Step 2: GPU detection and profile selection
    detect_and_select_gpu || exit 1

    # Step 3: Create directories
    create_directories

    # Step 4: Download models
    # Allow partial failures: exports will re-attempt individual downloads as needed
    local download_rc=0
    local step_start=$SECONDS
    download_models_step || download_rc=$?
    if [[ $download_rc -ne 0 ]]; then
        log_warn "Some model downloads failed. Setup will continue."
        log_info "Missing models can be re-downloaded: ./scripts/openprocessor.sh download all"
    fi
    log_info "Download step: $(format_elapsed $((SECONDS - step_start)))"

    # Step 5: Pull or build Docker images
    step_start=$SECONDS
    if ! build_docker_images; then
        log_error "Docker image setup failed. Cannot continue without container images."
        log_info "Troubleshooting:"
        log_info "  - Check internet connectivity"
        log_info "  - Check disk space: df -h"
        log_info "  - Try manually: docker compose pull"
        log_info "  - Or build locally: docker compose build"
        exit 1
    fi
    log_info "Docker image step: $(format_elapsed $((SECONDS - step_start)))"

    # Step 6: Generate configuration (before export so configs match engines)
    generate_configs_step

    # Step 7: Export to TensorRT
    step_start=$SECONDS
    export_models_step
    log_info "TensorRT export step: $(format_elapsed $((SECONDS - step_start)))"

    # Step 8: Start services
    step_start=$SECONDS
    local start_rc=0
    start_services_step || start_rc=$?
    if [[ $start_rc -ne 0 ]]; then
        log_warn "Some services may not be fully ready yet."
        log_info "Check status: ./scripts/openprocessor.sh status"
    fi
    log_info "Service startup step: $(format_elapsed $((SECONDS - step_start)))"

    # Step 9: Smoke tests
    if [[ "$SKIP_START" != "true" ]]; then
        run_smoke_test
    fi

    # Step 10: Success summary
    local total_elapsed=$((SECONDS - setup_start))
    print_success_summary
    log_info "Total setup time: $(format_elapsed $total_elapsed)"
}

main "$@"
