#!/bin/bash
# =============================================================================
# OpenProcessor Management CLI
# =============================================================================
# Unified command-line interface for managing OpenProcessor services
#
# Usage: ./scripts/openprocessor.sh <command> [options]
# =============================================================================

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

# Source optional libraries only when needed
load_download_lib() {
    source "${SCRIPT_DIR}/lib/download.sh"
}

load_export_lib() {
    source "${SCRIPT_DIR}/lib/export.sh"
}

load_config_lib() {
    source "${SCRIPT_DIR}/lib/config.sh"
}

# =============================================================================
# Service Management
# =============================================================================

cmd_start() {
    log_step "Starting OpenProcessor services..."
    cd "$PROJECT_DIR"
    docker compose up -d
    log_success "Services started"
    echo ""
    cmd_status
}

cmd_stop() {
    log_step "Stopping OpenProcessor services..."
    cd "$PROJECT_DIR"
    docker compose down
    log_success "Services stopped"
}

cmd_restart() {
    local service="${1:-}"

    if [[ -n "$service" ]]; then
        log_step "Restarting $service..."
        cd "$PROJECT_DIR"
        docker compose restart "$service"
        log_success "$service restarted"
    else
        log_step "Restarting all services..."
        cd "$PROJECT_DIR"
        docker compose restart
        log_success "All services restarted"
    fi
}

cmd_logs() {
    local service="${1:-}"
    local follow="${2:-}"

    cd "$PROJECT_DIR"

    if [[ -n "$service" ]]; then
        if [[ "$follow" == "-f" ]] || [[ "$follow" == "--follow" ]]; then
            docker compose logs -f "$service"
        else
            docker compose logs --tail=100 "$service"
        fi
    else
        if [[ "$1" == "-f" ]] || [[ "$1" == "--follow" ]]; then
            docker compose logs -f
        else
            docker compose logs --tail=50
        fi
    fi
}

cmd_status() {
    print_header "OpenProcessor Status"

    cd "$PROJECT_DIR"

    # Container status
    echo -e "${BOLD}Containers:${NC}"
    docker compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || \
        docker compose ps

    echo ""

    # Health checks
    echo -e "${BOLD}Health Checks:${NC}"

    # API health
    local api_status
    api_status=$(curl -s -o /dev/null -w "%{http_code}" localhost:4603/health 2>/dev/null || echo "000")
    if [[ "$api_status" == "200" ]]; then
        echo -e "  API (4603):       ${GREEN}healthy${NC}"
    else
        echo -e "  API (4603):       ${RED}unavailable${NC} (HTTP $api_status)"
    fi

    # Triton health
    local triton_status
    triton_status=$(curl -s -o /dev/null -w "%{http_code}" localhost:4600/v2/health/ready 2>/dev/null || echo "000")
    if [[ "$triton_status" == "200" ]]; then
        echo -e "  Triton (4600):    ${GREEN}healthy${NC}"
    else
        echo -e "  Triton (4600):    ${RED}unavailable${NC} (HTTP $triton_status)"
    fi

    # OpenSearch health
    local opensearch_status
    opensearch_status=$(curl -s -o /dev/null -w "%{http_code}" localhost:4607 2>/dev/null || echo "000")
    if [[ "$opensearch_status" == "200" ]]; then
        echo -e "  OpenSearch (4607): ${GREEN}healthy${NC}"
    else
        echo -e "  OpenSearch (4607): ${RED}unavailable${NC} (HTTP $opensearch_status)"
    fi

    # Grafana health
    local grafana_status
    grafana_status=$(curl -s -o /dev/null -w "%{http_code}" localhost:4605/api/health 2>/dev/null || echo "000")
    if [[ "$grafana_status" == "200" ]]; then
        echo -e "  Grafana (4605):   ${GREEN}healthy${NC}"
    else
        echo -e "  Grafana (4605):   ${YELLOW}unavailable${NC}"
    fi

    echo ""

    # GPU status
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${BOLD}GPU:${NC}"
        check_gpu_memory_usage 0
    fi
}

cmd_health() {
    log_info "Checking API health..."

    local response
    response=$(curl -s localhost:4603/health 2>/dev/null)

    if [[ -n "$response" ]]; then
        echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
    else
        log_error "API not responding"
        return 1
    fi
}

# =============================================================================
# Model Management
# =============================================================================

cmd_models() {
    log_info "Fetching loaded models from Triton..."

    local response
    response=$(curl -s -X POST localhost:4600/v2/repository/index 2>/dev/null)

    if [[ -n "$response" ]]; then
        echo ""
        echo -e "${BOLD}Loaded Models:${NC}"
        echo "$response" | python3 -c "
import sys, json
models = json.load(sys.stdin)
if not models:
    print('  No models loaded')
else:
    for m in sorted(models, key=lambda x: x.get('name', '')):
        name = m.get('name', 'unknown')
        state = m.get('state', 'UNKNOWN')
        if state == 'READY':
            print(f'  \033[32m[READY]\033[0m {name}')
        else:
            print(f'  \033[33m[{state}]\033[0m {name}')
" 2>/dev/null || echo "$response"
    else
        log_error "Triton not responding"
        return 1
    fi
}

cmd_export() {
    local model="${1:-all}"

    load_export_lib

    case "$model" in
        all)
            export_all_models
            ;;
        essential)
            export_essential_models
            ;;
        status)
            check_exported_models
            ;;
        *)
            export_model "$model"
            ;;
    esac
}

cmd_download() {
    local target="${1:-all}"

    load_download_lib

    case "$target" in
        all)
            download_all_models
            ;;
        essential)
            download_essential_models
            ;;
        ocr|paddleocr)
            download_paddleocr_models
            ;;
        status)
            check_downloaded_models
            ;;
        *)
            download_model "$target"
            ;;
    esac
}

# =============================================================================
# Profile Management
# =============================================================================

cmd_profile() {
    local profile="${1:-}"

    if [[ -z "$profile" ]]; then
        # Show current profile
        if [[ -f "$PROJECT_DIR/.env" ]]; then
            local current
            current=$(grep "^GPU_PROFILE=" "$PROJECT_DIR/.env" | cut -d'=' -f2)
            echo "Current profile: $current"
        else
            echo "No profile configured. Run: ./scripts/setup.sh"
        fi
        echo ""
        echo "Available profiles:"
        echo "  minimal   - 6-8GB VRAM (RTX 3060, RTX 4060)"
        echo "  standard  - 12-24GB VRAM (RTX 3080, RTX 4090)"
        echo "  full      - 48GB+ VRAM (A6000, A100)"
        return 0
    fi

    # Validate profile
    if [[ ! "$profile" =~ ^(minimal|standard|full)$ ]]; then
        log_error "Invalid profile: $profile"
        echo "Valid profiles: minimal, standard, full"
        return 1
    fi

    load_config_lib

    local gpu_id="${2:-0}"

    log_step "Switching to profile: $profile"

    # Generate new configs
    generate_all_configs "$profile" "$gpu_id"
    generate_env_file "$profile" "$gpu_id"
    generate_compose_override "$profile" "$gpu_id"

    log_success "Profile switched to: $profile"
    log_info "Restart services to apply: ./scripts/openprocessor.sh restart"
}

# =============================================================================
# Testing
# =============================================================================

cmd_test() {
    local test_type="${1:-quick}"

    cd "$PROJECT_DIR"

    case "$test_type" in
        quick|smoke)
            log_step "Running quick smoke tests..."
            # Simple curl tests
            echo ""
            echo "Health check:"
            curl -s localhost:4603/health | python3 -m json.tool 2>/dev/null || curl -s localhost:4603/health
            echo ""
            ;;
        full)
            log_step "Running full test suite..."
            if [[ -f "$PROJECT_DIR/tests/test_full_system.py" ]]; then
                source "$PROJECT_DIR/.venv/bin/activate" 2>/dev/null || true
                python "$PROJECT_DIR/tests/test_full_system.py"
            else
                log_error "Test suite not found"
                return 1
            fi
            ;;
        visual)
            log_step "Running visual validation..."
            if [[ -f "$PROJECT_DIR/tests/validate_visual_results.py" ]]; then
                source "$PROJECT_DIR/.venv/bin/activate" 2>/dev/null || true
                python "$PROJECT_DIR/tests/validate_visual_results.py"
            else
                log_error "Visual validation script not found"
                return 1
            fi
            ;;
        *)
            log_error "Unknown test type: $test_type"
            echo "Available: quick, full, visual"
            return 1
            ;;
    esac
}

cmd_bench() {
    local mode="${1:-quick}"

    cd "$PROJECT_DIR"

    if [[ ! -f "$PROJECT_DIR/benchmarks/triton_bench" ]]; then
        log_info "Building benchmark tool..."
        cd "$PROJECT_DIR/benchmarks"
        ./build.sh
        cd "$PROJECT_DIR"
    fi

    log_step "Running benchmark ($mode mode)..."
    "$PROJECT_DIR/benchmarks/triton_bench" --mode "$mode"
}

# =============================================================================
# Cleanup
# =============================================================================

cmd_clean() {
    local target="${1:-cache}"

    case "$target" in
        cache)
            log_step "Cleaning caches..."
            rm -rf "$PROJECT_DIR/.mypy_cache"
            rm -rf "$PROJECT_DIR/.ruff_cache"
            rm -rf "$PROJECT_DIR/.pytest_cache"
            find "$PROJECT_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
            log_success "Caches cleaned"
            ;;
        models)
            log_warn "This will remove downloaded PyTorch models."
            read -r -p "Continue? [y/N]: " reply
            if [[ "$reply" =~ ^[Yy]$ ]]; then
                rm -rf "$PROJECT_DIR/pytorch_models"/*.pt
                rm -rf "$PROJECT_DIR/pytorch_models"/*.onnx
                log_success "Models cleaned"
            fi
            ;;
        exports)
            log_warn "This will remove TensorRT exports."
            read -r -p "Continue? [y/N]: " reply
            if [[ "$reply" =~ ^[Yy]$ ]]; then
                find "$PROJECT_DIR/models" -name "model.plan" -delete
                log_success "Exports cleaned"
            fi
            ;;
        all)
            cmd_clean cache
            cmd_clean models
            cmd_clean exports
            ;;
        *)
            log_error "Unknown target: $target"
            echo "Available: cache, models, exports, all"
            return 1
            ;;
    esac
}

# =============================================================================
# Update
# =============================================================================

cmd_update() {
    log_step "Updating OpenProcessor..."

    cd "$PROJECT_DIR"

    # Pull latest images
    log_info "Pulling latest Docker images..."
    docker compose pull

    # Rebuild if needed
    log_info "Rebuilding containers..."
    docker compose build

    log_success "Update complete"
    log_info "Restart to apply: ./scripts/openprocessor.sh restart"
}

# =============================================================================
# Help
# =============================================================================

show_help() {
    cat << EOF
OpenProcessor Management CLI

Usage: ./scripts/openprocessor.sh <command> [options]

Service Commands:
  start               Start all services
  stop                Stop all services
  restart [service]   Restart service(s)
  logs [service] [-f] View logs (optionally follow)
  status              Show service status and health
  health              Check API health endpoint

Model Commands:
  models              List loaded Triton models
  export <model>      Export model to TensorRT
                      Options: yolo, scrfd, arcface, clip-image, clip-text, ocr, all, essential, status
  download <model>    Download model
                      Options: all, essential, ocr, status, <model-name>

Configuration:
  profile [name]      Show/switch GPU profile
                      Options: minimal, standard, full

Testing:
  test [type]         Run tests
                      Options: quick (default), full, visual
  bench [mode]        Run benchmarks
                      Options: quick (default), full

Maintenance:
  clean [target]      Clean files
                      Options: cache (default), models, exports, all
  update              Pull latest images and rebuild

Setup:
  setup               Run initial setup wizard

Examples:
  ./scripts/openprocessor.sh status
  ./scripts/openprocessor.sh logs yolo-api -f
  ./scripts/openprocessor.sh export all
  ./scripts/openprocessor.sh profile standard
  ./scripts/openprocessor.sh test full
  ./scripts/openprocessor.sh clean cache

EOF
}

# =============================================================================
# Main
# =============================================================================

main() {
    local command="${1:-help}"
    shift 2>/dev/null || true

    case "$command" in
        start)
            cmd_start "$@"
            ;;
        stop)
            cmd_stop "$@"
            ;;
        restart)
            cmd_restart "$@"
            ;;
        logs)
            cmd_logs "$@"
            ;;
        status)
            cmd_status "$@"
            ;;
        health)
            cmd_health "$@"
            ;;
        models)
            cmd_models "$@"
            ;;
        export)
            cmd_export "$@"
            ;;
        download)
            cmd_download "$@"
            ;;
        profile)
            cmd_profile "$@"
            ;;
        test)
            cmd_test "$@"
            ;;
        bench)
            cmd_bench "$@"
            ;;
        clean)
            cmd_clean "$@"
            ;;
        update)
            cmd_update "$@"
            ;;
        setup)
            exec "${SCRIPT_DIR}/setup.sh" "$@"
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Unknown command: $command"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

main "$@"
