#!/bin/bash
# =============================================================================
# cli_dev.sh - checkout-only commands: profile, test, bench, clean
# =============================================================================
# Sourced lazily by the `openprocessor` CLI, which supplies dc(), _env_value(),
# log_*(), require_checkout(), PROJECT_DIR, SCRIPT_DIR and IS_CHECKOUT.

# The CLI runs under `set -e`, so a failed cd already aborts the command.
# shellcheck disable=SC2164

[[ -n "${_CLI_DEV_SH_LOADED:-}" ]] && return 0
_CLI_DEV_SH_LOADED=1

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
    apply_profile_to_env "$profile"

    log_success "Profile switched to: $profile"
    log_info "Restart services to apply: ./openprocessor restart triton-server"
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
            curl -s "localhost:${API_PORT}/health" | python3 -m json.tool 2>/dev/null || curl -s "localhost:${API_PORT}/health"
            echo ""
            ;;
        full)
            log_step "Running full test suite..."
            if [[ ! -x "$PROJECT_DIR/.venv/bin/python" ]]; then
                log_error "$PROJECT_DIR/.venv/bin/python not found — create the venv first"
                return 1
            fi
            if [[ -f "$PROJECT_DIR/tests/test_full_system.py" ]]; then
                "$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/tests/test_full_system.py"
            else
                log_error "Test suite not found"
                return 1
            fi
            ;;
        visual)
            log_step "Running visual validation..."
            if [[ ! -x "$PROJECT_DIR/.venv/bin/python" ]]; then
                log_error "$PROJECT_DIR/.venv/bin/python not found — create the venv first"
                return 1
            fi
            if [[ -f "$PROJECT_DIR/tests/validate_visual_results.py" ]]; then
                "$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/tests/validate_visual_results.py"
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
