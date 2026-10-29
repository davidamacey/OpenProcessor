#!/bin/bash
# =============================================================================
# config.sh - Configuration generation for OpenProcessor
# =============================================================================
# Generates Triton config.pbtxt files based on GPU profile
# =============================================================================

# Avoid redefining if already sourced
[[ -n "${_CONFIG_SH_LOADED:-}" ]] && return 0
_CONFIG_SH_LOADED=1

# Source dependencies
_CONFIG_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=colors.sh
[[ -z "${NC:-}" ]] && source "${_CONFIG_SH_DIR}/colors.sh"
# shellcheck source=gpu.sh
source "${_CONFIG_SH_DIR}/gpu.sh"
# shellcheck source=opensearch_heap.sh
source "${_CONFIG_SH_DIR}/opensearch_heap.sh"
# shellcheck source=config_models.sh
source "${_CONFIG_SH_DIR}/config_models.sh"

# =============================================================================
# Configuration
# =============================================================================

PROJECT_DIR="${PROJECT_DIR:-$(dirname "$(dirname "$(dirname "${BASH_SOURCE[0]}")")")}"
MODELS_DIR="$PROJECT_DIR/models"

# =============================================================================
# Config Generation
# =============================================================================

# Generate config.pbtxt for a model based on profile
# Usage: generate_model_config MODEL_NAME PROFILE [GPU_ID]
generate_model_config() {
    local model_name="$1"
    local profile="${2:-standard}"
    local gpu_id="${3:-0}"

    # Load profile settings
    load_profile "$profile" || return 1

    local config_file="$MODELS_DIR/$model_name/config.pbtxt"
    local backup_file="$MODELS_DIR/$model_name/config.pbtxt.backup"

    # Backup existing config
    if [[ -f "$config_file" ]]; then
        cp "$config_file" "$backup_file"
    fi

    # Generate config based on model type
    case "$model_name" in
        yolov11_small_trt_end2end)
            generate_yolo_detection_config "$config_file" "$profile" "$gpu_id"
            ;;
        scrfd_10g_bnkps)
            generate_scrfd_config "$config_file" "$profile" "$gpu_id"
            ;;
        arcface_w600k_r50)
            generate_arcface_config "$config_file" "$profile" "$gpu_id"
            ;;
        mobileclip2_s2_image_encoder)
            generate_mobileclip_image_config "$config_file" "$profile" "$gpu_id"
            ;;
        mobileclip2_s2_text_encoder)
            generate_mobileclip_text_config "$config_file" "$profile" "$gpu_id"
            ;;
        paddleocr_det_trt)
            generate_paddleocr_det_config "$config_file" "$profile" "$gpu_id"
            ;;
        paddleocr_rec_trt)
            generate_paddleocr_rec_config "$config_file" "$profile" "$gpu_id"
            ;;
        ocr_pipeline)
            # Python BLS model: its instance count is the OCR concurrency and
            # has its own per-profile value, bounded by the profile's
            # python_backend_budget_mib (see tests/test_triton_instance_budget.py).
            update_config_instance_settings "$config_file" "$profile" "$gpu_id" \
                "$PROFILE_OCR_PIPELINE_INSTANCES"
            ;;
        gpu_image_prep)
            # Python model, one CUDA context and CuPy pool per instance (GPU_MIB_PER_INSTANCE
            # each): the profile decides how many copies the card holds (#268).
            update_config_instance_settings "$config_file" "$profile" "$gpu_id" \
                "$PROFILE_GPU_IMAGE_PREP_INSTANCES"
            ;;
        *)
            log_warn "No config template for: $model_name"
            return 1
            ;;
    esac

    log_debug "Generated config for $model_name (profile=$profile)"
    return 0
}

# =============================================================================
# Profile Settings Updater (preserves original input/output definitions)
# =============================================================================

# Update profile-specific settings in existing config
# This preserves the input/output names that match the TensorRT model
update_config_profile_settings() {
    local config_file="$1"
    local profile="$2"
    local gpu_id="$3"

    if [[ ! -f "$config_file" ]]; then
        log_error "Config file not found: $config_file"
        return 1
    fi

    local instance_count="$PROFILE_INSTANCE_COUNT"
    local max_batch="$PROFILE_MAX_BATCH"
    local batch_sizes
    batch_sizes=$(get_batch_sizes_string "$profile")

    # Create temp file for updates
    local temp_file
    temp_file=$(mktemp)

    # Update max_batch_size
    sed "s/^max_batch_size:.*/max_batch_size: $max_batch/" "$config_file" > "$temp_file"

    # Update instance_group count (handles multi-line)
    awk -v count="$instance_count" -v gpu="$gpu_id" '
    /instance_group/ { in_group=1 }
    in_group && /count:/ { gsub(/count: [0-9]+/, "count: " count) }
    in_group && /gpus:/ { gsub(/gpus: \[ [0-9]+ \]/, "gpus: [ " gpu " ]") }
    /^]$/ && in_group { in_group=0 }
    { print }
    ' "$temp_file" > "${temp_file}.2"

    # Update preferred_batch_size
    sed -i "s/preferred_batch_size:.*/preferred_batch_size: [ $batch_sizes ]/" "${temp_file}.2"

    mv "${temp_file}.2" "$config_file"
    rm -f "$temp_file"

    log_debug "Updated profile settings in $config_file"
    return 0
}

# Update only instance_count and gpu_id (for models with TRT-constrained batch sizes)
# PaddleOCR models have max_batch_size set by their TRT engine, not by the profile.
update_config_instance_settings() {
    local config_file="$1"
    local profile="$2"
    local gpu_id="$3"

    if [[ ! -f "$config_file" ]]; then
        log_error "Config file not found: $config_file"
        return 1
    fi

    local instance_count="${4:-$PROFILE_INSTANCE_COUNT}"

    # Only update instance_group count and gpu_id (not max_batch_size or preferred_batch_size)
    local temp_file
    temp_file=$(mktemp)

    awk -v count="$instance_count" -v gpu="$gpu_id" '
    /instance_group/ { in_group=1 }
    in_group && /count:/ { gsub(/count: [0-9]+/, "count: " count) }
    in_group && /gpus:/ { gsub(/gpus: \[ [0-9]+ \]/, "gpus: [ " gpu " ]") }
    /^]$/ && in_group { in_group=0 }
    { print }
    ' "$config_file" > "$temp_file"

    mv "$temp_file" "$config_file"

    log_debug "Updated instance settings in $config_file"
    return 0
}

# =============================================================================
# Helper Functions
# =============================================================================

# Get batch sizes as comma-separated string
get_batch_sizes_string() {
    local profile="$1"

    case "$profile" in
        minimal)
            echo "4, 8, 16"
            ;;
        standard)
            echo "8, 16, 32"
            ;;
        full)
            echo "8, 16, 32, 64"
            ;;
        *)
            echo "8, 16, 32"
            ;;
    esac
}

# =============================================================================
# Batch Config Generation
# =============================================================================

# Generate configs for all models
generate_all_configs() {
    local profile="${1:-standard}"
    local gpu_id="${2:-0}"

    log_step "Generating Triton configs (profile=$profile, GPU=$gpu_id)..."

    local models=(
        "yolov11_small_trt_end2end"
        "scrfd_10g_bnkps"
        "arcface_w600k_r50"
        "mobileclip2_s2_image_encoder"
        "mobileclip2_s2_text_encoder"
        "paddleocr_det_trt"
        "paddleocr_rec_trt"
        "ocr_pipeline"
        "gpu_image_prep"
    )

    local failed=0
    for model in "${models[@]}"; do
        if ! generate_model_config "$model" "$profile" "$gpu_id"; then
            failed=$((failed + 1))
        fi
    done

    if [[ $failed -eq 0 ]]; then
        log_success "All configs generated"
        return 0
    else
        log_warn "$failed config(s) failed to generate"
        return 1
    fi
}

# =============================================================================
# Environment File Generation
# =============================================================================

# Generate .env file from template
#
# G-04: NEVER overwrites an existing .env unless the caller opts in with
# force=true (setup.sh's --force) -- a shared host may already have a
# hand-edited .env (remapped ports, a second isolated stack's
# COMPOSE_PROJECT_NAME, a real HF_TOKEN); silently regenerating it here
# would blow that away on a routine re-run.
generate_env_file() {
    local profile="${1:-standard}"
    local gpu_id="${2:-0}"
    local force="${3:-false}"

    local env_file="$PROJECT_DIR/.env"

    if [[ -f "$env_file" && "$force" != "true" ]]; then
        log_info ".env already exists -- leaving it untouched (pass --force to regenerate)"
        return 0
    fi

    # Load profile
    load_profile "$profile" || return 1

    # A heap the user already set survives a forced regeneration.
    local heap=""
    [[ -f "$env_file" ]] && heap="$(sed -n 's/^OPENSEARCH_HEAP=//p' "$env_file" | tail -n1)"
    if [[ -n "$heap" ]]; then
        log_info "keeping your OPENSEARCH_HEAP=${heap}"
    else
        heap="$(opensearch_heap_for_host)" || { log_error "could not read host memory to size the OpenSearch heap"; return 1; }
    fi

    # An explicit GPU decode choice survives a forced regeneration too.
    local gpu_decode="$PROFILE_INGEST_GPU_DECODE" kept_gpu_decode=""
    [[ -f "$env_file" ]] && kept_gpu_decode="$(sed -n 's/^OP_INGEST_GPU_DECODE=//p' "$env_file" | tail -n1)"
    [[ -n "$kept_gpu_decode" ]] && gpu_decode="$kept_gpu_decode"

    log_step "Generating .env file..."

    cat > "$env_file" << EOF
# =============================================================================
# OpenProcessor Configuration
# Generated by setup.sh on $(date)
# =============================================================================

# GPU Profile
# Options: minimal (6-8GB), standard (12-24GB), full (40GB+)
GPU_PROFILE=$profile
GPU_ID=$gpu_id

# GPU decode: 1 only where the profile's card has ~8.5 GB of room beyond its models
OP_INGEST_GPU_DECODE=$gpu_decode

# OpenSearch JVM heap: host RAM/2, clamped to 2g..30g (scripts/lib/opensearch_heap.sh)
OPENSEARCH_HEAP=$heap

# Ports (change if these conflict with something else on your host, or
# to run a second isolated stack -- see env.template's "Isolation" section)
API_PORT=4603
TRITON_HTTP_PORT=4600
TRITON_GRPC_PORT=4601
TRITON_METRICS_PORT=4602
GRAFANA_PORT=4605
OPENSEARCH_PORT=4607
EOF

    log_success ".env file created: $env_file"
    return 0
}

# =============================================================================
# Switching the profile of an existing .env
# =============================================================================

# apply_profile_to_env PROFILE -- set the two profile keys something reads
# (GPU_PROFILE, OP_INGEST_GPU_DECODE) in the existing .env, leaving every other
# line alone. generate_env_file never rewrites an existing .env.
apply_profile_to_env() {
    local profile="${1:-standard}"
    local env_file="$PROJECT_DIR/.env"

    load_profile "$profile" || return 1
    [[ -f "$env_file" ]] || { log_error "no .env at $env_file to update"; return 1; }

    local key value
    for key in GPU_PROFILE OP_INGEST_GPU_DECODE; do
        case "$key" in
            GPU_PROFILE) value="$profile" ;;
            *) value="$PROFILE_INGEST_GPU_DECODE" ;;
        esac
        if grep -q "^${key}=" "$env_file"; then
            sed -i "s|^${key}=.*|${key}=${value}|" "$env_file"
        else
            printf '%s=%s\n' "$key" "$value" >> "$env_file"
        fi
    done
    log_success ".env updated: GPU_PROFILE=$profile OP_INGEST_GPU_DECODE=$PROFILE_INGEST_GPU_DECODE"
}
