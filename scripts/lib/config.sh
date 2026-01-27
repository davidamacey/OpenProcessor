#!/bin/bash
# =============================================================================
# config.sh - Configuration generation for triton-api
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

    local instance_count="$PROFILE_INSTANCE_COUNT"

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
# Model-Specific Config Generators (LEGACY - for fresh exports only)
# =============================================================================

generate_yolo_detection_config() {
    local config_file="$1"
    local profile="$2"
    local gpu_id="$3"

    # If config exists, just update profile settings
    if [[ -f "$config_file" ]]; then
        update_config_profile_settings "$config_file" "$profile" "$gpu_id"
        return $?
    fi

    local instance_count="$PROFILE_INSTANCE_COUNT"
    local max_batch="$PROFILE_MAX_BATCH"
    local batch_sizes
    batch_sizes=$(get_batch_sizes_string "$profile")

    cat > "$config_file" << EOF
name: "yolov11_small_trt_end2end"
platform: "tensorrt_plan"
max_batch_size: $max_batch

input [
  {
    name: "images"
    data_type: TYPE_FP32
    dims: [ 3, 640, 640 ]
  }
]

output [
  {
    name: "num_dets"
    data_type: TYPE_INT32
    dims: [ 1 ]
  },
  {
    name: "det_boxes"
    data_type: TYPE_FP32
    dims: [ 300, 4 ]
  },
  {
    name: "det_scores"
    data_type: TYPE_FP32
    dims: [ 300 ]
  },
  {
    name: "det_classes"
    data_type: TYPE_INT32
    dims: [ 300 ]
  }
]

dynamic_batching {
  preferred_batch_size: [ $batch_sizes ]
  max_queue_delay_microseconds: 25000
  preserve_ordering: false
  default_queue_policy {
    timeout_action: REJECT
    default_timeout_microseconds: 30000000
    allow_timeout_override: false
    max_queue_size: 4096
  }
}

instance_group [
  {
    count: $instance_count
    kind: KIND_GPU
    gpus: [ $gpu_id ]
  }
]
EOF
}

generate_scrfd_config() {
    local config_file="$1"
    local profile="$2"
    local gpu_id="$3"

    # If config exists, just update profile settings (preserves FPN output tensors)
    if [[ -f "$config_file" ]]; then
        update_config_profile_settings "$config_file" "$profile" "$gpu_id"
        return $?
    fi

    local instance_count="$PROFILE_INSTANCE_COUNT"
    local max_batch="$PROFILE_MAX_BATCH"
    local batch_sizes
    batch_sizes=$(get_batch_sizes_string "$profile")

    cat > "$config_file" << EOF
# SCRFD-10G Face Detection with 5-point Landmarks
# Input: RGB images, normalized (x-127.5)/128.0
# Output: 9 tensors across 3 FPN strides for CPU post-processing
name: "scrfd_10g_bnkps"
platform: "tensorrt_plan"
max_batch_size: $max_batch

input [
  {
    name: "input.1"
    data_type: TYPE_FP32
    dims: [ 3, 640, 640 ]
  }
]

output [
  {
    name: "score_8"
    data_type: TYPE_FP32
    dims: [ 12800, 1 ]
  },
  {
    name: "score_16"
    data_type: TYPE_FP32
    dims: [ 3200, 1 ]
  },
  {
    name: "score_32"
    data_type: TYPE_FP32
    dims: [ 800, 1 ]
  },
  {
    name: "bbox_8"
    data_type: TYPE_FP32
    dims: [ 12800, 4 ]
  },
  {
    name: "bbox_16"
    data_type: TYPE_FP32
    dims: [ 3200, 4 ]
  },
  {
    name: "bbox_32"
    data_type: TYPE_FP32
    dims: [ 800, 4 ]
  },
  {
    name: "kps_8"
    data_type: TYPE_FP32
    dims: [ 12800, 10 ]
  },
  {
    name: "kps_16"
    data_type: TYPE_FP32
    dims: [ 3200, 10 ]
  },
  {
    name: "kps_32"
    data_type: TYPE_FP32
    dims: [ 800, 10 ]
  }
]

dynamic_batching {
  preferred_batch_size: [ $batch_sizes ]
  max_queue_delay_microseconds: 5000
}

instance_group [
  {
    count: $instance_count
    kind: KIND_GPU
    gpus: [ $gpu_id ]
  }
]

version_policy {
  latest {
    num_versions: 1
  }
}
EOF
}

generate_arcface_config() {
    local config_file="$1"
    local profile="$2"
    local gpu_id="$3"

    # If config exists, just update profile settings (preserves correct input name)
    if [[ -f "$config_file" ]]; then
        update_config_profile_settings "$config_file" "$profile" "$gpu_id"
        return $?
    fi

    # ArcFace can handle more instances since it's smaller
    local instance_count=$((PROFILE_INSTANCE_COUNT * 2))
    local max_batch=128
    local batch_sizes
    batch_sizes=$(get_batch_sizes_string "$profile")

    cat > "$config_file" << EOF
name: "arcface_w600k_r50"
platform: "tensorrt_plan"
max_batch_size: $max_batch

input {
    name: "input"
    data_type: TYPE_FP32
    dims: [3, 112, 112]
}

output {
    name: "output"
    data_type: TYPE_FP32
    dims: [512]
}

dynamic_batching {
    preferred_batch_size: [ $batch_sizes ]
    max_queue_delay_microseconds: 15000
}

instance_group [{
    count: $instance_count
    kind: KIND_GPU
    gpus: [$gpu_id]
}]

version_policy {
    latest {
        num_versions: 1
    }
}
EOF
}

generate_mobileclip_image_config() {
    local config_file="$1"
    local profile="$2"
    local gpu_id="$3"

    # If config exists, just update profile settings (preserves correct input/output names)
    if [[ -f "$config_file" ]]; then
        update_config_profile_settings "$config_file" "$profile" "$gpu_id"
        return $?
    fi

    local instance_count="$PROFILE_INSTANCE_COUNT"
    local max_batch="$PROFILE_MAX_BATCH"
    local batch_sizes
    batch_sizes=$(get_batch_sizes_string "$profile")

    cat > "$config_file" << EOF
name: "mobileclip2_s2_image_encoder"
platform: "tensorrt_plan"
max_batch_size: $max_batch

input [
  {
    name: "image"
    data_type: TYPE_FP32
    dims: [3, 256, 256]
  }
]

output [
  {
    name: "embedding"
    data_type: TYPE_FP32
    dims: [512]
  }
]

dynamic_batching {
  preferred_batch_size: [ $batch_sizes ]
  max_queue_delay_microseconds: 10000
}

instance_group [
  {
    count: $instance_count
    kind: KIND_GPU
    gpus: [$gpu_id]
  }
]
EOF
}

generate_mobileclip_text_config() {
    local config_file="$1"
    local profile="$2"
    local gpu_id="$3"

    # If config exists, just update profile settings
    if [[ -f "$config_file" ]]; then
        update_config_profile_settings "$config_file" "$profile" "$gpu_id"
        return $?
    fi

    local instance_count="$PROFILE_INSTANCE_COUNT"
    local max_batch="$PROFILE_MAX_BATCH"
    local batch_sizes
    batch_sizes=$(get_batch_sizes_string "$profile")

    cat > "$config_file" << EOF
name: "mobileclip2_s2_text_encoder"
platform: "tensorrt_plan"
max_batch_size: $max_batch

input [
  {
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [77]
  }
]

output [
  {
    name: "text_embedding"
    data_type: TYPE_FP32
    dims: [512]
  }
]

dynamic_batching {
  preferred_batch_size: [ $batch_sizes ]
  max_queue_delay_microseconds: 5000
}

instance_group [
  {
    count: $instance_count
    kind: KIND_GPU
    gpus: [$gpu_id]
  }
]
EOF
}

generate_paddleocr_det_config() {
    local config_file="$1"
    local profile="$2"
    local gpu_id="$3"

    # If config exists, only update instance/GPU settings (NOT max_batch_size)
    # PaddleOCR det TRT engine is built with max_batch=4, this must not be overridden
    if [[ -f "$config_file" ]]; then
        update_config_instance_settings "$config_file" "$profile" "$gpu_id"
        return $?
    fi

    local instance_count="$PROFILE_INSTANCE_COUNT"

    cat > "$config_file" << EOF
# PP-OCRv5 Text Detection Model (TensorRT)
name: "paddleocr_det_trt"
platform: "tensorrt_plan"
max_batch_size: 4

input [
  {
    name: "x"
    data_type: TYPE_FP32
    dims: [ 3, -1, -1 ]
  }
]

output [
  {
    name: "fetch_name_0"
    data_type: TYPE_FP32
    dims: [ 1, -1, -1 ]
  }
]

instance_group [
  {
    count: $instance_count
    kind: KIND_GPU
    gpus: [$gpu_id]
  }
]

dynamic_batching {
  preferred_batch_size: [ 1, 2, 4 ]
  max_queue_delay_microseconds: 5000
}
EOF
}

generate_paddleocr_rec_config() {
    local config_file="$1"
    local profile="$2"
    local gpu_id="$3"

    # If config exists, only update instance/GPU settings (NOT max_batch_size)
    # PaddleOCR rec TRT engine is built with max_batch=64, this must not be overridden
    if [[ -f "$config_file" ]]; then
        update_config_instance_settings "$config_file" "$profile" "$gpu_id"
        return $?
    fi

    local instance_count="$PROFILE_INSTANCE_COUNT"

    cat > "$config_file" << EOF
# PP-OCRv5 Text Recognition Model (TensorRT)
# Multilingual model: Chinese + English + symbols (18385 character classes)
# TensorRT engine exported with dynamic batch (max=64) and dynamic width.
name: "paddleocr_rec_trt"
platform: "tensorrt_plan"
max_batch_size: 64

input [
  {
    name: "x"
    data_type: TYPE_FP32
    dims: [ 3, 48, -1 ]
  }
]

output [
  {
    name: "fetch_name_0"
    data_type: TYPE_FP32
    dims: [ -1, 18385 ]
  }
]

instance_group [
  {
    count: $instance_count
    kind: KIND_GPU
    gpus: [$gpu_id]
  }
]

dynamic_batching {
  preferred_batch_size: [ 16, 32, 64 ]
  max_queue_delay_microseconds: 10000
}
EOF
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
generate_env_file() {
    local profile="${1:-standard}"
    local gpu_id="${2:-0}"

    local env_file="$PROJECT_DIR/.env"

    # Load profile
    load_profile "$profile" || return 1

    log_step "Generating .env file..."

    cat > "$env_file" << EOF
# =============================================================================
# triton-api Configuration
# Generated by setup.sh on $(date)
# =============================================================================

# GPU Profile
# Options: minimal (6-8GB), standard (12-24GB), full (48GB+)
GPU_PROFILE=$profile
GPU_ID=$gpu_id

# Performance (auto-configured based on GPU_PROFILE)
TRITON_WORKERS=$PROFILE_WORKERS
MAX_BATCH_SIZE=$PROFILE_MAX_BATCH
SHM_SIZE=$PROFILE_SHM_SIZE

# OpenSearch vector database
OPENSEARCH_HEAP=$PROFILE_HEAP

# Ports (change if conflicts with other services)
API_PORT=4603
TRITON_HTTP=4600
TRITON_GRPC=4601
TRITON_METRICS=4602
GRAFANA_PORT=4605
OPENSEARCH_PORT=4607
EOF

    log_success ".env file created: $env_file"
    return 0
}

# =============================================================================
# Docker Compose Override Generation
# =============================================================================

# Generate docker-compose.override.yml for profile-specific settings
generate_compose_override() {
    local profile="${1:-standard}"
    local gpu_id="${2:-0}"

    local override_file="$PROJECT_DIR/docker-compose.override.yml"

    # Load profile
    load_profile "$profile" || return 1

    log_step "Generating docker-compose.override.yml..."

    cat > "$override_file" << EOF
# =============================================================================
# docker-compose.override.yml
# Generated by setup.sh - Profile: $profile
# =============================================================================
# This file overrides settings in docker-compose.yml based on your GPU profile.
# Regenerate with: ./scripts/openprocessor.sh profile $profile
# =============================================================================

services:
  triton-server:
    shm_size: ${PROFILE_SHM_SIZE}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['$gpu_id']
              capabilities:
                - gpu

  yolo-api:
    shm_size: ${PROFILE_SHM_SIZE}
    command:
      - uvicorn
      - src.main:app
      - --host=0.0.0.0
      - --port=8000
      - --workers=${PROFILE_WORKERS}
      - --limit-max-requests=10000
      - --loop=uvloop
      - --http=httptools
      - --backlog=4096
      - --limit-concurrency=512
      - --timeout-keep-alive=75
      - --timeout-graceful-shutdown=30
      - --access-log
      - --log-level=info
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['$gpu_id']
              capabilities:
                - gpu

  opensearch:
    environment:
      - discovery.type=single-node
      - bootstrap.memory_lock=true
      - "OPENSEARCH_JAVA_OPTS=-Xms${PROFILE_HEAP} -Xmx${PROFILE_HEAP}"
      - DISABLE_SECURITY_PLUGIN=true
EOF

    log_success "docker-compose.override.yml created"
    return 0
}
