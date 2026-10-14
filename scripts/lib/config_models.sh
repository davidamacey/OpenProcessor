#!/bin/bash
# =============================================================================
# config_models.sh - per-model config.pbtxt generators
# =============================================================================
# Sourced by config.sh, which supplies update_config_profile_settings(),
# update_config_instance_settings(), get_batch_sizes_string(), log_*() and
# the profile variables loaded by load_profile().

[[ -n "${_CONFIG_MODELS_SH_LOADED:-}" ]] && return 0
_CONFIG_MODELS_SH_LOADED=1

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

    # If config exists, update profile settings (batch size, instance count, GPU)
    # SCRFD uses implicit batching (max_batch_size > 0) with the batch-patched engine
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
