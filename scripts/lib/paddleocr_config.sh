#!/bin/bash
# =============================================================================
# paddleocr_config.sh - dictionary setup and Triton config.pbtxt creation
# =============================================================================
# Sourced by scripts/export_paddleocr.sh, which supplies the paths, ports,
# shape and TRT settings. Not runnable on its own.

# Variables are assigned by export_paddleocr.sh before this file is sourced.
# shellcheck disable=SC2154

[[ -n "${_PADDLEOCR_CONFIG_SH_LOADED:-}" ]] && return 0
_PADDLEOCR_CONFIG_SH_LOADED=1

# =============================================================================
# Config and Dictionary Functions
# =============================================================================

setup_dictionary() {
    log_info "Setting up dictionary file..."

    mkdir -p "$(dirname "$DICT_FILE")"

    # Check if dictionary already exists with correct character count
    if [ -f "$DICT_FILE" ]; then
        local char_count
        char_count=$(wc -l < "$DICT_FILE")
        log_info "Dictionary already exists: $char_count characters"
        return 0
    fi

    # Copy multilingual dictionary from downloaded models
    # The download_paddleocr.py script downloads ppocr_keys_v1.txt (6623 chars)
    # but the multilingual PP-OCRv5 model uses ppocrv5_dict.txt (18383 chars).
    # Check for ppocrv5_dict.txt first, then fall back to ppocr_keys_v1.txt.
    local dict_source=""
    if [ -f "$MODELS_DIR/paddleocr_rec_trt/ppocrv5_dict.txt" ]; then
        dict_source="$MODELS_DIR/paddleocr_rec_trt/ppocrv5_dict.txt"
    elif [ -f "$PYTORCH_MODELS_DIR/ppocrv5_dict.txt" ]; then
        dict_source="$PYTORCH_MODELS_DIR/ppocrv5_dict.txt"
    elif [ -f "$PYTORCH_MODELS_DIR/ppocr_keys_v1.txt" ]; then
        dict_source="$PYTORCH_MODELS_DIR/ppocr_keys_v1.txt"
    fi

    if [ -n "$dict_source" ]; then
        mkdir -p "$(dirname "$DICT_FILE")"
        cp "$dict_source" "$DICT_FILE"
        local char_count
        char_count=$(wc -l < "$DICT_FILE")
        log_success "Dictionary created: $char_count characters (from $dict_source)"
    else
        log_warn "Dictionary not found in downloaded models"
        log_info "Please place ppocrv5_dict.txt in $MODELS_DIR/paddleocr_rec_trt/"
    fi
}

create_configs() {
    log_info "Creating Triton configs..."

    # Detection config
    cat > "$MODELS_DIR/paddleocr_det_trt/config.pbtxt" << 'EOF'
# PP-OCRv5 Text Detection Model (TensorRT)
#
# DB++ architecture for text region detection
#
# Input:
#   - x: [B, 3, H, W] FP32, preprocessed (x / 127.5 - 1), BGR
#        H,W must be multiples of 32, max 960
#
# Output:
#   - fetch_name_0: [B, 1, H, W] FP32, probability map [0, 1]

name: "paddleocr_det_trt"
platform: "tensorrt_plan"
max_batch_size: 4

input [
  {
    name: "x"
    data_type: TYPE_FP32
    dims: [ 3, -1, -1 ]  # Dynamic H, W (multiples of 32)
  }
]

output [
  {
    name: "fetch_name_0"
    data_type: TYPE_FP32
    dims: [ 1, -1, -1 ]  # Same H, W as input
  }
]

instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [0]
  }
]

dynamic_batching {
  preferred_batch_size: [ 1, 2, 4 ]
  max_queue_delay_microseconds: 5000
}
EOF

    # Recognition config
    cat > "$MODELS_DIR/paddleocr_rec_trt/config.pbtxt" << 'EOF'
# PP-OCRv5 Text Recognition Model (TensorRT)
#
# SVTR-LCNet architecture for text sequence recognition
# Supports dynamic width for variable-length text recognition
#
# Input:
#   - x: [B, 3, 48, W] FP32, preprocessed (x / 127.5 - 1), BGR
#        Text crops with height=48, width=48-2048
#
# Output:
#   - fetch_name_0: [B, T, 18385] FP32, character probabilities
#        T timesteps (dynamic), 18385 character classes (multilingual dict)
#
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
    count: 2
    kind: KIND_GPU
    gpus: [0]
  }
]

dynamic_batching {
  preferred_batch_size: [ 16, 32, 64 ]
  max_queue_delay_microseconds: 10000
}
EOF

    log_success "Triton configs created"
}
