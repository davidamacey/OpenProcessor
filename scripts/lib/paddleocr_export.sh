#!/bin/bash
# =============================================================================
# paddleocr_export.sh - logging, compose helpers, ONNX download and TensorRT export steps
# =============================================================================
# Sourced by scripts/export_paddleocr.sh, which supplies the paths, ports,
# shape and TRT settings. Not runnable on its own.

# Variables are assigned by export_paddleocr.sh before this file is sourced.
# shellcheck disable=SC2154

[[ -n "${_PADDLEOCR_EXPORT_SH_LOADED:-}" ]] && return 0
_PADDLEOCR_EXPORT_SH_LOADED=1

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Compose with an explicit project: a bare `docker compose` takes the project
# from the directory name, which is not the stack's COMPOSE_PROJECT_NAME.
dc() {
    local project="${COMPOSE_PROJECT_NAME:-$(env_port COMPOSE_PROJECT_NAME openprocessor)}"
    local -a cmd=(docker compose -p "$project")
    [[ -f "$PROJECT_DIR/.env" ]] && cmd+=(--env-file "$PROJECT_DIR/.env")
    "${cmd[@]}" --project-directory "$PROJECT_DIR" "$@"
}

# True when the compose SERVICE has a running container. Container names are
# ${COMPOSE_PROJECT_NAME}-<service>, so a bare name match never hits.
check_container() {
    [[ -n "$(dc ps -q "$1" 2>/dev/null)" ]]
}

wait_for_triton() {
    log_info "Waiting for Triton to be ready..."
    for _ in {1..30}; do
        if curl -s "localhost:${TRITON_HTTP_PORT}/v2/health/ready" > /dev/null 2>&1; then
            log_success "Triton is ready"
            return 0
        fi
        sleep 2
    done
    log_error "Triton did not become ready in time"
    return 1
}

unload_models_for_memory() {
    log_info "Unloading models to free GPU memory..."

    local models_to_unload=(
        "ocr_pipeline"
        "paddleocr_det_trt"
        "paddleocr_rec_trt"
        "yolov11_small_trt_end2end"
        "scrfd_10g_bnkps"
        "arcface_w600k_r50"
        "mobileclip2_s2_image_encoder"
        "mobileclip2_s2_text_encoder"
    )

    for model in "${models_to_unload[@]}"; do
        curl -s -X POST "localhost:${TRITON_HTTP_PORT}/v2/repository/models/${model}/unload" > /dev/null 2>&1 || true
    done

    sleep 3
    log_success "Models unloaded"
}

# =============================================================================
# Download Functions
# =============================================================================

download_models() {
    log_info "Downloading PP-OCRv5 ONNX models (detection + multilingual recognition)..."

    local download_ok=false

    # Try host .venv first (no Docker dependency)
    local venv_python="$PROJECT_DIR/.venv/bin/python"
    if [ -f "$venv_python" ]; then
        log_info "Using host .venv for download..."
        cd "$PROJECT_DIR" || return 1
        if "$venv_python" export/download_paddleocr.py 2>&1; then
            download_ok=true
        fi
    fi

    # Fall back to Docker container if host .venv didn't work
    if [ "$download_ok" = false ]; then
        if check_container "api"; then
            log_info "Using running api container for download..."
            if dc exec -T api python /app/export/download_paddleocr.py; then
                download_ok=true
            fi
        else
            log_info "Using temporary api container for download..."
            if dc run --rm --no-deps -T api python /app/export/download_paddleocr.py; then
                download_ok=true
            fi
        fi
    fi

    if [ "$download_ok" = true ]; then
        log_success "ONNX models downloaded (detection + multilingual recognition)"
    else
        log_error "Model download failed"
        log_info "Ensure either .venv or api container image is built"
        return 1
    fi
}

# =============================================================================
# TensorRT Export Functions
# =============================================================================

export_detection() {
    log_info "Exporting detection model to TensorRT..."

    local onnx_path="/models/$DET_ONNX_NAME"
    local plan_path="/models/paddleocr_det_trt/1/model.plan"

    # Check ONNX exists
    if [ ! -f "$PYTORCH_MODELS_DIR/$DET_ONNX_NAME" ]; then
        log_error "Detection ONNX not found: $PYTORCH_MODELS_DIR/$DET_ONNX_NAME"
        log_info "Run: ./scripts/export_paddleocr.sh download"
        return 1
    fi

    # Copy ONNX to models dir for container access
    cp "$PYTORCH_MODELS_DIR/$DET_ONNX_NAME" "$MODELS_DIR/$DET_ONNX_NAME"

    # Create output directory
    mkdir -p "$MODELS_DIR/paddleocr_det_trt/1"

    # Remove old plan
    rm -f "$DET_PLAN"

    log_info "Running trtexec for detection model..."
    log_info "  Min shapes: $DET_MIN_SHAPES"
    log_info "  Opt shapes: $DET_OPT_SHAPES"
    log_info "  Max shapes: $DET_MAX_SHAPES"
    log_info "  Workspace: $TRT_WORKSPACE"

    # Use 'docker compose run' instead of 'exec' to avoid requiring triton-server
    # to be running with all models loaded (chicken-and-egg on fresh install).
    # --rm: clean up container after exit
    # --no-deps: don't start dependent services
    # -T: disable pseudo-TTY (needed for non-interactive/piped output)
    dc run --rm --no-deps -T triton-server trtexec \
        --onnx="$onnx_path" \
        --saveEngine="$plan_path" \
        --minShapes="$DET_MIN_SHAPES" \
        --optShapes="$DET_OPT_SHAPES" \
        --maxShapes="$DET_MAX_SHAPES" \
        --memPoolSize=workspace:$TRT_WORKSPACE \
        $TRT_FP16 \
        2>&1 | tee /tmp/trtexec_det.log

    if [ -f "$DET_PLAN" ] && [ -s "$DET_PLAN" ]; then
        local size
        size=$(du -h "$DET_PLAN" | cut -f1)
        log_success "Detection TensorRT engine created: $size"
    else
        log_error "Detection TensorRT conversion failed"
        log_info "Check log: /tmp/trtexec_det.log"
        return 1
    fi
}

export_recognition() {
    log_info "Exporting multilingual recognition model to TensorRT..."

    local onnx_path="/models/$REC_ONNX_NAME"
    local plan_path="/models/paddleocr_rec_trt/1/model.plan"

    # Check ONNX exists in pytorch_models/paddleocr (downloaded by download_paddleocr.py)
    if [ ! -f "$PYTORCH_MODELS_DIR/$REC_ONNX_NAME" ]; then
        log_error "Recognition ONNX not found: $PYTORCH_MODELS_DIR/$REC_ONNX_NAME"
        log_info "Run: ./scripts/export_paddleocr.sh download"
        return 1
    fi

    local onnx_source="$PYTORCH_MODELS_DIR/$REC_ONNX_NAME"

    # Copy ONNX to models dir for container access
    cp "$onnx_source" "$MODELS_DIR/$REC_ONNX_NAME"

    # Create output directory
    mkdir -p "$MODELS_DIR/paddleocr_rec_trt/1"

    # Remove old plan
    rm -f "$REC_PLAN"

    log_info "Running trtexec for recognition model..."
    log_info "  Min shapes: $REC_MIN_SHAPES"
    log_info "  Opt shapes: $REC_OPT_SHAPES"
    log_info "  Max shapes: $REC_MAX_SHAPES"
    log_info "  Workspace: $TRT_WORKSPACE"
    log_warn "This may take 10-20 minutes for dynamic width optimization..."

    # Use 'docker compose run' instead of 'exec' to avoid chicken-and-egg problem.
    dc run --rm --no-deps -T triton-server trtexec \
        --onnx="$onnx_path" \
        --saveEngine="$plan_path" \
        --minShapes="$REC_MIN_SHAPES" \
        --optShapes="$REC_OPT_SHAPES" \
        --maxShapes="$REC_MAX_SHAPES" \
        --memPoolSize=workspace:$TRT_WORKSPACE \
        $TRT_FP16 \
        2>&1 | tee /tmp/trtexec_rec.log

    if [ -f "$REC_PLAN" ] && [ -s "$REC_PLAN" ]; then
        local size
        size=$(du -h "$REC_PLAN" | cut -f1)
        log_success "Recognition TensorRT engine created: $size"
    else
        log_error "Recognition TensorRT conversion failed"
        log_info "Check log: /tmp/trtexec_rec.log"
        return 1
    fi
}

export_all_trt() {
    log_info "Converting all models to TensorRT..."

    # Try to unload models if Triton is running (frees GPU memory)
    if check_container "triton-server"; then
        unload_models_for_memory
    else
        log_info "Triton not running - using docker compose run for trtexec"
    fi

    # Export detection
    export_detection || return 1

    # Export recognition
    export_recognition || return 1

    log_success "All TensorRT exports complete"
}
