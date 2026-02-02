#!/bin/bash
# =============================================================================
# export.sh - TensorRT export utilities for OpenProcessor
# =============================================================================
# Wraps existing Python export scripts with progress feedback and error handling
# =============================================================================

# Avoid redefining if already sourced
[[ -n "${_EXPORT_SH_LOADED:-}" ]] && return 0
_EXPORT_SH_LOADED=1

# Source dependencies
_EXPORT_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=colors.sh
[[ -z "${NC:-}" ]] && source "${_EXPORT_SH_DIR}/colors.sh"

# =============================================================================
# Configuration
# =============================================================================

PROJECT_DIR="${PROJECT_DIR:-$(dirname "$(dirname "$(dirname "${BASH_SOURCE[0]}")")")}"
MODELS_DIR="$PROJECT_DIR/models"
PYTORCH_MODELS_DIR="$PROJECT_DIR/pytorch_models"

# TensorRT export settings
TRT_WORKSPACE="${TRT_WORKSPACE:-4G}"
TRT_FP16="${TRT_FP16:---fp16}"

# Estimated export times (minutes) for progress display
declare -gA EXPORT_TIMES=(
    ["yolov11_small_trt_end2end"]="5"
    ["scrfd_10g_bnkps"]="3"
    ["arcface_w600k_r50"]="3"
    ["mobileclip2_s2_image_encoder"]="10"
    ["mobileclip2_s2_text_encoder"]="5"
    ["paddleocr_det_trt"]="10"
    ["paddleocr_rec_trt"]="15"
)

# =============================================================================
# Container Helpers
# =============================================================================

# Run a command in a temporary yolo-api container (no dependencies needed).
# Uses 'docker compose run --rm --no-deps -T' so the export works without
# Triton running and without the yolo-api container already up.
run_in_api_container() {
    docker compose run --rm --no-deps -T yolo-api "$@"
}

check_triton_container() {
    if ! docker ps --format '{{.Names}}' | grep -q "^triton-server$"; then
        log_error "triton-server container not running"
        log_info "Start with: docker compose up -d triton-server"
        return 1
    fi
    return 0
}

# =============================================================================
# GPU Memory Management
# =============================================================================

# Unload models to free GPU memory for export
unload_models_for_export() {
    log_info "Unloading models to free GPU memory..."

    local models_to_unload=(
        "yolov11_small_trt_end2end"
        "scrfd_10g_bnkps"
        "arcface_w600k_r50"
        "mobileclip2_s2_image_encoder"
        "mobileclip2_s2_text_encoder"
        "paddleocr_det_trt"
        "paddleocr_rec_trt"
        "ocr_pipeline"
    )

    for model in "${models_to_unload[@]}"; do
        curl -s -X POST "localhost:4600/v2/repository/models/${model}/unload" > /dev/null 2>&1 || true
    done

    sleep 3
    log_success "Models unloaded"
}

# =============================================================================
# Individual Model Exports
# =============================================================================

# Export YOLO object detection model
export_yolo_detection() {
    local model_name="yolov11_small_trt_end2end"
    local source_model="yolo11s.pt"
    local plan_path="$MODELS_DIR/$model_name/1/model.plan"

    log_step "Exporting YOLO detection model..."
    log_info "  Source: $source_model"
    log_info "  Output: $plan_path"
    log_info "  Estimated time: ~${EXPORT_TIMES[$model_name]} minutes"

    # Check source exists
    if [[ ! -f "$PYTORCH_MODELS_DIR/$source_model" ]]; then
        log_error "Source model not found: $PYTORCH_MODELS_DIR/$source_model"
        return 1
    fi

    # Create output directory
    mkdir -p "$MODELS_DIR/$model_name/1"

    # Run export via export_models.py (needs all formats for end2end dependency)
    if run_in_api_container python /app/export/export_models.py \
        --models small \
        --formats onnx_end2end trt_end2end \
        --normalize-boxes \
        --generate-config \
        --save-labels \
        2>&1 | tee /tmp/export_yolo.log; then

        if [[ -f "$plan_path" ]] && [[ -s "$plan_path" ]]; then
            local size
            size=$(du -h "$plan_path" | cut -f1)
            log_success "YOLO detection exported: $size"
            return 0
        fi
    fi

    log_error "YOLO detection export failed"
    log_info "Check log: /tmp/export_yolo.log"
    return 1
}

# Export SCRFD face detection model
export_scrfd() {
    local model_name="scrfd_10g_bnkps"
    local source_model="scrfd_10g_bnkps.onnx"
    local plan_path="$MODELS_DIR/$model_name/1/model.plan"

    log_step "Exporting SCRFD face detection model..."
    log_info "  Source: $source_model"
    log_info "  Output: $plan_path"
    log_info "  Estimated time: ~${EXPORT_TIMES[$model_name]} minutes"

    mkdir -p "$MODELS_DIR/$model_name/1"

    # export_scrfd.py handles download + ONNX patching + TensorRT conversion
    if run_in_api_container python /app/export/export_scrfd.py \
        2>&1 | tee /tmp/export_scrfd.log; then

        if [[ -f "$plan_path" ]] && [[ -s "$plan_path" ]]; then
            local size
            size=$(du -h "$plan_path" | cut -f1)
            log_success "SCRFD face detection exported: $size"
            return 0
        fi
    fi

    log_error "SCRFD export failed"
    log_info "Check log: /tmp/export_scrfd.log"
    return 1
}

# Export ArcFace model
export_arcface() {
    local model_name="arcface_w600k_r50"
    local source_model="arcface_w600k_r50.onnx"
    local plan_path="$MODELS_DIR/$model_name/1/model.plan"

    log_step "Exporting ArcFace model..."
    log_info "  Source: $source_model"
    log_info "  Output: $plan_path"
    log_info "  Estimated time: ~${EXPORT_TIMES[$model_name]} minutes"

    # Check source exists
    if [[ ! -f "$PYTORCH_MODELS_DIR/$source_model" ]]; then
        log_error "Source model not found: $PYTORCH_MODELS_DIR/$source_model"
        return 1
    fi

    mkdir -p "$MODELS_DIR/$model_name/1"

    # Copy ONNX to models dir for trtexec access
    cp "$PYTORCH_MODELS_DIR/$source_model" "$MODELS_DIR/$source_model"

    log_info "Running ArcFace export (Python TensorRT API)..."

    if run_in_api_container python /app/export/export_face_recognition.py \
        --onnx "/app/pytorch_models/$source_model" \
        --plan "/app/models/$model_name/1/model.plan" \
        --max-batch 128 \
        2>&1 | tee /tmp/export_arcface.log; then

        if [[ -f "$plan_path" ]] && [[ -s "$plan_path" ]]; then
            local size
            size=$(du -h "$plan_path" | cut -f1)
            log_success "ArcFace exported: $size"
            # Cleanup ONNX from models dir
            rm -f "$MODELS_DIR/$source_model"
            return 0
        fi
    fi

    log_error "ArcFace export failed"
    log_info "Check log: /tmp/export_arcface.log"
    return 1
}

# Export MobileCLIP image encoder
export_mobileclip_image() {
    local model_name="mobileclip2_s2_image_encoder"
    local plan_path="$MODELS_DIR/$model_name/1/model.plan"

    log_step "Exporting MobileCLIP image encoder..."
    log_info "  Output: $plan_path"
    log_info "  Estimated time: ~${EXPORT_TIMES[$model_name]} minutes"

    mkdir -p "$MODELS_DIR/$model_name/1"

    if run_in_api_container python /app/export/export_mobileclip_image_encoder.py \
        --model S2 \
        2>&1 | tee /tmp/export_mobileclip_image.log; then

        if [[ -f "$plan_path" ]] && [[ -s "$plan_path" ]]; then
            local size
            size=$(du -h "$plan_path" | cut -f1)
            log_success "MobileCLIP image encoder exported: $size"
            return 0
        fi
    fi

    log_error "MobileCLIP image encoder export failed"
    log_info "Check log: /tmp/export_mobileclip_image.log"
    return 1
}

# Export MobileCLIP text encoder
export_mobileclip_text() {
    local model_name="mobileclip2_s2_text_encoder"
    local plan_path="$MODELS_DIR/$model_name/1/model.plan"

    log_step "Exporting MobileCLIP text encoder..."
    log_info "  Output: $plan_path"
    log_info "  Estimated time: ~${EXPORT_TIMES[$model_name]} minutes"

    mkdir -p "$MODELS_DIR/$model_name/1"

    if run_in_api_container python /app/export/export_mobileclip_text_encoder.py \
        --model S2 \
        2>&1 | tee /tmp/export_mobileclip_text.log; then

        if [[ -f "$plan_path" ]] && [[ -s "$plan_path" ]]; then
            local size
            size=$(du -h "$plan_path" | cut -f1)
            log_success "MobileCLIP text encoder exported: $size"
            return 0
        fi
    fi

    log_error "MobileCLIP text encoder export failed"
    log_info "Check log: /tmp/export_mobileclip_text.log"
    return 1
}

# Export PaddleOCR models (uses dedicated script)
export_paddleocr() {
    local det_plan="$MODELS_DIR/paddleocr_det_trt/1/model.plan"
    local rec_plan="$MODELS_DIR/paddleocr_rec_trt/1/model.plan"

    log_step "Exporting PaddleOCR models..."
    log_info "  Estimated time: ~25 minutes (detection + recognition)"

    local script="$PROJECT_DIR/scripts/export_paddleocr.sh"

    if [[ ! -x "$script" ]]; then
        # Make executable if it exists but isn't executable
        if [[ -f "$script" ]]; then
            chmod +x "$script"
        else
            log_error "PaddleOCR export script not found: $script"
            return 1
        fi
    fi

    # Use 'all' command to run full pipeline:
    # 1. Download detection + multilingual recognition ONNX
    # 2. Setup dictionary
    # 3. Convert both to TensorRT (via docker compose run)
    # 4. Create Triton configs
    "$script" all 2>&1 | tee /tmp/export_paddleocr.log

    # Verify both plan files exist (pipeline exit code can be masked by tee)
    if [[ -f "$det_plan" ]] && [[ -s "$det_plan" ]] && \
       [[ -f "$rec_plan" ]] && [[ -s "$rec_plan" ]]; then
        log_success "PaddleOCR models exported"
        return 0
    fi

    log_error "PaddleOCR export failed (missing model.plan files)"
    log_info "Check log: /tmp/export_paddleocr.log"
    return 1
}

# =============================================================================
# Batch Export Functions
# =============================================================================

# Export all models (with per-step timing)
export_all_models() {
    print_header "TensorRT Model Export"

    log_info "This will export all models to TensorRT format."
    log_info "Total estimated time: 45-60 minutes"
    log_warn "GPU will be heavily utilized during export."
    echo ""

    # Unload existing models to free memory
    unload_models_for_export

    local failed=0
    local total=6
    local export_start step_start elapsed
    export_start=$SECONDS

    # Arrays for timing summary
    local -a model_names=()
    local -a model_times=()
    local -a model_status=()

    # Export each model with timing
    log_step "Exporting model 1/$total: YOLO detection"
    step_start=$SECONDS
    if export_yolo_detection; then
        model_status+=("OK")
    else
        failed=$((failed + 1))
        model_status+=("FAIL")
    fi
    elapsed=$((SECONDS - step_start))
    model_names+=("YOLO detection")
    model_times+=("$elapsed")
    log_info "  Elapsed: $(format_elapsed $elapsed)"

    log_step "Exporting model 2/$total: SCRFD face detection"
    step_start=$SECONDS
    if export_scrfd; then
        model_status+=("OK")
    else
        failed=$((failed + 1))
        model_status+=("FAIL")
    fi
    elapsed=$((SECONDS - step_start))
    model_names+=("SCRFD face detection")
    model_times+=("$elapsed")
    log_info "  Elapsed: $(format_elapsed $elapsed)"

    log_step "Exporting model 3/$total: ArcFace"
    step_start=$SECONDS
    if export_arcface; then
        model_status+=("OK")
    else
        failed=$((failed + 1))
        model_status+=("FAIL")
    fi
    elapsed=$((SECONDS - step_start))
    model_names+=("ArcFace embedding")
    model_times+=("$elapsed")
    log_info "  Elapsed: $(format_elapsed $elapsed)"

    log_step "Exporting model 4/$total: MobileCLIP image"
    step_start=$SECONDS
    if export_mobileclip_image; then
        model_status+=("OK")
    else
        failed=$((failed + 1))
        model_status+=("FAIL")
    fi
    elapsed=$((SECONDS - step_start))
    model_names+=("MobileCLIP image encoder")
    model_times+=("$elapsed")
    log_info "  Elapsed: $(format_elapsed $elapsed)"

    log_step "Exporting model 5/$total: MobileCLIP text"
    step_start=$SECONDS
    if export_mobileclip_text; then
        model_status+=("OK")
    else
        failed=$((failed + 1))
        model_status+=("FAIL")
    fi
    elapsed=$((SECONDS - step_start))
    model_names+=("MobileCLIP text encoder")
    model_times+=("$elapsed")
    log_info "  Elapsed: $(format_elapsed $elapsed)"

    log_step "Exporting model 6/$total: PaddleOCR"
    step_start=$SECONDS
    if export_paddleocr; then
        model_status+=("OK")
    else
        failed=$((failed + 1))
        model_status+=("FAIL")
    fi
    elapsed=$((SECONDS - step_start))
    model_names+=("PaddleOCR (det + rec)")
    model_times+=("$elapsed")
    log_info "  Elapsed: $(format_elapsed $elapsed)"

    # Print timing summary table
    local total_elapsed=$((SECONDS - export_start))
    echo ""
    print_header "Export Timing Summary"
    printf "  %-28s  %8s  %s\n" "Model" "Time" "Status"
    printf "  %-28s  %8s  %s\n" "----------------------------" "--------" "------"
    for i in "${!model_names[@]}"; do
        local status_color="$GREEN"
        [[ "${model_status[$i]}" == "FAIL" ]] && status_color="$RED"
        printf "  %-28s  %8s  ${status_color}%s${NC}\n" \
            "${model_names[$i]}" "$(format_elapsed "${model_times[$i]}")" "${model_status[$i]}"
    done
    printf "  %-28s  %8s\n" "----------------------------" "--------"
    printf "  %-28s  %8s\n" "Total" "$(format_elapsed $total_elapsed)"
    echo ""

    if [[ $failed -eq 0 ]]; then
        log_success "All models exported successfully!"
        return 0
    else
        log_error "$failed/$total exports failed"
        return 1
    fi
}

# Export essential models only (for minimal profile, with timing)
export_essential_models() {
    print_header "TensorRT Export (Essential Models)"

    log_info "Exporting essential models only (no OCR)"
    log_info "Estimated time: 25-35 minutes"
    echo ""

    unload_models_for_export

    local failed=0
    local total=5
    local export_start step_start elapsed
    export_start=$SECONDS

    local -a model_names=()
    local -a model_times=()
    local -a model_status=()

    log_step "Exporting model 1/$total: YOLO detection"
    step_start=$SECONDS
    if export_yolo_detection; then model_status+=("OK"); else failed=$((failed + 1)); model_status+=("FAIL"); fi
    elapsed=$((SECONDS - step_start))
    model_names+=("YOLO detection")
    model_times+=("$elapsed")
    log_info "  Elapsed: $(format_elapsed $elapsed)"

    log_step "Exporting model 2/$total: SCRFD face detection"
    step_start=$SECONDS
    if export_scrfd; then model_status+=("OK"); else failed=$((failed + 1)); model_status+=("FAIL"); fi
    elapsed=$((SECONDS - step_start))
    model_names+=("SCRFD face detection")
    model_times+=("$elapsed")
    log_info "  Elapsed: $(format_elapsed $elapsed)"

    log_step "Exporting model 3/$total: ArcFace"
    step_start=$SECONDS
    if export_arcface; then model_status+=("OK"); else failed=$((failed + 1)); model_status+=("FAIL"); fi
    elapsed=$((SECONDS - step_start))
    model_names+=("ArcFace embedding")
    model_times+=("$elapsed")
    log_info "  Elapsed: $(format_elapsed $elapsed)"

    log_step "Exporting model 4/$total: MobileCLIP image"
    step_start=$SECONDS
    if export_mobileclip_image; then model_status+=("OK"); else failed=$((failed + 1)); model_status+=("FAIL"); fi
    elapsed=$((SECONDS - step_start))
    model_names+=("MobileCLIP image encoder")
    model_times+=("$elapsed")
    log_info "  Elapsed: $(format_elapsed $elapsed)"

    log_step "Exporting model 5/$total: MobileCLIP text"
    step_start=$SECONDS
    if export_mobileclip_text; then model_status+=("OK"); else failed=$((failed + 1)); model_status+=("FAIL"); fi
    elapsed=$((SECONDS - step_start))
    model_names+=("MobileCLIP text encoder")
    model_times+=("$elapsed")
    log_info "  Elapsed: $(format_elapsed $elapsed)"

    # Print timing summary table
    local total_elapsed=$((SECONDS - export_start))
    echo ""
    print_header "Export Timing Summary"
    printf "  %-28s  %8s  %s\n" "Model" "Time" "Status"
    printf "  %-28s  %8s  %s\n" "----------------------------" "--------" "------"
    for i in "${!model_names[@]}"; do
        local status_color="$GREEN"
        [[ "${model_status[$i]}" == "FAIL" ]] && status_color="$RED"
        printf "  %-28s  %8s  ${status_color}%s${NC}\n" \
            "${model_names[$i]}" "$(format_elapsed "${model_times[$i]}")" "${model_status[$i]}"
    done
    printf "  %-28s  %8s\n" "----------------------------" "--------"
    printf "  %-28s  %8s\n" "Total" "$(format_elapsed $total_elapsed)"
    echo ""

    if [[ $failed -eq 0 ]]; then
        log_success "Essential models exported!"
        return 0
    else
        log_error "$failed exports failed"
        return 1
    fi
}

# Export single model by name
export_model() {
    local model_name="$1"

    case "$model_name" in
        yolo|yolov11|detection)
            export_yolo_detection
            ;;
        face|scrfd)
            export_scrfd
            ;;
        arcface)
            export_arcface
            ;;
        clip-image|mobileclip-image)
            export_mobileclip_image
            ;;
        clip-text|mobileclip-text)
            export_mobileclip_text
            ;;
        ocr|paddleocr)
            export_paddleocr
            ;;
        all)
            export_all_models
            ;;
        essential)
            export_essential_models
            ;;
        *)
            log_error "Unknown model: $model_name"
            echo "Available: yolo, scrfd, arcface, clip-image, clip-text, ocr, all, essential"
            return 1
            ;;
    esac
}

# =============================================================================
# Status Functions
# =============================================================================

# Check which models have TensorRT exports
check_exported_models() {
    echo ""
    echo -e "${BOLD}TensorRT Model Status${NC}"
    echo "-------------------------------------------"

    local models=(
        "yolov11_small_trt_end2end"
        "scrfd_10g_bnkps"
        "arcface_w600k_r50"
        "mobileclip2_s2_image_encoder"
        "mobileclip2_s2_text_encoder"
        "paddleocr_det_trt"
        "paddleocr_rec_trt"
    )

    for model in "${models[@]}"; do
        local plan_path="$MODELS_DIR/$model/1/model.plan"
        if [[ -f "$plan_path" ]] && [[ -s "$plan_path" ]]; then
            local size
            size=$(du -h "$plan_path" | cut -f1)
            echo -e "  ${GREEN}[OK]${NC} $model ($size)"
        else
            echo -e "  ${RED}[--]${NC} $model"
        fi
    done

    echo "-------------------------------------------"
    echo ""
}

# Check if all required models are exported
all_models_exported() {
    local required=(
        "yolov11_small_trt_end2end"
        "scrfd_10g_bnkps"
        "arcface_w600k_r50"
        "mobileclip2_s2_image_encoder"
        "mobileclip2_s2_text_encoder"
        "paddleocr_det_trt"
        "paddleocr_rec_trt"
    )

    for model in "${required[@]}"; do
        local plan_path="$MODELS_DIR/$model/1/model.plan"
        if [[ ! -f "$plan_path" ]] || [[ ! -s "$plan_path" ]]; then
            return 1
        fi
    done

    return 0
}
