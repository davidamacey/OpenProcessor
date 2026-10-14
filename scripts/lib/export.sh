#!/bin/bash
# =============================================================================
# export.sh - TensorRT export utilities for OpenProcessor
# =============================================================================
# Wraps existing Python export scripts with progress feedback and error handling
# =============================================================================

# PYTORCH_MODELS_DIR and EXPORT_TIMES are read by the sourced export_steps.sh.
# shellcheck disable=SC2034

# Avoid redefining if already sourced
[[ -n "${_EXPORT_SH_LOADED:-}" ]] && return 0
_EXPORT_SH_LOADED=1

# Source dependencies
_EXPORT_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=colors.sh
[[ -z "${NC:-}" ]] && source "${_EXPORT_SH_DIR}/colors.sh"
# shellcheck source=export_steps.sh
source "${_EXPORT_SH_DIR}/export_steps.sh"

# =============================================================================
# Configuration
# =============================================================================

PROJECT_DIR="${PROJECT_DIR:-$(dirname "$(dirname "$(dirname "${BASH_SOURCE[0]}")")")}"
MODELS_DIR="$PROJECT_DIR/models"
PYTORCH_MODELS_DIR="$PROJECT_DIR/pytorch_models"

# TensorRT export settings
TRT_WORKSPACE="${TRT_WORKSPACE:-4G}"
# TRT_FP16 (G-21): removed. trtexec --fp16 no longer exists on TRT 11.1
# (strongly-typed builds follow the ONNX dtypes); this var was defined but
# never read by anything in this file. See scripts/export_paddleocr.sh for
# the current pattern (empty placeholder, precision baked into the ONNX).

# Estimated export times (minutes) for progress display
declare -gA EXPORT_TIMES=(
    ["yolov11_small_trt_end2end"]="5"
    ["scrfd_10g_bnkps"]="3"
    ["arcface_w600k_r50"]="3"
    ["mobileclip2_s2_image_encoder"]="10"
    ["mobileclip2_s2_text_encoder"]="5"
    ["paddleocr_det_trt"]="10"
    ["paddleocr_rec_trt"]="15"
    # G-07: pe_image_encoder is now in docker-compose.yml's default
    # --load-model list (required, not curation-only -- see
    # docs/CURATION.md "Models you must supply"), so it must be built
    # before Triton's first start, same as every other model above.
    ["pe_image_encoder"]="12"
)

# =============================================================================
# Container Helpers
# =============================================================================

# Run a command in a temporary api container (no dependencies needed).
# Uses 'docker compose run --rm --no-deps -T' so the export works without
# Triton running and without the api container already up.
run_in_api_container() {
    dc run --rm --no-deps -T api "$@"
}

check_triton_container() {
    # G-03: check by compose SERVICE state, not a hardcoded container
    # name -- container_name is now ${COMPOSE_PROJECT_NAME}-triton, which
    # differs on a second isolated stack (or any COMPOSE_PROJECT_NAME).
    if [[ -z "$(dc ps --status running --services triton-server 2>/dev/null)" ]]; then
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

    # G-03: read the port THIS deployment's .env/shell actually set, not
    # the hardcoded default -- a remapped TRITON_HTTP_PORT must still hit
    # the right Triton.
    local triton_http_port="${TRITON_HTTP_PORT:-4600}"
    for model in "${models_to_unload[@]}"; do
        curl -s -X POST "localhost:${triton_http_port}/v2/repository/models/${model}/unload" > /dev/null 2>&1 || true
    done

    sleep 3
    log_success "Models unloaded"
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
    local total=7
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

    log_step "Exporting model 7/$total: PE-Core image encoder"
    step_start=$SECONDS
    if export_pe; then
        model_status+=("OK")
    else
        failed=$((failed + 1))
        model_status+=("FAIL")
    fi
    elapsed=$((SECONDS - step_start))
    model_names+=("PE-Core image encoder")
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

    log_info "Exporting essential models only (no OCR; Triton skips the missing OCR models)"
    log_info "Estimated time: 25-35 minutes"
    echo ""

    unload_models_for_export

    local failed=0
    local total=6
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

    # PE-Core backs curation ingest embeddings, search and item scores.
    log_step "Exporting model 6/$total: PE-Core image encoder"
    step_start=$SECONDS
    if export_pe; then model_status+=("OK"); else failed=$((failed + 1)); model_status+=("FAIL"); fi
    elapsed=$((SECONDS - step_start))
    model_names+=("PE-Core image encoder")
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
