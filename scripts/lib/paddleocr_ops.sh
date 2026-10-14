#!/bin/bash
# =============================================================================
# paddleocr_ops.sh - restart, reload, test, status and clean commands
# =============================================================================
# Sourced by scripts/export_paddleocr.sh, which supplies the paths, ports,
# shape and TRT settings. Not runnable on its own.

# Variables are assigned by export_paddleocr.sh before this file is sourced.
# shellcheck disable=SC2154

[[ -n "${_PADDLEOCR_OPS_SH_LOADED:-}" ]] && return 0
_PADDLEOCR_OPS_SH_LOADED=1

# =============================================================================
# Control Functions
# =============================================================================

restart_triton() {
    log_info "Restarting Triton server..."
    dc restart triton-server
    wait_for_triton
}

reload_models() {
    log_info "Reloading OCR models..."

    local models=("paddleocr_det_trt" "paddleocr_rec_trt" "ocr_pipeline")

    for model in "${models[@]}"; do
        log_info "Loading $model..."
        curl -s -X POST "localhost:${TRITON_HTTP_PORT}/v2/repository/models/${model}/load" > /dev/null 2>&1 || true
    done

    sleep 2
    log_success "Models reloaded"
}

run_test() {
    log_info "Running OCR test..."

    if [ -f "$PROJECT_DIR/scripts/test_ocr_pipeline.py" ]; then
        python "$PROJECT_DIR/scripts/test_ocr_pipeline.py"
    else
        # Quick curl test
        local test_image="$PROJECT_DIR/test_images/ocr-synthetic/hello_world.jpg"
        if [ -f "$test_image" ]; then
            log_info "Testing with: $test_image"
            curl -s -X POST "http://localhost:${API_PORT}/ocr/predict" \
                -F "image=@$test_image" | python -m json.tool
        else
            log_warn "No test image found at: $test_image"
            log_info "Testing model health..."
            curl -s "localhost:${TRITON_HTTP_PORT}/v2/models/paddleocr_det_trt" | python -m json.tool
        fi
    fi
}

show_status() {
    echo ""
    echo "=========================================="
    echo "PP-OCRv5 Model Status"
    echo "=========================================="
    echo ""

    # Check files
    echo "Model Files:"
    echo "-----------"

    if [ -f "$DET_PLAN" ]; then
        local det_size
        det_size=$(du -h "$DET_PLAN" | cut -f1)
        echo -e "  Detection TRT:   ${GREEN}OK${NC} ($det_size)"
    else
        echo -e "  Detection TRT:   ${RED}MISSING${NC}"
    fi

    if [ -f "$REC_PLAN" ]; then
        local rec_size
        rec_size=$(du -h "$REC_PLAN" | cut -f1)
        echo -e "  Recognition TRT: ${GREEN}OK${NC} ($rec_size)"
    else
        echo -e "  Recognition TRT: ${RED}MISSING${NC}"
    fi

    if [ -f "$DICT_FILE" ]; then
        local char_count
        char_count=$(wc -l < "$DICT_FILE")
        echo -e "  Dictionary:      ${GREEN}OK${NC} ($char_count chars)"
    else
        echo -e "  Dictionary:      ${RED}MISSING${NC}"
    fi

    echo ""
    echo "Triton Models:"
    echo "-------------"

    for model in paddleocr_det_trt paddleocr_rec_trt ocr_pipeline; do
        local status
        status=$(curl -s "localhost:${TRITON_HTTP_PORT}/v2/models/$model" 2>/dev/null | grep -o '"state":"[^"]*"' | cut -d'"' -f4)
        if [ "$status" = "READY" ]; then
            echo -e "  $model: ${GREEN}READY${NC}"
        elif [ -n "$status" ]; then
            echo -e "  $model: ${YELLOW}$status${NC}"
        else
            echo -e "  $model: ${RED}NOT LOADED${NC}"
        fi
    done

    echo ""
}

clean_all() {
    log_warn "This will remove all OCR model files. Continue? [y/N]"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        rm -rf "$MODELS_DIR/paddleocr_det_trt/1/model.plan"
        rm -rf "$MODELS_DIR/paddleocr_rec_trt/1/model.plan"
        rm -f "$MODELS_DIR/$DET_ONNX_NAME"
        rm -f "$MODELS_DIR/$REC_ONNX_NAME"
        log_success "OCR model files removed"
    else
        log_info "Cancelled"
    fi
}
