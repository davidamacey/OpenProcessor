#!/bin/bash
# =============================================================================
# export_steps.sh - per-model TensorRT export steps
# =============================================================================
# Sourced by export.sh, which supplies run_in_api_container(),
# check_triton_container(), unload_models_for_export(), the path and TRT
# settings and log_*().

[[ -n "${_EXPORT_STEPS_SH_LOADED:-}" ]] && return 0
_EXPORT_STEPS_SH_LOADED=1

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

# G-07: PE-Core-L14-336 image encoder. Required (not curation-only) --
# docker-compose.yml's default --load-model list includes
# pe_image_encoder, and Triton's explicit model-control-mode exits at
# startup if any listed model fails to load. So this must run in the
# same automated flow as every other model above, not be left to a
# separate `make export-pe` a core-install user would never know to run.
export_pe() {
    local plan_path="$MODELS_DIR/pe_image_encoder/1/model.plan"
    local onnx_path="$PYTORCH_MODELS_DIR/pe_image_encoder.onnx"

    log_step "Exporting PE-Core image encoder (pe_image_encoder)..."
    log_info "  Estimated time: ~${EXPORT_TIMES[pe_image_encoder]} minutes"

    mkdir -p "$MODELS_DIR/pe_image_encoder/1"

    if ! run_in_api_container python /app/export/download_pe_weights.py 2>&1 | tee /tmp/export_pe_download.log; then
        log_error "PE-Core weight download failed"
        log_info "Check log: /tmp/export_pe_download.log"
        return 1
    fi

    if ! run_in_api_container python /app/export/export_pe_image_encoder.py 2>&1 | tee /tmp/export_pe_image.log; then
        log_error "PE-Core image tower ONNX export failed"
        log_info "Check log: /tmp/export_pe_image.log"
        return 1
    fi

    # Path 1: TensorRT. Known to fail on some TRT builds (attention-pool
    # ops) -- fall back to Path 2 (ONNX Runtime via Triton) rather than
    # failing the whole export, matching `make export-pe`'s own note.
    if ONNX_PATH="$onnx_path" bash "$PROJECT_DIR/export/build_pe_trt.sh" 2>&1 | tee /tmp/export_pe_trt.log \
        && [[ -f "$plan_path" ]] && [[ -s "$plan_path" ]]; then
        log_success "PE-Core image encoder exported (TensorRT)"
    else
        log_warn "PE-Core TensorRT build failed or produced no plan -- falling back to ONNX Runtime"
        if ! ONNX_PATH="$onnx_path" bash "$PROJECT_DIR/export/build_pe_ort_fallback.sh" 2>&1 | tee /tmp/export_pe_ort.log; then
            log_error "PE-Core export failed (both TensorRT and ONNX Runtime paths)"
            log_info "Check logs: /tmp/export_pe_trt.log, /tmp/export_pe_ort.log"
            return 1
        fi
        log_success "PE-Core image encoder exported (ONNX Runtime fallback)"
    fi

    # Text tower: runs in-process (ONNX Runtime), not on Triton -- no
    # --load-model entry needed, so a failure here is a warning, not a
    # hard failure of the whole export step (semantic-search TEXT
    # queries degrade to PyTorch eager; embeddings/near-dup/clustering,
    # which only need the image tower above, are unaffected).
    if ! run_in_api_container python /app/export/export_pe_text_encoder.py 2>&1 | tee /tmp/export_pe_text.log; then
        log_warn "PE-Core text tower export failed -- semantic-search text queries will fall back to PyTorch eager"
        log_info "Check log: /tmp/export_pe_text.log"
    fi

    return 0
}
