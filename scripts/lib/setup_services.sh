#!/bin/bash
# =============================================================================
# setup_services.sh - service start, readiness wait, smoke test and summary steps
# =============================================================================
# Sourced by scripts/setup.sh (checkout only), which supplies dc(), log_*(),
# the option flags, PROJECT_DIR and SCRIPT_DIR.

[[ -n "${_SETUP_SERVICES_SH_LOADED:-}" ]] && return 0
_SETUP_SERVICES_SH_LOADED=1

# =============================================================================
# Service Start
# =============================================================================

start_services_step() {
    if [[ "$SKIP_START" == "true" ]]; then
        log_info "Skipping service start (--skip-start)"
        return 0
    fi

    print_section "Starting Services"

    if [[ "$AUTO_YES" != "true" ]]; then
        if ! confirm "Start OpenProcessor services?" "Y"; then
            log_info "Skipping. Start manually with: docker compose up -d"
            return 0
        fi
    fi

    # Stop any running containers first. G-01/G-04: scoped to THIS
    # project only -- docker-compose.yml no longer pins a fixed `name:`,
    # so compose resolves the project from COMPOSE_PROJECT_NAME (.env or
    # shell) or, absent that, this directory's name. It never reaches a
    # different project's containers, even on a host running several
    # OpenProcessor stacks.
    local compose_project
    compose_project="$(dc config --format json 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin).get("name",""))' 2>/dev/null || true)"
    log_info "Stopping existing containers in project '${compose_project:-<unresolved>}'..."
    dc stop 2>/dev/null || true
    dc rm -f 2>/dev/null || true

    # Start all services (now that model.plan files exist, triton-server will load them)
    log_info "Starting all services..."
    dc up -d --remove-orphans

    # Wait for services to be ready
    wait_for_services
}

wait_for_services() {
    print_section "Waiting for Services"

    # Wait for Triton HTTP endpoint (up to 5 minutes for first-time model loading)
    log_info "Waiting for Triton HTTP endpoint..."
    local triton_http=false
    for i in {1..150}; do
        if curl -s localhost:${TRITON_HTTP_PORT}/v2/health/ready > /dev/null 2>&1; then
            triton_http=true
            break
        fi
        printf "\r  Waiting for Triton... %ds " "$((i*2))"
        sleep 2
    done
    echo ""

    if [[ "$triton_http" != "true" ]]; then
        log_warn "Triton HTTP endpoint not responding after 300s"
        log_info "Triton may still be loading models. Check: docker compose logs triton-server"
        log_info "Continuing setup - smoke tests may fail."
        # Don't return 1 here; let smoke tests diagnose specific failures
    else
        log_success "Triton HTTP endpoint ready"
    fi

    # Wait for required models to load
    log_info "Waiting for TensorRT models to load..."
    local required_models=("yolov11_small_trt_end2end" "scrfd_10g_bnkps" "arcface_w600k_r50" "mobileclip2_s2_image_encoder" "mobileclip2_s2_text_encoder" "paddleocr_det_trt" "paddleocr_rec_trt")
    local all_loaded=false

    for i in {1..90}; do  # Wait up to 3 minutes for models
        local loaded=0
        local total=${#required_models[@]}

        for model in "${required_models[@]}"; do
            # Triton returns HTTP 200 with empty body when model is ready
            if curl -s -o /dev/null -w "%{http_code}" "localhost:${TRITON_HTTP_PORT}/v2/models/${model}/ready" 2>/dev/null | grep -q "200"; then
                loaded=$((loaded + 1))
            fi
        done

        printf "\r  Models loaded: %d/%d " "$loaded" "$total"

        if [[ $loaded -eq $total ]]; then
            all_loaded=true
            break
        fi
        sleep 2
    done
    echo ""

    if [[ "$all_loaded" == "true" ]]; then
        log_success "All TensorRT models loaded"
    else
        log_warn "Some models may not be loaded yet - continuing anyway"
        log_info "Loaded models:"
        for model in "${required_models[@]}"; do
            if curl -s -o /dev/null -w "%{http_code}" "localhost:${TRITON_HTTP_PORT}/v2/models/${model}/ready" 2>/dev/null | grep -q "200"; then
                log_success "  $model"
            else
                log_warn "  $model (not ready)"
            fi
        done
    fi

    # Wait for API
    log_info "Waiting for API service..."
    local api_ready=false
    for i in {1..30}; do
        if curl -s localhost:${API_PORT}/health 2>/dev/null | grep -qE "ready|healthy"; then
            api_ready=true
            break
        fi
        printf "\r  Waiting for API... %ds " "$((i*2))"
        sleep 2
    done
    echo ""

    if [[ "$api_ready" == "true" ]]; then
        log_success "API is ready"
    else
        log_warn "API not fully ready yet"
    fi

    # Wait for OpenSearch
    log_info "Waiting for OpenSearch..."
    local opensearch_ready=false
    for i in {1..30}; do
        if curl -s localhost:${OPENSEARCH_PORT} > /dev/null 2>&1; then
            opensearch_ready=true
            break
        fi
        printf "\r  Waiting for OpenSearch... %ds " "$((i*2))"
        sleep 2
    done
    echo ""

    if [[ "$opensearch_ready" == "true" ]]; then
        log_success "OpenSearch is ready"
    else
        log_warn "OpenSearch not ready yet"
    fi
}

# =============================================================================
# Smoke Test
# =============================================================================

run_smoke_test() {
    print_section "Running Smoke Tests"

    local passed=0
    local failed=0
    local test_image="$PROJECT_DIR/test_images/bus.jpg"
    local face_image="$PROJECT_DIR/test_images/zidane.jpg"

    # Check if test images exist
    if [[ ! -f "$test_image" ]]; then
        log_warn "Test image not found: $test_image"
        log_info "Skipping image-based smoke tests"
        return 0
    fi

    # Test 1: Health endpoint
    log_info "Testing /health endpoint..."
    if curl -s localhost:${API_PORT}/health | grep -qE "ready|healthy"; then
        log_success "[1/8] Health check - PASS"
        passed=$((passed + 1))
    else
        log_error "[1/8] Health check - FAIL"
        failed=$((failed + 1))
    fi

    # Test 2: Object detection
    log_info "Testing object detection..."
    local detect_result
    detect_result=$(curl -s -X POST localhost:${API_PORT}/detect -F "image=@$test_image" 2>/dev/null)
    if echo "$detect_result" | grep -q "detections"; then
        local det_count
        det_count=$(echo "$detect_result" | grep -o '"class_name"' | wc -l)
        log_success "[2/8] Object detection - PASS ($det_count objects detected)"
        passed=$((passed + 1))
    else
        log_error "[2/8] Object detection - FAIL"
        echo "  Response: $(echo "$detect_result" | head -c 200)"
        failed=$((failed + 1))
    fi

    # Test 3: Face detection
    if [[ -f "$face_image" ]]; then
        log_info "Testing face detection..."
        local face_result
        face_result=$(curl -s -X POST localhost:${API_PORT}/faces/recognize -F "image=@$face_image" 2>/dev/null)
        if echo "$face_result" | grep -q "faces"; then
            local face_count
            face_count=$(echo "$face_result" | grep -o '"box"' | wc -l)
            log_success "[3/8] Face detection - PASS ($face_count faces detected)"
            passed=$((passed + 1))
        else
            log_error "[3/8] Face detection - FAIL"
            echo "  Response: $(echo "$face_result" | head -c 200)"
            failed=$((failed + 1))
        fi
    else
        log_info "[3/8] Face detection - SKIPPED (no test image)"
    fi

    # Test 4: CLIP embedding
    log_info "Testing CLIP embedding..."
    local embed_result
    embed_result=$(curl -s -X POST localhost:${API_PORT}/embed/image -F "image=@$test_image" 2>/dev/null)
    if echo "$embed_result" | grep -q "embedding"; then
        log_success "[4/8] CLIP embedding - PASS"
        passed=$((passed + 1))
    else
        log_error "[4/8] CLIP embedding - FAIL"
        echo "  Response: $(echo "$embed_result" | head -c 200)"
        failed=$((failed + 1))
    fi

    # Test 5: Full analysis
    log_info "Testing full analysis..."
    local analyze_result
    analyze_result=$(curl -s -X POST localhost:${API_PORT}/analyze -F "image=@$test_image" 2>/dev/null)
    if echo "$analyze_result" | grep -q "status"; then
        log_success "[5/8] Full analysis - PASS"
        passed=$((passed + 1))
    else
        log_error "[5/8] Full analysis - FAIL"
        echo "  Response: $(echo "$analyze_result" | head -c 200)"
        failed=$((failed + 1))
    fi

    # Test 6: Triton model count
    log_info "Checking Triton models..."
    local model_count
    model_count=$(curl -s -X POST localhost:${TRITON_HTTP_PORT}/v2/repository/index 2>/dev/null | grep -o '"name"' | wc -l)
    if [[ "$model_count" -ge 5 ]]; then
        log_success "[6/8] Triton models - PASS ($model_count models loaded)"
        passed=$((passed + 1))
    else
        log_warn "[6/8] Triton models - WARN (only $model_count models)"
    fi

    # Test 7: Ingest (indexes image into OpenSearch)
    log_info "Testing image ingestion..."
    local ingest_result
    ingest_result=$(curl -s -X POST localhost:${API_PORT}/ingest -F "image=@$test_image" 2>/dev/null)
    if echo "$ingest_result" | grep -q '"global": true'; then
        local face_count
        face_count=$(echo "$ingest_result" | grep -o '"faces": [0-9]*' | grep -o '[0-9]*')
        log_success "[7/8] Image ingest - PASS (indexed: global + ${face_count:-0} faces)"
        passed=$((passed + 1))
    else
        log_error "[7/8] Image ingest - FAIL"
        echo "  Response: $(echo "$ingest_result" | head -c 200)"
        failed=$((failed + 1))
    fi

    # Test 8: OpenSearch index verification
    log_info "Checking OpenSearch indexes..."
    local index_count
    index_count=$(curl -s localhost:${OPENSEARCH_PORT}/_cat/indices 2>/dev/null | grep -c "visual_search" || echo "0")
    if [[ "$index_count" -ge 1 ]]; then
        log_success "[8/8] OpenSearch indexes - PASS ($index_count visual_search indexes)"
        passed=$((passed + 1))
    else
        log_error "[8/8] OpenSearch indexes - FAIL (no visual_search indexes found)"
        failed=$((failed + 1))
    fi

    # Summary
    echo ""
    print_header "Smoke Test Results"
    local total=$((passed + failed))
    if [[ $failed -eq 0 ]]; then
        echo -e "${GREEN}All tests passed: $passed/$total${NC}"
    else
        echo -e "${YELLOW}Tests: $passed passed, $failed failed${NC}"
        echo ""
        log_info "Some tests failed. This may be because models are still loading."
        log_info "Wait a minute and run: make test-all"
    fi
}

# =============================================================================
# Summary
# =============================================================================

print_success_summary() {
    print_header "Setup Complete!"

    echo -e "${GREEN}OpenProcessor is ready to use!${NC}"
    echo ""
    echo "Services running:"
    echo "  API:         http://localhost:${API_PORT}"
    echo "  API Docs:    http://localhost:${API_PORT}/docs"
    echo "  Triton:      http://localhost:${TRITON_HTTP_PORT}"
    echo "  Grafana:     http://localhost:${GRAFANA_PORT} (admin/admin)"
    echo "  OpenSearch:  http://localhost:${OPENSEARCH_PORT}"
    echo ""
    echo "Quick test:"
    echo "  curl http://localhost:${API_PORT}/health"
    echo ""
    echo "Documentation:"
    echo "  README.md        - API overview and examples"
    echo "  INSTALLATION.md  - Detailed installation guide"
    echo "  CLAUDE.md        - Full technical reference"
    echo ""
    echo "Management:"
    echo "  ./scripts/openprocessor.sh status   - Check service status"
    echo "  ./scripts/openprocessor.sh logs     - View logs"
    echo "  ./scripts/openprocessor.sh restart  - Restart services"
    echo ""

    if [[ "$SHOW_CURATION_NEXT_STEPS" == "true" ]]; then
        print_curation_next_steps
    else
        log_info "Curation/labeling subsystem next steps: re-run with --curation, or see docs/CURATION.md"
    fi
}

# G-04: this script only ever brings up the core inference stack (detect/
# faces/embed/ocr). The curation/active-learning subsystem is a separate,
# opt-in stack (--profile curation) with its own prerequisites this
# script does NOT run for you -- print the exact next commands instead of
# leaving it a silent gap. See docs/CURATION.md for the full walkthrough.
print_curation_next_steps() {
    print_header "Curation / Labeling Subsystem — Next Steps"
    cat << EOF
The steps above set up core inference only. To also get curation
(active-learning labeling, clustering, region detection, training):

  1. Export the semantic-search encoder (required, not optional):
       make export-pe
     Then add pe_image_encoder to triton-server's --load-model list in
     docker-compose.yml (already listed by default in this repo) and
     restart Triton: make restart-triton

  2. Seed a class registry (there is no default one):
       Add classes via POST {API}/curation/projects/default/classes, or via the
       Cropwright UI's /classes page. See docs/CURATION.md "Quick start for
       a fresh install".

  3. VLM (auto-labeling / region verification) -- pick one:
       - docker compose --profile vlm up -d      # runs vLLM in this compose
       - or set OP_VLM_URL in .env to a VLM you already run elsewhere
     See docs/CURATION.md "Choosing a VLM" and env.template's VLM section.

  4. Segmenter (region-of-interest cascade), if your use case needs one:
       HF_TOKEN=<your-hf-token-with-SAM3-access> \\
         docker compose --profile curation --profile segmenter up -d
     See docs/CURATION.md "Workers and compose profiles"
     and env.template's Segmenter section.

  5. Start the curation workers:
       docker compose --profile curation up -d

  6. Bring up Cropwright (the labeling UI) against this API -- see
     docs/CURATION.md "Wiring up Cropwright" for the exact env vars
     (API_UPSTREAM, PUBLIC_API_PREFIX, docker network name).

Full walkthrough: docs/CURATION.md
EOF
}
