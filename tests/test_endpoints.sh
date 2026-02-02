#!/usr/bin/env bash
# =============================================================================
# Endpoint Integration Tests
#
# Tests all API endpoints and models. Exits with non-zero code on any failure.
#
# Usage:
#   ./tests/test_endpoints.sh              # Run all tests
#   ./tests/test_endpoints.sh detect       # Run only detection tests
#   ./tests/test_endpoints.sh faces        # Run only face tests
#   ./tests/test_endpoints.sh embed        # Run only embedding tests
#   ./tests/test_endpoints.sh ocr          # Run only OCR tests
#   ./tests/test_endpoints.sh analyze      # Run only analyze tests
#   ./tests/test_endpoints.sh models       # Run only model status checks
#   ./tests/test_endpoints.sh health       # Run only health check
# =============================================================================

set -euo pipefail

API_URL="${API_URL:-http://localhost:4603}"
TRITON_URL="${TRITON_URL:-http://localhost:4600}"
TEST_IMAGE="${TEST_IMAGE:-test_images/zidane.jpg}"
TEST_IMAGE_BUS="${TEST_IMAGE_BUS:-test_images/bus.jpg}"
TEST_IMAGE_OCR="${TEST_IMAGE_OCR:-test_images/ocr-synthetic/caution_sign.jpg}"

# Auto-download test images if missing
download_if_missing() {
    local file="$1" url="$2"
    if [ ! -f "$file" ]; then
        echo "Downloading $(basename "$file")..."
        mkdir -p "$(dirname "$file")"
        curl -sL "$url" -o "$file" || { echo "WARNING: Failed to download $file"; return 1; }
    fi
}
download_if_missing "$TEST_IMAGE_BUS" "https://ultralytics.com/images/bus.jpg"
download_if_missing "$TEST_IMAGE" "https://ultralytics.com/images/zidane.jpg"

PASS=0
FAIL=0
SKIP=0

# Colors (if terminal supports it)
if [ -t 1 ]; then
    GREEN='\033[0;32m'
    RED='\033[0;31m'
    YELLOW='\033[0;33m'
    NC='\033[0m'
else
    GREEN='' RED='' YELLOW='' NC=''
fi

pass() { PASS=$((PASS + 1)); echo -e "  ${GREEN}PASS${NC}: $1"; }
fail() { FAIL=$((FAIL + 1)); echo -e "  ${RED}FAIL${NC}: $1"; }
skip() { SKIP=$((SKIP + 1)); echo -e "  ${YELLOW}SKIP${NC}: $1"; }

# Helper: make a request and validate JSON field
# Usage: check_endpoint "label" "url" "method" "form_args" "jq_expr" "expected_pattern"
check_endpoint() {
    local label="$1"
    local url="$2"
    local form_args="$3"
    local jq_expr="$4"
    local expected="$5"

    local response
    response=$(curl -s -w "\n%{http_code}" -X POST "$url" $form_args 2>/dev/null) || {
        fail "$label (curl error)"
        return
    }

    local http_code
    http_code=$(echo "$response" | tail -1)
    local body
    body=$(echo "$response" | sed '$d')

    if [ "$http_code" != "200" ]; then
        fail "$label (HTTP $http_code)"
        return
    fi

    local value
    value=$(echo "$body" | jq -r "$jq_expr" 2>/dev/null) || {
        fail "$label (jq parse error)"
        return
    }

    if echo "$value" | grep -qE "$expected"; then
        pass "$label = $value"
    else
        fail "$label: got '$value', expected pattern '$expected'"
    fi
}

# =============================================================================
# Health & Models
# =============================================================================
test_health() {
    echo ""
    echo "=== Health Check ==="

    local resp
    resp=$(curl -s "$API_URL/health" 2>/dev/null) || { fail "health endpoint unreachable"; return; }
    local status
    status=$(echo "$resp" | jq -r '.status' 2>/dev/null)
    if [ "$status" = "healthy" ]; then
        pass "API healthy"
    else
        fail "API status: $status"
    fi
}

test_models() {
    echo ""
    echo "=== Triton Model Status ==="

    local expected_models="yolov11_small_trt_end2end scrfd_10g_bnkps arcface_w600k_r50 mobileclip2_s2_image_encoder mobileclip2_s2_text_encoder paddleocr_det_trt paddleocr_rec_trt ocr_pipeline"

    local repo
    repo=$(curl -s -X POST "$TRITON_URL/v2/repository/index" 2>/dev/null) || {
        fail "Triton repository unreachable"
        return
    }

    for model in $expected_models; do
        local state
        state=$(echo "$repo" | jq -r ".[] | select(.name==\"$model\") | .state // \"NOT_FOUND\"" 2>/dev/null)
        if [ "$state" = "READY" ]; then
            pass "$model: READY"
        else
            fail "$model: $state"
        fi
    done

    # Verify SCRFD is loaded (face detection)
    local scrfd_state
    scrfd_state=$(echo "$repo" | jq -r '.[] | select(.name=="scrfd_10g_bnkps") | .state // "NOT_FOUND"' 2>/dev/null)
    if [ "$scrfd_state" = "READY" ]; then
        pass "scrfd_10g_bnkps: READY"
    else
        fail "scrfd_10g_bnkps: $scrfd_state"
    fi
}

# =============================================================================
# Detection
# =============================================================================
test_detect() {
    echo ""
    echo "=== Object Detection ==="

    if [ ! -f "$TEST_IMAGE_BUS" ]; then
        skip "detect: $TEST_IMAGE_BUS not found"
        return
    fi

    check_endpoint \
        "/v1/detect: detections found" \
        "$API_URL/v1/detect" \
        "-F image=@$TEST_IMAGE_BUS" \
        ".num_detections // .detections | length" \
        "^[1-9]"
}

# =============================================================================
# Face Detection & Recognition
# =============================================================================
test_faces() {
    echo ""
    echo "=== Face Detection & Recognition (SCRFD + ArcFace) ==="

    if [ ! -f "$TEST_IMAGE" ]; then
        skip "faces: $TEST_IMAGE not found"
        return
    fi

    # /v1/faces/detect
    check_endpoint \
        "/v1/faces/detect: faces found" \
        "$API_URL/v1/faces/detect" \
        "-F image=@$TEST_IMAGE -F confidence=0.3" \
        ".num_faces" \
        "^[1-9]"

    # Check landmarks are non-zero
    local detect_resp
    detect_resp=$(curl -s -X POST "$API_URL/v1/faces/detect" \
        -F "image=@$TEST_IMAGE" -F "confidence=0.3" 2>/dev/null)
    local lm_count
    lm_count=$(echo "$detect_resp" | jq '[.faces[0].landmarks[] | select(. > 0.001)] | length' 2>/dev/null)
    if [ "$lm_count" = "10" ]; then
        pass "/v1/faces/detect: all 10 landmarks non-zero"
    else
        fail "/v1/faces/detect: only $lm_count/10 landmarks non-zero"
    fi

    # /v1/faces/recognize
    check_endpoint \
        "/v1/faces/recognize: embedding dim" \
        "$API_URL/v1/faces/recognize" \
        "-F image=@$TEST_IMAGE -F confidence=0.3" \
        ".embeddings[0] | length" \
        "^512$"

    # /v1/faces/verify (same image = should match)
    local verify_resp
    verify_resp=$(curl -s -X POST "$API_URL/v1/faces/verify" \
        -F "image1=@$TEST_IMAGE" -F "image2=@$TEST_IMAGE" 2>/dev/null)
    local match
    match=$(echo "$verify_resp" | jq -r '.match' 2>/dev/null)
    if [ "$match" = "true" ]; then
        pass "/v1/faces/verify: same image matches"
    else
        fail "/v1/faces/verify: same image should match, got match=$match"
    fi
}

# =============================================================================
# Embeddings
# =============================================================================
test_embed() {
    echo ""
    echo "=== CLIP Embeddings ==="

    if [ ! -f "$TEST_IMAGE_BUS" ]; then
        skip "embed: $TEST_IMAGE_BUS not found"
        return
    fi

    check_endpoint \
        "/v1/embed/image: embedding dim" \
        "$API_URL/v1/embed/image" \
        "-F image=@$TEST_IMAGE_BUS" \
        ".embedding | length" \
        "^512$"

    # Text embedding
    local text_resp
    text_resp=$(curl -s -X POST "$API_URL/v1/embed/text" \
        -H "Content-Type: application/json" \
        -d '{"text": "a bus on the street"}' 2>/dev/null)
    local text_dim
    text_dim=$(echo "$text_resp" | jq '.embedding | length' 2>/dev/null)
    if [ "$text_dim" = "512" ]; then
        pass "/v1/embed/text: embedding dim = 512"
    else
        fail "/v1/embed/text: got dim=$text_dim, expected 512"
    fi
}

# =============================================================================
# OCR
# =============================================================================
test_ocr() {
    echo ""
    echo "=== OCR (PP-OCRv5) ==="

    if [ ! -f "$TEST_IMAGE_OCR" ]; then
        skip "ocr: $TEST_IMAGE_OCR not found"
        return
    fi

    check_endpoint \
        "/v1/ocr/predict: texts found" \
        "$API_URL/v1/ocr/predict" \
        "-F image=@$TEST_IMAGE_OCR" \
        ".num_texts // (.texts | length)" \
        "^[1-9]"
}

# =============================================================================
# Analyze (combined pipeline)
# =============================================================================
test_analyze() {
    echo ""
    echo "=== Combined Analysis ==="

    if [ ! -f "$TEST_IMAGE" ]; then
        skip "analyze: $TEST_IMAGE not found"
        return
    fi

    local resp
    resp=$(curl -s -X POST "$API_URL/v1/analyze" -F "image=@$TEST_IMAGE" 2>/dev/null)
    local http_ok=$?

    if [ $http_ok -ne 0 ]; then
        fail "/v1/analyze: curl error"
        return
    fi

    local dets
    dets=$(echo "$resp" | jq '.num_detections // (.detections | length)' 2>/dev/null)
    if [ "$dets" -ge 1 ] 2>/dev/null; then
        pass "/v1/analyze: $dets detections"
    else
        fail "/v1/analyze: no detections"
    fi

    local faces
    faces=$(echo "$resp" | jq '.num_faces // (.faces | length)' 2>/dev/null)
    if [ "$faces" -ge 1 ] 2>/dev/null; then
        pass "/v1/analyze: $faces faces"
    else
        fail "/v1/analyze: no faces"
    fi

    local lm_count
    lm_count=$(echo "$resp" | jq '[.faces[0].landmarks[] | select(. > 0.001)] | length' 2>/dev/null)
    if [ "$lm_count" = "10" ]; then
        pass "/v1/analyze: face landmarks non-zero"
    else
        fail "/v1/analyze: only $lm_count/10 landmarks non-zero"
    fi
}

# =============================================================================
# Main
# =============================================================================
echo "============================================="
echo " Visual Search API - Endpoint Tests"
echo " API: $API_URL"
echo " Triton: $TRITON_URL"
echo "============================================="

target="${1:-all}"

case "$target" in
    health)  test_health ;;
    models)  test_models ;;
    detect)  test_detect ;;
    faces)   test_faces ;;
    embed)   test_embed ;;
    ocr)     test_ocr ;;
    analyze) test_analyze ;;
    all)
        test_health
        test_models
        test_detect
        test_faces
        test_embed
        test_ocr
        test_analyze
        ;;
    *)
        echo "Unknown target: $target"
        echo "Valid targets: all, health, models, detect, faces, embed, ocr, analyze"
        exit 1
        ;;
esac

echo ""
echo "============================================="
echo " Results: ${GREEN}$PASS passed${NC}, ${RED}$FAIL failed${NC}, ${YELLOW}$SKIP skipped${NC}"
echo "============================================="

exit $FAIL
