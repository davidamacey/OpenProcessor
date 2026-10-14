#!/bin/bash
# =============================================================================
# PP-OCRv5 Model Export Script for Triton Inference Server
# =============================================================================
#
# This script automates the full export pipeline for PP-OCRv5 OCR models:
# 1. Download ONNX models (detection + recognition)
# 2. Convert to TensorRT engines with optimal settings
# 3. Setup dictionary files
# 4. Restart Triton to load new models
#
# Usage:
#   ./scripts/export_paddleocr.sh [command]
#
# Commands:
#   all        - Run complete export pipeline (default)
#   download   - Download ONNX models only
#   trt        - Convert ONNX to TensorRT only
#   det        - Export detection model only
#   rec        - Export recognition model only
#   restart    - Restart Triton server
#   test       - Run OCR test
#   clean      - Remove all OCR model files
#   status     - Check model status
#
# Requirements:
#   - Docker with triton-server and api containers running
#   - NVIDIA GPU with sufficient memory (4GB+ for TRT build)
#
# =============================================================================

# Paths, shapes and ports are read by the sourced lib/paddleocr_*.sh functions.
# shellcheck disable=SC2034
set -eo pipefail  # Exit on error; propagate pipe failures

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/models"
PYTORCH_MODELS_DIR="$PROJECT_DIR/pytorch_models/paddleocr"

# shellcheck source=lib/ports.sh
source "${SCRIPT_DIR}/lib/ports.sh"

# resolve from THIS deployment's .env instead of hardcoding
# localhost:4600/4603 -- a remapped port (second isolated stack) must
# still export/test against the right containers.
API_PORT="$(env_port API_PORT 4603)"
TRITON_HTTP_PORT="$(env_port TRITON_HTTP_PORT 4600)"

# Model paths
DET_ONNX_NAME="ppocr_det_v5_mobile.onnx"
REC_ONNX_NAME="ppocr_rec_v5_mobile.onnx"
DET_PLAN="$MODELS_DIR/paddleocr_det_trt/1/model.plan"
REC_PLAN="$MODELS_DIR/paddleocr_rec_trt/1/model.plan"
DICT_FILE="$MODELS_DIR/paddleocr_rec_trt/ppocrv5_dict.txt"

# TensorRT configuration
# CRITICAL: Use --memPoolSize=workspace:8G syntax (not --workspace)
TRT_WORKSPACE="8G"  # 8GB workspace for TRT optimization (required for dynamic shapes)
TRT_FP16=""   # trtexec --fp16 was removed in TensorRT 11 (strongly-typed builds follow the ONNX dtypes)

# Detection model shapes (H,W are multiples of 32, max 960)
DET_MIN_SHAPES="x:1x3x32x32"
DET_OPT_SHAPES="x:1x3x736x736"
DET_MAX_SHAPES="x:4x3x960x960"

# Recognition model shapes (height=48, width=48-2048 dynamic)
# NOTE: min width=48 (not 8) matches the export_paddleocr_rec.py working config
REC_MIN_SHAPES="x:1x3x48x48"
REC_OPT_SHAPES="x:32x3x48x320"
REC_MAX_SHAPES="x:64x3x48x2048"

# shellcheck source=lib/paddleocr_export.sh
source "${SCRIPT_DIR}/lib/paddleocr_export.sh"
# shellcheck source=lib/paddleocr_config.sh
source "${SCRIPT_DIR}/lib/paddleocr_config.sh"
# shellcheck source=lib/paddleocr_ops.sh
source "${SCRIPT_DIR}/lib/paddleocr_ops.sh"

# =============================================================================
# Full Pipeline
# =============================================================================

run_full_pipeline() {
    echo ""
    echo "=========================================="
    echo "PP-OCRv5 Full Export Pipeline"
    echo "=========================================="
    echo ""

    # 1. Download detection ONNX + generate English recognition ONNX
    log_info "Step 1/5: Downloading/generating ONNX models..."
    download_models || return 1

    # 2. Setup dictionary
    log_info "Step 2/5: Setting up dictionary..."
    setup_dictionary

    # 3. Export to TensorRT (uses docker compose run, no running Triton needed)
    log_info "Step 3/5: Exporting to TensorRT..."
    export_all_trt || return 1

    # 4. Create configs
    log_info "Step 4/5: Creating Triton configs..."
    create_configs

    # 5. Cleanup temporary ONNX files from models dir
    log_info "Step 5/5: Cleaning up..."
    rm -f "$MODELS_DIR/$DET_ONNX_NAME"
    rm -f "$MODELS_DIR/$REC_ONNX_NAME"

    echo ""
    log_success "Export pipeline complete!"
    show_status
}

# =============================================================================
# Main
# =============================================================================

show_help() {
    echo "PP-OCRv5 Model Export Script"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  all        Run complete export pipeline (default)"
    echo "  download   Download ONNX models only"
    echo "  trt        Convert ONNX to TensorRT only"
    echo "  det        Export detection model only"
    echo "  rec        Export recognition model only"
    echo "  config     Create Triton configs only"
    echo "  dict       Setup dictionary file only"
    echo "  restart    Restart Triton server"
    echo "  reload     Reload OCR models without restart"
    echo "  test       Run OCR test"
    echo "  status     Check model status"
    echo "  clean      Remove all OCR model files"
    echo "  help       Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 all           # Full pipeline: download + TRT + configs + restart"
    echo "  $0 trt           # Convert existing ONNX to TensorRT"
    echo "  $0 det           # Export detection model only"
    echo "  $0 status        # Check current status"
    echo ""
}

main() {
    local command="${1:-all}"

    case "$command" in
        all)
            run_full_pipeline
            ;;
        download)
            download_models
            ;;
        trt)
            export_all_trt
            ;;
        det)
            unload_models_for_memory
            export_detection
            ;;
        rec)
            unload_models_for_memory
            export_recognition
            ;;
        config)
            create_configs
            ;;
        dict)
            setup_dictionary
            ;;
        restart)
            restart_triton
            ;;
        reload)
            reload_models
            ;;
        test)
            run_test
            ;;
        status)
            show_status
            ;;
        clean)
            clean_all
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
