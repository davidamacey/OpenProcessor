#!/bin/bash
# =============================================================================
# PE-Core-L14-336 image encoder: ONNX Runtime fallback (Path 2)
# =============================================================================
#
# Use this when export/build_pe_trt.sh (Path 1) fails -- PE's attention
# pooling has hit ops that some TensorRT releases do not support. Triton
# serves the raw ONNX through its onnxruntime_onnx backend on the CUDA
# execution provider: identical numerics, lower throughput, no engine build.
#
# No GPU, no trtexec and no TensorRT are needed to run this script; it just
# installs the ONNX and writes the matching config.pbtxt.
#
# Environment:
#   ONNX_PATH     ONNX produced by export/export_pe_image_encoder.py
#                 (default: <repo>/pytorch_models/pe_image_encoder.onnx)
#   MODELS_DIRS   Space-separated Triton model repositories to install into
#                 (default: <repo>/models)
#   MAX_BATCH     config.pbtxt max_batch_size (default: 32)
#   IMAGE_SIZE    Input resolution (default: 336)
#   PYTHON        Interpreter used to render config.pbtxt (default:
#                 <repo>/.venv/bin/python if present, else python3)
#
# Usage:
#   bash export/build_pe_ort_fallback.sh
#   ONNX_PATH=/tmp/pe.onnx bash export/build_pe_ort_fallback.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

ONNX_PATH="${ONNX_PATH:-$REPO_DIR/pytorch_models/pe_image_encoder.onnx}"
MODELS_DIRS="${MODELS_DIRS:-$REPO_DIR/models}"
MAX_BATCH="${MAX_BATCH:-32}"
IMAGE_SIZE="${IMAGE_SIZE:-336}"
MODEL_NAME="pe_image_encoder"

if [ -x "$REPO_DIR/.venv/bin/python" ]; then
    PYTHON="${PYTHON:-$REPO_DIR/.venv/bin/python}"
else
    PYTHON="${PYTHON:-python3}"
fi

log() { echo "[INFO]  $1"; }
err() { echo "[ERROR] $1" >&2; }

if [ ! -f "$ONNX_PATH" ]; then
    err "ONNX not found: $ONNX_PATH"
    err "Run export/export_pe_image_encoder.py first."
    exit 1
fi

log "Installing $ONNX_PATH as the $MODEL_NAME Triton model (Path 2 / ORT)"

for models_dir in $MODELS_DIRS; do
    mkdir -p "$models_dir/$MODEL_NAME/1"
    install -m 0644 "$ONNX_PATH" "$models_dir/$MODEL_NAME/1/model.onnx"
    # Path 1 and Path 2 must never coexist -- Triton loads whichever the
    # platform names and silently ignores the other artifact.
    rm -f "$models_dir/$MODEL_NAME/1/model.plan"
    log "installed: $models_dir/$MODEL_NAME/1/model.onnx"

    "$PYTHON" "$SCRIPT_DIR/export_pe_image_encoder.py" \
        --config-only \
        --platform onnxruntime_onnx \
        --models-dir "$models_dir" \
        --max-batch "$MAX_BATCH" \
        --image-size "$IMAGE_SIZE"
done

log "Done. Restart or reload Triton to pick up $MODEL_NAME."
log "  curl -X POST localhost:4600/v2/repository/models/$MODEL_NAME/load"
log ""
log "Reminder: dynamic_batching requires a DYNAMIC leading dimension in the"
log "ONNX. A batch-1 trace bakes it static and Triton will reject batches > 1."
log "export_pe_image_encoder.py traces with a batch-2 dummy and reports this"
log "in its validation step -- re-export rather than editing config.pbtxt."
