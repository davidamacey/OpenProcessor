#!/bin/bash
# =============================================================================
# PE-Core-L14-336 image encoder: TensorRT engine build (Path 1)
# =============================================================================
#
# Step 1: export/export_pe_image_encoder.py produces the ONNX.
# Step 2: this script runs trtexec over it, installs the resulting plan as
#         <models dir>/pe_image_encoder/1/model.plan, and writes the matching
#         config.pbtxt (platform: tensorrt_plan).
#
# If the engine build fails -- PE's attention pooling has hit ops that some
# TensorRT releases do not support -- fall back to Path 2:
#     export/build_pe_ort_fallback.sh
#
# trtexec lives in the Triton container, not in the API image. With Docker
# available this script shells out to `docker compose run --rm --no-deps
# triton-server trtexec`; set TRTEXEC=trtexec to use a host binary instead.
#
# Environment:
#   ONNX_PATH        ONNX produced by the exporter (default:
#                    <repo>/pytorch_models/pe_image_encoder.onnx)
#   MODELS_DIRS      Space-separated Triton model repositories to install
#                    into (default: <repo>/models)
#   MAX_BATCH        Engine + config max batch size (default: 32)
#   OPT_BATCH        Engine optimization-profile opt batch (default: 8)
#   IMAGE_SIZE       Input resolution (default: 336)
#   WORKSPACE        trtexec workspace memory pool (default: 8G)
#   PYTHON           Interpreter used to render config.pbtxt (default:
#                    <repo>/.venv/bin/python if present, else python3)
#   TRTEXEC          trtexec invocation override (default: docker compose)
#
# Usage:
#   bash export/build_pe_trt.sh
#   ONNX_PATH=/tmp/pe.onnx MAX_BATCH=16 bash export/build_pe_trt.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

ONNX_PATH="${ONNX_PATH:-$REPO_DIR/pytorch_models/pe_image_encoder.onnx}"
MODELS_DIRS="${MODELS_DIRS:-$REPO_DIR/models}"
PLAN_TMP="${PLAN_TMP:-$(dirname "$ONNX_PATH")/pe_image_encoder.plan}"
MAX_BATCH="${MAX_BATCH:-32}"
OPT_BATCH="${OPT_BATCH:-8}"
IMAGE_SIZE="${IMAGE_SIZE:-336}"
WORKSPACE="${WORKSPACE:-8G}"
MODEL_NAME="pe_image_encoder"
INPUT_TENSOR="images"

if [ -x "$REPO_DIR/.venv/bin/python" ]; then
    PYTHON="${PYTHON:-$REPO_DIR/.venv/bin/python}"
else
    PYTHON="${PYTHON:-python3}"
fi

log()  { echo "[INFO]  $1"; }
err()  { echo "[ERROR] $1" >&2; }

if [ ! -f "$ONNX_PATH" ]; then
    err "ONNX not found: $ONNX_PATH"
    err "Run export/export_pe_image_encoder.py first."
    exit 1
fi

# -----------------------------------------------------------------------------
# bake FP16 into the ONNX (TRT 11 typed builds have no --fp16 flag;
# reduced precision must live in the graph -- trt_utils.bake_fp16_onnx, the
# same helper export_face_recognition.py / export_scrfd.py /
# export_mobileclip_image_encoder.py call). PE was shipping FP32-only
# (~2x the VRAM, slower) because this build script never called it.
#
# FP16 is required. If the bake falls back to FP32 (trt_utils.py reports
# 'fp32') or fails, or the FP16 ONNX fails to build in trtexec below, this
# script FAILS (exit 4) rather than ship an FP32 engine nobody asked for.
# ALLOW_FP32=1 is the explicit opt-in to an FP32 engine (it says so loudly);
# BAKE_FP16=0 builds straight from the FP32 ONNX and needs ALLOW_FP32=1 too.
# -----------------------------------------------------------------------------
BAKE_FP16="${BAKE_FP16:-1}"
ALLOW_FP32="${ALLOW_FP32:-0}"

# _fp32_refused WHY: stop unless the caller opted in to FP32.
_fp32_refused() {
    if [ "$ALLOW_FP32" != "1" ]; then
        err "$1"
        err "Refusing to build an FP32 engine (about twice the GPU memory, slower)."
        err "Fix the cause, or set ALLOW_FP32=1 to accept an FP32 engine knowingly."
        exit 4
    fi
    err "WARNING: $1 -- continuing with an FP32 engine because ALLOW_FP32=1."
}
BUILD_ONNX_PATH="$ONNX_PATH"
PE_PRECISION="fp32"
if [ "$BAKE_FP16" = "1" ]; then
    log "Baking FP16 into the ONNX (trt_utils.bake_fp16_onnx)..."
    FP16_ONNX_PATH="$(dirname "$ONNX_PATH")/$(basename "$ONNX_PATH" .onnx).fp16.onnx"
    BAKE_OUT="$(mktemp)"
    BAKE_ERR="$(mktemp)"
    if "$PYTHON" "$SCRIPT_DIR/trt_utils.py" "$ONNX_PATH" "$FP16_ONNX_PATH" \
            >"$BAKE_OUT" 2>"$BAKE_ERR"; then
        BUILD_ONNX_PATH="$(cat "$BAKE_OUT")"
        if [ "$(cat "$BAKE_ERR")" = "fp16" ]; then
            PE_PRECISION="fp16"
            log "  FP16 ONNX: $BUILD_ONNX_PATH"
        else
            _fp32_refused "FP16 bake fell back to FP32: $(cat "$BAKE_ERR")"
        fi
    else
        _fp32_refused "FP16 bake step failed: $(cat "$BAKE_ERR")"
        BUILD_ONNX_PATH="$ONNX_PATH"
    fi
    rm -f "$BAKE_OUT" "$BAKE_ERR"
else
    _fp32_refused "BAKE_FP16=0 asks for an FP32 build"
fi

# -----------------------------------------------------------------------------
# Resolve how to call trtexec
# -----------------------------------------------------------------------------
if [ -n "${TRTEXEC:-}" ]; then
    TRTEXEC_CMD=("$TRTEXEC")
elif command -v trtexec >/dev/null 2>&1; then
    TRTEXEC_CMD=(trtexec)
elif command -v docker >/dev/null 2>&1; then
    TRTEXEC_CMD=(docker compose run --rm --no-deps -T triton-server trtexec)
else
    err "Neither trtexec nor docker is available."
    err "Run this inside the Triton container, or use Path 2:"
    err "  export/build_pe_ort_fallback.sh"
    exit 2
fi

# The engine takes uint8 pixels and applies PE-Core's own Normalize([0.5]*3, [0.5]*3) itself
# (export/uint8_input.py), so the request carries a quarter of the bytes of the float tensor.
PE_NORMALIZE="--mean 0.5 0.5 0.5 --std 0.5 0.5 0.5"

# _stage_and_resolve_args ONNX_TO_BUILD -- sets ONNX_ARG/PLAN_ARG/PLAN_TMP
# (and STAGE_DIR when docker-staging) for whichever ONNX path (FP16 or
# FP32) is about to be built, since the docker path needs the file copied
# into the mounted models dir under a fixed name each time. The uint8 input
# is added after the FP16 bake, which keeps the graph I/O float32.
_stage_and_resolve_args() {
    local src_onnx="$1"
    # shellcheck disable=SC2086
    src_onnx="$("$PYTHON" "$SCRIPT_DIR/uint8_input.py" "$src_onnx" "${src_onnx%.onnx}.u8.onnx" $PE_NORMALIZE)" || {
        err "could not add the uint8 input to the engine ONNX; config.pbtxt declares uint8, so Triton would refuse the engine"
        exit 4
    }
    if [ -n "${TRTEXEC:-}" ] || command -v trtexec >/dev/null 2>&1; then
        ONNX_ARG="$src_onnx"
        PLAN_ARG="$PLAN_TMP"
    else
        STAGE_DIR="$(printf '%s\n' "$MODELS_DIRS" | awk '{print $1}')"
        cp "$src_onnx" "$STAGE_DIR/$MODEL_NAME.onnx"
        ONNX_ARG="/models/$MODEL_NAME.onnx"
        PLAN_ARG="/models/$MODEL_NAME.plan"
        PLAN_TMP="$STAGE_DIR/$MODEL_NAME.plan"
    fi
}

_stage_and_resolve_args "$BUILD_ONNX_PATH"

# -----------------------------------------------------------------------------
# Build
# -----------------------------------------------------------------------------
log "Building the $MODEL_NAME TensorRT engine from $BUILD_ONNX_PATH ($PE_PRECISION)"
log "  profile: min=1 opt=$OPT_BATCH max=$MAX_BATCH at ${IMAGE_SIZE}x${IMAGE_SIZE}"
log "  workspace: $WORKSPACE"
log "  this typically takes several minutes"

# NOTE: no --fp16 flag. TensorRT 11 builds are strongly typed and follow the
# ONNX dtypes; the flag was removed (see export/trt_utils.py::enable_fp16).
# Precision comes entirely from which ONNX (FP16-baked or original FP32) is
# passed in above.
if ! "${TRTEXEC_CMD[@]}" \
        --onnx="$ONNX_ARG" \
        --saveEngine="$PLAN_ARG" \
        --minShapes="$INPUT_TENSOR:1x3x${IMAGE_SIZE}x${IMAGE_SIZE}" \
        --optShapes="$INPUT_TENSOR:${OPT_BATCH}x3x${IMAGE_SIZE}x${IMAGE_SIZE}" \
        --maxShapes="$INPUT_TENSOR:${MAX_BATCH}x3x${IMAGE_SIZE}x${IMAGE_SIZE}" \
        --memPoolSize=workspace:"$WORKSPACE" \
        --skipInference ; then
    if [ "$PE_PRECISION" = "fp16" ]; then
        err ""
        _fp32_refused "trtexec failed to build the FP16 engine"
        PE_PRECISION="fp32"
        _stage_and_resolve_args "$ONNX_PATH"
        if ! "${TRTEXEC_CMD[@]}" \
                --onnx="$ONNX_ARG" \
                --saveEngine="$PLAN_ARG" \
                --minShapes="$INPUT_TENSOR:1x3x${IMAGE_SIZE}x${IMAGE_SIZE}" \
                --optShapes="$INPUT_TENSOR:${OPT_BATCH}x3x${IMAGE_SIZE}x${IMAGE_SIZE}" \
                --maxShapes="$INPUT_TENSOR:${MAX_BATCH}x3x${IMAGE_SIZE}x${IMAGE_SIZE}" \
                --memPoolSize=workspace:"$WORKSPACE" \
                --skipInference ; then
            err ""
            err "trtexec failed on the FP32 fallback too. PE's attention pooling may use"
            err "ops this TensorRT release does not support. Fall back to Path 2:"
            err "  ONNX_PATH=$ONNX_PATH bash export/build_pe_ort_fallback.sh"
            exit 3
        fi
    else
        err ""
        err "trtexec failed to build the engine. PE's attention pooling may use"
        err "ops this TensorRT release does not support. Fall back to Path 2:"
        err "  ONNX_PATH=$ONNX_PATH bash export/build_pe_ort_fallback.sh"
        exit 3
    fi
fi

if [ ! -s "$PLAN_TMP" ]; then
    err "trtexec reported success but produced no engine at $PLAN_TMP"
    exit 3
fi
log "Engine built ($PE_PRECISION): $PLAN_TMP ($(du -h "$PLAN_TMP" | cut -f1))"

# -----------------------------------------------------------------------------
# Install into every configured model repository
# -----------------------------------------------------------------------------
for models_dir in $MODELS_DIRS; do
    mkdir -p "$models_dir/$MODEL_NAME/1"
    install -m 0644 "$PLAN_TMP" "$models_dir/$MODEL_NAME/1/model.plan"
    # Path 1 and Path 2 must never be present at once -- Triton would load
    # whichever the platform names and silently ignore the other.
    rm -f "$models_dir/$MODEL_NAME/1/model.onnx"
    log "installed: $models_dir/$MODEL_NAME/1/model.plan"

    "$PYTHON" "$SCRIPT_DIR/export_pe_image_encoder.py" \
        --config-only \
        --platform tensorrt_plan \
        --models-dir "$models_dir" \
        --max-batch "$MAX_BATCH" \
        --image-size "$IMAGE_SIZE"
done

# Clean up the staging copy used for the container round-trip.
if [ -n "${STAGE_DIR:-}" ]; then
    rm -f "$STAGE_DIR/$MODEL_NAME.onnx" "$STAGE_DIR/$MODEL_NAME.plan"
fi

log "Done. Restart or reload Triton to pick up $MODEL_NAME."
log "  curl -X POST localhost:4600/v2/repository/models/$MODEL_NAME/load"
