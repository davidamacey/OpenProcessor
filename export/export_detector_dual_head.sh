#!/bin/bash
# =============================================================================
# Dual-head detector: TensorRT engine build + model-repository install
# =============================================================================
#
# Companion to export/export_detector_dual_head.py for environments that
# have `trtexec` but not the TensorRT Python bindings (the triton-server
# container, for example). Step 1 produces the two-output ONNX:
#
#   python export/export_detector_dual_head.py \
#       --weights <checkpoint.pt> --triton-name <name> --imgsz <size>
#
# Step 2 (this script) builds the .plan and installs it as
# <model repo>/<name>/1/model.plan for every repository given.
#
# Usage:
#   export/export_detector_dual_head.sh --onnx <path> --name <triton-name> [options]
#
# Options (all have env-var equivalents, shown in brackets):
#   --onnx PATH          Two-output ONNX produced by step 1      [ONNX_PATH]
#   --name NAME          Triton model name = model dir name      [MODEL_NAME]
#   --model-repo DIR     Model repository; repeatable            [MODEL_REPOS]
#   --input-size N       Network input size (default 640)        [INPUT_SIZE]
#   --min-batch N        Min profile batch (default 1)           [MIN_BATCH]
#   --opt-batch N        Opt profile batch (default 4)           [OPT_BATCH]
#   --max-batch N        Max profile batch (default 8)           [MAX_BATCH]
#   --input-name NAME    Input tensor name (default images)      [INPUT_NAME]
#   --workspace-mb N     trtexec workspace in MB (default 4096)  [WORKSPACE_MB]
#   --plan PATH          Intermediate .plan path                 [PLAN_TMP]
#   -h, --help           Show this help
#
# Exit codes: 1 bad arguments · 2 trtexec missing · 3 ONNX missing
# =============================================================================
set -euo pipefail

ONNX_PATH="${ONNX_PATH:-}"
MODEL_NAME="${MODEL_NAME:-}"
INPUT_SIZE="${INPUT_SIZE:-640}"
MIN_BATCH="${MIN_BATCH:-1}"
OPT_BATCH="${OPT_BATCH:-4}"
MAX_BATCH="${MAX_BATCH:-8}"
INPUT_NAME="${INPUT_NAME:-images}"
WORKSPACE_MB="${WORKSPACE_MB:-4096}"
PLAN_TMP="${PLAN_TMP:-}"

# Repositories to install into. MODEL_REPOS is a whitespace-separated list;
# --model-repo appends. Defaults to ./models relative to the repo root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
read -r -a REPOS <<<"${MODEL_REPOS:-}"

usage() {
    sed -n '2,33p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

die() {
    echo "ERROR: $1" >&2
    echo "Run with --help for usage." >&2
    exit "${2:-1}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --onnx) ONNX_PATH="${2:-}"; shift 2 ;;
        --name) MODEL_NAME="${2:-}"; shift 2 ;;
        --model-repo) REPOS+=("${2:-}"); shift 2 ;;
        --input-size) INPUT_SIZE="${2:-}"; shift 2 ;;
        --min-batch) MIN_BATCH="${2:-}"; shift 2 ;;
        --opt-batch) OPT_BATCH="${2:-}"; shift 2 ;;
        --max-batch) MAX_BATCH="${2:-}"; shift 2 ;;
        --input-name) INPUT_NAME="${2:-}"; shift 2 ;;
        --workspace-mb) WORKSPACE_MB="${2:-}"; shift 2 ;;
        --plan) PLAN_TMP="${2:-}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

[[ -n "$ONNX_PATH" ]] || die "--onnx is required (the ONNX from export_detector_dual_head.py)"
[[ -n "$MODEL_NAME" ]] || die "--name is required (the Triton model name)"
[[ "$MODEL_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] ||
    die "--name '$MODEL_NAME' is not a valid Triton model/directory name"
for n in "$INPUT_SIZE" "$MIN_BATCH" "$OPT_BATCH" "$MAX_BATCH" "$WORKSPACE_MB"; do
    [[ "$n" =~ ^[0-9]+$ ]] && [[ "$n" -gt 0 ]] || die "expected a positive integer, got '$n'"
done
(( INPUT_SIZE % 32 == 0 )) || die "--input-size must be a multiple of 32, got $INPUT_SIZE"
(( MIN_BATCH <= OPT_BATCH && OPT_BATCH <= MAX_BATCH )) ||
    die "batch profile must satisfy min <= opt <= max (got $MIN_BATCH/$OPT_BATCH/$MAX_BATCH)"

if [[ ${#REPOS[@]} -eq 0 ]]; then
    REPOS=("$(dirname "$SCRIPT_DIR")/models")
fi
PLAN_TMP="${PLAN_TMP:-/tmp/${MODEL_NAME}_dual_head.plan}"

[[ -f "$ONNX_PATH" ]] && [[ -r "$ONNX_PATH" ]] ||
    die "ONNX not found or unreadable at $ONNX_PATH — run export_detector_dual_head.py first" 3
command -v trtexec >/dev/null 2>&1 ||
    die "trtexec is not on PATH. Run inside the triton-server container (or any TensorRT env)." 2

echo "=== Building TRT engine from $ONNX_PATH ==="
trtexec \
    --onnx="$ONNX_PATH" \
    --saveEngine="$PLAN_TMP" \
    --minShapes="${INPUT_NAME}:${MIN_BATCH}x3x${INPUT_SIZE}x${INPUT_SIZE}" \
    --optShapes="${INPUT_NAME}:${OPT_BATCH}x3x${INPUT_SIZE}x${INPUT_SIZE}" \
    --maxShapes="${INPUT_NAME}:${MAX_BATCH}x3x${INPUT_SIZE}x${INPUT_SIZE}" \
    --memPoolSize=workspace:"${WORKSPACE_MB}M" \
    --skipInference

echo "=== Installing $PLAN_TMP ==="
for repo in "${REPOS[@]}"; do
    dst="$repo/$MODEL_NAME/1/model.plan"
    mkdir -p "$(dirname "$dst")"
    install -m 0644 "$PLAN_TMP" "$dst"
    echo "  installed: $dst"
done

echo "=== Done ==="
echo "Each model directory needs a config.pbtxt declaring BOTH outputs"
echo "(the detection tensor and the backbone feature map) — Triton serves"
echo "only the tensors its config names. export_detector_dual_head.py"
echo "--formats onnx trt writes that file for you."
