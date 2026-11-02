#!/bin/bash
# =============================================================================
# trt_settings.sh - TensorRT build target and workspace for the shell builds
# =============================================================================
# The shell twin of export/trt_build_settings.py (a test keeps the numbers
# equal): trtexec builds and the engine build stamp need the same answers the
# Python exporters give.
#   OP_TRT_BUILD_TARGET  default|small (GPU_PROFILE=small implies small)
#   OP_TRT_WORKSPACE_GB  whole number >= 1, overrides every model

[[ -n "${_TRT_SETTINGS_SH_LOADED:-}" ]] && return 0
_TRT_SETTINGS_SH_LOADED=1

# trt_target: prints default or small; returns 1 (message on stderr) for any other value.
trt_target() {
    case "${OP_TRT_BUILD_TARGET:-}" in
        "") if [[ "${GPU_PROFILE:-}" == small ]]; then echo small; else echo default; fi ;;
        default|small) echo "$OP_TRT_BUILD_TARGET" ;;
        *) echo "OP_TRT_BUILD_TARGET must be default or small, got '${OP_TRT_BUILD_TARGET}'" >&2; return 1 ;;
    esac
}

# trt_workspace_gb MODEL: whole GB for the workspace pool limit of MODEL.
trt_workspace_gb() {
    local model="$1" target
    if [[ -n "${OP_TRT_WORKSPACE_GB:-}" ]]; then
        if [[ ! "$OP_TRT_WORKSPACE_GB" =~ ^[0-9]+$ ]] || (( OP_TRT_WORKSPACE_GB < 1 )); then
            echo "OP_TRT_WORKSPACE_GB must be a whole number >= 1, got '${OP_TRT_WORKSPACE_GB}'" >&2
            return 1
        fi
        echo "$((10#$OP_TRT_WORKSPACE_GB))"
        return 0
    fi
    target="$(trt_target)" || return 1
    if [[ "$target" == small ]]; then echo 2; return 0; fi
    case "$model" in
        pe_image_encoder|paddleocr_rec_trt) echo 8 ;;
        *) echo 4 ;;
    esac
}

# trt_max_batch MODEL DEFAULT: engine max batch; halved on the small target, never
# below the small profile's Triton max_batch_size (16) and never above DEFAULT.
trt_max_batch() {
    local default="$2" target half floor=16
    target="$(trt_target)" || return 1
    if [[ "$target" != small ]]; then echo "$default"; return 0; fi
    half=$(( default / 2 ))
    (( half < floor )) && half=$floor
    (( half > default )) && half=$default
    echo "$half"
}
