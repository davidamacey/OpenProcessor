#!/bin/bash
# triton_load.sh - the unload-before-load rule for a model that is already loaded.
#
# Triton's model management guide: in explicit mode a loaded model "should be explicitly
# unloaded prior to the updated version being loaded". A reload in place crashed the server
# (SIGSEGV, TensorRT backend) when the model's files were rewritten; unload, wait, load did not.
# Sourced by model_setup.sh and paddleocr_ops.sh. The Python twin is
# src/services/models/triton_control.py load_model_safely.

[[ -n "${_TRITON_LOAD_SH_LOADED:-}" ]] && return 0
_TRITON_LOAD_SH_LOADED=1

# triton_unload_if_ready BASE_URL NAME [TIMEOUT_S]
# 0: NAME is not loaded (it was not, or was unloaded and the server stopped reporting it ready).
# 1: it is still ready after TIMEOUT_S. 2: its state is unknown (nothing is sent after that).
# KServe v2 ready endpoint: 200 is ready, a 4xx is not.
triton_unload_if_ready() {
    local base="$1" name="$2" timeout="${3:-30}" waited=0 code
    code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 5 "${base}/v2/models/${name}/ready" 2>/dev/null)" || return 2
    [[ "$code" == 200 ]] || { [[ "$code" == 4?? ]] && return 0; return 2; }
    curl -sS -o /dev/null --max-time 30 -X POST "${base}/v2/repository/models/${name}/unload" >/dev/null 2>&1 || true
    while (( waited < timeout )); do
        code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 5 "${base}/v2/models/${name}/ready" 2>/dev/null)" || return 2
        [[ "$code" == 200 ]] || { [[ "$code" == 4?? ]] && return 0; return 2; }
        sleep 1
        waited=$((waited + 1))
    done
    return 1
}
