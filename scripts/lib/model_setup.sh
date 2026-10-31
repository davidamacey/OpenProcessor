#!/bin/bash
# =============================================================================
# model_setup.sh - model export/load groups, retry classifier, skip logic
# =============================================================================
# Shared by setup-openprocessor.sh and the `openprocessor models` CLI. One
# implementation, per the installer plan
# (the installer design section 4).
#
# Every compose invocation here goes through the caller's dc() wrapper;
# this file never calls `docker compose` directly
# (tests/installer/test_static.py enforces this).
#
# Caller contract (all read at call time, never at source time):
#   dc()                  compose wrapper (always passes -p and --env-file)
#   OP_DIR                install dir (holds models/, pytorch_models/, .install/)
#   OP_DRY_RUN            "1": print the plan, run nothing, record "planned"
#   TRITON_HTTP_PORT      host port Triton's HTTP endpoint is published on
#   OP_HEALTH_HOST        host to reach published ports on (default 127.0.0.1)
#   MODEL_SETUP_LOGDIR    step logs (default $OP_DIR/.install/logs, mode 700)
#   MODEL_SETUP_STATE     state.json path (default $OP_DIR/.install/state.json)
#   MODEL_SETUP_GROUPS_FILE  per-group results, TSV name<TAB>status<TAB>seconds
#   MODEL_SETUP_TRITON_DIGEST  the Triton image digest engines are built with
#   GPU_PROFILE           minimal|standard|full (preflight's config pass)

[[ -n "${_MODEL_SETUP_SH_LOADED:-}" ]] && return 0
_MODEL_SETUP_SH_LOADED=1

# Logging: reuse the caller's log_* if it has them (the installer defines
# its own self-contained set); otherwise load the repo's colors.sh.
if ! declare -F log_info >/dev/null 2>&1; then
    # shellcheck source=colors.sh
    source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/colors.sh"
fi

# -----------------------------------------------------------------------------
# Redaction (plan section 7): every step log goes through this filter.
# -----------------------------------------------------------------------------
model_setup_redact() {
    sed -u -E \
        -e 's/hf_[A-Za-z0-9]{10,}/hf_***REDACTED***/g' \
        -e 's/(Bearer)[[:space:]]+[^[:space:]"]+/\1 ***REDACTED***/g' \
        -e 's/(^|[^A-Za-z0-9_])([A-Z0-9_]*(_API_KEY|_TOKEN|_SECRET|_PASSWORD))=[^[:space:]]*/\1\2=***REDACTED***/g'
}

# -----------------------------------------------------------------------------
# Failure classification (section 4.2)
# -----------------------------------------------------------------------------
# classify_failure LOGFILE
# Prints one of: transient | permanent:<why> | unknown
classify_failure() {
    local logfile="$1"
    [[ -f "$logfile" ]] || { echo "unknown"; return 0; }

    # Permanent, checked first: no point retrying these.
    if grep -qE 'ModuleNotFoundError' "$logfile"; then
        echo "permanent:stale_image"
        return 0
    fi
    # 401/403 only in an HTTP/status context: a bare " 403 " (e.g.
    # "downloaded 403 files") is not an auth failure.
    if grep -qE 'GatedRepoError|HTTP[^0-9]{0,12}(401|403)([^0-9]|$)|[Ss]tatus( code)?[: =]+(401|403)([^0-9]|$)|(401|403) (Client Error|Unauthorized|Forbidden)' "$logfile"; then
        echo "permanent:gated"
        return 0
    fi
    if grep -qiE 'no space left' "$logfile"; then
        echo "permanent:disk"
        return 0
    fi
    # TensorRT reports an exhausted card as "CUDA initialization failure with
    # error: 2" (cudaErrorMemoryAllocation), not as "out of memory". Retrying
    # does not help: the memory is held by Triton's loaded engines (#111).
    if grep -qiE 'out of memory|CUDA initialization failure with error: 2([^0-9]|$)|cudaErrorMemoryAllocation' "$logfile"; then
        echo "permanent:oom"
        return 0
    fi

    # Transient: known GPU-init / network races that clear on retry.
    if grep -qE \
        'CUDA initialization failure|error 100|factory function returned nullptr|CUDA error: an illegal memory access|cudaErrorDevicesUnavailable|Connection reset|Read timed out|HTTP/1\.1" 5[0-9]{2}' \
        "$logfile"; then
        echo "transient"
        return 0
    fi

    echo "unknown"
}

# permanent_hint CLASS
permanent_hint() {
    case "$1" in
        permanent:stale_image)
            echo "image is stale or wrong; run 'openprocessor repair --images'" ;;
        permanent:gated)
            echo "token or licence issue: accept the model's licence and re-check the token" ;;
        permanent:disk)
            echo "no space left on the install filesystem" ;;
        permanent:oom)
            echo "out of GPU memory (Triton and the export share a card with too little free VRAM); move Triton with --gpu-plan triton=N, or free the card, then re-run" ;;
        *)
            echo "" ;;
    esac
}

_model_setup_logdir() {
    local d="${MODEL_SETUP_LOGDIR:-${OP_DIR:?OP_DIR not set}/.install/logs}"
    ( umask 077; mkdir -p "$d" )
    chmod 700 "$d"
    printf '%s\n' "$d"
}

# -----------------------------------------------------------------------------
# Retry/backoff (section 4.2)
# -----------------------------------------------------------------------------
# retry_step NAME MAX_ATTEMPTS -- CMD...
# Runs CMD, retrying on a "transient" classification with 15s/45s backoff.
# An "unknown" failure gets exactly one retry. A "permanent" failure fails
# immediately with its hint. Returns 0 on success, the last exit code
# otherwise. The caller's errexit setting is left exactly as it was.
# The last attempt's output is written, redacted and mode 600, to
# <logdir>/<NAME>.log.
retry_step() {
    local name="$1" max="${2:-3}"
    shift 2
    [[ "${1:-}" == "--" ]] && shift

    local logdir logfile
    logdir="$(_model_setup_logdir)"
    logfile="${logdir}/${name}.log"

    local attempt=1 last_rc=1 cls sleep_s
    local backoffs=(15 45)

    while (( attempt <= max )); do
        # `if` keeps errexit out of play without touching the caller's
        # shell options; PIPESTATUS[0] is the step's own exit code.
        if ( umask 077; "$@" 2>&1 | model_setup_redact > "$logfile"; exit "${PIPESTATUS[0]}" ); then
            log_success "step ${name}: ok (attempt ${attempt})"
            return 0
        else
            last_rc=$?
        fi

        cls="$(classify_failure "$logfile")"

        if [[ "$cls" == permanent:* ]]; then
            log_error "step ${name}: permanent failure (${cls#permanent:}): $(permanent_hint "$cls")"
            return "$last_rc"
        fi
        if [[ "$cls" == "unknown" ]] && (( attempt >= 2 )); then
            log_error "step ${name}: unknown failure, no more retries (log: ${logfile})"
            return "$last_rc"
        fi
        if (( attempt >= max )); then
            log_error "step ${name}: exhausted ${max} attempts (log: ${logfile})"
            return "$last_rc"
        fi

        sleep_s="${backoffs[$((attempt - 1))]:-45}"
        log_warn "step ${name}: ${cls} failure (attempt ${attempt}/${max}), retrying in ${sleep_s}s"
        if declare -F wait_for_gpu_memory >/dev/null 2>&1; then
            wait_for_gpu_memory || true
        fi
        sleep "$sleep_s"
        attempt=$((attempt + 1))
    done

    return "$last_rc"
}

# -----------------------------------------------------------------------------
# Triton load + READY wait (section 4.2)
# -----------------------------------------------------------------------------
_ms_triton_url() {
    printf 'http://%s:%s' "${OP_HEALTH_HOST:-127.0.0.1}" "${TRITON_HTTP_PORT:-4600}"
}

# triton_model_ready NAME
triton_model_ready() {
    curl -fsS --max-time 5 "$(_ms_triton_url)/v2/models/${1}/ready" >/dev/null 2>&1
}

# triton_load_and_wait NAME [TIMEOUT_S]
triton_load_and_wait() {
    local name="$1" timeout="${2:-120}" waited=0
    if ! curl -fsS --max-time 60 -X POST "$(_ms_triton_url)/v2/repository/models/${name}/load" >/dev/null 2>&1; then
        log_warn "load request for ${name} was rejected; waiting for READY anyway"
    fi
    while (( waited < timeout )); do
        if triton_model_ready "$name"; then
            return 0
        fi
        sleep 2
        waited=$((waited + 2))
    done
    return 1
}

# -----------------------------------------------------------------------------
# Skip logic (section 4.2, "Idempotent skip")
# -----------------------------------------------------------------------------
# group_should_skip OUTPUTS TRITON_MODELS STATE_FILE TRITON_DIGEST
# OUTPUTS: space-separated file paths. TRITON_MODELS: space-separated model
# names ("" for none). Returns 0 (skip) only if every output exists, every
# model reports READY, and state.json records the same Triton digest.
group_should_skip() {
    local outputs="$1" models="$2" state_file="$3" digest="$4" f m recorded
    local -a out_list model_list
    read -ra out_list <<< "$outputs"
    read -ra model_list <<< "$models"

    for f in "${out_list[@]}"; do
        [[ -e "$f" ]] || return 1
    done
    for m in "${model_list[@]}"; do
        triton_model_ready "$m" || return 1
    done

    [[ -f "$state_file" && -n "$digest" ]] || return 1
    recorded="$(sed -n -E 's/.*"triton_image_digest":[[:space:]]*"([^"]*)".*/\1/p' "$state_file" | head -n1)"
    [[ -n "$recorded" && "$recorded" == "$digest" ]]
}

# Groups (section 4.2 table): the tables live in model_groups.sh.
# shellcheck source=model_groups.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/model_groups.sh"

# One-shot API-image and Triton-image containers (never need a running API).
_ms_api() { dc run --rm --no-deps -T -w /app api "$@"; }
_ms_triton() { dc run --rm --no-deps -T triton-server "$@"; }

# Same as PE_NORMALIZE in export/build_pe_trt.sh (a test keeps them equal).
_MS_PE_NORMALIZE="--mean 0.5 0.5 0.5 --std 0.5 0.5 0.5"

# _ms_pe_uint8_prepare: uint8 input prefix after the FP16 bake (which keeps the I/O
# float32), then fail unless the input datatype is the one config.pbtxt declares.
_ms_pe_uint8_prepare() {
    # shellcheck disable=SC2086
    _ms_api python /app/export/uint8_input.py /app/models/pe_image_encoder.build.onnx \
        /app/models/pe_image_encoder.build.u8.onnx $_MS_PE_NORMALIZE \
        && _ms_api python /app/export/check_input_dtype.py \
            /app/models/pe_image_encoder.build.u8.onnx /app/models/pe_image_encoder/config.pbtxt \
        || { log_error "pe_image_encoder: the engine input would not match config.pbtxt; Triton would refuse to load it"; return 1; }
}

# _ms_config_input_dtype MODEL: the input datatype (UINT8, FP32, ...) the shipped
# config.pbtxt declares, empty when there is no config.
_ms_config_input_dtype() {
    local cfg="${OP_DIR:?OP_DIR not set}/models/$1/config.pbtxt"
    [[ -f "$cfg" ]] || return 0
    awk '/^input[[:space:]]*\[/ {in_in=1} in_in && /data_type:/ {sub(/.*TYPE_/, ""); print $1; exit}' "$cfg"
}

_ms_pe_trtexec() {
    local onnx="$1"
    _ms_triton trtexec \
        --onnx="/models/${onnx}" \
        --saveEngine=/models/pe_image_encoder/1/model.plan \
        --minShapes=images:1x3x336x336 \
        --optShapes=images:8x3x336x336 \
        --maxShapes=images:32x3x336x336 \
        --memPoolSize=workspace:8G \
        --skipInference
}

# Per-model precision ledger (.install/precision.tsv, tab separated:
# model, precision, reason, tensorrt, compute_cap, onnx_sha256, flags). It is
# also the engine build stamp: what was actually built, with which TensorRT,
# for which GPU architecture, from which ONNX. Every engine of every group
# carries one (model_setup_group_engines). "fp16" and "fp32" are clean builds
# at the precision the model is designed for; "*_fallback" is a recorded
# fallback that model_setup_run_group reports loudly and `openprocessor
# models status` lists, so a silent FP32 engine can never pass as the FP16
# default. A stamp that no longer matches the host
# (another architecture, a re-exported ONNX, no row at all) makes the group
# rebuild instead of being skipped as up to date.
_ms_precision_file() {
    printf '%s' "${MODEL_SETUP_PRECISION_FILE:-${OP_DIR:?OP_DIR not set}/.install/precision.tsv}"
}

# The PE engine's trtexec profile (_ms_pe_trtexec), recorded in its stamp.
_MS_PE_FLAGS="min1/opt8/max32 336x336 workspace8G strongly-typed input=uint8"
_MS_PE_ONNX_REL="pytorch_models/pe_image_encoder.onnx"

# _ms_compute_cap: compute capability of the lowest GPU id given to Triton.
_ms_compute_cap() {
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i "${TRITON_GPU_ID:-0}" 2>/dev/null \
        | head -n1 | tr -d '[:space:]'
}

# _ms_trt_version: the TensorRT build inside the Triton image (e.g. v110100).
_ms_trt_version() {
    _ms_triton trtexec --help 2>&1 | sed -n -E 's/.*TensorRT (v[0-9]+).*/\1/p' | head -n1
}

# _ms_onnx_sha256 REL_PATH: sha256 of an ONNX under OP_DIR, empty when absent.
_ms_onnx_sha256() {
    local f="${OP_DIR:?OP_DIR not set}/$1"
    [[ -f "$f" ]] || return 0
    sha256sum "$f" | cut -d' ' -f1
}

# _ms_record_precision MODEL PRECISION REASON [ONNX_REL FLAGS]
# With an ONNX path and flags the row is a full build stamp.
_ms_record_precision() {
    [[ "${OP_DRY_RUN:-0}" == "1" ]] && return 0
    local model="$1" precision="$2" reason="${3//$'\t'/ }" onnx_rel="${4:-}" flags="${5:-}" f tmp
    local trt="" cc="" sha=""
    if [[ -n "$onnx_rel" ]]; then
        trt="$(_ms_trt_version)"
        cc="$(_ms_compute_cap)"
        sha="$(_ms_onnx_sha256 "$onnx_rel")"
    fi
    f="$(_ms_precision_file)"
    ( umask 077; mkdir -p "$(dirname "$f")" )
    tmp="$(mktemp "${f}.XXXXXX")"
    if [[ -f "$f" ]]; then
        awk -F'\t' -v m="$model" '$1 != m' "$f" > "$tmp"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$model" "$precision" "$reason" "$trt" "$cc" "$sha" "$flags" >> "$tmp"
    chmod 600 "$tmp"
    mv -f "$tmp" "$f"
}

# model_setup_precision_report: one line per model whose engine is not the
# designed-precision build ("model<TAB>precision<TAB>reason"); empty when all clean.
model_setup_precision_report() {
    local f
    f="$(_ms_precision_file)"
    [[ -f "$f" ]] || return 0
    awk -F'\t' -v OFS='\t' '$2 ~ /_fallback$/ { print $1, $2, $3 }' "$f"
}

# model_setup_stamp_report: every recorded engine stamp, one line per model
# ("model<TAB>precision<TAB>tensorrt<TAB>compute_cap<TAB>onnx_sha256_prefix").
model_setup_stamp_report() {
    local f
    f="$(_ms_precision_file)"
    [[ -f "$f" ]] || return 0
    awk -F'\t' -v OFS='\t' '$4 != "" { print $1, $2, $4, $5, substr($6, 1, 12) }' "$f"
}

# model_setup_stamp_stale MODEL ONNX_REL: prints why and returns 0 when the
# recorded stamp of MODEL no longer describes what this host would build
# (no stamp, a fallback precision, another GPU architecture or another ONNX);
# returns 1 when the engine is current. The TensorRT build is covered by the
# Triton image digest that group_should_skip already compares.
model_setup_stamp_stale() {
    local model="$1" onnx_rel="$2" f row precision cc sha now_cc now_sha want
    f="$(_ms_precision_file)"
    row="$(awk -F'\t' -v m="$model" '$1 == m' "$f" 2>/dev/null | tail -n1)"
    if [[ -z "$row" ]]; then echo "no build stamp"; return 0; fi
    precision="$(cut -f2 <<< "$row")"
    cc="$(cut -f5 <<< "$row")"
    sha="$(cut -f6 <<< "$row")"
    if [[ "$precision" == *_fallback ]]; then echo "built as ${precision}"; return 0; fi
    want="$(_ms_config_input_dtype "$model")"
    if [[ -n "$want" && " $(cut -f7 <<< "$row") " != *" input=${want,,} "* ]]; then
        echo "engine input datatype is not the config's ${want}"
        return 0
    fi
    if [[ -z "$sha" ]]; then echo "stamp has no ONNX hash"; return 0; fi
    now_cc="$(_ms_compute_cap)"
    if [[ -n "$now_cc" && "$cc" != "$now_cc" ]]; then
        echo "built for compute capability ${cc:-unknown}, GPU is ${now_cc}"
        return 0
    fi
    now_sha="$(_ms_onnx_sha256 "$onnx_rel")"
    if [[ -n "$now_sha" && "$sha" != "$now_sha" ]]; then echo "ONNX changed since the build"; return 0; fi
    return 1
}

# _ms_group_stamp_stale GROUP: logs and returns 0 when any engine of the group
# has a stamp that no longer matches, so the group is rebuilt instead of
# skipped (a skipped group is what lets its models stay READY).
_ms_group_stamp_stale() {
    local model onnx_rel _prec why stale=1
    while IFS=$'\t' read -r model onnx_rel _prec; do
        [[ -n "$model" ]] || continue
        if why="$(model_setup_stamp_stale "$model" "$onnx_rel")"; then
            log_info "group $1: engine stamp is stale (${why}; ${model}), rebuilding"
            stale=0
        fi
    done < <(model_setup_group_engines "$1")
    return "$stale"
}

# _ms_record_group_stamps GROUP: stamps every engine of a group whose build
# runs inside the export scripts (all but pe, whose _ms_pe_build_engine
# records its own precision, FP32 fallback included). The input datatype is
# taken from the config the engine was just built for.
_ms_record_group_stamps() {
    local model onnx_rel precision dtype
    while IFS=$'\t' read -r model onnx_rel precision; do
        [[ -n "$model" ]] || continue
        dtype="$(_ms_config_input_dtype "$model")"
        _ms_record_precision "$model" "$precision" "" "$onnx_rel" \
            "trt-python-api ${precision}${dtype:+ input=${dtype,,}}"
    done < <(model_setup_group_engines "$1")
}

# _ms_warn_if_degraded GROUP: prints a warning per fallback-precision model of
# the group; returns 0 when the group is degraded, 1 when it is clean.
_ms_warn_if_degraded() {
    local group="$1" model precision reason degraded=1 m
    local -a group_models
    read -ra group_models <<< "$(model_setup_group_models "$group")"
    while IFS=$'\t' read -r model precision reason; do
        [[ -n "$model" ]] || continue
        for m in "${group_models[@]}"; do
            [[ "$m" == "$model" ]] || continue
            degraded=0
            log_warn "model ${model}: shipped as FP32, NOT the FP16 default (${precision}). Reason: ${reason}"
            log_warn "  Consequence: a roughly twice-as-large engine, about twice the GPU memory and lower throughput than FP16."
            log_warn "  Group '${group}' is recorded as 'degraded'. Fix the cause, then run: ./openprocessor models install --only ${group}"
        done
    done < <(model_setup_precision_report)
    return "$degraded"
}

# _ms_pe_build_engine: FP16-baked ONNX first, FP32 retry (the same
# fallback export/build_pe_trt.sh has), entirely inside the containers. The
# FP32 retry is kept (an FP32 engine beats none) but is recorded in the
# precision ledger with the FP16 failure reason, never as a clean build.
_ms_pe_build_engine() {
    local attempt_log raw reason
    if [[ "${OP_DRY_RUN:-0}" == "1" ]]; then
        _ms_api python /app/export/trt_utils.py /app/pytorch_models/pe_image_encoder.onnx \
            /app/models/pe_image_encoder.build.onnx
        _ms_pe_uint8_prepare
        _ms_pe_trtexec pe_image_encoder.build.u8.onnx
        return 0
    fi
    attempt_log="$(_model_setup_logdir)/pe_image_trt.fp16_attempt.log"
    raw="$(mktemp "${attempt_log}.XXXXXX")"
    _ms_api mkdir -p /app/models/pe_image_encoder/1
    if { _ms_api python /app/export/trt_utils.py /app/pytorch_models/pe_image_encoder.onnx \
            /app/models/pe_image_encoder.build.onnx \
        && _ms_pe_uint8_prepare \
        && _ms_pe_trtexec pe_image_encoder.build.u8.onnx; } >"$raw" 2>&1; then
        ( umask 077; model_setup_redact < "$raw" > "$attempt_log" )
        rm -f "$raw"
        cat "$attempt_log"
        _ms_api rm -f /app/models/pe_image_encoder.build.onnx /app/models/pe_image_encoder.build.u8.onnx
        _ms_record_precision pe_image_encoder fp16 "" "$_MS_PE_ONNX_REL" "$_MS_PE_FLAGS"
        return 0
    fi
    ( umask 077; model_setup_redact < "$raw" > "$attempt_log" )
    rm -f "$raw"
    cat "$attempt_log"
    reason="$(grep -m1 -E 'bake_fp16_onnx failed|\[E\]|Error' "$attempt_log" | cut -c1-300)"
    reason="${reason:-FP16 build failed, see ${attempt_log}}"
    echo "FP16 engine build failed (${reason}); retrying from the FP32 ONNX"
    _ms_api cp /app/pytorch_models/pe_image_encoder.onnx /app/models/pe_image_encoder.build.onnx
    if _ms_pe_uint8_prepare && _ms_pe_trtexec pe_image_encoder.build.u8.onnx; then
        _ms_api rm -f /app/models/pe_image_encoder.build.onnx /app/models/pe_image_encoder.build.u8.onnx
        _ms_record_precision pe_image_encoder fp32_fallback "FP16 build failed: ${reason}" "$_MS_PE_ONNX_REL" "$_MS_PE_FLAGS"
        return 0
    fi
    return 1
}

# _ms_run_group_steps GROUP RUNNER
# Calls RUNNER STEP_NAME CMD... once per step. The installer passes a
# retrying runner; dry-run passes one that only prints, so both paths walk
# exactly the same step list.
_ms_run_group_steps() {
    local group="$1" runner="$2"
    case "$group" in
        preflight)
            "$runner" preflight_image _ms_api python -m export.preflight || return 1
            "$runner" preflight_seed dc run --rm --no-deps -T --entrypoint python api \
                /app/export/sync_model_seed.py /opt/openprocessor/model_repo_seed /app/models || return 1
            "$runner" preflight_profile _ms_api bash -c \
                'PROJECT_DIR=/app; source /app/scripts/lib/config.sh && generate_all_configs "$1" 0' \
                _ "${GPU_PROFILE:-standard}" || return 1
            ;;
        base)
            "$runner" base_weights _ms_api bash -c 'source scripts/lib/download.sh && download_essential_models' || return 1
            ;;
        yolo)
            "$runner" yolo_export _ms_api python /app/export/export_models.py --models small \
                --formats trt_end2end --normalize-boxes --save-labels --generate-config || return 1
            "$runner" yolo_stamp _ms_record_group_stamps yolo || return 1
            ;;
        mobileclip)
            "$runner" mobileclip_image _ms_api python /app/export/export_mobileclip_image_encoder.py || return 1
            "$runner" mobileclip_text _ms_api python /app/export/export_mobileclip_text_encoder.py || return 1
            "$runner" mobileclip_stamp _ms_record_group_stamps mobileclip || return 1
            ;;
        faces)
            "$runner" faces_download _ms_api python /app/export/download_face_models.py || return 1
            "$runner" faces_arcface _ms_api python /app/export/export_face_recognition.py || return 1
            "$runner" faces_scrfd _ms_api python /app/export/export_scrfd.py || return 1
            "$runner" faces_stamp _ms_record_group_stamps faces || return 1
            ;;
        ocr)
            "$runner" ocr_download _ms_api python /app/export/download_paddleocr.py || return 1
            "$runner" ocr_det _ms_api python /app/export/export_paddleocr_det.py || return 1
            "$runner" ocr_rec _ms_api python /app/export/export_paddleocr_rec.py || return 1
            "$runner" ocr_stamp _ms_record_group_stamps ocr || return 1
            ;;
        pe)
            "$runner" pe_weights _ms_api python /app/export/download_pe_weights.py || return 1
            "$runner" pe_image_onnx _ms_api python /app/export/export_pe_image_encoder.py || return 1
            "$runner" pe_image_trt _ms_pe_build_engine || return 1
            "$runner" pe_text _ms_api python /app/export/export_pe_text_encoder.py \
                --install-triton --models-dir /app/models || return 1
            ;;
        *)
            log_error "unknown model group: ${group}"
            return 2
            ;;
    esac
}

_ms_record() {
    local group="$1" status="$2" secs="$3"
    local f="${MODEL_SETUP_GROUPS_FILE:-${OP_DIR:?OP_DIR not set}/.install/groups.tsv}"
    local tmp
    tmp="$(mktemp "${f}.XXXXXX")"
    if [[ -f "$f" ]]; then
        awk -F'\t' -v g="$group" '$1 != g' "$f" > "$tmp"
    fi
    printf '%s\t%s\t%s\n' "$group" "$status" "$secs" >> "$tmp"
    chmod 600 "$tmp"
    mv -f "$tmp" "$f"
}

# model_setup_group_status NAME -> recorded status or ""
model_setup_group_status() {
    local f="${MODEL_SETUP_GROUPS_FILE:-${OP_DIR:?OP_DIR not set}/.install/groups.tsv}"
    [[ -f "$f" ]] || return 0
    awk -F'\t' -v g="$1" '$1 == g { s = $2 } END { if (s != "") print s }' "$f"
}

_ms_dry_step() {
    local name="$1"
    shift
    echo "DRY: step ${name}:"
    "$@"
}

_ms_retry3() {
    local name="$1"
    shift
    retry_step "$name" 3 -- "$@"
}

# model_setup_run_group NAME
# Returns 0 on success or skip, non-zero on failure. Records the result.
model_setup_run_group() {
    local group="$1" started models m outputs
    local state="${MODEL_SETUP_STATE:-${OP_DIR:?OP_DIR not set}/.install/state.json}"
    models="$(model_setup_group_models "$group")"
    outputs="$(model_setup_group_outputs "$group")"

    if [[ "${OP_DRY_RUN:-0}" == "1" ]]; then
        log_info "group ${group}: planned (dry-run, nothing executed)"
        _ms_run_group_steps "$group" _ms_dry_step
        for m in $models; do
            echo "DRY: POST $(_ms_triton_url)/v2/repository/models/${m}/load"
        done
        _ms_record "$group" planned 0
        return 0
    fi

    # A degraded group (fallback precision) is never "up to date": re-running
    # retries the preferred precision.
    if [[ -n "$outputs" ]] && [[ "$(model_setup_group_status "$group")" != degraded ]] \
        && ! _ms_group_stamp_stale "$group" \
        && group_should_skip "$outputs" "$models" "$state" "${MODEL_SETUP_TRITON_DIGEST:-}"; then
        log_info "group ${group}: up to date, skipped"
        _ms_record "$group" skipped 0
        return 0
    fi

    started=$SECONDS
    log_step "model group: ${group}"
    if ! _ms_run_group_steps "$group" _ms_retry3; then
        _ms_record "$group" failed $((SECONDS - started))
        return 1
    fi
    for m in $models; do
        if ! triton_load_and_wait "$m" 120; then
            log_warn "model ${m} did not reach READY after load"
            _ms_record "$group" load_failed $((SECONDS - started))
            return 3
        fi
    done
    if _ms_warn_if_degraded "$group"; then
        _ms_record "$group" degraded $((SECONDS - started))
    else
        _ms_record "$group" ok $((SECONDS - started))
    fi
    return 0
}

# model_setup_restart_triton_fallback API_PORT
# Section 4.2: a failed load falls back to a Triton restart, a READY wait
# and the API's reload_promoted.
model_setup_restart_triton_fallback() {
    local api_port="${1:-${API_PORT:-4603}}" waited=0
    log_warn "falling back to a Triton restart after a failed load"
    dc restart triton-server || return 1
    while (( waited < 180 )); do
        if curl -fsS --max-time 5 "$(_ms_triton_url)/v2/health/ready" >/dev/null 2>&1; then
            break
        fi
        sleep 3
        waited=$((waited + 3))
    done
    curl -fsS --max-time 30 -X POST \
        "http://${OP_HEALTH_HOST:-127.0.0.1}:${api_port}/curation/projects/${OP_CURATION_PROJECT:-default}/train/reload_promoted" \
        >/dev/null 2>&1 || true
    return 0
}

# model_setup_run_groups "TIERS" [ONLY_GROUP]
# Runs every group for the tiers (or just ONLY_GROUP). A failed group does
# not stop later independent groups. Prints the failed groups on stdout's
# last line as "failed_groups=a,b" and returns the number that failed.
model_setup_run_groups() {
    local tiers="$1" only="${2:-}" g rc failed=() load_failed=()
    local groups=()
    if [[ -n "$only" ]]; then
        groups=("$only")
    else
        mapfile -t groups < <(model_setup_groups_for_tiers "$tiers")
    fi
    for g in "${groups[@]}"; do
        if model_setup_run_group "$g"; then
            continue
        else
            rc=$?
        fi
        if (( rc == 3 )); then
            load_failed+=("$g")
        else
            failed+=("$g")
        fi
        # A failed preflight means the image is unusable: nothing after it
        # can succeed, so stop here instead of burning retries.
        if [[ "$g" == "preflight" ]]; then
            break
        fi
    done

    if (( ${#load_failed[@]} > 0 )) && [[ "${OP_DRY_RUN:-0}" != "1" ]]; then
        model_setup_restart_triton_fallback "${API_PORT:-4603}" || true
        for g in "${load_failed[@]}"; do
            local m ok=1
            for m in $(model_setup_group_models "$g"); do
                triton_load_and_wait "$m" 120 || ok=0
            done
            if (( ok == 1 )); then
                if _ms_warn_if_degraded "$g"; then
                    _ms_record "$g" degraded 0
                else
                    _ms_record "$g" ok 0
                fi
            else
                _ms_record "$g" failed 0
                failed+=("$g")
            fi
        done
    fi

    local IFS=,
    echo "failed_groups=${failed[*]}"
    return "${#failed[@]}"
}

# model_setup_sample_coco [--full]
# Public COCO sample (plan section 6.3), fetched inside the API image.
model_setup_sample_coco() {
    local manifest=scripts/datasets/manifests/coco_va_200.json n=200
    if [[ "${1:-}" == "--full" ]]; then
        manifest=scripts/datasets/manifests/coco_va_800.json
        n=800
    fi
    _ms_api python -m scripts.datasets.fetch_coco_subset \
        --out /app/data/samples/coco_va_readme --n "$n" \
        --manifest "$manifest" --cache-dir /app/data/.cache/datasets
}
