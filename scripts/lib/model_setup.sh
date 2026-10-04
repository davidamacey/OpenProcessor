#!/bin/bash
# =============================================================================
# model_setup.sh - model export/load groups, retry classifier, skip logic
# =============================================================================
# Shared by setup-openprocessor.sh and the `openprocessor models` CLI. One
# implementation, per the installer plan
# (docs/design/openprocessor_internal/one_line_installer_plan.md section 4).
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

# -----------------------------------------------------------------------------
# Groups (section 4.2 table)
# -----------------------------------------------------------------------------
MODEL_SETUP_ALL_GROUPS=(preflight base yolo mobileclip faces ocr pe)

# model_setup_groups_for_tiers "core curation ..." -> ordered group names
model_setup_groups_for_tiers() {
    local tiers=" $1 " g
    for g in "${MODEL_SETUP_ALL_GROUPS[@]}"; do
        case "$g" in
            pe) [[ "$tiers" == *" curation "* ]] && echo "$g" ;;
            *) [[ "$tiers" == *" core "* ]] && echo "$g" ;;
        esac
    done
    return 0
}

# model_setup_group_models NAME -> Triton models loaded after the group
model_setup_group_models() {
    case "$1" in
        yolo) echo "yolov11_small_trt_end2end" ;;
        mobileclip) echo "mobileclip2_s2_image_encoder mobileclip2_s2_text_encoder" ;;
        faces) echo "scrfd_10g_bnkps arcface_w600k_r50" ;;
        ocr) echo "paddleocr_det_trt paddleocr_rec_trt ocr_pipeline" ;;
        pe) echo "pe_image_encoder pe_text_encoder" ;;
        *) echo "" ;;
    esac
}

# model_setup_group_outputs NAME -> host paths that prove the group ran
model_setup_group_outputs() {
    local m="${OP_DIR:?OP_DIR not set}/models"
    case "$1" in
        yolo) echo "$m/yolov11_small_trt_end2end/1/model.plan" ;;
        mobileclip) echo "$m/mobileclip2_s2_image_encoder/1/model.plan $m/mobileclip2_s2_text_encoder/1/model.plan" ;;
        faces) echo "$m/scrfd_10g_bnkps/1/model.plan $m/arcface_w600k_r50/1/model.plan" ;;
        ocr) echo "$m/paddleocr_det_trt/1/model.plan $m/paddleocr_rec_trt/1/model.plan" ;;
        pe) echo "$m/pe_image_encoder/1/model.plan $m/pe_text_encoder/1/model.onnx" ;;
        *) echo "" ;;
    esac
}

# One-shot API-image and Triton-image containers (never need a running API).
_ms_api() { dc run --rm --no-deps -T -w /app yolo-api "$@"; }
_ms_triton() { dc run --rm --no-deps -T triton-server "$@"; }

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

# _ms_pe_build_engine: FP16-baked ONNX first, FP32 retry (the same
# fallback export/build_pe_trt.sh has), entirely inside the containers.
_ms_pe_build_engine() {
    _ms_api mkdir -p /app/models/pe_image_encoder/1
    if _ms_api python /app/export/trt_utils.py /app/pytorch_models/pe_image_encoder.onnx \
            /app/models/pe_image_encoder.build.onnx \
        && _ms_pe_trtexec pe_image_encoder.build.onnx; then
        _ms_api rm -f /app/models/pe_image_encoder.build.onnx
        return 0
    fi
    echo "FP16 engine build failed; retrying from the FP32 ONNX"
    _ms_api cp /app/pytorch_models/pe_image_encoder.onnx /app/models/pe_image_encoder.build.onnx
    if _ms_pe_trtexec pe_image_encoder.build.onnx; then
        _ms_api rm -f /app/models/pe_image_encoder.build.onnx
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
            "$runner" preflight_seed dc run --rm --no-deps -T --entrypoint cp yolo-api \
                -rn /opt/openprocessor/model_repo_seed/. /app/models/ || return 1
            "$runner" preflight_profile _ms_api bash -c \
                'PROJECT_DIR=/app; source /app/scripts/lib/config.sh && generate_all_configs "$1" 0' \
                _ "${GPU_PROFILE:-standard}" || return 1
            ;;
        base)
            "$runner" base_weights _ms_api bash -c 'source scripts/lib/download.sh && download_essential_models' || return 1
            ;;
        yolo)
            "$runner" yolo_export _ms_api python /app/export/export_models.py --models small \
                --formats trt trt_end2end --normalize-boxes --save-labels --generate-config || return 1
            ;;
        mobileclip)
            "$runner" mobileclip_image _ms_api python /app/export/export_mobileclip_image_encoder.py || return 1
            "$runner" mobileclip_text _ms_api python /app/export/export_mobileclip_text_encoder.py || return 1
            ;;
        faces)
            "$runner" faces_download _ms_api python /app/export/download_face_models.py || return 1
            "$runner" faces_arcface _ms_api python /app/export/export_face_recognition.py || return 1
            "$runner" faces_scrfd _ms_api python /app/export/export_scrfd.py || return 1
            ;;
        ocr)
            "$runner" ocr_download _ms_api python /app/export/download_paddleocr.py || return 1
            "$runner" ocr_det _ms_api python /app/export/export_paddleocr_det.py || return 1
            "$runner" ocr_rec _ms_api python /app/export/export_paddleocr_rec.py || return 1
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

    if [[ -n "$outputs" ]] && group_should_skip "$outputs" "$models" "$state" "${MODEL_SETUP_TRITON_DIGEST:-}"; then
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
    _ms_record "$group" ok $((SECONDS - started))
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
                _ms_record "$g" ok 0
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
