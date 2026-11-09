#!/bin/bash
# =============================================================================
# model_stamps.sh - the engine build stamp ledger (.install/precision.tsv):
# recording, reporting, staleness and adoption of engines built without a stamp
# =============================================================================
# Sourced by model_setup.sh, which supplies _ms_triton, _ms_config_input_dtype,
# triton_model_ready, log_info and the group tables of model_groups.sh.

[[ -n "${_MODEL_STAMPS_SH_LOADED:-}" ]] && return 0
_MODEL_STAMPS_SH_LOADED=1

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

# _ms_recipe MODEL: the export recipe of an engine whose ONNX is not the whole story. An upgrade
# keeps the old ONNX, so its hash matches and nothing would rebuild: the recipe token in the stamp
# is what makes an engine built by an older recipe stale. Bump the value when the recipe changes.
#   fused-v1: export/ultralytics_export.py, one output0 [300, 6] (the `nms` export argument)
_ms_recipe() {
    case "$1" in
        yolov11_small_trt_end2end) printf 'fused-v1' ;;
    esac
}

# _ms_build_flags MODEL: the build settings recorded in (and compared with) an engine
# stamp: the build target and workspace of scripts/lib/trt_settings.sh, and the export recipe
# where the model has one. A profile switch (standard <-> full) leaves the first two
# unchanged, so it does not rebuild; standard <-> small changes them, so it does.
_ms_build_flags() {
    local target ws recipe
    target="$(trt_target)" || return 1
    ws="$(trt_workspace_gb "$1")" || return 1
    recipe="$(_ms_recipe "$1")"
    printf 'target=%s workspace=%sG%s' "$target" "$ws" "${recipe:+ recipe=$recipe}"
}

# The PE engine's trtexec profile (_ms_pe_trtexec), recorded in its stamp.
_ms_pe_flags() {
    local build
    build="$(_ms_build_flags pe_image_encoder)" || return 1
    printf 'min1/opt8/max%s 336x336 %s strongly-typed input=uint8' \
        "$(trt_max_batch pe_image_encoder 32)" "$build"
}
_MS_PE_ONNX_REL="pytorch_models/pe_image_encoder.onnx"

# _ms_build_gpu_id MODEL: the card MODEL's engine is built on. The PE trtexec runs in a
# triton-server container (TRITON_GPU_ID); every other engine is built by the model-export
# container (API_GPU_ID). Two cards of different architectures must not share one stamp value.
_ms_build_gpu_id() {
    local id
    if [[ "${1:-}" == pe_image_encoder ]]; then
        id="${TRITON_GPU_ID:-0}"
    else
        id="${API_GPU_ID:-${TRITON_GPU_ID:-0}}"
    fi
    printf '%s' "${id%%,*}"
}

# _ms_compute_cap [MODEL]: compute capability of the card MODEL's engine is built on.
_ms_compute_cap() {
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i "$(_ms_build_gpu_id "${1:-}")" 2>/dev/null \
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

# _ms_record_precision MODEL PRECISION REASON [ONNX_REL FLAGS [SHA]]
# With an ONNX path and flags the row is a full build stamp. SHA overrides the hash read
# from the ONNX file ("-" = the ONNX was not on this host when the engine was adopted).
_ms_record_precision() {
    [[ "${OP_DRY_RUN:-0}" == "1" ]] && return 0
    local model="$1" precision="$2" reason="${3//$'\t'/ }" onnx_rel="${4:-}" flags="${5:-}" sha_arg="${6:-}" f tmp
    local trt="" cc="" sha=""
    if [[ -n "$onnx_rel" ]]; then
        trt="$(_ms_trt_version)"
        cc="$(_ms_compute_cap "$model")"
        sha="${sha_arg:-$(_ms_onnx_sha256 "$onnx_rel")}"
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

# _ms_flags_stale MODEL FLAGS: prints why and returns 0 when the build target or
# workspace recorded in FLAGS is not what this host would build now. A stamp without
# the fields (written before #42) was built with the default target and workspace, so it
# stays current under those and is never rebuilt just for lacking them.
_ms_flags_stale() {
    local model="$1" flags=" $2 " want_target want_ws want_recipe have
    want_recipe="$(_ms_recipe "$model")"
    if [[ -n "$want_recipe" && "$flags" != *" recipe=${want_recipe} "* ]]; then
        have="$(sed -n -E 's/.* recipe=([^ ]+) .*/\1/p' <<< "$flags")"
        echo "exported by recipe ${have:-none}, now ${want_recipe}"
        return 0
    fi
    want_target="$(trt_target)" || { echo "invalid OP_TRT_BUILD_TARGET"; return 0; }
    want_ws="$(trt_workspace_gb "$model")" || { echo "invalid OP_TRT_WORKSPACE_GB"; return 0; }
    if [[ "$flags" != *" target="* ]]; then
        # Written before #42: built with the default target and that model's default workspace.
        flags="${flags}target=default workspace=$(OP_TRT_BUILD_TARGET=default GPU_PROFILE='' OP_TRT_WORKSPACE_GB='' trt_workspace_gb "$model")G "
    fi
    if [[ "$flags" != *" target=${want_target} "* ]]; then
        have="$(sed -n -E 's/.* target=([a-z]+) .*/\1/p' <<< "$flags")"
        echo "built for target ${have:-unknown}, now ${want_target}"
        return 0
    fi
    if [[ "$flags" != *" workspace=${want_ws}G "* ]]; then
        have="$(sed -n -E 's/.* workspace=([0-9]+G) .*/\1/p' <<< "$flags")"
        echo "built with workspace ${have:-unknown}, now ${want_ws}G"
        return 0
    fi
    return 1
}

# model_setup_stamp_stale MODEL ONNX_REL: prints why and returns 0 when the
# recorded stamp of MODEL no longer describes what this host would build
# (no stamp, a fallback precision, another GPU architecture or another ONNX);
# returns 1 when the engine is current. The TensorRT build is covered by the
# Triton image digest that group_should_skip already compares.
model_setup_stamp_stale() {
    local model="$1" onnx_rel="$2" f row precision cc sha now_cc now_sha want why
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
    if why="$(_ms_flags_stale "$model" "$(cut -f7 <<< "$row")")"; then echo "$why"; return 0; fi
    if [[ -z "$sha" ]]; then echo "stamp has no ONNX hash"; return 0; fi
    now_cc="$(_ms_compute_cap "$model")"
    if [[ -n "$now_cc" && "$cc" != "$now_cc" ]]; then
        echo "built for compute capability ${cc:-unknown}, GPU is ${now_cc}"
        return 0
    fi
    now_sha="$(_ms_onnx_sha256 "$onnx_rel")"
    if [[ "$sha" != "-" && -n "$now_sha" && "$sha" != "$now_sha" ]]; then echo "ONNX changed since the build"; return 0; fi
    return 1
}

# _ms_adopt_unstamped MODEL ONNX_REL PRECISION: writes a build stamp for an engine
# that has none (every engine except the PE image encoder was built unstamped before
# per-engine stamps existed), but only when the engine is provably valid for this
# host; returns 1 otherwise and the caller rebuilds. Proof, all required:
#   - the engine file exists and is non-empty;
#   - the model is READY in Triton: the current TensorRT loaded the engine, which
#     also proves the GPU architecture and the input datatype match config.pbtxt;
#   - the config declares an input datatype (recorded as the input= flag);
#   - the host's compute capability and the Triton image's TensorRT version are
#     known and, when the PE image encoder has a stamp (same install, same Triton
#     image built it), equal to that stamp's.
# The precision recorded is the one the build is designed for; it cannot be read
# back from an engine, hence the reason text. The ONNX is provenance, not proof: an
# engine whose ONNX is not on this host (a fresh install seeded with engines) is
# adopted with the hash "-", which no ONNX can contradict until _ms_pin_adopted_sha
# replaces it with the real hash the first time the ONNX is seen.
_ms_adopt_unstamped() {
    local model="$1" onnx_rel="$2" precision="$3" plan dtype cc trt pe pe_trt pe_cc have_onnx
    [[ "$model" == pe_image_encoder ]] && return 1
    # No stamp proves no recipe: an engine of a model with a recipe is rebuilt, never adopted.
    [[ -z "$(_ms_recipe "$model")" ]] || return 1
    plan="${OP_DIR:?OP_DIR not set}/models/${model}/1/model.plan"
    [[ -s "$plan" ]] || return 1
    dtype="$(_ms_config_input_dtype "$model")"
    [[ -n "$dtype" ]] || return 1
    # Pre-stamp engines were all built with the default target and workspaces: they can
    # only be adopted while the host still asks for exactly that.
    [[ "$(trt_target)" == default && -z "${OP_TRT_WORKSPACE_GB:-}" ]] || return 1
    cc="$(_ms_compute_cap "$model")"
    [[ -n "$cc" ]] || return 1
    trt="$(_ms_trt_version)"
    [[ -n "$trt" ]] || return 1
    pe="$(awk -F'\t' '$1 == "pe_image_encoder"' "$(_ms_precision_file)" 2>/dev/null | tail -n1)"
    if [[ -n "$pe" ]]; then
        pe_trt="$(cut -f4 <<< "$pe")"
        pe_cc="$(cut -f5 <<< "$pe")"
        [[ -n "$pe_trt" && "$pe_trt" == "$trt" ]] || return 1
        # The compute capabilities are only comparable when both engines were built on one card.
        [[ "$(_ms_build_gpu_id "$model")" != "$(_ms_build_gpu_id pe_image_encoder)" || "$pe_cc" == "$cc" ]] || return 1
    fi
    triton_model_ready "$model" || return 1
    have_onnx="$(_ms_onnx_sha256 "$onnx_rel")"
    _ms_record_precision "$model" "$precision" "adopted: built before per-engine stamps" "$onnx_rel" \
        "adopted input=${dtype,,} $(_ms_build_flags "$model")" "${have_onnx:--}"
}

# _ms_pin_adopted_sha MODEL ONNX_REL: replaces the placeholder hash "-" of an adopted
# engine by the hash of its ONNX once that file is on the host. Trust on first sight: the
# engine is the one Triton serves (READY, which proved it valid at adoption and still
# does), and rebuilding it just because its ONNX arrived would be a needless multi-minute
# build; what the pin buys is that every later change of the ONNX is detected. Left as
# "-" while the model is not READY or the ONNX is still absent.
_ms_pin_adopted_sha() {
    local model="$1" now_sha f tmp
    [[ "${OP_DRY_RUN:-0}" == "1" ]] && return 0
    f="$(_ms_precision_file)"
    [[ "$(awk -F'\t' -v m="$model" '$1 == m { s = $6 } END { print s }' "$f" 2>/dev/null)" == "-" ]] || return 0
    now_sha="$(_ms_onnx_sha256 "$2")"
    [[ -n "$now_sha" ]] && triton_model_ready "$model" || return 0
    tmp="$(mktemp "${f}.XXXXXX")"
    awk -F'\t' -v OFS='\t' -v m="$model" -v s="$now_sha" '$1 == m && $6 == "-" { $6 = s } 1' "$f" > "$tmp"
    chmod 600 "$tmp"
    mv -f "$tmp" "$f"
    log_info "pinned the ONNX hash of the adopted ${model} engine"
}

# _ms_group_stamp_stale GROUP: logs and returns 0 when any engine of the group
# has a stamp that no longer matches, so the group is rebuilt instead of
# skipped (a skipped group is what lets its models stay READY).
_ms_group_stamp_stale() {
    local model onnx_rel prec why stale=1
    while IFS=$'\t' read -r model onnx_rel prec; do
        [[ -n "$model" ]] || continue
        if [[ -z "$(awk -F'\t' -v m="$model" '$1 == m' "$(_ms_precision_file)" 2>/dev/null)" ]]; then
            _ms_adopt_unstamped "$model" "$onnx_rel" "$prec" && log_info "group $1: adopted the unstamped ${model} engine"
        fi
        _ms_pin_adopted_sha "$model" "$onnx_rel"
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
            "trt-python-api ${precision}${dtype:+ input=${dtype,,}} $(_ms_build_flags "$model")"
    done < <(model_setup_group_engines "$1")
}
