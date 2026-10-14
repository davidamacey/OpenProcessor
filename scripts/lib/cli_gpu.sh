#!/bin/bash
# =============================================================================
# cli_gpu.sh - gpu plan, vlm and train-mode commands
# =============================================================================
# Sourced lazily by the `openprocessor` CLI, which supplies dc(), _env_value(),
# log_*(), require_checkout(), PROJECT_DIR, SCRIPT_DIR and IS_CHECKOUT.

[[ -n "${_CLI_GPU_SH_LOADED:-}" ]] && return 0
_CLI_GPU_SH_LOADED=1

# Explain where the PE image encoder runs and refuse a TRITON_GPU_ID that this
# host cannot honour (missing card, mixed architectures, no room).
_gpu_plan_placement() {
    local ids pe_mib running=0 project table out line rc=0 primary_mib
    ids="$(_env_value TRITON_GPU_ID)"
    ids="${ids:-0}"
    if ! command -v nvidia-smi &> /dev/null; then
        log_warn "nvidia-smi not found: the placement of TRITON_GPU_ID=${ids} is not verified"
        return 0
    fi
    pe_mib="$(sed -n -E '/GPU_MIB_PER_INSTANCE/{n;s/.*string_value: *"([0-9]+)".*/\1/p}' \
        "${PROJECT_DIR}/models/pe_image_encoder/config.pbtxt" 2>/dev/null | head -1)" || true
    pe_mib="${pe_mib:-1536}"
    project="$(_env_value COMPOSE_PROJECT_NAME)"
    if command -v docker &> /dev/null \
        && docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "${project:-openprocessor}-triton"; then
        running=1
    fi
    case "$(_env_value GPU_PROFILE)" in
        minimal) primary_mib=8192 ;;
        full) primary_mib=17408 ;;
        *) primary_mib=12288 ;;
    esac
    table="$(nvidia-smi --query-gpu=index,name,memory.total,memory.used,compute_cap \
        --format=csv,noheader,nounits | awk -F', ' '{ print $1 "|" $2 "|" $3 "|" $4 "|" $5 }')"
    out="$(gpu_pe_placement "$ids" "$pe_mib" "$table" "$running" "$primary_mib")" || rc=$?
    echo "  PE image encoder (about ${pe_mib} MiB per card):"
    while IFS= read -r line; do
        case "$line" in
            gpu=*) echo "    GPU ${line#gpu=}" ;;
            warn=*) log_warn "${line#warn=}" ;;
            problem=*) log_error "${line#problem=}" ;;
        esac
    done <<< "$out"
    echo "  every other Triton model runs on the primary card (the lowest id in TRITON_GPU_ID)"
    _gpu_plan_decode "$ids" "$table" "$running" "$primary_mib"
    return "$rc"
}

# Say whether the GPU decode is on and why. Refuses nothing: the GPU decode loads on first
# use and falls back to the CPU path per image, so a forced-on card without room only warns.
_gpu_plan_decode() {
    local ids="$1" table="$2" running="$3" primary_mib="$4" setting profile source primary
    local decode_mib=8704
    setting="$(_env_value OP_INGEST_GPU_DECODE)"
    profile="$(_env_value GPU_PROFILE)"
    profile="${profile:-standard}"
    if [[ -n "$setting" ]]; then
        source="OP_INGEST_GPU_DECODE=${setting} in .env"
    else
        setting=1
        source="code default (on); the installer writes 0 for profiles without room"
    fi
    case "${setting,,}" in
        1|true|yes|on) setting=1 ;;
        *) setting=0 ;;
    esac
    if (( setting == 0 )); then
        echo "  GPU decode (ingest): off (${source}); photos decode on the CPU. Needs about ${decode_mib} MiB more on the primary card"
        return 0
    fi
    echo "  GPU decode (ingest): on (${source}); about ${decode_mib} MiB more on the primary card, loaded on first use"
    primary="$(tr ',' '\n' <<< "$ids" | sort -n | head -1)"
    local idx name total used cc need=$(( primary_mib + decode_mib )) have
    # shellcheck disable=SC2034  # name and cc are positional fields of the nvidia-smi row
    while IFS='|' read -r idx name total used cc; do
        [[ "$idx" == "$primary" ]] || continue
        # A running Triton already holds its models, so only the card's total can be judged then.
        if [[ "$running" == 1 ]]; then have=$total; else have=$(( total - used )); fi
        if (( have < need )); then
            log_warn "GPU ${idx} has ${have} MiB $([[ "$running" == 1 ]] && echo total || echo free); the ${profile} profile's models (${primary_mib} MiB) plus the GPU decode (${decode_mib} MiB) need about ${need} MiB. Without room the GPU decode fails to load and ingest stays on the CPU path; set OP_INGEST_GPU_DECODE=0 to silence this"
        fi
    done <<< "$table"
}

cmd_gpu() {
    local sub="${1:-plan}"
    case "$sub" in
        plan)
            log_step "GPU plan (from .env)"
            for key in TRITON_GPU_ID API_GPU_ID SEGMENTER_GPU_ID VLM_GPU_ID EVALUATOR_GPU_ID GPU_PROFILE; do
                local value
                value="$(_env_value "$key")"
                echo "  ${key}=${value:-<unset>}"
            done
            _gpu_plan_placement
            ;;
        *)
            log_error "unknown gpu subcommand: $sub"
            return 1
            ;;
    esac
}

cmd_vlm() {
    local sub="${1:-list}"
    shift 2>/dev/null || true
    case "$sub" in
        list)
            log_step "VLM catalog"
            printf '%-20s %-10s %-8s %-10s\n' "id" "licence" "vram_gb" "status"
            while IFS=$'\t' read -r id _repo licence vram _sctx _mctx _maximg status _rank _gated _key; do
                printf '%-20s %-10s %-8s %-10s\n' "$id" "$licence" "$vram" "$status"
            done < <(vlm_catalog_rows)
            ;;
        status) _vlm_status ;;
        use) _vlm_use "$@" ;;
        apply) _vlm_apply "$@" ;;
        probe) _vlm_probe ;;
        key)
            local action="${1:-}" slug="${2:-}"
            if [[ "$action" != "set" || -z "$slug" ]]; then
                log_error "usage: openprocessor vlm key set <slug>"
                return 2
            fi
            if [[ ! "$slug" =~ ^[a-z0-9][a-z0-9_-]{0,63}$ ]]; then
                log_error "slug must match [a-z0-9][a-z0-9_-]{0,63}"
                return 2
            fi
            if ! { : </dev/tty; } 2>/dev/null; then
                log_error "no terminal to read the key from"
                return 2
            fi
            ( umask 077; mkdir -p "${PROJECT_DIR}/secrets/vlm" )
            chmod 700 "${PROJECT_DIR}/secrets" "${PROJECT_DIR}/secrets/vlm"
            local key_value=""
            printf "VLM API key for '%s' (input hidden): " "$slug" >/dev/tty
            IFS= read -rs key_value </dev/tty || true
            printf '\n' >/dev/tty
            [[ -n "$key_value" ]] || { log_error "empty key"; return 2; }
            ( umask 077; printf '%s' "$key_value" > "${PROJECT_DIR}/secrets/vlm/${slug}" )
            key_value=""
            chmod 600 "${PROJECT_DIR}/secrets/vlm/${slug}"
            log_success "stored secrets/vlm/${slug} (mode 600); reference it as secret:${slug}"
            ;;
        *)
            log_error "unknown vlm subcommand: $sub"
            return 1
            ;;
    esac
}

cmd_train_mode() {
    local mode="${1:-}" state="${PROJECT_DIR}/.install/train_mode" trn seg services=(vlm)
    trn="$(_env_value OP_TRAIN_GPU_ORDER)"
    seg="$(_env_value SEGMENTER_GPU_ID)"
    if [[ -n "$trn" && "$trn" == "$seg" ]]; then
        services+=(segmenter)
    fi
    case "$mode" in
        on)
            log_step "train-mode on: stopping ${services[*]} for the training run"
            dc stop "${services[@]}"
            ( umask 077; mkdir -p "${PROJECT_DIR}/.install"; printf 'on %s\n' "${services[*]}" > "$state" )
            log_success "train-mode on (stopped: ${services[*]})"
            ;;
        off)
            if [[ -f "$state" ]]; then
                read -r _ services_line < "$state" || true
                read -ra services <<< "${services_line:-vlm}"
            fi
            log_step "train-mode off: starting ${services[*]}"
            dc up -d "${services[@]}"
            ( umask 077; printf 'off\n' > "$state" )
            log_success "train-mode off (started: ${services[*]})"
            ;;
        *)
            log_error "usage: openprocessor train-mode on|off"
            return 2
            ;;
    esac
}
