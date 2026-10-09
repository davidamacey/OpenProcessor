#!/bin/bash
# =============================================================================
# vlm_switch.sh - `openprocessor vlm status|use|apply|probe` (any-domain W9.7)
# =============================================================================
# Sourced by the `openprocessor` CLI, which supplies dc(), _env_value(),
# log_*(), PROJECT_DIR and API_PORT. Switching the local model recreates the
# vlm container with new arguments, which is a host-side compose operation:
# the API only records the *desired* model (POST /curation/vlm/local/select)
# and reports restart_required. Nothing here talks to a container except
# through dc(), and no secret is ever placed in an argv.

[[ -n "${_VLM_SWITCH_SH_LOADED:-}" ]] && return 0
_VLM_SWITCH_SH_LOADED=1

# The alias the API always sends, whatever model the container serves.
VLM_ALIAS="local-vlm"
# Where the shared state volume is mounted inside the API container.
_VLM_STATE_IN_CONTAINER='${OP_STATE_DIR:-/var/lib/openprocessor}'

# _vlm_host -> the address the host reaches published ports on
_vlm_host() {
    local bind
    bind="$(_env_value OP_BIND_ADDRESS)"
    if [[ -z "$bind" || "$bind" == "0.0.0.0" ]]; then
        echo 127.0.0.1
    else
        echo "$bind"
    fi
}

# _vlm_api METHOD PATH [JSON_BODY] -> prints the response body; status 0 only
# on 2xx (curl never sees a secret: none is sent to these routes)
_vlm_api() {
    local method="$1" path="$2" body="${3:-}" out code
    local -a args=(-sS -o "" -w '%{http_code}' --max-time 60 -X "$method")
    out="$(mktemp)"
    args[2]="$out"
    [[ -n "$body" ]] && args+=(-H 'Content-Type: application/json' -d "$body")
    code="$(curl "${args[@]}" "http://$(_vlm_host):${API_PORT}${path}" 2>/dev/null)" || code=000
    cat "$out"
    rm -f "$out"
    [[ "$code" =~ ^2[0-9][0-9]$ ]]
}

# _vlm_json_field JSON KEY -> the first string value of "KEY":"..." (the API
# answers compact JSON; enough for the few ids read here, no host jq/python)
_vlm_json_field() {
    printf '%s' "$1" | sed -n "s/.*\"$2\":\"\\([^\"]*\\)\".*/\\1/p" | head -1
}

# _vlm_env_backup -> path of a mode-600 copy of .env
_vlm_env_backup() {
    local backup
    backup="$(umask 077; mktemp "${PROJECT_DIR}/.env.vlm-backup.XXXXXX")"
    cp -p "${PROJECT_DIR}/.env" "$backup"
    chmod 600 "$backup"
    echo "$backup"
}

# _vlm_env_set KEY VALUE -- pure-bash atomic rewrite of .env (mode 600), the
# value never in an argv (mirrors setup-openprocessor.sh _env_write)
_vlm_env_set() {
    local key="$1" value="$2" file="${PROJECT_DIR}/.env" tmp line found=0
    [[ "$key" =~ ^[A-Z_][A-Z0-9_]*$ ]] || { log_error "invalid .env key: ${key}"; return 1; }
    if [[ "$value" == *$'\n'* || "$value" == *$'\r'* ]]; then
        log_error "refusing a multi-line value for ${key}"
        return 1
    fi
    tmp="$(umask 077; mktemp "${PROJECT_DIR}/.env.XXXXXX")" || return 1
    chmod 600 "$tmp"
    if ! {
        while IFS= read -r line || [[ -n "$line" ]]; do
            if [[ "$line" == "${key}="* ]]; then
                (( found == 0 )) && printf '%s=%s\n' "$key" "$value"
                found=1
            else
                printf '%s\n' "$line"
            fi
        done < "$file"
        (( found == 0 )) && printf '%s=%s\n' "$key" "$value"
        true
    } > "$tmp" || ! mv -f "$tmp" "$file"; then
        rm -f "$tmp"
        return 1
    fi
}

# _vlm_lock_ref KEY -> the digest-pinned image for an images.lock key
_vlm_lock_ref() {
    awk -F= -v k="$1" '$1 == k { print substr($0, length(k) + 2); exit }' "${PROJECT_DIR}/images.lock" 2>/dev/null
}

# _vlm_card_mib INDEX -> "total free" in MiB for that GPU
_vlm_card_mib() {
    nvidia-smi -i "$1" --query-gpu=memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null \
        | head -1 | tr -d ' ' | tr ',' ' '
}

# _vlm_own_mib INDEX -> MiB the running vlm container holds on GPU INDEX (0 when
# it is not running or nvidia-smi cannot attribute it). The switch recreates that
# container, so this memory is free by the time the new model loads. Same pid ->
# uuid -> index attribution as gpu_own_usage in setup-openprocessor.sh (that
# script is standalone, so it cannot be sourced here), narrowed to the vlm service.
_vlm_own_mib() {
    local gpu="$1" cid pids apps idx
    cid="$(dc ps -q vlm 2>/dev/null | head -1)"
    [[ -n "$cid" ]] || { echo 0; return 0; }
    pids="$(docker top "$cid" -eo pid 2>/dev/null | awk 'NR > 1 { print $1 }' | tr '\n' ' ')"
    [[ -n "${pids// /}" ]] || { echo 0; return 0; }
    apps="$(nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null || true)"
    idx="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null || true)"
    awk -F', *' -v pids="$pids" -v idxmap="$idx" -v gpu="$gpu" '
        BEGIN {
            n = split(pids, a, " "); for (i = 1; i <= n; i++) mine[a[i]] = 1
            m = split(idxmap, rows, "\n")
            for (i = 1; i <= m; i++) { split(rows[i], f, ", *"); if (f[2] != "") byuuid[f[2]] = f[1] }
        }
        ($1 in mine) && ($2 in byuuid) && (byuuid[$2] == gpu) { sum += $3 }
        END { print sum + 0 }' <<< "$apps"
}

# _vlm_in_api SUBCOMMAND -> runs one of the three fixed state-volume actions
# inside the API container (the state volume is not visible from the host)
_vlm_in_api() {
    dc exec -T api sh -c '
        d="'"${_VLM_STATE_IN_CONTAINER}"'"
        s="$d/vlm_worker/pause.sentinel"
        case "$1" in
            lock-check) [ -e "$d/training_gpus.lock" ] ;;
            pause-create)
                mkdir -p "$d/vlm_worker"
                if [ -e "$s" ]; then echo present
                else printf "{\"owner\": \"vlm-switch\", \"created_at\": %s}\n" "$(date +%s)" > "$s"; echo created; fi ;;
            pause-remove) rm -f "$s" ;;
            *) exit 2 ;;
        esac' _ "$1"
}

# _vlm_hf_ok REPO -- a gated entry needs an HF_TOKEN that can see the repo
# (the token goes to curl through a mode-600 header file, never argv)
_vlm_hf_ok() {
    local repo="$1" token hdr code
    token="$(_env_value HF_TOKEN)"
    if [[ -z "$token" ]]; then
        log_error "${repo} is gated: set HF_TOKEN in ${PROJECT_DIR}/.env first"
        return 1
    fi
    hdr="$(umask 077; mktemp)"
    ( umask 077; printf 'Authorization: Bearer %s\n' "$token" > "$hdr" )
    code="$(curl --proto '=https' --tlsv1.2 -sS -o /dev/null -w '%{http_code}' \
        --max-time 30 -H "@${hdr}" "https://huggingface.co/api/models/${repo}" 2>/dev/null)" || true
    rm -f "$hdr"
    if [[ "$code" != "200" ]]; then
        log_error "Hugging Face answered ${code:-000} for ${repo}: accept its terms and check HF_TOKEN"
        return 1
    fi
}

# _vlm_wait_served ID HF_REPO -- vLLM healthy AND serving HF_REPO as the alias
_vlm_wait_served() {
    local hf_repo="$1" limit="${OP_VLM_WAIT_S:-1200}" poll="${OP_VLM_POLL_S:-5}"
    local base waited=0 models
    base="http://$(_vlm_host):$(_env_value VLM_PORT)"
    [[ "$base" == *: ]] && base="http://$(_vlm_host):4612"
    while (( waited <= limit )); do
        if curl -sf --max-time 5 "${base}/health" >/dev/null 2>&1; then
            models="$(curl -sf --max-time 5 "${base}/v1/models" 2>/dev/null || true)"
            if [[ "$models" == *"\"id\":\"${VLM_ALIAS}\""* || "$models" == *"\"id\": \"${VLM_ALIAS}\""* ]] \
                && [[ "$models" == *"\"root\":\"${hf_repo}\""* || "$models" == *"\"root\": \"${hf_repo}\""* ]]; then
                return 0
            fi
        fi
        (( waited % 60 == 0 )) && log_info "waiting for the vlm container to serve ${hf_repo} (${waited}s; the first start downloads the model)"
        sleep "$poll"
        waited=$(( waited + poll ))
    done
    return 1
}

# _vlm_recreate_targets CLIENTS -> services to recreate: vlm plus (when CLIENTS
# is 1: the served alias or the per-call image cap changes) every
# already-running service that reads OP_VLM_MODEL / OP_VLM_MAX_IMAGES_PER_CALL
_vlm_recreate_targets() {
    local clients="$1" svc running
    echo vlm
    [[ "$clients" == 1 ]] || return 0
    running="$(dc ps --services --status running 2>/dev/null || true)"
    for svc in api curation-detection-worker curation-vlm-worker curation-auto-label-worker; do
        [[ $'\n'"$running"$'\n' == *$'\n'"$svc"$'\n'* ]] && echo "$svc"
    done
    return 0
}

# _vlm_no_endpoint ACTION -> the refusal for an install without an in-compose vlm,
# with the steps that enable one on an existing install (docs: VLM selection).
_vlm_no_endpoint() {
    log_error "OP_LOCAL_VLM_ENDPOINT is not set: this install has no in-compose vlm${1:+ to $1}"
    log_info "To enable one on this install, either:"
    log_info "  - re-run the installer with the vlm tier added: ./setup-openprocessor.sh --tiers <your tiers>,vlm"
    log_info "    (it picks a catalog model for your GPU and writes the VLM_* settings and OP_LOCAL_VLM_ENDPOINT=env), or"
    log_info "  - set OP_LOCAL_VLM_ENDPOINT=env in ${PROJECT_DIR}/.env, keep 'vlm' in COMPOSE_PROFILES, then start it:"
    log_info "    docker compose --profile vlm up -d vlm   (or 'openprocessor update' on a checkout)"
    log_info "Docs: Guides > VLM selection, section 'CLI'."
}

_vlm_status() {
    local endpoint body
    endpoint="$(_env_value OP_LOCAL_VLM_ENDPOINT)"
    log_step "Local VLM"
    printf '%-22s %s\n' "catalog id (.env):" "$(_env_value VLM_CATALOG_ID)"
    printf '%-22s %s\n' "model (.env):" "$(_env_value VLM_MODEL)"
    printf '%-22s %s\n' "endpoint:" "${endpoint:-none (no in-compose vlm)}"
    if ! body="$(_vlm_api GET /curation/vlm/local)"; then
        log_warn "the API is not answering; only .env values are shown"
        return 0
    fi
    printf '%-22s %s\n' "serving (probed):" "$(_vlm_json_field "$body" root)"
    local desired
    desired="$(printf '%s' "$body" | sed -n 's/.*"desired":{"catalog_id":"\([^"]*\)".*/\1/p' | head -1)"
    printf '%-22s %s\n' "desired:" "${desired:-none requested}"
    if [[ "$body" == *'"restart_required":true'* ]]; then
        log_warn "a different model is desired: run 'openprocessor vlm apply'"
    fi
}

_vlm_probe() {
    local endpoint body
    endpoint="$(_env_value OP_LOCAL_VLM_ENDPOINT)"
    [[ -n "$endpoint" ]] || { _vlm_no_endpoint ""; return 1; }
    if ! body="$(_vlm_api POST "/curation/vlm/endpoints/${endpoint}/probe")"; then
        log_error "probe of '${endpoint}' failed: ${body}"
        return 1
    fi
    log_success "probed '${endpoint}'"
    printf '%s' "$body" | grep -o '"message":"[^"]*"' | sed 's/^"message":"/  warning: /;s/"$//' || true
}

# State the interrupt handler needs: what this run changed so far.
_VLM_PAUSE_OWNED=0
_VLM_ENV_BACKUP=""

# Ctrl-C / SIGTERM mid-switch: remove the pause this run created and put the
# previous .env back, so the workers are not left paused.
_vlm_on_signal() {
    trap - INT TERM
    log_warn "interrupted: undoing what this switch changed"
    if [[ -n "$_VLM_ENV_BACKUP" && -f "$_VLM_ENV_BACKUP" ]]; then
        cp -p "$_VLM_ENV_BACKUP" "${PROJECT_DIR}/.env" && chmod 600 "${PROJECT_DIR}/.env"
        rm -f "$_VLM_ENV_BACKUP"
    fi
    (( _VLM_PAUSE_OWNED == 1 )) && _vlm_in_api pause-remove
    log_info "re-run 'openprocessor vlm use <id>' (or 'vlm apply') to finish the switch"
    exit 130
}

# _vlm_use ID [--force] [--yes]
_vlm_use() {
    local rc
    _VLM_PAUSE_OWNED=0
    _VLM_ENV_BACKUP=""
    trap _vlm_on_signal INT TERM
    _vlm_switch "$@"
    rc=$?
    trap - INT TERM
    return "$rc"
}

_vlm_switch() {
    local id="" force=0 yes=0 arg
    for arg in "$@"; do
        case "$arg" in
            --force) force=1 ;;
            --yes) yes=1 ;;
            -*) log_error "unknown option: ${arg}"; return 2 ;;
            *) id="$arg" ;;
        esac
    done
    [[ -n "$id" ]] || { log_error "usage: openprocessor vlm use <id> [--force] [--yes]"; return 2; }
    [[ -f "${PROJECT_DIR}/.env" ]] || { log_error "no .env in ${PROJECT_DIR}"; return 1; }

    local hf_repo vram_gb gated image_key
    hf_repo="$(vlm_catalog_field "$id" hf_repo)" \
        || { log_error "'${id}' is not in the catalog (openprocessor vlm list)"; return 1; }
    vram_gb="$(vlm_catalog_field "$id" vram_gb)"
    gated="$(vlm_catalog_field "$id" gated)"
    image_key="$(vlm_catalog_field "$id" vllm_image_key)"
    local endpoint gpu_id total_mib free_mib
    endpoint="$(_env_value OP_LOCAL_VLM_ENDPOINT)"
    [[ -n "$endpoint" ]] || { _vlm_no_endpoint "switch"; return 1; }
    gpu_id="$(_env_value VLM_GPU_ID)"; gpu_id="${gpu_id:-0}"

    # 1. fits the card
    read -r total_mib free_mib <<< "$(_vlm_card_mib "$gpu_id")"
    if [[ -z "${total_mib:-}" ]]; then
        log_error "cannot read GPU ${gpu_id} (nvidia-smi)"
        return 1
    fi
    local total_gb need_mib
    total_gb="$(awk -v t="$total_mib" 'BEGIN { printf "%.1f", t / 1024 }')"
    need_mib="$(awk -v v="$vram_gb" 'BEGIN { printf "%d", v * 1024 }')"
    if (( total_mib < need_mib )); then
        if (( force == 0 )); then
            log_error "${id} needs about ${vram_gb} GB; GPU ${gpu_id} has ${total_gb} GB (--force to try anyway)"
            return 1
        fi
        log_warn "${id} needs about ${vram_gb} GB; GPU ${gpu_id} has ${total_gb} GB (forced)"
    fi
    local own_mib
    own_mib="$(_vlm_own_mib "$gpu_id")"
    free_mib=$(( free_mib + own_mib ))
    if (( free_mib < need_mib )) && (( force == 0 )); then
        log_error "${id} needs about ${vram_gb} GB free; GPU ${gpu_id} would have $(( free_mib / 1024 )) GB free once the running vlm (${own_mib} MiB) is replaced (--force to try anyway)"
        return 1
    fi
    if [[ "$gated" == "true" ]]; then
        _vlm_hf_ok "$hf_repo" || return 1
    fi
    local image
    image="$(_vlm_lock_ref "$image_key")"
    if [[ -z "$image" ]]; then
        log_error "images.lock has no '${image_key}' entry; cannot pin the vLLM image for ${id}"
        return 1
    fi

    # 2. a training run owns the GPUs
    if _vlm_in_api lock-check; then
        log_error "a training run holds the GPUs (training_gpus.lock); switch the VLM after it finishes"
        return 1
    fi

    local migrate=0 current_alias clients=0 old_max new_max
    current_alias="$(_env_value VLM_SERVED_MODEL_NAME)"
    [[ "$current_alias" == "$VLM_ALIAS" ]] || migrate=1
    # The API and workers read the image cap when they start, so a changed cap
    # needs them recreated too (the compose default is 8).
    old_max="$(_env_value OP_VLM_MAX_IMAGES_PER_CALL)"; old_max="${old_max:-8}"
    new_max="$(vlm_catalog_field "$id" max_images)"
    { (( migrate == 1 )) || [[ "$old_max" != "$new_max" ]]; } && clients=1
    if (( migrate == 1 )); then
        log_warn "this install still serves the model as '${current_alias:-the compose default}': this one time the API and workers are recreated too, so they send '${VLM_ALIAS}'"
    fi
    if (( yes == 0 )) && [[ -t 0 ]]; then
        local answer
        read -r -p "Switch the local VLM to ${id} (${hf_repo})? [y/N] " answer
        [[ "$answer" =~ ^[Yy] ]] || { log_info "cancelled"; return 1; }
    fi

    # 3. pause the workers (only if nobody else already did)
    local created_pause=0 paused
    paused="$(_vlm_in_api pause-create 2>/dev/null || true)"
    if [[ "$paused" != created && "$paused" != present ]]; then
        log_error "cannot reach the api container to pause the workers; is the stack running?"
        return 1
    fi
    [[ "$paused" == created ]] && created_pause=1
    _VLM_PAUSE_OWNED=$created_pause

    # 4. .env, with a backup to restore on failure
    local backup targets
    backup="$(_vlm_env_backup)" || {
        log_error "cannot back up .env; nothing was changed"
        (( created_pause == 1 )) && _vlm_in_api pause-remove
        return 1
    }
    _VLM_ENV_BACKUP="$backup"
    _vlm_write_env "$id" "$hf_repo" "$image" "$gpu_id" "$total_mib" "$migrate" || {
        _vlm_fail "$backup" "$created_pause" "$clients"
        return 1
    }
    readarray -t targets < <(_vlm_recreate_targets "$clients")

    # 5. recreate and wait until the new model is really served
    log_info "recreating: ${targets[*]}"
    if ! dc up -d "${targets[@]}"; then
        log_error "docker compose could not recreate ${targets[*]}"
        _vlm_fail "$backup" "$created_pause" "$clients"
        return 1
    fi
    if ! _vlm_wait_served "$hf_repo"; then
        log_error "the vlm container did not serve ${hf_repo} as '${VLM_ALIAS}' in time"
        _vlm_fail "$backup" "$created_pause" "$clients"
        return 1
    fi

    # 6. record the new identity, then drop the request
    local probe_out
    if ! probe_out="$(_vlm_api POST "/curation/vlm/endpoints/${endpoint}/probe")"; then
        log_error "the new model is up but probing '${endpoint}' failed: ${probe_out}"
        _vlm_fail "$backup" "$created_pause" "$clients"
        return 1
    fi
    _vlm_api DELETE /curation/vlm/local/select >/dev/null || log_warn "could not clear the desired-model request"

    # 7. unpause (only what this command paused) and report
    (( created_pause == 1 )) && _vlm_in_api pause-remove || true
    rm -f "$backup"
    log_success "the local VLM now serves ${id} (${hf_repo})"
    printf '%s' "$probe_out" | grep -o '"message":"[^"]*"' | sed 's/^"message":"/  warning: /;s/"$//' || true
}

# _vlm_write_env ID HF_REPO IMAGE GPU_ID TOTAL_MIB MIGRATE
_vlm_write_env() {
    local id="$1" hf_repo="$2" image="$3" total_mib="$5" migrate="$6"
    local util total_gb
    total_gb="$(awk -v t="$total_mib" 'BEGIN { printf "%.1f", t / 1024 }')"
    util="$(vlm_gpu_memory_utilization "$(vlm_catalog_field "$id" vram_gb)" "$total_gb")"
    _vlm_env_set VLM_CATALOG_ID "$id" \
        && _vlm_env_set VLM_MODEL "$hf_repo" \
        && _vlm_env_set VLM_IMAGE "$image" \
        && _vlm_env_set VLM_REASONING_PARSER "$(vlm_catalog_field "$id" reasoning_parser)" \
        && _vlm_env_set VLM_CHAT_TEMPLATE "$(vlm_catalog_field "$id" chat_template)" \
        && _vlm_env_set VLM_EXTRA_ARGS "$(vlm_catalog_field "$id" extra_args)" \
        && _vlm_env_set VLM_MAX_MODEL_LEN "$(vlm_catalog_field "$id" served_context)" \
        && _vlm_env_set VLM_LIMIT_MM_IMAGES "$(vlm_catalog_field "$id" max_images)" \
        && _vlm_env_set VLM_GPU_MEMORY_UTILIZATION "$util" \
        && _vlm_env_set OP_VLM_MAX_IMAGES_PER_CALL "$(vlm_catalog_field "$id" max_images)" \
        || return 1
    if [[ "$migrate" == 1 ]]; then
        _vlm_env_set VLM_SERVED_MODEL_NAME "$VLM_ALIAS" && _vlm_env_set OP_VLM_MODEL "$VLM_ALIAS" || return 1
    fi
}

# _vlm_fail BACKUP CREATED_PAUSE CLIENTS -- put .env and the container back
_vlm_fail() {
    local backup="$1" created_pause="$2" clients="$3" targets
    log_warn "restoring the previous .env and recreating the vlm container on it"
    cp -p "$backup" "${PROJECT_DIR}/.env" && chmod 600 "${PROJECT_DIR}/.env"
    rm -f "$backup"
    readarray -t targets < <(_vlm_recreate_targets "$clients")
    dc up -d "${targets[@]}" || log_error "could not recreate ${targets[*]} on the old values; run 'docker compose up -d vlm' yourself"
    (( created_pause == 1 )) && _vlm_in_api pause-remove || true
}

_vlm_apply() {
    local body id
    body="$(_vlm_api GET /curation/vlm/local)" || { log_error "the API is not answering"; return 1; }
    id="$(printf '%s' "$body" | sed -n 's/.*"desired":{"catalog_id":"\([^"]*\)".*/\1/p' | head -1)"
    [[ -n "$id" ]] || { log_info "nothing to apply: no local model has been requested"; return 0; }
    _vlm_use "$id" "$@"
}
