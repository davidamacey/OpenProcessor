#!/bin/bash
# =============================================================================
# cli_core.sh - models, curation, update, help, sample, config and lifecycle commands
# =============================================================================
# Sourced lazily by the `openprocessor` CLI, which supplies dc(), _env_value(),
# log_*(), require_checkout(), PROJECT_DIR, SCRIPT_DIR and IS_CHECKOUT.

# The CLI runs under `set -e`, so a failed cd already aborts the command.
# shellcheck disable=SC2164

[[ -n "${_CLI_CORE_SH_LOADED:-}" ]] && return 0
_CLI_CORE_SH_LOADED=1

# =============================================================================
# Model Management
# =============================================================================

cmd_models() {
    local sub="${1:-status}"
    shift 2>/dev/null || true
    case "$sub" in
        status|list) cmd_models_status ;;
        install|repair)
            local only=""
            while [[ $# -gt 0 ]]; do
                case "$1" in
                    --only) only="${2:?--only needs a group name}"; shift 2 ;;
                    --build-target) OP_TRT_BUILD_TARGET="${2:?--build-target needs default or small}"; shift 2 ;;
                    --workspace) OP_TRT_WORKSPACE_GB="${2:?--workspace needs a whole number of GB}"; shift 2 ;;
                    --yes|-y) shift ;; # never prompts; accepted so scripted callers can pass the same flag everywhere
                    *) log_error "unknown option: $1"; return 2 ;;
                esac
            done
            export OP_TRT_BUILD_TARGET OP_TRT_WORKSPACE_GB
            cmd_models_install "$only"
            ;;
        *)
            log_error "usage: openprocessor models status|install|repair [--only GROUP] [--build-target default|small] [--workspace GB] [--yes]"
            return 2
            ;;
    esac
}

# Engines that were not built at the preferred precision (the installer's
# precision ledger). A local file, so it prints even when Triton is down.
_models_precision_section() {
    local degraded model precision reason
    # shellcheck source=scripts/lib/model_setup.sh
    source "${SCRIPT_DIR}/lib/model_setup.sh"
    local stamps trt cc sha
    stamps="$(OP_DIR="$PROJECT_DIR" model_setup_stamp_report)"
    if [[ -n "$stamps" ]]; then
        echo -e "${BOLD}Engine build stamps:${NC}"
        while IFS=$'\t' read -r model precision trt cc sha; do
            echo "  ${model}: ${precision}, TensorRT ${trt}, compute capability ${cc}, ONNX ${sha}"
        done <<< "$stamps"
        echo ""
    fi
    degraded="$(OP_DIR="$PROJECT_DIR" model_setup_precision_report)"
    [[ -n "$degraded" ]] || return 0
    echo -e "${BOLD}Degraded precision:${NC}"
    while IFS=$'\t' read -r model precision reason; do
        echo -e "  ${YELLOW}[${precision^^}]${NC} ${model}: not the FP16 default. ${reason}"
    done <<< "$degraded"
    echo "  Retry FP16 after fixing the cause: ./openprocessor models install --only pe"
    echo ""
}

cmd_models_status() {
    _models_precision_section
    log_info "Fetching the model repository index from Triton..."
    local response
    response=$(curl -s -X POST "localhost:${TRITON_HTTP_PORT}/v2/repository/index" 2>/dev/null)
    if [[ -z "$response" ]]; then
        log_error "Triton not responding"
        return 1
    fi
    echo ""
    echo -e "${BOLD}Models:${NC}"
    # No host Python: split the JSON array into objects with sed.
    printf '%s\n' "$response" | sed 's/},{/}\n{/g' | while IFS= read -r obj; do
        local name state
        name="$(printf '%s' "$obj" | sed -n -E 's/.*"name":"([^"]*)".*/\1/p')"
        state="$(printf '%s' "$obj" | sed -n -E 's/.*"state":"([^"]*)".*/\1/p')"
        [[ -z "$name" ]] && continue
        if [[ "$state" == READY ]]; then
            echo -e "  ${GREEN}[READY]${NC} ${name}"
        else
            echo -e "  ${YELLOW}[${state:-UNAVAILABLE}]${NC} ${name}"
        fi
    done
}

# One implementation with the installer: scripts/lib/model_setup.sh.
cmd_models_install() {
    local only="$1" tiers="core" profiles
    # shellcheck source=scripts/lib/model_setup.sh
    source "${SCRIPT_DIR}/lib/model_setup.sh"
    profiles="$(_env_value COMPOSE_PROFILES)"
    [[ ",${profiles}," == *",curation,"* ]] && tiers="core curation"
    OP_DIR="$PROJECT_DIR"
    OP_FACES_ENABLED="$(_env_value OP_FACES_ENABLED)"
    OP_HEALTH_HOST=127.0.0.1
    GPU_PROFILE="$(_env_value GPU_PROFILE)"
    TRITON_GPU_ID="$(_env_value TRITON_GPU_ID)"
    API_GPU_ID="$(_env_value API_GPU_ID)"
    # A flag beats .env; .env (what the model-export-worker also reads) beats the profile.
    OP_TRT_BUILD_TARGET="${OP_TRT_BUILD_TARGET:-$(_env_value OP_TRT_BUILD_TARGET)}"
    OP_TRT_WORKSPACE_GB="${OP_TRT_WORKSPACE_GB:-$(_env_value OP_TRT_WORKSPACE_GB)}"
    MODEL_SETUP_TRITON_DIGEST=""
    if [[ -f "$PROJECT_DIR/.install/state.json" ]]; then
        MODEL_SETUP_TRITON_DIGEST="$(sed -n -E 's/^  "triton_image_digest": "([^"]*)",?$/\1/p' "$PROJECT_DIR/.install/state.json" | head -n1)"
    else
        log_info "no .install/state.json (a checkout, not an installed directory): every model group is built"
    fi
    export OP_DIR OP_FACES_ENABLED OP_HEALTH_HOST GPU_PROFILE MODEL_SETUP_TRITON_DIGEST \
        TRITON_GPU_ID API_GPU_ID OP_TRT_BUILD_TARGET OP_TRT_WORKSPACE_GB
    dc up -d triton-server
    if model_setup_run_groups "$tiers" "$only"; then
        log_success "model groups done"
    else
        log_error "some model groups failed; see ${PROJECT_DIR}/.install/logs"
        return 1
    fi
    if [[ -z "$only" || "$only" == pe ]] && [[ "$tiers" == *curation* ]]; then
        dc restart api
    fi
}

# faces purge [--dry-run] [--yes]: forwards exactly the flags given to the purge script. A real
# purge is refused unless faces are off in .env AND in the running api (an edited .env that was
# never applied would let the api keep writing face data while it is deleted).
cmd_faces() {
    local sub="${1:-}" flag
    if [[ "$sub" != purge ]]; then
        log_error "usage: openprocessor faces purge [--dry-run] [--yes]"
        return 2
    fi
    shift
    local -a flags=()
    for flag in "$@"; do
        case "$flag" in
            --dry-run|--yes) flags+=("$flag") ;;
            *)
                log_error "unknown option '${flag}'; usage: openprocessor faces purge [--dry-run] [--yes]"
                return 2
                ;;
        esac
    done
    if [[ " ${flags[*]} " == *" --yes "* ]]; then
        local configured running
        configured="$(_env_value OP_FACES_ENABLED | tr '[:upper:]' '[:lower:]')"
        case "$configured" in
            0|false|no|off) ;;
            *)
                log_error "refusing to purge while faces are on: set OP_FACES_ENABLED=false in ${PROJECT_DIR}/.env and restart the api first"
                return 2
                ;;
        esac
        if [[ -n "$(dc ps --status running -q api 2>/dev/null)" ]]; then
            running="$(dc exec -T api printenv OP_FACES_ENABLED 2>/dev/null | tr '[:upper:]' '[:lower:]' | tr -d '\r')"
            case "$running" in
                0|false|no|off) ;;
                *)
                    log_error "the running api still has faces on: ./openprocessor restart api, then purge again"
                    return 2
                    ;;
            esac
        fi
    fi
    dc run --rm --no-deps -T -w /app api python -m scripts.purge_face_data "${flags[@]}"
}

cmd_export() {
    local model="${1:-all}"

    load_export_lib
    OP_FACES_ENABLED="$(_env_value OP_FACES_ENABLED)"
    export OP_FACES_ENABLED

    case "$model" in
        all)
            export_all_models
            ;;
        essential)
            export_essential_models
            ;;
        status)
            check_exported_models
            ;;
        *)
            export_model "$model"
            ;;
    esac
}

cmd_download() {
    local target="${1:-all}"

    load_download_lib

    case "$target" in
        all)
            download_all_models
            ;;
        essential)
            download_essential_models
            ;;
        ocr|paddleocr)
            download_paddleocr_models
            ;;
        status)
            check_downloaded_models
            ;;
        *)
            download_model "$target"
            ;;
    esac
}

# =============================================================================
# Curation subsystem — EXPERIMENTAL, opt-in compose profile
# =============================================================================
# Mirrors the `curation-*` Makefile targets. The curation worker services
# (curation-detection-worker, curation-vlm-worker, curation-auto-label-worker,
# curation-assist-loop-worker, curation-cluster-refresh) all carry `profiles: [curation]` in
# docker-compose.yml, so `start`/`stop` above never touch them.

# Every service in the curation profile, plus the on-demand evaluator (profile
# `evaluator`, so a bare `curation up` leaves it alone).
CURATION_SERVICES="curation-detection-worker curation-vlm-worker curation-auto-label-worker curation-assist-loop-worker curation-cluster-refresh curation-evaluator"

cmd_curation() {
    local subcommand="${1:-status}"
    shift 2>/dev/null || true

    cd "$PROJECT_DIR"

    case "$subcommand" in
        up)
            if (( $# > 0 )); then
                local svc
                for svc in "$@"; do
                    case " $CURATION_SERVICES " in
                        *" $svc "*) ;;
                        *)
                            log_error "Not a curation service: $svc"
                            echo "Curation services: $CURATION_SERVICES"
                            return 1
                            ;;
                    esac
                done
                log_step "Starting curation services: $*..."
            else
                log_step "Starting curation subsystem (profile: curation)..."
            fi
            dc --profile curation up -d "$@"
            log_success "Curation services starting. Check with: $0 curation status"
            ;;
        down)
            log_step "Stopping curation subsystem (profile: curation)..."
            dc --profile curation down
            log_success "Curation services stopped"
            ;;
        logs)
            dc --profile curation logs -f \
                curation-detection-worker curation-vlm-worker \
                curation-auto-label-worker curation-assist-loop-worker curation-cluster-refresh
            ;;
        status)
            dc --profile curation ps \
                curation-detection-worker curation-vlm-worker \
                curation-auto-label-worker curation-assist-loop-worker curation-cluster-refresh curation-evaluator
            ;;
        seed)
            log_warn "curation seed: not yet implemented."
            echo "scripts/curation/seed_live_harness.py does not exist on this branch yet"
            echo "(live write-path verification harness). Nothing was run."
            ;;
        *)
            log_error "Unknown curation subcommand: $subcommand"
            echo "Available: up [service...], down, logs, status, seed"
            return 1
            ;;
    esac
}

# =============================================================================
# Update
# =============================================================================

cmd_update() {
    if [[ "$IS_CHECKOUT" != 1 ]]; then
        cmd_upgrade "$@"
        return
    fi
    log_step "Updating OpenProcessor (checkout)..."

    cd "$PROJECT_DIR"

    # Pull latest images
    log_info "Pulling latest Docker images..."
    dc pull

    # Rebuild if needed
    log_info "Rebuilding containers..."
    dc build
    # The one-shot export image sits in the `setup` profile, which a plain build skips.
    dc --profile setup build model-export

    log_success "Update complete"
    log_info "Restart to apply: ./scripts/openprocessor.sh restart"
}

# =============================================================================
# Help
# =============================================================================

show_help() {
    cat << EOF
OpenProcessor Management CLI

Usage: ./scripts/openprocessor.sh <command> [options]

Service Commands:
  start               Start all services
  stop                Stop all services
  restart [service]   Restart service(s)
  logs [service] [-f] View logs (optionally follow)
  status              Show service status and health
  health              Check API health endpoint

Model Commands:
  models [status]     List Triton models and their state
  models install [--only GROUP] [--build-target default|small] [--workspace GB] [--yes]
                      Export/load model groups (skips up-to-date ones)
  models repair       Same as install: re-runs failed or missing groups
  export <model>      Export model to TensorRT
                      Options: yolo, scrfd, arcface, clip-image, clip-text, ocr, all, essential, status
  download <model>    Download model
                      Options: all, essential, ocr, status, <model-name>

Configuration:
  profile [name]      Show/switch GPU profile
                      Options: minimal, small, standard, full

Testing:
  test [type]         Run tests
                      Options: quick (default), full, visual

Face data (OP_FACES_ENABLED):
  faces purge [--dry-run] [--yes]
                      Report (default, or --dry-run) or delete (--yes) every stored face
                      record, person id and name; --yes needs OP_FACES_ENABLED=false first

Curation (experimental, opt-in compose profile):
  curation [action]   Manage the curation worker services
                      Options: up [service...], down, logs, status (default), seed

Maintenance:
  clean [target]      Clean files
                      Options: cache (default), models, exports, all
  update              Checkout: pull + rebuild. Install dir: same as upgrade
  upgrade [--version vX.Y.Z]   Upgrade this install (setup-openprocessor.sh)
  repair              Re-verify files, re-pull images, re-run failed model groups
  uninstall [--purge-volumes] [--purge-data] [--remove-images]

Deploy:
  sample coco [--full]  Fetch the public COCO sample (inside the API image)
  config show           Print .env with every secret redacted
  gpu plan              Show the GPU placement from .env
  cropwright password   Set the Cropwright basic-auth login (an empty answer generates one)
  vlm list|status|use <id> [--force] [--yes]|apply|probe|key set <slug>
  train-mode on|off     Stop/start the VLM (and a co-located segmenter) for training

Setup:
  setup               Run initial setup wizard

Examples:
  ./scripts/openprocessor.sh status
  ./scripts/openprocessor.sh logs api -f
  ./scripts/openprocessor.sh export all
  ./scripts/openprocessor.sh profile standard
  ./scripts/openprocessor.sh test full
  ./scripts/openprocessor.sh clean cache

EOF
}

# =============================================================================
# New in the deploy-mode CLI (installer plan section 5.5)
# =============================================================================

cmd_sample() {
    local sub="${1:-coco}"
    shift 2>/dev/null || true
    case "$sub" in
        coco)
            # shellcheck source=scripts/lib/model_setup.sh
            source "${SCRIPT_DIR}/lib/model_setup.sh"
            log_step "Fetching the public COCO sample (writes to /app/data/samples, no host Python)..."
            model_setup_sample_coco "${1:-}"
            ;;
        *)
            log_error "unknown sample: $sub (only 'coco' is supported)"
            return 1
            ;;
    esac
}

# cropwright password: set the basic-auth login in front of Cropwright (an empty answer
# generates one). The password goes to secrets/cropwright/ only, never to the terminal log.
cmd_cropwright() {
    local sub="${1:-}" pw=""
    if [[ "$sub" != "password" ]]; then
        log_error "usage: openprocessor cropwright password"
        return 2
    fi
    [[ -f "${PROJECT_DIR}/.env" ]] || { log_error "no .env at ${PROJECT_DIR}/.env"; return 1; }
    if { : </dev/tty; } 2>/dev/null; then
        printf "New Cropwright password (input hidden, empty = generate one): " >/dev/tty
        IFS= read -rs pw </dev/tty || true
        printf '\n' >/dev/tty
    fi
    if [[ -z "$pw" ]]; then
        pw="$(cropwright_generate_password)" || { log_error "cannot generate a password (is openssl installed?)"; return 1; }
    fi
    cropwright_write_login "$PROJECT_DIR" "$pw" || return 1
    pw=""
    _vlm_env_set OP_CROPWRIGHT_BASIC_AUTH true \
        && _vlm_env_set OP_CROPWRIGHT_BASIC_AUTH_FILE ./secrets/cropwright/htpasswd || return 1
    log_success "Cropwright login '${CROPWRIGHT_AUTH_USER}' stored (password in secrets/cropwright/password, mode 600)"
    log_info "apply it: ./openprocessor restart cropwright"
}

cmd_config() {
    local sub="${1:-show}"
    case "$sub" in
        show)
            local env_file="${PROJECT_DIR}/.env"
            [[ -f "$env_file" ]] || { log_error "no .env at ${env_file}"; return 1; }
            # Redact every token/key/secret/password/credential/auth value, with or
            # without a leading `export`.
            sed -E 's/^([[:space:]]*(export[[:space:]]+)?[A-Z0-9_]*(TOKEN|KEY|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH)[A-Z0-9_]*)=.*/\1=***/' "$env_file"
            ;;
        *)
            log_error "unknown config subcommand: $sub"
            return 1
            ;;
    esac
}

# Explain where the PE image encoder runs and refuse a TRITON_GPU_ID that this
# host cannot honour (missing card, mixed architectures, no room).
# One code path: lifecycle commands run this install's own verified
# setup-openprocessor.sh against this directory.
_run_setup() {
    if [[ "$IS_CHECKOUT" == 1 || ! -x "$PROJECT_DIR/setup-openprocessor.sh" ]]; then
        log_error "this is not an installer-managed directory (use git + make in a checkout)"
        return 1
    fi
    resolve_project
    exec "$PROJECT_DIR/setup-openprocessor.sh" --dir "$PROJECT_DIR" "$@"
}

cmd_upgrade() { _run_setup "$@"; }
cmd_repair() { _run_setup --repair "$@"; }
cmd_uninstall() { _run_setup --uninstall "$@"; }
