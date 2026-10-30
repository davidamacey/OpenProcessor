#!/bin/bash
# =============================================================================
# install_summary.sh - the installer's closing summary (print_summary)
# =============================================================================
# Sourced by setup-openprocessor.sh after the release files are installed, so
# it runs with the installer's globals (ENV_FILE, INSTALL_REF, SELECTED_TIERS,
# OP_* flags) and helpers (read_env_var, _has_tier, is_loopback_ipv4) in scope.
# Kept out of the installer to hold that file to its size baseline.

print_summary() {
    local h="$1" api tri seg vlmp ml failed degraded
    api="$(read_env_var "$ENV_FILE" API_PORT || true)"
    tri="$(read_env_var "$ENV_FILE" TRITON_HTTP_PORT || true)"
    log_step "Summary"
    echo "  install dir : ${OP_REAL_DIR}"
    echo "  release     : $(_short_ref "$INSTALL_REF") (${RESOLVED_MODE}; images: ${IMAGE_MODE})"
    echo "  project     : ${OP_PROJECT}"
    echo "  tiers       : ${SELECTED_TIERS}"
    echo "  GPU plan    : ${GPU_PLAN_SUMMARY:-none}"
    echo "  health      : ${HEALTH_RESULT}"
    local heap per_gb budget
    heap="$(read_env_var "$ENV_FILE" OPENSEARCH_HEAP || true)"
    per_gb="$(read_env_var "$ENV_FILE" OP_SHARDS_PER_HEAP_GB || true)"
    [[ "$per_gb" =~ ^[0-9]+$ ]] || per_gb=20
    budget="$(opensearch_shard_budget "$heap" "$per_gb" || echo unknown)"
    echo "  OpenSearch  : heap ${heap:-unset}, soft shard budget ${budget} (${per_gb} shards per heap GB)"
    echo ""
    echo "  URLs:"
    echo "    API docs    http://${h}:${api}/docs"
    if [[ "$OP_CONTROL_PLANE_ONLY" == 1 ]]; then
        echo "    API alive   http://${h}:${api}/   (/health answers 503 by design: there is no Triton)"
    else
        echo "    API health  http://${h}:${api}/health"
    fi
    [[ "$OP_CONTROL_PLANE_ONLY" == 1 ]] || echo "    Triton      http://${h}:${tri}/v2/health/ready"
    if _has_tier segmenter; then seg="$(read_env_var "$ENV_FILE" SEGMENTER_PORT)"; echo "    Segmenter   http://${h}:${seg}/health"; fi
    if _has_tier vlm; then vlmp="$(read_env_var "$ENV_FILE" VLM_PORT)"; echo "    VLM         http://${h}:${vlmp}/v1/models (${VLM_PICK})"; fi
    if _has_tier trainer; then ml="$(read_env_var "$ENV_FILE" MLFLOW_PORT)"; echo "    MLflow      http://${h}:${ml}"; fi
    if _has_tier cropwright; then
        echo "    Cropwright  http://${h}:${OP_CROPWRIGHT_PORT}"
        if ! is_loopback_ipv4 "${OP_CROPWRIGHT_BIND:-127.0.0.1}"; then
            local lan="$OP_CROPWRIGHT_BIND"
            [[ "$lan" == 0.0.0.0 ]] && lan="$(hostname -I 2>/dev/null | awk '{ print $1 }' || true)"
            echo "    Cropwright on your LAN: http://${lan:-<this-computer-ip>}:${OP_CROPWRIGHT_PORT}"
            log_warn "Cropwright is reachable from your LAN and has NO login: use it only on a trusted network,"
            log_warn "never port-forward it to the internet, and put a reverse proxy with authentication in front"
            log_warn "for anything wider (SECURITY.md). Re-run with --local-only to keep it on this computer."
        fi
    fi
    if [[ "$OP_WITH_MONITORING" == 1 ]]; then
        echo "    Grafana     http://${h}:$(read_env_var "$ENV_FILE" GRAFANA_PORT) (default admin/admin: change it)"
    fi
    [[ -n "$OP_DOCS_URL" ]] && echo "    Docs        ${OP_DOCS_URL}"
    echo ""
    if [[ "$OP_CONTROL_PLANE_ONLY" == 1 ]]; then
        echo "  NOTE: control-plane-only install: NOT a functional inference install (no Triton)."
    fi
    if ! is_loopback_ipv4 "$OP_BIND_ADDRESS"; then
        log_warn "SECURITY: services are published on ${OP_BIND_ADDRESS} with NO authentication. See SECURITY.md."
    elif _has_tier cropwright && ! is_loopback_ipv4 "${OP_CROPWRIGHT_BIND:-127.0.0.1}"; then
        echo "  Security: every OpenProcessor API port is bound to ${OP_BIND_ADDRESS}; Cropwright is the exception (your LAN, above)."
        echo "            The API has no auth; use a reverse proxy before exposing it."
    else
        echo "  Security: every port is bound to ${OP_BIND_ADDRESS} only. The API has no auth; use a reverse proxy before exposing it."
    fi
    failed="$(awk -F'\t' '$2 != "ok" && $2 != "skipped" && $2 != "planned" && $2 != "degraded" { printf "%s ", $1 }' "${OP_DIR}/.install/groups.tsv" 2>/dev/null || true)"
    if [[ -n "$failed" ]]; then
        echo ""
        echo "  Failed model groups: ${failed}"
        for g in $failed; do echo "    re-run: ./openprocessor models install --only ${g}"; done
    fi
    degraded="$(awk -F'\t' '$2 == "degraded" { printf "%s ", $1 }' "${OP_DIR}/.install/groups.tsv" 2>/dev/null || true)"
    if [[ -n "$degraded" ]]; then
        echo ""
        echo "  DEGRADED model groups (an FP32 fallback engine was built instead of FP16): ${degraded}"
        echo "    details: ./openprocessor models status"
        for g in $degraded; do echo "    retry FP16: ./openprocessor models install --only ${g}"; done
    fi
    echo ""
    echo "  Manage: cd ${OP_REAL_DIR} && ./openprocessor status|logs|health|stop|start"
    echo "          ./setup-openprocessor.sh --repair   (also --rollback, --uninstall)"
    [[ "$OP_NO_START" == 1 ]] && echo "  Nothing was started (--no-start): run ./openprocessor start$([[ "$OP_CONTROL_PLANE_ONLY" == 1 ]] || echo ", then ./openprocessor models install")"
    return 0
}
