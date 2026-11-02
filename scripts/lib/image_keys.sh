#!/bin/bash
# =============================================================================
# image_keys.sh - the one table of images.lock keys
# =============================================================================
# Sourced by scripts/release/build_and_publish.sh (which writes images.lock)
# and by setup-openprocessor.sh (which reads and verifies it), so the two can
# never disagree on a key name. Each row is
#   key|env_var|kind|spec
# kind "build": spec is "dockerfile|image_name|target" (pushed as
#   ${OP_IMAGE_NAMESPACE}/<image_name>:<version>; target is the Dockerfile stage built with
#   `docker build --target`, empty for a single-stage Dockerfile)
# kind "third": spec is the upstream image ref the release resolves to a
#   digest; docker-compose.yml reads <env_var> with that ref as its default.
#
# vlm_generic points at the same vLLM build as vlm_gemma4 until a mainline vLLM release is
# verified for the Qwen catalog rows.

[[ -n "${_IMAGE_KEYS_SH_LOADED:-}" ]] && return 0
_IMAGE_KEYS_SH_LOADED=1

IMAGE_KEYS_TABLE=(
    "api|OP_API_IMAGE|build|Dockerfile|openprocessor|api"
    "export|OP_EXPORT_IMAGE|build|Dockerfile|openprocessor-export|export"
    "evaluator|OP_EVALUATOR_IMAGE|build|Dockerfile|openprocessor-evaluator|evaluator"
    "triton|OP_TRITON_IMAGE|build|Dockerfile.triton|openprocessor-triton|"
    "segmenter|OP_SEGMENTER_IMAGE|build|docker/segmenter/Dockerfile|openprocessor-segmenter|"
    "trainer|OP_TRAINER_IMAGE|build|docker/trainer/Dockerfile|openprocessor-trainer|"
    "cropwright|OP_CROPWRIGHT_IMAGE|build|frontend/Dockerfile|cropwright|"
    "vlm_gemma4|VLM_IMAGE|third|vllm/vllm-openai:gemma4-cu130"
    "vlm_generic|VLM_IMAGE|third|vllm/vllm-openai:gemma4-cu130"
    "opensearch|OPENSEARCH_IMAGE|third|opensearchproject/opensearch:3.6.0"
    "opensearch_dashboards|OPENSEARCH_DASHBOARDS_IMAGE|third|opensearchproject/opensearch-dashboards:3.6.0"
    "mlflow|MLFLOW_IMAGE|third|ghcr.io/mlflow/mlflow:v3.17.0"
    "prometheus|PROMETHEUS_IMAGE|third|prom/prometheus:v3.12.0"
    "grafana|GRAFANA_IMAGE|third|grafana/grafana:13.1.0"
    "loki|LOKI_IMAGE|third|grafana/loki:3.6.12"
    "alloy|ALLOY_IMAGE|third|grafana/alloy:v1.17.1"
    "node_exporter|NODE_EXPORTER_IMAGE|third|prom/node-exporter:v1.10.2"
    "dcgm_exporter|DCGM_EXPORTER_IMAGE|third|nvcr.io/nvidia/k8s/dcgm-exporter:3.3.5-3.4.0-ubuntu22.04"
)

# image_keys [KIND] -> every key (of KIND "build" or "third"), table order
image_keys() {
    local row key kind
    for row in "${IMAGE_KEYS_TABLE[@]}"; do
        IFS='|' read -r key _ kind _ <<< "$row"
        if [[ -z "${1:-}" || "$kind" == "$1" ]]; then
            echo "$key"
        fi
    done
}

# image_key_field KEY FIELD -> env | kind | dockerfile | image | target | source
image_key_field() {
    local want="$1" field="$2" row key env kind a b c
    for row in "${IMAGE_KEYS_TABLE[@]}"; do
        IFS='|' read -r key env kind a b c <<< "$row"
        [[ "$key" == "$want" ]] || continue
        case "$field" in
            env) echo "$env" ;;
            kind) echo "$kind" ;;
            dockerfile) [[ "$kind" == build ]] && echo "$a" ;;
            image) [[ "$kind" == build ]] && echo "$b" ;;
            target) [[ "$kind" == build ]] && echo "$c" ;;
            source) [[ "$kind" == third ]] && echo "$a" ;;
            *) return 1 ;;
        esac
        return 0
    done
    return 1
}

# compose_image_refs -> every image an install pulls, one per line: the active profiles' images
# plus the one-shot model-export image (profile setup), so it is pulled and digest-verified with
# the rest instead of unverified on the first `compose run`. A --profile flag replaces
# COMPOSE_PROFILES from the .env, so the .env profiles are passed explicitly too. Runs inside the
# installer: needs its dc(), read_env_var() and ENV_FILE.
compose_image_refs() {
    local -a args=(--profile setup)
    local p
    while IFS= read -r p; do
        [[ -n "$p" ]] && args+=(--profile "$p")
    done < <(read_env_var "$ENV_FILE" COMPOSE_PROFILES 2>/dev/null | tr ',' '\n' || true)
    dc "${args[@]}" config --images
}
