#!/bin/bash
# =============================================================================
# image_keys.sh - the one table of images.lock keys
# =============================================================================
# Sourced by scripts/release/build_and_publish.sh (which writes images.lock)
# and by setup-openprocessor.sh (which reads and verifies it), so the two can
# never disagree on a key name. Each row is
#   key|env_var|kind|spec
# kind "build": spec is "dockerfile|image_name" (pushed as
#   ${OP_IMAGE_NAMESPACE}/<image_name>:<version>)
# kind "third": spec is the upstream image ref the release resolves to a
#   digest; docker-compose.yml reads <env_var> with that ref as its default.
#
# vlm_generic points at the same vLLM build as vlm_gemma4 until any-domain
# W9 verifies a mainline vLLM release for the Qwen catalog rows.

[[ -n "${_IMAGE_KEYS_SH_LOADED:-}" ]] && return 0
_IMAGE_KEYS_SH_LOADED=1

IMAGE_KEYS_TABLE=(
    "api|OP_API_IMAGE|build|Dockerfile|openprocessor"
    "triton|OP_TRITON_IMAGE|build|Dockerfile.triton|openprocessor-triton"
    "evaluator|OP_EVALUATOR_IMAGE|build|docker/evaluator/Dockerfile|openprocessor-evaluator"
    "segmenter|OP_SEGMENTER_IMAGE|build|docker/segmenter/Dockerfile|openprocessor-segmenter"
    "trainer|OP_TRAINER_IMAGE|build|docker/trainer/Dockerfile|openprocessor-trainer"
    "vlm_gemma4|VLM_IMAGE|third|vllm/vllm-openai:gemma4-cu130"
    "vlm_generic|VLM_IMAGE|third|vllm/vllm-openai:gemma4-cu130"
    "opensearch|OPENSEARCH_IMAGE|third|opensearchproject/opensearch:3.6.0"
    "opensearch_dashboards|OPENSEARCH_DASHBOARDS_IMAGE|third|opensearchproject/opensearch-dashboards:3.6.0"
    "mlflow|MLFLOW_IMAGE|third|ghcr.io/mlflow/mlflow:v2.19.0"
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

# image_key_field KEY FIELD -> env | kind | dockerfile | image | source
image_key_field() {
    local want="$1" field="$2" row key env kind a b
    for row in "${IMAGE_KEYS_TABLE[@]}"; do
        IFS='|' read -r key env kind a b <<< "$row"
        [[ "$key" == "$want" ]] || continue
        case "$field" in
            env) echo "$env" ;;
            kind) echo "$kind" ;;
            dockerfile) [[ "$kind" == build ]] && echo "$a" ;;
            image) [[ "$kind" == build ]] && echo "$b" ;;
            source) [[ "$kind" == third ]] && echo "$a" ;;
            *) return 1 ;;
        esac
        return 0
    done
    return 1
}
