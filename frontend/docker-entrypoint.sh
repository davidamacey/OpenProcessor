#!/bin/sh
# Replaces __RUNTIME__ placeholder in built JS files with the actual
# PUBLIC_TRITON_API_URL at container start. Lets us bake one image and
# point it at any OpenProcessor URL via env var.
#
# Default is empty — that makes the JS issue *relative* fetches, which
# the labeler's nginx then proxies to API_UPSTREAM over the docker network.
# Works regardless of which host or LAN IP the browser uses to reach
# the labeler. The previous `http://localhost:4603` default broke any
# browser not running on the host (LAN IPs hit their own localhost,
# which has no API). Override the env var only when the labeler needs
# to call a OpenProcessor on a different origin.
#
# Also replaces __API_PREFIX__ (from PUBLIC_API_PREFIX) with the backend
# path prefix, in both the bundled JS/HTML and nginx.conf's proxy
# `location`, so the two stay in lockstep. See API_PREFIX below.
set -eu
TARGET_URL="${PUBLIC_TRITON_API_URL-}"
# Empty is the default (relative URLs). Anything else is substituted into a
# JS string literal through sed, so it is restricted to characters that are
# inert in both (no `&`, `|`, quotes or whitespace).
if [ -n "$TARGET_URL" ] && ! printf '%s' "$TARGET_URL" | grep -Eq '^https?://[][A-Za-z0-9.:/_-]+$'; then
  echo "[entrypoint] PUBLIC_TRITON_API_URL must be empty or http(s)://host[:port][/path] using only A-Za-z0-9.:/_-[], got: $TARGET_URL" >&2
  exit 1
fi

# Unlike the API URL, an EMPTY prefix is never valid — it would produce
# request paths like /health instead of /curation/health. So this defaults
# here as well as in api.ts (normalizeApiPrefix), because the bundle
# sees whatever this substitutes, not the unset env var.
API_PREFIX="${PUBLIC_API_PREFIX:-/curation}"
case "$API_PREFIX" in /*) ;; *) API_PREFIX="/$API_PREFIX" ;; esac
API_PREFIX="${API_PREFIX%/}"
# Also substituted into JS strings and an nginx regex `location`.
if ! printf '%s' "$API_PREFIX" | grep -Eq '^/[A-Za-z0-9/_-]+$'; then
  echo "[entrypoint] PUBLIC_API_PREFIX must be a non-empty path of A-Za-z0-9/_-, got: $API_PREFIX" >&2
  exit 1
fi

# Where nginx proxies API_PREFIX/* to — the OpenProcessor API container,
# reached by name over a shared docker network. Restricted to a plain
# http(s)://host[:port] so it can't break out of the sed below.
API_UPSTREAM="${API_UPSTREAM:-http://op-api:8000}"
case "$API_UPSTREAM" in
  http://*|https://*) ;;
  *) echo "[entrypoint] API_UPSTREAM must be http(s)://host[:port], got: $API_UPSTREAM" >&2; exit 1 ;;
esac
if printf '%s' "$API_UPSTREAM" | grep -q '[^A-Za-z0-9.:/_-]'; then
  echo "[entrypoint] API_UPSTREAM contains unsupported characters: $API_UPSTREAM" >&2
  exit 1
fi

# Where nginx proxies /OpenProcessor/ (the one docs site) to. Same shape
# and validation as API_UPSTREAM.
DOCS_UPSTREAM="${DOCS_UPSTREAM:-http://docs:8080}"
case "$DOCS_UPSTREAM" in
  http://*|https://*) ;;
  *) echo "[entrypoint] DOCS_UPSTREAM must be http(s)://host[:port], got: $DOCS_UPSTREAM" >&2; exit 1 ;;
esac
if printf '%s' "$DOCS_UPSTREAM" | grep -q '[^A-Za-z0-9.:/_-]'; then
  echo "[entrypoint] DOCS_UPSTREAM contains unsupported characters: $DOCS_UPSTREAM" >&2
  exit 1
fi

# Deployment-owned upload/body-size cap (docs/design/
# ingest-ui-and-acceptance-plan-2026-09-24.md §A.4) — ours, not the
# backend's. Substituted into both nginx.conf's client_max_body_size
# (as `<n>m`) and window.__CROPWRIGHT_INGEST_MAX_REQUEST_MB__
# (src/app.html) so the client-side chunk planner and the proxy's actual
# limit can never drift apart. Must be a bare positive integer (MB) —
# it's interpolated directly into nginx config.
INGEST_MAX_REQUEST_MB="${CROPWRIGHT_INGEST_MAX_REQUEST_MB:-256}"
case "$INGEST_MAX_REQUEST_MB" in
  ''|*[!0-9]*)
    echo "[entrypoint] CROPWRIGHT_INGEST_MAX_REQUEST_MB must be a positive integer, got: $INGEST_MAX_REQUEST_MB" >&2
    exit 1
    ;;
esac

# The dataset-archive upload cap (W10 `POST .../datasets/uploads`):
# nginx's client_max_body_size for that one location, and
# window.__CROPWRIGHT_DATASET_UPLOAD_MAX_MB__ for the client's pre-check.
DATASET_UPLOAD_MAX_MB="${CROPWRIGHT_DATASET_UPLOAD_MAX_MB:-2048}"
case "$DATASET_UPLOAD_MAX_MB" in
  ''|*[!0-9]*)
    echo "[entrypoint] CROPWRIGHT_DATASET_UPLOAD_MAX_MB must be a positive integer, got: $DATASET_UPLOAD_MAX_MB" >&2
    exit 1
    ;;
esac

# Gateway mode: serve the monitoring UIs under sub-paths of this one origin.
# Same spelling as the API's OP_GATEWAY_SUBPATHS (CurationConfig._bool), plus an
# explicit refusal of anything else so a typo cannot silently leave it off.
GATEWAY_DIR=/etc/nginx/gateway.d
GATEWAY_SNIPPET=/etc/nginx/snippets/gateway-subpaths.conf
GATEWAY_ON=0
case "$(printf '%s' "${OP_GATEWAY_SUBPATHS:-}" | tr 'A-Z' 'a-z')" in
  1|true|yes|on) GATEWAY_ON=1 ;;
  ''|0|false|no|off) ;;
  *)
    echo "[entrypoint] OP_GATEWAY_SUBPATHS must be true/false (1, true, yes, on / 0, false, no, off), got: ${OP_GATEWAY_SUBPATHS}" >&2
    exit 1
    ;;
esac

# A UI upstream is a bare compose service name over plain http: no IP, no
# dotted name, no path. That is what keeps the gateway from proxying to
# anything outside the compose network.
gateway_upstream() {
  var="$1"; value="$2"
  if ! printf '%s' "$value" | grep -Eq '^http://[a-z][a-z0-9-]*:[0-9]{1,5}$'; then
    echo "[entrypoint] $var must be http://<compose-service-name>:<port> (single-label name, no IP or path), got: $value" >&2
    exit 1
  fi
  printf '%s' "$value"
}

mkdir -p "$GATEWAY_DIR"
rm -f "$GATEWAY_DIR/gateway-subpaths.conf"
if [ "$GATEWAY_ON" != 1 ]; then
  # Off: these paths are not the SPA's. Without this, `try_files` would answer a
  # /grafana/ link with the app shell (200) as if something were served there.
  for sub in grafana prometheus dashboards mlflow; do
    printf 'location ^~ /%s/ {\n    return 404;\n}\nlocation = /%s {\n    return 404;\n}\n' "$sub" "$sub"
  done > "$GATEWAY_DIR/gateway-subpaths.conf"
else
  GRAFANA_UPSTREAM="$(gateway_upstream GRAFANA_UPSTREAM "${GRAFANA_UPSTREAM:-http://grafana:3000}")"
  PROMETHEUS_UPSTREAM="$(gateway_upstream PROMETHEUS_UPSTREAM "${PROMETHEUS_UPSTREAM:-http://prometheus:9090}")"
  DASHBOARDS_UPSTREAM="$(gateway_upstream DASHBOARDS_UPSTREAM "${DASHBOARDS_UPSTREAM:-http://opensearch-dashboards:5601}")"
  MLFLOW_UPSTREAM="$(gateway_upstream MLFLOW_UPSTREAM "${MLFLOW_UPSTREAM:-http://curation-mlflow:5000}")"
  sed "s|__GRAFANA_UPSTREAM__|${GRAFANA_UPSTREAM}|g; s|__PROMETHEUS_UPSTREAM__|${PROMETHEUS_UPSTREAM}|g; s|__DASHBOARDS_UPSTREAM__|${DASHBOARDS_UPSTREAM}|g; s|__MLFLOW_UPSTREAM__|${MLFLOW_UPSTREAM}|g" \
    "$GATEWAY_SNIPPET" > "$GATEWAY_DIR/gateway-subpaths.conf"
fi

find /usr/share/nginx/html -type f \( -name '*.js' -o -name '*.html' \) \
    -exec sed -i "s|__RUNTIME__|${TARGET_URL}|g; s|__API_PREFIX__|${API_PREFIX}|g; s|__INGEST_MAX_REQUEST_MB__|${INGEST_MAX_REQUEST_MB}|g; s|__DATASET_UPLOAD_MAX_MB__|${DATASET_UPLOAD_MAX_MB}|g" {} +

# nginx's proxy `location` must track the same prefix, or the SPA asks
# for {prefix}/... and nginx answers with index.html. This runs as
# /docker-entrypoint.d/40-runtime-config.sh, i.e. before nginx starts.
sed -i "s|__API_PREFIX__|${API_PREFIX}|g; s|__API_UPSTREAM__|${API_UPSTREAM}|g; s|__DOCS_UPSTREAM__|${DOCS_UPSTREAM}|g; s|__INGEST_MAX_REQUEST_MB__|${INGEST_MAX_REQUEST_MB}|g; s|__DATASET_UPLOAD_MAX_MB__|${DATASET_UPLOAD_MAX_MB}|g" /etc/nginx/conf.d/default.conf

echo "[entrypoint] PUBLIC_TRITON_API_URL=${TARGET_URL:-<empty - relative URLs via nginx proxy>}"
echo "[entrypoint] PUBLIC_API_PREFIX=${API_PREFIX}"
echo "[entrypoint] API_UPSTREAM=${API_UPSTREAM}"
echo "[entrypoint] DOCS_UPSTREAM=${DOCS_UPSTREAM}"
echo "[entrypoint] CROPWRIGHT_INGEST_MAX_REQUEST_MB=${INGEST_MAX_REQUEST_MB}"
echo "[entrypoint] CROPWRIGHT_DATASET_UPLOAD_MAX_MB=${DATASET_UPLOAD_MAX_MB}"
echo "[entrypoint] OP_GATEWAY_SUBPATHS=${GATEWAY_ON} (1 = monitoring UIs proxied under /grafana/, /prometheus/, /dashboards/, /mlflow/)"
