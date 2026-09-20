#!/bin/sh
# Replaces __RUNTIME__ placeholder in built JS files with the actual
# PUBLIC_TRITON_API_URL at container start. Lets us bake one image and
# point it at any openprocessor URL via env var.
#
# Default is empty — that makes the JS issue *relative* fetches, which
# the labeler's nginx then proxies to op-api over the docker network.
# Works regardless of which host or LAN IP the browser uses to reach
# the labeler. The previous `http://localhost:4603` default broke any
# browser not running on the host (LAN IPs hit their own localhost,
# which has no API). Override the env var only when the labeler needs
# to call a openprocessor on a different origin.
#
# Also replaces __API_PREFIX__ (from PUBLIC_API_PREFIX) with the backend
# path prefix, in both the bundled JS/HTML and nginx.conf's proxy
# `location`, so the two stay in lockstep. See API_PREFIX below.
set -eu
TARGET_URL="${PUBLIC_TRITON_API_URL-}"

# Unlike the API URL, an EMPTY prefix is never valid — it would produce
# request paths like /health instead of /curation/health. So this defaults
# here as well as in api.ts (normalizeApiPrefix), because the bundle
# sees whatever this substitutes, not the unset env var.
API_PREFIX="${PUBLIC_API_PREFIX:-/curation}"
case "$API_PREFIX" in /*) ;; *) API_PREFIX="/$API_PREFIX" ;; esac
API_PREFIX="${API_PREFIX%/}"

find /usr/share/nginx/html -type f \( -name '*.js' -o -name '*.html' \) \
    -exec sed -i "s|__RUNTIME__|${TARGET_URL}|g; s|__API_PREFIX__|${API_PREFIX}|g" {} +

# nginx's proxy `location` must track the same prefix, or the SPA asks
# for {prefix}/... and nginx answers with index.html. This runs as
# /docker-entrypoint.d/40-runtime-config.sh, i.e. before nginx starts.
sed -i "s|__API_PREFIX__|${API_PREFIX}|g" /etc/nginx/conf.d/default.conf

echo "[entrypoint] PUBLIC_TRITON_API_URL=${TARGET_URL:-<empty - relative URLs via nginx proxy>}"
echo "[entrypoint] PUBLIC_API_PREFIX=${API_PREFIX}"
