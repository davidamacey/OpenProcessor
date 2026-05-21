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
set -eu
TARGET_URL="${PUBLIC_TRITON_API_URL-}"
find /usr/share/nginx/html -type f \( -name '*.js' -o -name '*.html' \) \
    -exec sed -i "s|__RUNTIME__|${TARGET_URL}|g" {} +
echo "[entrypoint] PUBLIC_TRITON_API_URL=${TARGET_URL:-<empty - relative URLs via nginx proxy>}"
