#!/bin/sh
# Replaces __RUNTIME__ placeholder in built JS files with the actual
# PUBLIC_TRITON_API_URL at container start. Lets us bake one image and
# point it at any openprocessor URL via env var.
set -eu
TARGET_URL="${PUBLIC_TRITON_API_URL:-http://localhost:4603}"
find /usr/share/nginx/html -type f \( -name '*.js' -o -name '*.html' \) \
    -exec sed -i "s|__RUNTIME__|${TARGET_URL}|g" {} +
echo "[entrypoint] PUBLIC_TRITON_API_URL=${TARGET_URL}"
