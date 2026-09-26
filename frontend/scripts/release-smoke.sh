#!/bin/bash
# Smoke-test a built Cropwright image before it is published:
#   scripts/release-smoke.sh <image> [platform]
# Starts the container with no reachable API (the UI must still serve),
# waits for the healthcheck, and checks the page, a JS asset, the
# security headers, the hidden nginx version and the non-root user.
set -euo pipefail

IMAGE="${1:?usage: release-smoke.sh <image> [platform]}"
PLATFORM="${2:-}"
NAME="cropwright-smoke-$$"
PORT="${SMOKE_PORT:-18080}"

cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

fail() {
  echo "FAIL: $*" >&2
  docker logs "$NAME" 2>&1 | tail -20 >&2 || true
  exit 1
}

run_args=(-d --name "$NAME" -p "127.0.0.1:${PORT}:8080"
  --cap-drop ALL --security-opt no-new-privileges:true)
if [[ -n "$PLATFORM" ]]; then
  run_args+=(--platform "$PLATFORM")
fi
docker run "${run_args[@]}" "$IMAGE" >/dev/null

status=""
for _ in $(seq 1 30); do
  status="$(docker inspect -f '{{.State.Health.Status}}' "$NAME" 2>/dev/null || echo missing)"
  [[ "$status" == "healthy" ]] && break
  [[ "$status" == "unhealthy" ]] && fail "healthcheck reported unhealthy"
  sleep 2
done
[[ "$status" == "healthy" ]] || fail "not healthy after 60s (status: $status)"

base="http://127.0.0.1:${PORT}"
headers="$(curl -fsS -D - -o /tmp/cropwright-smoke-index.html "$base/")" \
  || fail "GET / did not return 2xx"
grep -q '<div' /tmp/cropwright-smoke-index.html || fail "GET / returned no HTML body"

for h in X-Frame-Options X-Content-Type-Options Referrer-Policy Permissions-Policy; do
  grep -qi "^${h}:" <<<"$headers" || fail "missing header on /: $h"
done
if grep -qiE '^Server: nginx/[0-9]' <<<"$headers"; then
  fail "Server header exposes the nginx version"
fi

asset="$(grep -oE '/_app/immutable/[^"]+\.js' /tmp/cropwright-smoke-index.html | head -1 || true)"
[[ -n "$asset" ]] || fail "no JS asset referenced from index.html"
curl -fsS -o /dev/null "$base$asset" || fail "GET $asset failed"
curl -fsS -o /dev/null "$base/review" || fail "GET /review (SPA fallback) failed"

uid="$(docker exec "$NAME" id -u)"
[[ "$uid" == "101" ]] || fail "container runs as uid $uid, expected 101"

arch="$(docker image inspect -f '{{.Architecture}}' "$IMAGE")"
echo "OK: $IMAGE ($arch) healthy, headers present, uid 101"
