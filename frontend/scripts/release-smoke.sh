#!/bin/bash
# Smoke-test a built Cropwright image before it is published:
#   scripts/release-smoke.sh <image> [platform]
# Starts the container with no reachable API (the UI must still serve),
# waits for the healthcheck, and checks the page, a JS asset, the
# security headers, the hidden nginx version and the non-root user.
#
# Every HTTP check runs INSIDE the container via `docker exec ... wget`
# rather than curl against a published host port. This makes the script
# location-agnostic: `DOCKER_CONTEXT=remote-arm64 scripts/release-smoke.sh
# <image>` runs the whole check against a remote docker engine (used by
# 60-smoke.sh to test the arm64 leg natively on the Mac Studio builder
# node) with no port-forwarding or SSH tunnel needed — `docker exec`
# already crosses that boundary for every other command here.
set -euo pipefail

IMAGE="${1:?usage: release-smoke.sh <image> [platform]}"
PLATFORM="${2:-}"
NAME="cropwright-smoke-$$"

cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

fail() {
  echo "FAIL: $*" >&2
  docker logs "$NAME" 2>&1 | tail -20 >&2 || true
  exit 1
}

run_args=(-d --name "$NAME" --cap-drop ALL --security-opt no-new-privileges:true)
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

wget_get() {
  # $1=path -> prints "<headers>\n---BODY---\n<body>" from inside the container,
  # exiting with wget's own status (not cat's — `sh -c` last-command-wins would
  # otherwise mask a 404/500 as success once cat prints something).
  docker exec "$NAME" sh -c \
    'wget -qS -O /tmp/smoke-body "http://127.0.0.1:8080'"$1"'" 2>/tmp/smoke-headers; rc=$?; cat /tmp/smoke-headers; echo ---BODY---; cat /tmp/smoke-body; exit $rc'
}

out="$(wget_get /)" || fail "GET / did not return 2xx"
headers="${out%%---BODY---*}"
body="${out#*---BODY---$'\n'}"
grep -q '<div' <<<"$body" || fail "GET / returned no HTML body"

for h in X-Frame-Options X-Content-Type-Options Referrer-Policy Permissions-Policy; do
  grep -qi "^ *${h}:" <<<"$headers" || fail "missing header on /: $h"
done
if grep -qiE '^ *Server: nginx/[0-9]' <<<"$headers"; then
  fail "Server header exposes the nginx version"
fi

asset="$(grep -oE '/_app/immutable/[^"]+\.js' <<<"$body" | head -1 || true)"
[[ -n "$asset" ]] || fail "no JS asset referenced from index.html"
docker exec "$NAME" wget -q -O /dev/null "http://127.0.0.1:8080$asset" || fail "GET $asset failed"
docker exec "$NAME" wget -q -O /dev/null "http://127.0.0.1:8080/review" \
  || fail "GET /review (SPA fallback) failed"

uid="$(docker exec "$NAME" id -u)"
[[ "$uid" == "101" ]] || fail "container runs as uid $uid, expected 101"

arch="$(docker image inspect -f '{{.Architecture}}' "$IMAGE")"
echo "OK: $IMAGE ($arch) healthy, headers present, uid 101"
