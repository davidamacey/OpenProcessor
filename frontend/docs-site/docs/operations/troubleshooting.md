---
sidebar_position: 4
title: Troubleshooting
---

# Troubleshooting

## The status chip stays red / "API OK" never shows

- Confirm `PUBLIC_API_PREFIX` matches the backend's own `OP_API_PREFIX`
  exactly.
- Confirm `API_UPSTREAM` resolves — Cropwright's container must be on the
  same docker network as the backend (`OP_DOCKER_NETWORK`).
- `curl http://localhost:<CROPWRIGHT_PORT>/curation/health` (swap in your
  prefix) from the host to bypass the browser entirely.

## A route/section I expect isn't showing up

Nearly every optional feature is gated on the backend advertising it at
runtime (see [Backend feature flags](../configuration/backend-feature-flags.md)).
Absent is the expected behavior for a backend that hasn't enabled that
feature — it is not a bug.

## `/ingest` shows only an error

The page renders from the served `GET {API_PREFIX}/ingest/config`; the
error line is that read's own failure. Check the backend is reachable
and on a current OpenProcessor release.

## Upload fails with a 413

Either the server-path batch's own item cap, or nginx's
`client_max_body_size` (`CROPWRIGHT_INGEST_MAX_REQUEST_MB`). The upload
run controller retries once at half the chunk size on the backend's own
JSON 413; an nginx HTML 413 stops the run outright rather than retrying —
raise the cap and re-run.

## A class hotkey binding was silently rejected

The letter is in the reserved set (global actions + every registered
slot's keymap). Pick a different letter — `/classes` shows the backend's
detail text verbatim when a bind is rejected server-side.
