---
sidebar_position: 1
title: Environment variables
---

# Environment variables

From `.env.example`. Copy it to `.env` before `docker compose up`.

## Read by `npm run dev` only (not Docker Compose)

| Variable | Purpose | Default |
| --- | --- | --- |
| `PUBLIC_TRITON_API_URL` | Points the dev server's browser bundle directly at a reachable OpenProcessor API origin, bypassing the nginx proxy. Leave unset for `docker compose up` — Compose passes it through to the browser bundle, which would then call this origin directly instead of the proxy. | empty |

## Read by both

| Variable | Purpose | Default |
| --- | --- | --- |
| `PUBLIC_APP_NAME` | Top-bar wordmark, for a white-label deployment | `Cropwright` |
| `PUBLIC_APP_BADGE` | Top-bar badge letters | `CW` |
| `PUBLIC_API_PREFIX` | Path prefix the backend serves curation endpoints under. Must equal the backend's own `OP_API_PREFIX` — the API builds some URLs (region thumbnails) from its own prefix. | `/curation` |

## Read by `docker-compose.yml` / `docker-entrypoint.sh` only

| Variable | Purpose | Default |
| --- | --- | --- |
| `API_UPSTREAM` | Where nginx proxies `PUBLIC_API_PREFIX/*` — the OpenProcessor API container, by name over the shared docker network | `http://op-api:8000` |
| `OP_DOCKER_NETWORK` | The OpenProcessor API's docker network name (must already exist) | `openprocessor_triton_net` |
| `CROPWRIGHT_PORT` | Host port this instance publishes nginx on | `5184` |
| `CROPWRIGHT_CONTAINER_NAME` | Container name for `docker compose up` — give a second instance its own value | `cropwright` |
| `CROPWRIGHT_INGEST_MAX_REQUEST_MB` | Deployment-owned upload cap for `/ingest` (nginx `client_max_body_size`, in MB); substituted into both nginx.conf and the served client config so the two can't drift | `256` |

See [Running a second instance](./second-instance.md) for the multi-instance
pattern, and [Backend feature flags](./backend-feature-flags.md) for what
the *backend* needs to enable per feature — none of that is a Cropwright
env var.
