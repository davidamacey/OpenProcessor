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
| `CROPWRIGHT_BIND_ADDRESS` | Host address the port is published on. The default serves this machine and the local network; set `127.0.0.1` to allow this machine only. The API has no authentication, so keep it on a trusted network. | `0.0.0.0` |
| `CROPWRIGHT_TAG` | Which published image tag to pull; pin a version for a reproducible deploy | `latest` |
| `CROPWRIGHT_CONTAINER_NAME` | Container name for `docker compose up` — give a second instance its own value | `cropwright` |
| `CROPWRIGHT_DATASET_UPLOAD_MAX_MB` | Upload cap for a dataset archive on [dataset import](../user-guide/dataset-import.md) (nginx `client_max_body_size` for the dataset-upload route, in MB), with a one-hour read timeout. Substituted into both nginx.conf and the served client config; the client refuses a larger file before sending it. | `2048` |
| `CROPWRIGHT_INGEST_MAX_REQUEST_MB` | Deployment-owned upload cap for `/ingest` (nginx `client_max_body_size`, in MB); substituted into both nginx.conf and the served client config so the two can't drift | `256` |

Both upload caps must be positive integers; the container refuses to start
otherwise. Values are substituted when the container starts, so one image
fits any deployment and a change needs only a restart, not a rebuild.

## Settings that are not environment variables

Almost everything else you might want to change is **not** configured on
Cropwright. It is either backend configuration or a per-project setting you
edit in the app. See [Runtime settings](./runtime-settings.md) for where each
one lives.

See [Running a second instance](./second-instance.md) for the multi-instance
pattern, and [Backend feature flags](./backend-feature-flags.md) for what
the *backend* needs to enable per feature — none of that is a Cropwright
env var.
