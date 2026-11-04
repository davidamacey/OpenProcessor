---
sidebar_position: 1
title: Environment variables
---

# Environment variables

These are read from the stack's `.env` (the repository root `env.template`). The frontend-only
`frontend/.env.example` covers `npm run dev`.

## Read by `npm run dev` only (not Docker Compose)

| Variable | Purpose | Default |
| --- | --- | --- |
| `PUBLIC_TRITON_API_URL` | Points the dev server's browser bundle directly at a reachable OpenProcessor API origin, bypassing the nginx proxy. Leave unset for `docker compose up` — Compose passes it through to the browser bundle, which would then call this origin directly instead of the proxy. | empty |

## Read by `npm run dev` and Docker Compose

| Variable | Purpose | Default |
| --- | --- | --- |
| `PUBLIC_API_PREFIX` | Path prefix the backend serves curation endpoints under. Must equal the backend's own `OP_API_PREFIX` — the API builds some URLs (region thumbnails) from its own prefix. | `/curation` |

## Build time only (source builds)

Vite inlines these into the bundle when the app is built, so they take
effect in `npm run dev` or a host `npm run build`, **not** as a Docker
Compose setting: the published image and the Compose stack never read them.

| Variable | Purpose | Default |
| --- | --- | --- |
| `PUBLIC_APP_NAME` | Top-bar wordmark, for a white-label deployment | `Cropwright` |
| `PUBLIC_APP_BADGE` | Top-bar badge letters | `CW` |

## Read by `docker-compose.yml` / `docker-entrypoint.sh` only

| Variable | Purpose | Default |
| --- | --- | --- |
| `API_UPSTREAM` | Where nginx proxies `PUBLIC_API_PREFIX/*` — the OpenProcessor API container, by name over the shared docker network | `http://op-api:8000` |
| `OP_CROPWRIGHT_IMAGE` | Full image reference (for example a digest pin); wins over `OP_IMAGE_REPO` and `OP_IMAGE_TAG` | `davidamacey/cropwright:<OP_IMAGE_TAG>` |
| `DOCS_UPSTREAM` | Where nginx proxies `/OpenProcessor/` (the docs service of `docker-compose.docs.yml`, when it runs) | `http://docs:8080` |
| `OP_GATEWAY_SUBPATHS` | `true` makes this nginx also serve the monitoring UIs under `/grafana/`, `/prometheus/`, `/dashboards/` and `/mlflow/` of the one origin. | `false` |
| `OP_CROPWRIGHT_PORT` | Host port this instance publishes nginx on | `5184` |
| `OP_CROPWRIGHT_BIND_ADDRESS` | Host address for the Cropwright port alone; wins over `OP_UI_BIND_ADDRESS`. The installer sets it on a LAN bind. | unset |
| `OP_UI_BIND_ADDRESS` | Host address the Cropwright port is published on (the optional monitoring UIs follow it). Used when `OP_CROPWRIGHT_BIND_ADDRESS` is unset; defaults to `OP_BIND_ADDRESS`, then loopback. Set `0.0.0.0` to allow your local network; the API has no authentication, so keep it on a trusted network. | `127.0.0.1` |
| `OP_CROPWRIGHT_BASIC_AUTH` | `true` puts HTTP basic auth on every route except `/nginx-health`. The login is the htpasswd file below, never an env value. The container refuses to start when this is on and the file is missing, empty or malformed. The installer sets it on a LAN bind; `./openprocessor cropwright password` sets the login. | `false` |
| `OP_CROPWRIGHT_BASIC_AUTH_FILE` | Host path of the htpasswd file (`user:apr1-or-bcrypt-hash`), mounted read-only into the container. Compose reads it; the installer writes `./secrets/cropwright/htpasswd`. | `/dev/null` (empty) |
| `OP_CROPWRIGHT_DATASET_UPLOAD_MAX_MB` | Upload cap for a dataset archive on [dataset import](../user-guide/dataset-import.md) (nginx `client_max_body_size` for the dataset-upload route, in MB), with a one-hour read timeout. Substituted into both nginx.conf and the served client config; the client refuses a larger file before sending it. | `2048` |
| `OP_CROPWRIGHT_INGEST_MAX_REQUEST_MB` | Deployment-owned upload cap for `/ingest` (nginx `client_max_body_size`, in MB); substituted into both nginx.conf and the served client config so the two can't drift | `256` |

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
