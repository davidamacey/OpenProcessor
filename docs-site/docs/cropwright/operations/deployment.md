---
sidebar_position: 1
title: Deployment
---

# Deployment

Cropwright ships as a single Docker image: SvelteKit's static adapter build
served by nginx.

- **Non-root**: the image runs nginx as uid 101, listening on container
  port `8080`. Compose maps `CROPWRIGHT_PORT` (default `5184`) to it.
- **Docs on the same origin**: nginx proxies the API's Swagger UI (`/docs`), ReDoc
  (`/redoc`) and `/openapi.json`, and serves the one documentation site at `/OpenProcessor/`
  when the `docs` service of `docker-compose.docs.yml` is running. Point `DOCS_UPSTREAM`
  elsewhere to host the docs yourself; if the docs container is down, only `/OpenProcessor/`
  answers 502.
- **One compose project**: the `cropwright` service is part of the OpenProcessor compose project
  and shares its network, so nginx proxies `PUBLIC_API_PREFIX/*` to the API by service name
  (`op-api`) — no CORS, and the browser never sees the backend's real hostname.
- **No database, no persistent volumes**: Cropwright itself is stateless.
  Everything it shows comes from the backend on each request.

## One deployment, many projects

A single Cropwright container and a single backend API serve **every
project**. Projects are separated by the URL (`/p/<project>/...`) and by the
prefix the backend serves for each; you do not run a container per project.
The one API prefix and one upstream in your `.env` cover them all. See
[Projects](../user-guide/projects.md) for creating, pausing, archiving and
deleting them, and [Project administration](./project-administration.md)
for the operator view.

## Reverse proxy behavior

nginx proxies `PUBLIC_API_PREFIX/*` to `API_UPSTREAM`, with three
differences worth knowing when you put another proxy in front:

- The general API location has a 120 s read timeout.
- Ingest routes (`.../projects/<slug>/ingest/...`) get a 600 s read timeout
  and the `CROPWRIGHT_INGEST_MAX_REQUEST_MB` body cap, because a batch with
  detection and embedding can run long.
- The dataset-archive upload route (`.../projects/<slug>/datasets/uploads`)
  gets a one-hour read timeout and the `CROPWRIGHT_DATASET_UPLOAD_MAX_MB`
  body cap.

An authenticating proxy in front of Cropwright must pass these through
unchanged, or large uploads will fail with a 413 or a gateway timeout.

## Network exposure

The compose file publishes on `${OP_UI_BIND_ADDRESS:-${OP_BIND_ADDRESS:-127.0.0.1}}:${CROPWRIGHT_PORT:-5184}`,
so by default Cropwright answers on this machine only. Set `OP_UI_BIND_ADDRESS=0.0.0.0` (or a LAN
address) to let your local network open it; that also opens the optional monitoring UIs, and the API
port stays on `OP_BIND_ADDRESS`.
See [Security](./security.md): the API has no authentication.

## Production Dockerfile conventions

Multi-stage build (Node build stage discarded, `nginx:alpine`-family
runtime), non-root user, healthcheck against `127.0.0.1:8080/` (not
`localhost` — that resolves to `::1` and nginx only listens on IPv4 inside
the container).

## Registry

Released images are published to Docker Hub as `davidamacey/cropwright`
(`X.Y.Z`, `X.Y` and `latest`), multi-arch for `linux/amd64` and
`linux/arm64`. `docker-compose.yml` only pulls; set `CROPWRIGHT_TAG` (or a full `CROPWRIGHT_IMAGE`
reference) to pin a version. To build from the checkout instead, add the dev overlay:
`docker compose -f docker-compose.yml -f docker-compose.dev.yml --profile cropwright up -d --build cropwright`.
