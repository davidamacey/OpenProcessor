---
sidebar_position: 1
title: Deployment
---

# Deployment

Cropwright ships as a single Docker image: SvelteKit's static adapter build
served by nginx.

- **Non-root**: the image runs nginx as uid 101, listening on container
  port `8080`. Compose maps `CROPWRIGHT_PORT` (default `5184`) to it.
- **One docker network**: the container joins the OpenProcessor API's own
  docker network so nginx can proxy `PUBLIC_API_PREFIX/*` to it by
  container name — no CORS, and the browser never sees the backend's real
  hostname.
- **No database, no persistent volumes**: Cropwright itself is stateless.
  Everything it shows comes from the backend on each request.

## Production Dockerfile conventions

Multi-stage build (Node build stage discarded, `nginx:alpine`-family
runtime), non-root user, healthcheck against `127.0.0.1:8080/` (not
`localhost` — that resolves to `::1` and nginx only listens on IPv4 inside
the container).

## Registry

Released images are published to Docker Hub as `davidamacey/cropwright`
(`X.Y.Z`, `X.Y` and `latest`), multi-arch for `linux/amd64` and
`linux/arm64`. `docker-compose.yml` only pulls; set `CROPWRIGHT_TAG` to pin
a version. To build from source instead, add the build overlay:
`docker compose -f docker-compose.yml -f docker-compose.build.yml up -d --build`.
