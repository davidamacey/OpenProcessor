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

Published to Docker Hub once public releases start; until then, build
locally with `docker compose up -d --build`.
