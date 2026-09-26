---
sidebar_position: 2
title: Quick start
---

# Quick start

## 1. Have an OpenProcessor backend running

Cropwright needs a reachable [OpenProcessor](https://github.com/example-org/OpenProcessor)
instance, started with `OP_API_PREFIX=/curation` (the default), on its own
Docker network. Note that network's name — you'll need it below. See that
project's own README for bringing it up.

## 2. Get the compose file and configure it

No checkout needed: Cropwright runs from the published image
`davidamacey/cropwright`, which is multi-arch (`linux/amd64` and
`linux/arm64`, so it runs on Intel/AMD hosts and on ARM machines such as
Apple Silicon). Download the compose file and the example env file:

```bash
mkdir cropwright && cd cropwright
curl -fsSLO https://raw.githubusercontent.com/davidamacey/OpenProcessor/main/docker-compose.yml
curl -fsSL https://raw.githubusercontent.com/davidamacey/OpenProcessor/main/.env.example -o .env
```

Edit `.env` and set, at minimum:

| Variable            | Meaning                                                                                  | Default                             |
| ------------------- | ----------------------------------------------------------------------------------------- | ------------------------------------ |
| `API_UPSTREAM`      | Where nginx proxies API calls, by container name over the shared docker network           | `http://op-api:8000`                 |
| `PUBLIC_API_PREFIX` | Must equal the backend's own `OP_API_PREFIX`                                              | `/curation`                          |
| `OP_DOCKER_NETWORK` | The OpenProcessor backend's docker network name (`docker network ls`)                     | `openprocessor_triton_net`           |
| `CROPWRIGHT_PORT`   | Host port to publish                                                                       | `5184`                               |
| `CROPWRIGHT_TAG`    | Image version to run; pin one (e.g. `0.1.0`) for reproducible deploys                     | `latest`                             |

See [Environment variables](../configuration/environment-variables.md) for
the full list, including white-label and ingest-upload-cap options.

## 3. Start it

```bash
docker compose pull && docker compose up -d
```

Open `http://localhost:5184` (or whatever `CROPWRIGHT_PORT` you set). The
image runs nginx as a non-root user (uid 101) listening on container port
8080; Compose maps `CROPWRIGHT_PORT` to it.

Building from source instead is for development — see
[Development setup](../developer-guide/development-setup.md).

## 4. Verify the connection

- Look at the top-bar status chip — it polls `{PUBLIC_API_PREFIX}/health`
  and turns green when the backend answers.
- Or from a shell:

```bash
curl http://localhost:5184/curation/health
```

(swap in your own port/prefix.)

## Trying it with sample data

For a new install with no images yet, OpenProcessor ships a script to fetch a
small public COCO val2017 subset as sample data (never bundled with
Cropwright itself). From the OpenProcessor checkout:

```bash
make sample-coco-readme   # 200 images, 20 per class
# or: make sample-coco    # the larger 800-image set
```

Then ingest it through `/ingest`'s server-path panel, pointing at the sample
folder the backend mounts under its configured batch source roots — that
panel only appears when the backend advertises at least one source root.

## The end-to-end workflow

```
/ingest  ->  /review + /clusters  ->  /classes  ->  /export  ->  /train  ->  /bakeoff  ->  back to /review
```

Bring images in, label and triage them, manage the class registry, freeze a
test holdout and export a YOLO dataset, train and promote a model, compare
models, then loop back — **Model Disagreements** on `/review` surfaces where
a newly promoted model and the human label diverge, feeding the next cycle.

## Next steps

- [Architecture overview](./architecture-overview.md)
- The [User Guide](../user-guide/dashboard.md)
- [Security](../operations/security.md) — read this before putting
  Cropwright anywhere near a shared network.
