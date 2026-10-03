---
sidebar_position: 2
title: Quick start
---

# Quick start

## 1. Have an OpenProcessor backend running

Cropwright needs a reachable [OpenProcessor](https://github.com/davidamacey/OpenProcessor)
instance, started with `OP_API_PREFIX=/curation` (the default), on its own
Docker network. Note that network's name — you'll need it below. See that
project's own README for bringing it up.

## 2. Get the code and configure it

Cropwright is not published as a standalone image yet; it ships with the
OpenProcessor 0.5.0 release as the `ui` service of OpenProcessor's own
compose project. Until then, run it from a source checkout, which builds both
the app and these docs:

```bash
cp .env.example .env
```

Edit `.env` and set, at minimum:

| Variable            | Meaning                                                                                  | Default                             |
| ------------------- | ----------------------------------------------------------------------------------------- | ------------------------------------ |
| `API_UPSTREAM`      | Where nginx proxies API calls, by container name over the shared docker network           | `http://op-api:8000`                 |
| `PUBLIC_API_PREFIX` | Must equal the backend's own `OP_API_PREFIX`                                              | `/curation`                          |
| `OP_DOCKER_NETWORK` | The OpenProcessor backend's docker network name (`docker network ls`)                     | `openprocessor_triton_net`           |
| `CROPWRIGHT_PORT`   | Host port to publish                                                                       | `5184`                               |

See [Environment variables](../configuration/environment-variables.md) for
the full list, including white-label, bind-address and upload-cap options.

## 3. Start it

```bash
docker compose -f docker-compose.yml -f docker-compose.build.yml up -d --build
```

Open `http://localhost:5184` (or whatever `CROPWRIGHT_PORT` you set). The
image runs nginx as a non-root user (uid 101) listening on container port
8080; Compose maps `CROPWRIGHT_PORT` to it. The same origin also serves these
docs (`/cropwright/`) and the backend's API reference (`/docs`, `/redoc`,
`/openapi.json`); the top-bar **Resources** menu links to them.

For frontend development see
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

## 5. Pick or create a project

Cropwright opens in the deployment's default project. Use the project
switcher in the top bar, or `/projects`, to create another; see
[Projects](../user-guide/projects.md). Already have labeled data? Use
[Dataset import](../user-guide/dataset-import.md).

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
