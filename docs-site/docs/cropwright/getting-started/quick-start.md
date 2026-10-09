---
sidebar_position: 2
title: Quick start
---

# Quick start

## 1. Set up OpenProcessor

Cropwright is the labeling frontend of [OpenProcessor](https://github.com/davidamacey/OpenProcessor)
and lives in the same repository (`frontend/`). It runs as the `cropwright` profile of
OpenProcessor's own compose project, on the project's network, so there is no separate
network, image repository or `.env` to set up. Install or start OpenProcessor first
(see [Getting started](../../getting-started/quick-start.mdx)); the API prefix is
`OP_API_PREFIX=/curation` by default.

## 2. Configure it

The defaults work out of the box. These optional settings go in the same `.env` as the rest of
the stack:

| Variable             | Meaning                                                                           | Default                          |
| -------------------- | --------------------------------------------------------------------------------- | -------------------------------- |
| `API_UPSTREAM`       | Where nginx proxies API calls, by service name over the compose network          | `http://op-api:8000`             |
| `PUBLIC_API_PREFIX`  | Must equal the backend's own `OP_API_PREFIX`                                      | `/curation`                      |
| `CROPWRIGHT_PORT`    | Host port to publish                                                              | `5184`                           |
| `OP_UI_BIND_ADDRESS` | Address the Cropwright port (and the monitoring UIs) is published on              | `OP_BIND_ADDRESS`, then loopback |

See [Environment variables](../configuration/environment-variables.md) for
the full list, including white-label, bind-address and upload-cap options.

## 3. Start it

```bash
docker compose --profile cropwright up -d
```

From a source checkout, `docker-compose.dev.yml` builds the image from `./frontend`:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml --profile cropwright up -d --build cropwright
```

Open `http://localhost:5184` (or whatever `CROPWRIGHT_PORT` you set). The
image runs nginx as a non-root user (uid 101) listening on container port
8080; Compose maps `CROPWRIGHT_PORT` to it. The same origin also serves the backend's API
reference (`/docs`, `/redoc`, `/openapi.json`) and, when the docs service is running
(`docker-compose.docs.yml`), this documentation; the top-bar **Resources** menu links to them.

<Screenshot name="cropwright/resources-menu-1600.png" alt="Resources menu open in the top bar listing the bundled docs and the served service links" caption="Resources menu — the bundled docs, then exactly the links the backend serves" />

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
the app itself). From the repository root:

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
