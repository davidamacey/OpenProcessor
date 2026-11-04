# OpenProcessor API - Source Code

One FastAPI service on port **4603** serves the inference API and the curation
subsystem. All inference runs on NVIDIA Triton with TensorRT engines.

Route lists and request examples are in the [top-level README](../README.md#api-endpoints);
the curation routes are in [`contracts/openapi/curation.json`](../contracts/openapi/curation.json)
and [docs/design/curation_api_contract.md](../docs/design/curation_api_contract.md).
Response shapes are in the live schema at `http://localhost:4603/docs`.

---

## File structure

```
src/
├── main.py              # Builds the `app` that `uvicorn src.main:app` serves
├── app/                 # factory.py (create_app, middleware, router mounting),
│                        #   lifespan.py (startup and shutdown), resources.py
├── routers/             # Thin HTTP adapters, one module per capability
│   ├── ingest/, search/ # Packages for the larger core routes
│   └── curation/        # The /curation routers
├── services/            # Logic with no FastAPI dependency
│   ├── curation/, config_store/, projects/, labeling/ (VLM), detection/, training/
│   ├── face_identity/, visual_search/, duplicate_detection/
│   └── inference.py, embedding.py, ocr_service.py, analyze_pipeline.py, ...
├── clients/             # Triton clients and pool, OpenSearch (opensearch/,
│                        #   curation_opensearch/), OCC helpers, PE encoder
├── domain/              # Pure rules shared by clients and services
├── schemas/             # Pydantic wire models
├── config/              # settings.py, curation config, project context, retired-env guard
├── core/                # CORS, dependencies, error handlers, logging, metrics, upload limits
├── utils/               # Image decode, preprocessing, atomic writes (see utils/README.md)
├── ultralytics_patches/ # Vendored YOLO export patches (see ultralytics_patches/README.md)
└── static/docs/         # Self-hosted Swagger UI and ReDoc assets
```

The layer rules (routers import services and clients, never the reverse;
`domain`, `config`, `utils` and `schemas` import neither) are enforced by
`scripts/codegen/check_layering.py`; see
[CONTRIBUTING.md](../CONTRIBUTING.md#guards-and-baselines).

---

## Application assembly

`src/app/factory.py` builds the app in `create_app()`: middleware, error
handlers, the core routers (`/health`, `/detect`, `/faces`, `/persons`,
`/embed`, `/search`, `/ingest`, `/analyze`, `/clusters`, `/query`, `/ocr`,
`/models`), the `/v1` twin of the core routes and the `/curation` routers.
Shared resources (the Triton pool, the executor) live in `src/app/resources.py`;
`src/core/dependencies.py` hands them to handlers.

---

## Configuration

Core settings are in `src/config/settings.py` and read from the environment;
`src/config/env.py` is the one place `src/` reads it (a malformed value raises
naming the variable). The template is [`env.template`](../env.template) and
the reference is the
[configuration docs](../docs-site/docs/configuration/basic.mdx). Retired
variables are rejected at startup (`src/config/retired_env.py`).

| Variable | Default | Description |
|----------|---------|-------------|
| `TRITON_URL` | `triton-server:8001` | Triton gRPC address |
| `OPENSEARCH_URL` | `http://opensearch:9200` | OpenSearch REST address |
| `MAX_FILE_SIZE_MB` | `50` | Maximum size of each uploaded file |
| `SLOW_REQUEST_THRESHOLD_MS` | `100` | Log slow requests |

---

## Running the service

Use the stack from the repository root (`make up`, see
[CLAUDE.md](../CLAUDE.md#commands)); a checkout mounts `src/` into the `api`
container, so most edits are live. Restart `api` after route changes.

Standalone, with Triton and OpenSearch already running (call the venv binary
directly, never `source .venv/bin/activate`):

```bash
.venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 4603 --workers 2 --reload
```

Measured latency and throughput are in [docs/PERFORMANCE.md](../docs/PERFORMANCE.md).

---

## Related documentation

- [../README.md](../README.md) - project overview and API reference
- [../CLAUDE.md](../CLAUDE.md) - repository layout and rules
- [utils/README.md](utils/README.md) - utilities
- [../docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) - components and data model
