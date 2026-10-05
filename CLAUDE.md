# CLAUDE.md

This file guides AI coding agents working in this repository: what it is,
where things live, and the commands and rules that matter.

## IMPORTANT: Read the project vision first

Before making design decisions or judgment calls, read
[`docs/VISION_AND_GOALS.md`](docs/VISION_AND_GOALS.md): what OpenProcessor is
for, the v0.4.1 feature scope, and the standards the code is held to (no dead
code, no compatibility shims, fail-closed isolation, class identity by name
never index, review depth matched to real risk).

## IMPORTANT: Python Environment

**Call the venv binaries directly. Never `source` the venv.**

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/pre-commit run --all-files
.venv/bin/python -m ruff check src/
.venv/bin/python -m mypy src/
```

- Correct: `.venv/bin/python tests/test_full_system.py`
- Wrong: `source .venv/bin/activate && python ...` (some harnesses flag it)
- Wrong: bare `python` or `python3` (system Python, missing dependencies)

Create the venv with `python3 -m venv .venv` and
`.venv/bin/pip install -r requirements.txt -r requirements-test.txt`
(see [CONTRIBUTING.md](CONTRIBUTING.md)). The offline suite needs no Docker
and no GPU: `.venv/bin/python -m pytest tests/ -q --no-cov -m 'not live'`.

## Project overview

OpenProcessor is a backend for computer-vision dataset curation and training
with a fast inference API on NVIDIA Triton underneath. The API runs on FastAPI
at port 4603.

- **Inference API** (`/detect`, `/faces`, `/embed`, `/search`, `/ingest`,
  `/ocr`, `/analyze`, `/clusters`, `/query`, `/models`, `/health`): YOLO11 and
  YOLO26 detection, SCRFD + ArcFace faces, MobileCLIP embeddings, PP-OCRv5.
  Also served under `/v1`.
- **Curation subsystem** (`/curation`, no `/v1` twin): isolated projects,
  ingest, a region stage producing one or many boxes per item, VLM labeling,
  review, dataset import, combine, export, training and promotion. Cropwright
  is the separate frontend.

## Repository layout

| Path | What is there |
|---|---|
| `src/main.py` | FastAPI app, router mounting, lifespan |
| `src/routers/` | Core routers; `src/routers/curation/` holds the curation routers (thin HTTP adapters) |
| `src/services/` | Logic with no FastAPI dependency: `curation/`, `config_store/`, `projects/`, `labeling/` (VLM), `detection/`, `training/` |
| `src/clients/` | Triton, OpenSearch (`curation_opensearch.py` has the index bodies), OCC helpers (`occ*.py`, the lock rule in `occ_locks.py`), PE encoder |
| `src/config/` | `CurationConfig`, `RegionFields`, `DetectionProfile`, `RegionStatus`, project records and context, retired-env guard |
| `scripts/curation/` | Worker entry points, ingest walkers, import client, backfills, `bakeoff/` |
| `scripts/` | `setup.sh`, `lib/` (shell libraries, VLM catalog reader), `release/`, `codegen/`, `datasets/`, `docs/`, `examples/` |
| `openprocessor`, `setup-openprocessor.sh` | Management CLI and the one-line installer (both bash) |
| `models/` | Triton model repository (`config.pbtxt` per model; `.plan` engines are built, not committed) |
| `export/` | Model export scripts, see [export/README.md](export/README.md) |
| `docker/` | Side-car images: `segmenter/`, `trainer/`, `evaluator/`, `test/` (live harness), `hardened/` |
| `contracts/` | Generated OpenAPI and TypeScript/JSON wire contracts. Do not hand-edit |
| `examples/` | Region profiles, prompt packs, bake-off profiles, the VLM catalog (`vlm/catalog.tsv`) |
| `tests/` | Offline pytest suite; `tests/live/` needs the live harness |
| `docs/`, `docs-site/` | Markdown docs; Docusaurus site |

## Services and compose

`docker-compose.yml` is deploy-safe on its own: no `build:` blocks and no
source mounts. `docker-compose.dev.yml` adds local builds and hot-reload mounts
for a checkout. The Makefile, `scripts/setup.sh` and the `openprocessor` CLI
add the dev overlay automatically when `src/main.py` exists next to the compose
file. `docker-compose.gpu-arbiter.yml` is an opt-in overlay that mounts the
Docker socket (read its header first).

| Service | Profile | Port | Role |
|---|---|---|---|
| `triton-server` | none | 4600 HTTP, 4601 gRPC, 4602 metrics | TensorRT models, explicit model control |
| `yolo-api` | none | 4603 | FastAPI, all routes |
| `opensearch` | none | 4607 | k-NN store and the curation datastore |
| `curation-detection-worker`, `curation-vlm-worker`, `curation-auto-label-worker`, `curation-cluster-refresh`, `curation-evaluator` | `curation` | | Region cascade, VLM loop, auto-label driver, clustering refresh, bake-off runner |
| `segmenter` | `segmenter` | 4611 | Region proposals from a text prompt |
| `vlm` | `vlm` | 4612 | Local vLLM serving a catalog model |
| `curation-trainer`, `curation-mlflow` | `training` | 4609 (MLflow) | Training jobs through a file protocol, tracking |
| Prometheus, Grafana, Loki, Alloy, DCGM, node exporter, OpenSearch Dashboards | `monitoring` | 4604, 4605, 4606, 4610, 4608 | Opt in: `make up-monitoring` |
| `triton-sdk` | `benchmark` | | Benchmark client |

GPU placement comes from `.env` (`TRITON_GPU_ID`, `API_GPU_ID`,
`SEGMENTER_GPU_ID`, `EVALUATOR_GPU_ID`, `VLM_GPU_ID`); the compose files pin no
GPU. The compose project name is `COMPOSE_PROJECT_NAME` (default
`openprocessor`). From a git worktree always pass `-p <name>` so you do not
act on a live stack with the default name.

## Commands

```bash
make up                  # core stack (Triton + API + OpenSearch)
make up-monitoring       # core plus monitoring
make down
make logs-api            # also logs-triton, logs-opensearch
make restart-api         # route/endpoint changes need an API restart
make status              # health of all services
make curation-up         # start the curation workers
make test                # pytest suite
make contracts           # regenerate contracts/ after an API or wire change
make contracts-check
make models-list         # Triton model states
make help                # every target
```

The management CLI works in a checkout (`./openprocessor`, or the shim
`./scripts/openprocessor.sh`) and in an installed directory. Subcommands:
`start`, `stop`, `restart [service]`, `logs [service] [-f]`, `status`,
`health`, `models [status|install [--only GROUP]|repair]`, `export`,
`download`, `profile`, `test`, `curation [up|down|logs|status]`, `bench`,
`clean`, `update`, `sample coco [--full]`, `config show`, `gpu plan`,
`vlm list|status|use <id>|apply|probe|key set <slug>`, `train-mode on|off`,
`upgrade`, `repair`, `uninstall`, `setup`, `version`. `profile`, `test`,
`bench` and `setup` need a checkout. Details:
[INSTALLATION.md](INSTALLATION.md#the-openprocessor-cli).

Python source is mounted into `yolo-api` in a checkout, so most edits are live;
restart `yolo-api` after route changes. Rebuild images only when a Dockerfile or
`requirements.txt` changes.

## Core API shape

Coordinates are normalized `0.0-1.0`. Face landmarks are a flat list of 10
floats.

```json
{
  "faces": [
    {
      "box": {"x1": 0.30, "y1": 0.10, "x2": 0.50, "y2": 0.40},
      "confidence": 0.98,
      "landmarks": [0.35, 0.20, 0.45, 0.20, 0.40, 0.28, 0.36, 0.35, 0.44, 0.35],
      "embedding": [0.012, -0.034, ...]
    }
  ],
  "inference_time_ms": 18.3
}
```

`POST /search/text` takes `text` and `top_k` as query parameters, not a JSON
body. The core route list is in [README.md](README.md#api-endpoints).

## Curation subsystem: rules that matter when editing

- **Projects.** Every curation route is `/curation/projects/{project}/...`.
  A project owns its OpenSearch indexes
  (`{OP_PROJECT_INDEX_PREFIX}{slug}__{role}`), directories and class registry.
  Code reads names through the bound project (`src/config/project_context.py`),
  never through a constant. The OpenSearch guard
  (`src/services/projects/guard.py`) refuses any request that reaches an index
  the bound project does not own. Construct OpenSearch clients only through
  the factory and `make_script_opensearch`. A script binds a project with
  `--project` (default `$OP_CURATION_PROJECT`, else `default`).
- **Regions are lists.** `region_boxes` is a nested list per item; one region
  is a list of one. Box elements have fixed keys (`box_id`, `bbox_norm`, `state`,
  `score`, `detector`, `text*`, `cluster_*`). Writes go through
  `src/services/curation/region_boxes.py` and the OCC helpers. Per-box vectors
  live in the sibling `region_box_embeddings`. `tests/test_no_legacy_region_scalars.py`
  guards against single-box scalar fields.
- **Lock rule.** Automated writers never overwrite a human- or
  import-validated class or box, or an item in the frozen test holdout
  (`src/clients/occ_locks.py`). Use the OCC read/merge/write helpers, not a
  blind update.
- **Class identity is the name.** Never carry a raw class index across a
  boundary (import, combine, export, train, promote, model sharing).
- **Config store.** Prompt packs, region profiles and the VLM activation are
  revisioned documents (`src/services/config_store/`). The VLM endpoint
  registry is deployment-wide; activation is per project.
- **Wire contract.** One serializer (`src/services/curation/wire.py`) maps
  storage to the fixed wire names. After changing a route or model run
  `make contracts` and commit `contracts/` in the same commit; a pre-commit
  hook rejects stale contracts.
- **No dead code or shims.** Delete a superseded route or field in the same
  change. Retired env vars are rejected at startup
  (`src/config/retired_env.py`).
- **Fail closed.** An unbound project, a stale registry or an ambiguous state
  refuses the operation.

## Pre-commit and checks

Run `.venv/bin/pre-commit run --all-files` before every commit. Notable
hooks: ruff, mypy, bandit, the per-file size ratchet
(`scripts/codegen/check_file_size.py`), the region-field literal ratchet
(`check_no_literal_region_fields.py`), the naming-leak scan
(`scripts/codegen/check_naming_leaks.py`: no private product or domain names
in the tracked tree) and the contract-drift hooks. Docs are checked against
the code:

```bash
.venv/bin/python scripts/docs/check_docs_vs_code.py            # routes, OP_* vars, links
.venv/bin/python scripts/docs/check_docs_vs_code.py --only README.md
```

Docs may only name routes that exist in `contracts/openapi/curation.json` or
the app, `OP_*` variables read by code, and links that resolve.

## Testing

```bash
.venv/bin/python -m pytest tests/ -q --no-cov -m 'not live'   # offline suite
.venv/bin/python tests/test_full_system.py                     # needs the running stack
.venv/bin/python tests/validate_visual_results.py              # annotated images in test_results/
```

The live write-path harness (`docker/test/compose.yml`, `tests/live/`) runs
under its own compose project name; see
[`docker/test/README.md`](docker/test/README.md). For UI work, open the page in
a browser; type checks do not prove a feature works.

## Configuration

Settings live in `.env` (template `env.template`). The "Curation quick-config"
block lists the keys the curation tiers need. Index names are not
configurable: they come from the project. Ports: API 4603, Triton 4600-4602,
Prometheus 4604, Grafana 4605, Loki 4606, OpenSearch 4607, Dashboards 4608,
MLflow 4609, DCGM 4610, segmenter 4611, VLM 4612. Set `OP_BIND_ADDRESS` to
publish beyond loopback (the installer asks for consent).

The API has no authentication. Do not expose it. See [SECURITY.md](SECURITY.md).

## Key documents

- [README.md](README.md), [INSTALLATION.md](INSTALLATION.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): components, data model, topology
- [docs/CURATION.md](docs/CURATION.md): curation user guide
- [docs/design/curation_api_contract.md](docs/design/curation_api_contract.md): routes and wire models
- [docs/opensearch_schema_design.md](docs/opensearch_schema_design.md): index schemas
- [contracts/README.md](contracts/README.md): generated contracts
- [ATTRIBUTION.md](ATTRIBUTION.md): third-party licensing
