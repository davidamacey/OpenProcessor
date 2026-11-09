# System Architecture

How OpenProcessor is built: the services, the inference path, the curation
data model (projects, items, region boxes, the config store, embedding spaces,
cluster partitions, near-duplicate groups) and the runtime topology. For the user-facing guide see [CURATION.md](CURATION.md); for the
wire contract see [design/curation_api_contract.md](design/curation_api_contract.md).

---

## Table of Contents

1. [Overview](#overview)
2. [Services and compose profiles](#services-and-compose-profiles)
3. [Inference path](#inference-path)
4. [Curation subsystem](#curation-subsystem)
5. [Projects and isolation](#projects-and-isolation)
6. [Data model](#data-model)
7. [Regions: the multi-box cascade](#regions-the-multi-box-cascade)
8. [Config store](#config-store)
9. [VLM endpoints](#vlm-endpoints)
10. [Datasets, reprocess and combine](#datasets-reprocess-and-combine)
11. [Embedding spaces](#embedding-spaces)
12. [Cluster partitions](#cluster-partitions)
13. [Near-duplicates](#near-duplicates)
14. [The auto-label loop record](#the-auto-label-loop-record)
15. [Serving profile and the Triton router](#serving-profile-and-the-triton-router)
16. [Workers, jobs and events](#workers-jobs-and-events)
17. [Concurrency and the lock rule](#concurrency-and-the-lock-rule)
18. [Export, training and promotion](#export-training-and-promotion)
19. [Contracts](#contracts)
20. [Security boundary](#security-boundary)
21. [Per-process state](#per-process-state)
22. [Scaling notes](#scaling-notes)

---

## Overview

```
 Client / Cropwright
        |
        v  :4603
  +-----------+  gRPC   +----------------+
  |   api     |-------->| triton-server  |  TensorRT engines (GPU)
  | (FastAPI) |         +----------------+
  |           |  HTTP   +----------------+
  |           |-------->| opensearch     |  k-NN: items, images, vectors
  |           |  SQL    +----------------+
  |           |-------->| postgres       |  projects, classes, configs, jobs,
  |           |  Valkey |                |  events, spaces, partitions
  |           |-------->| valkey         |  task broker, event wake-ups
  +-----------+         +----------------+
        ^                      |  tasks (Celery)
        |                      v
  +---------------------------------------------+
  | jobs-worker, periodic-worker, jobs-beat,    |
  | open-vocab-worker (always); profile         |
  | `curation`: cluster, auto-label, assist     |
  | loop, detection and VLM stream workers;     |
  | GPU queues: trainer, evaluator, export      |
  +---------------------------------------------+
        |  HTTP                      |  HTTP
        v                            v
  segmenter (profile `segmenter`)   vlm (profile `vlm`) or any remote
  MLflow (profile `training`)       OpenAI-compatible endpoint
```

Two surfaces share one process:

- The **inference API** (`/detect`, `/faces`, `/embed`, `/search`, `/ingest`,
  `/ocr`, `/analyze`, `/clusters`, `/query`, `/models`, `/health`), also
  mounted under `/v1`. It writes to the global `visual_search_*` indexes.
- The **curation API** (`/curation`, no `/v1` twin). Everything project scoped
  lives under `/curation/projects/{project}/`. Project-bound code can only see
  that project's indexes and directories.

The service layer (`src/services/`) has no FastAPI dependency. Routers under
`src/routers/` and `src/routers/curation/` are thin HTTP adapters over it.

---

## Services and compose profiles

`docker-compose.yml` is deploy-safe by itself (no `build:`, no source mounts).
`docker-compose.dev.yml` adds local builds and hot-reload mounts for a
checkout. GPU placement comes from `.env` keys, not from the compose files.

| Service | Profile | Port | Role |
|---|---|---|---|
| `triton-server` | none | 4600 HTTP, 4601 gRPC, 4602 metrics | Serves the TensorRT models. Explicit model control; loads the list in the compose command; a missing engine leaves that model unloaded instead of killing the server |
| `api` | none | 4603 | FastAPI, uvicorn with `API_WORKERS` worker processes (the installer writes `min(nproc, 32)`), each recycled after about 10,000 requests (limit 10,000 with 1,000 jitter) |
| `opensearch` | none | 4607 | k-NN indexes for items, images and vectors; the control data is not here |
| `postgres` | none | | PostgreSQL 18: projects, classes, config documents, settings, jobs, events, embedding spaces, cluster partitions, near-duplicate groups, the auto-label loop record. Row-level security scopes every project query. Internal only |
| `valkey` | none | | Valkey 9: the Celery broker and the event wake-ups. Internal only, password required |
| `db-migrate` | none | | One-shot: Alembic upgrade and OpenSearch index migration; the API and every worker wait for it and refuse to start at another schema revision |
| `jobs-worker`, `periodic-worker`, `open-vocab-worker` | none | | Celery workers for the `jobs` (default concurrency 4), `periodic` and `open_vocab` queues |
| `jobs-beat` | none | | The beat scheduler, one replica (`src/tasks/beat_schedule.py`: reaper, ingest dispatch, open-vocabulary sweep, serving inventory, near-duplicate rescore, cluster refresh and the rest) |
| `cluster-worker` | `curation` | | Celery worker for the `cluster` queue (clustering runs, one task per child process) |
| `curation-detection-worker`, `curation-vlm-worker` | `curation` | | Stream workers: the region cascade over pending items and the VLM class and verification loop. Replicas share projects through claims in PostgreSQL |
| `curation-auto-label-worker`, `curation-assist-loop-worker` | `curation` | | Celery workers for the `autolabel` and `assist_loop` queues |
| `model-export` | `setup` | | One-shot engine-build toolchain (TensorRT, ModelOpt, the export scripts and the model-repo seed): the `export` target of the root `Dockerfile`, which shares its runtime layers with `api`. `models install`, `export` and the `make` export targets run it |
| `model-export-worker` | `export` | | The same image as a long-running Celery worker (queue `gpu_export`) for `POST /models/upload` and `POST /models/{model_name}/export`: the API queues a job, the worker builds the engine and loads it into Triton (503 from the routes while it is down) |
| `curation-evaluator` | `evaluator` | | Runs bake-off jobs (queue `gpu_eval`) |
| `segmenter` | `segmenter` | 4611 | Region proposals from a text prompt, `docker/segmenter/` |
| `curation-trainer` | `training` | | Runs training jobs (queue `gpu_train`), `docker/trainer/` |
| `curation-mlflow` | `training` | 4609 | Experiment tracking, optional |
| `vlm` | `vlm` | 4612 | Local vLLM serving one catalog model |
| `cropwright` | `cropwright` | 5184 (`OP_CROPWRIGHT_PORT`) | The Cropwright web UI (nginx serving `frontend/`); proxies the API, and with `OP_GATEWAY_SUBPATHS=true` the monitoring UIs, so that one port is the only one published beyond loopback |
| Prometheus, Grafana, Loki, Alloy, DCGM exporter, node exporter, Valkey exporter, Flower, OpenSearch Dashboards | `monitoring` | 4604, 4605, 4606, 4610, 4613 (Flower), 4608 | Opt in only |
| `triton-sdk` | `benchmark` | | Benchmark client |

Named volumes: `openprocessor-pg` (PostgreSQL), `openprocessor-queue` (Valkey's append-only
file), `openprocessor-state` (uploads, training runs and other state files), the crop cache
(tmpfs). Per-project exports and bake-off data
live under `./data/projects/<slug>/`. Container names are
`<project>-<service>-<n>`; no service sets `container_name`.

`docker-compose.gpu-arbiter.yml` is an opt-in overlay that adds the one-process
`gpu-arbiter-worker`, the only service that mounts the Docker socket, so the
GPU arbiter can stop and restart sibling containers around a training run. It
gives that container control of the Docker host and is single-host; read its
header before using it. `docker-compose.pgbouncer.yml` is an opt-in PgBouncer
in front of PostgreSQL.

---

## Inference path

- **Triton models** (`models/`): `yolov11_small_trt_end2end` (GPU NMS),
  `yolo26_small_trt` (NMS-free), `scrfd_10g_bnkps`, `arcface_w600k_r50`,
  `mobileclip2_s2_image_encoder`, `mobileclip2_s2_text_encoder`,
  `paddleocr_det_trt`, `paddleocr_rec_trt`, `ocr_pipeline` (a BLS pipeline;
  see [OCR.md](OCR.md)), `gpu_image_prep` (decode-once ingest, loaded on first use) and
  the curation encoders `pe_image_encoder` (TensorRT) and `pe_text_encoder`
  (CPU). Dynamic batching is on; preferred sizes and instance counts are in each
  `config.pbtxt`. The API reads a detector's output format from Triton
  metadata, so YOLO11 and YOLO26 are interchangeable at `/detect`.
- **One shared gRPC client pool** (`src/clients/triton_pool_core.py`) per process.
  Opening a connection per request defeats Triton's dynamic batching, so route
  code takes the shared client, never builds its own.
- **Preprocessing** (letterbox, normalization) runs in the API on CPU before
  the request. Keep the API layer to validation and decoding; the models own the
  rest. The exception is photo ingest: with `OP_INGEST_GPU_DECODE` on, a photo at or
  above `OP_INGEST_GPU_MIN_MEGAPIXELS` goes to the `gpu_image_prep` model, which
  decodes it once on the GPU and builds the detector input, the PE vectors, the blur
  scores and the crop JPEGs there. A file it declines falls back to the CPU path.
  Details and numbers: [PERFORMANCE.md](PERFORMANCE.md).
- **Thread safety.** Ultralytics model objects keep internal state. The code
  creates a thin client wrapper per request rather than sharing one instance
  across threads. The wrapper is only a gRPC front, not a loaded model.
- **PE-Core text** embeddings (`OP_PE_TEXT_BACKEND`, default `auto`) prefer the
  Triton `pe_text_encoder` so all uvicorn workers share one CPU instance, then
  fall back to an in-process PyTorch load. See
  [export/README.md](../export/README.md#pe-core-encoders-curation-embeddings).

---

## Curation subsystem

A generic active-learning stack for building a labeled image dataset in any
domain. A domain is configured with data (class registry, region profile,
prompt pack, VLM endpoint), not by forking code. Four dataclasses carry the
deployment-specific parts: `CurationConfig` (roots, prefixes, dimensions),
`RegionFields` (storage names of the region fields), `DetectionProfile` (the
region stage as data) and `RegionStatus` (the pipeline state machine). The
design rationale is in
[design/curation_design_rationale.md](design/curation_design_rationale.md).

| Area | Path | Responsibility |
|---|---|---|
| Config | `src/config/` | Curation config, region fields, detection profile, project records and context, retired-env guard |
| Projects | `src/services/projects/` | Registry, lifecycle, the OpenSearch guard, clone, combine, capacity |
| Config store | `src/services/config_store/` | Prompt packs, region profiles, activations, VLM endpoint registry |
| Control plane | `src/db/`, `src/tasks/` | SQLAlchemy models and Alembic revisions, repositories, `project_tx`, the Celery app, task modules, the beat schedule |
| OpenSearch client | `src/clients/curation_opensearch/` | Index bodies, mapping migrations, item helpers |
| OCC | `src/clients/occ.py`, `occ_locks.py` | Optimistic-concurrency writes, the lock rule |
| Ingest and import | `src/services/curation/ingest*.py`, `dataset_import/` | Item creation, duplicate detection, labeled-dataset import |
| Regions | `src/services/curation/region_*.py`, `src/services/detection/` | Box list, edits, verification, text, embeddings, detector and segmenter cascade |
| Clustering | `src/services/clustering/`, `src/domain/clustering_vocab.py` | Partition runs (IVF k-means with FAISS, `MiniBatchKMeans` for boxes), the assigner, calibration, the served projection onto items |
| Embedding spaces | `src/services/embedding_spaces/`, `src/domain/embedding_spaces.py` | The registry of model versions and spaces, builds, reindex runs, promotion |
| Near-duplicates | `src/services/near_duplicates/`, `src/domain/near_dup.py` | Edges, grouping, decisions, the holdout gate |
| Scoring, selection, search | `src/services/curation/item_scores/`, `selection/`, `semantic_search.py` | Mistakenness and uniqueness, diverse sampling, PE-Core text-to-image search |
| Review | `src/services/curation/review_*.py`, `holdout.py` | Queue queries, sorts, the frozen test set |
| VLM | `src/services/labeling/` | Transport, endpoints and probes, prompts, class and region labelers |
| Reprocess | `src/services/curation/reprocess*.py` | One selection and lock rule behind every re-run |
| Assist runs | `src/services/curation/assist/` | Scope, seeded sampling, the VLM call and the `assist` slot of an item; a run is a job |
| Auto-label loops | `src/services/curation/assist_loop/`, `src/db/repos/assist_loop*.py` | Pinned partition, exact sample gate, propagate or split, budgets, audit and rollback; the record is in PostgreSQL, run by `curation-assist-loop-worker` |
| Export and training | `src/services/curation/export*.py`, `src/services/training/` | YOLO export, job lifecycle, preflight, promotion |
| Routers | `src/routers/curation/` (including `images.py`, `umap.py`, `train/`) | HTTP surface |
| Workers | `src/tasks/`, `scripts/curation/` | Celery task modules for each job family; stream workers for detection and VLM; evaluator |

The wire uses one vocabulary (`region_*`, `vlm_*`, ...) that does not change
with storage names; one serializer, `src/services/curation/wire.py`, maps
storage to wire.

---

## Projects and isolation

A project is a named, isolated dataset workspace. The project records are rows in PostgreSQL.

| Resource | Where |
|---|---|
| Control data | PostgreSQL rows carrying the project id (classes, config documents and activations, settings, keymap, jobs, events, embedding spaces, cluster partitions, near-duplicate groups, loop records); row-level security limits a transaction to its project (`project_tx`) |
| Indexes | `{OP_PROJECT_INDEX_PREFIX}{slug}__{role}`, default prefix `op_prj_`, roles `images` and `items` |
| Exports, bake-off eval sets | `OP_PROJECTS_DATA_ROOT/<slug>/` (default `./data/projects/<slug>/`) |
| Uploads | `OP_STATE_DIR/projects/<slug>/` |
| MLflow experiment | `openprocessor-<slug>` |
| Promoted Triton models | prefixed `<slug>__` (the `default` project has no prefix) |

Names are computed once when the project is created and stored in the record;
a later env change never remaps a live project.

**Slugs** are 2-32 characters: lowercase letters, digits and single hyphens,
starting with a letter. `combine`, `new`, `all`, `none`, `projects`, `global`,
`settings`, `vlm` and `health` are reserved, and a deleted slug cannot be reused
(`slug_retired`).

**Lifecycle.** Statuses: `building`, `active`, `archived`, `deleting`, `failed`.

| Action | Route | Notes |
|---|---|---|
| Create | `POST /curation/projects` | Answers 202 with the project in status `building` and the `project_create` job in `job_id`; the directories, indexes and settings clone (optional `clone_settings_from` and `clone_axes`) run in that job, which ends the project `active` (or `failed`). A repeat with the same `Idempotency-Key` answers 200 with the first job; `GET /curation/jobs/{job_id}` reads a job by id |
| Rename, describe | `PATCH /curation/projects/{project}` | Needs `expected_revision`; the slug is immutable |
| Archive, unarchive | `POST /curation/projects/{project}/archive`, `POST /curation/projects/{project}/unarchive` | An archived project refuses writes (`project_archived`); workers skip it |
| Copy settings | `POST /curation/projects/{project}/clone_settings` | Axes: `settings_defaults`, `classes`, `activations`, `keymap`, `prompt_packs`, `vlm_activation` |
| Delete | `DELETE /curation/projects/{project}` | `?dry_run=true` reports what would go; a real delete needs `?confirm=<slug>` and answers 202 while it drains and removes indexes and directories. `default` cannot be deleted |

**The guard.** Every OpenSearch client is wrapped by a transport-level guard
(`src/services/projects/guard.py`). It allows only request shapes the code
really sends, aimed at the bound project's concrete index names. Index-less
searches, wildcards, `_all`, aliases, `_reindex`, `_sql`, snapshots and any
request that names another project's index raise `CrossProjectAccess`. Unbound
code can touch only indexes no project owns (the `visual_search_*` set).
Workers are not bound to one project: each cycle they list active projects,
skip paused ones and bind one project around that project's work. Scripts bind
with `--project`.

**Combine.** `POST /curation/projects/combine/preview` checks one to eight
source projects and returns a mapping suggestion; `POST /curation/projects/combine`
starts a job that builds a new target project (status `building` while it fills,
then `active`). Sources are not modified. Class mapping is by name, duplicate
images are merged by content hash with their boxes attached, and the holdout can
be preserved as a union, recomputed or dropped. Progress arrives on the global
event stream and at `GET /curation/projects/combine/{job_id}`; a job can be
cancelled and resumed. Undoing a combine is deleting the target project.

---

## Data model

Two stores, split by what they hold. **PostgreSQL** holds the control data: what
exists and how it is configured. **OpenSearch** holds the bulk data: images,
items, boxes and their vectors. A row in one store names the other's documents
by id only.

| PostgreSQL (`src/db/models/`) | Holds |
|---|---|
| `projects`, `project_slugs`, `project_indexes`, `class_registries`, `project_classes`, `project_settings`, `project_keymaps`, `project_state` | Project records and retired slugs, the class registry (classes, names, hotkeys, the next class id), shared defaults, keymaps, the pause flag |
| `config_documents`, `config_revisions`, `config_activations` | Prompt packs, region and ingest profiles, VLM endpoints, serving, cluster, near-duplicate and loop-preset documents |
| `jobs`, `job_attempts`, `job_items`, `events`, `stream_worker_claims` | Every job, its attempts and items, the event stream, the stream workers' project claims |
| `embedding_model_versions`, `embedding_spaces`, `embedding_reindex_runs` | The embedding registry |
| `cluster_partitions`, `cluster_identities`, `cluster_memberships`, `cluster_activations` | Cluster partitions and their stable identities |
| `image_contents`, `near_dup_*` | The exact-duplicate key and near-duplicate runs, groups, edges and decisions |
| `assist_loop_*`, `vlm_usage_hourly` | The auto-label loop record and VLM usage |
| `privacy_purges` | Counts-only record of every purge of stored faces or text |
| `models`, `model_shares`, `training_runs`, `bakeoff_runs`, `serving_reports` | Promoted models, sharing, training and bake-off runs, which Triton server holds which model |

| OpenSearch index role | Holds | Key fields |
|---|---|---|
| `images` | One document per ingested source image | `image_id`, `image_path`, `imohash`, `phash`, `capture`, `embedding` (512-d), `pe_embedding` (1024-d) |
| `items` | One document per crop (item) | see below |

**Items** carry the class (`class_id`, `class_name`, `class_source`,
`class_validated`), the crop `bbox_norm` in the source frame, the served
cluster (`cluster_partition_id`, `cluster_uid`, `cluster_state`, `cluster_score`,
`cluster_margin`), the near-duplicate projection (`near_dup_group_id`,
`near_dup_is_representative`, ...), quality and score fields, VLM
provenance (`vlm_endpoint`, `vlm_model`, `vlm_prompt_pack`), import and combine
provenance, and the region fields described next. The exact wire keys are in
[`contracts/json/item_wire.json`](../contracts/json/item_wire.json).

**The class registry is a PostgreSQL table per project**, written with optimistic
concurrency: every change re-reads and re-applies on a conflict, so two users, an
import, a combine and the open-vocabulary pass can add classes at once without
losing one or giving two classes one id. `ensure_class` is idempotent by name for
the automated writers. A 0.6.0 `class_registry.json` is loaded into the database
by the one-step control-data upgrade (`scripts/db/upgrade_from_060.py`, run by
`openprocessor upgrade`); the file is only an export snapshot and the training pin.
See [opensearch_schema_design.md](opensearch_schema_design.md#the-class-registry).

**Class identity is the name.** The class registry assigns indexes, but every boundary maps by
name: dataset import and export, combine, training, promotion and sharing a
trained model with another project. An index is only a position inside one
registry at one moment. The ingest secondary detector, and the primary when
`OP_INGEST_PRIMARY_ASSIGNS_CLASS` is set, resolve the registry class named like the model's
output, never the output index.

**Schema revisions.** One Alembic head (`src/db/migrations/versions/`):
`0001_control_plane`, `0002_research_seams`, `0003_near_duplicates`,
`0004_privacy_purges` (the `privacy_purges` table, whose CHECK keeps it to counts)
and `0006_model_sharing_revision` (a monotonic `sharing_revision` on `models`).
`db-migrate` applies them; the API and workers refuse to start at another revision.
Revision `0005_region_text` is planned for the plate-text work and not on this branch.

The global indexes (`visual_search_global`, `_vehicles`, `_people`, `_faces`,
`_ocr`) belong to the inference API and are described in
[opensearch_schema_design.md](opensearch_schema_design.md).

---

## Regions: the multi-box cascade

A **region** is a sub-annotation of an item: a wheel on a car crop, text on a
sign. Regions are always a list.

```
ingest -> primary detector -> items (crops)
       -> region stage (items in the profile's parent_classes)
            detector leg  +  segmenter leg   -> candidates
            merge + NMS, keep up to max_regions_per_item
            VLM verifies each box -> per-box verdict
            optional text read -> per-box text
            embed + cluster each box
       -> review: per-box accept / reject / false positive / edit
```

**Storage.** `region_boxes` is a `nested` field on the item, so a query such as
"a box with detector X and state accepted" means one box. N=1 is a list of one
box; there is no second code path. Element keys are fixed: `box_id` (`b1`, `b2`,
never reused inside an item), `bbox_norm`, `state`, `score`, `detector`,
`detector_version`, `source`, `rejection_reason`, `text`, `text_raw`,
`text_source`, `text_confidence`, `cluster_id`, `cluster_subid`,
`cluster_distance`, `detected_at` and more. Per-box vectors are in the sibling
nested field `region_box_embeddings` (`box_id`, `bbox_norm`, `embedding`), so
an edit that rewrites the box list cannot silently delete embeddings. A vector
whose box moved since it was computed is stale and dropped.

**Item summary fields**: `region_status`, `region_count`, `region_rejected_count`,
`region_max_score`, `region_set_complete`, `region_revision` (bumped by any
write that changes a box's state, geometry or text; cluster-only writes do not
bump it), plus the profile stamps `region_profile` and `region_profile_revision`.

**Box states**: `proposed`, `accepted`, `rejected`, `false_positive`. The item
`region_status` is derived: any accepted box gives `detected`, else any
false-positive box gives `false_positive`, else any proposed box gives
`pending_verification`, else any rejected box gives `verify_rejected`. Pipeline
statuses `pending_detection`, `no_region_box`, `no_region_visible` and
`detection_failed` cover items with no boxes.

**Which items.** The profile's `parent_classes` (matched by name, case
insensitive, against `class_name` or the detector's own label `proposal_name`)
select items for the region stage; an empty list means every item. The profile's
`max_regions_per_item` (default 1) caps the boxes kept.

**Human edits** go through four routes and are the only writers of box geometry
and state:

| Route | Use |
|---|---|
| `PUT /curation/projects/{project}/crops/{crop_id}/regions` | Replace or extend an item's boxes; a box named by `box_id` alone is left untouched; takes `expected_region_revision` |
| `PATCH /curation/projects/{project}/crops/{crop_id}/regions/{box_id}` | Change one box |
| `PUT /curation/projects/{project}/crops/batch_regions` | The same over many items |
| `POST /curation/projects/{project}/regions/batch_box_state` | Set many boxes' state |

`POST /curation/projects/{project}/crops/{crop_id}/region/undo` reverts the last
region write. `GET /curation/projects/{project}/regions` lists boxes (one row
per box) with filters that all apply to the same box.

---

## Config store

Prompt packs, region profiles and the VLM activation are **versioned data**,
stored per project in PostgreSQL (`config_documents`, `config_revisions`,
`config_activations`); VLM endpoints are deployment-wide documents of the same tables.

- **Revisions.** Saving a named pack or profile writes a new revision; old
  revisions stay readable.
- **Activation.** Exactly one pack, one profile and one VLM endpoint is active
  per project. Activating applies it to the API and to the workers, which poll
  the store every `OP_CONFIG_POLL_S` seconds. Activation takes
  `expected_active` so a stale editor gets a 409 instead of overwriting.
- **Impact.** `GET /curation/projects/{project}/region_profiles/active/impact`
  reports how many items a profile change touches and the explicit
  `POST /curation/projects/{project}/reprocess` request that re-runs the
  unlocked ones. Activation never reprocesses anything by itself.
- **Rollback.** `POST /curation/projects/{project}/region_profiles/active/rollback`
  and `POST /curation/projects/{project}/prompt_packs/active/rollback` reactivate
  the previous revision.
- **Test on crop.** `POST /curation/projects/{project}/prompt_packs/test` runs a
  pack against stored crops through the real VLM path and returns the prompt,
  the raw reply, parsed results and a preview item.
  `POST /curation/projects/{project}/region_profiles/test` runs a profile's
  detector and segmenter legs (and optionally verification) on one stored crop
  and returns every candidate with its selection or drop reason. Neither writes
  anything.
- **Validation.** Both have `validate` routes that return a report without
  saving, and `schema` routes that describe the editor fields.
- **Settings.** `GET /curation/projects/{project}/settings` and
  `PUT /curation/projects/{project}/settings` hold shared defaults per axis
  (`cluster`, `sort`, `prompt_pack`, `detection_profile`, `vlm`).
  `GET /curation/projects/{project}/config/vocabulary` serves the enum choices
  the editors render (detectors, segmenters, OCR models, registry classes, text
  reader modes, VLM endpoints).
- **Keymap.** `GET /curation/projects/{project}/keymap` and `PUT` hold per-project
  shortcut overrides on top of the server's action table
  (`contracts/json/keymap_actions.json`); `POST .../keymap/validate` and
  `.../keymap/reset` check and restore.

Packs and profiles can also be loaded from files at startup
(`OP_PROMPT_PACK_PATH`, `OP_PROMPT_PACK_PATHS`, `OP_REGION_PROFILE_PATH`); the
examples live in `examples/`.

---

## VLM endpoints

A VLM endpoint is an OpenAI-compatible chat URL plus a model name, limits and a
key reference. Endpoints are a **deployment-wide registry** (any project sees
them), activation is **per project**.

- Sources: the built-in `env` endpoint (from `OP_VLM_URL` and `OP_VLM_MODEL`),
  endpoints saved through `POST /curation/vlm/endpoints`, and the in-compose
  local `vlm` service.
- A **probe** (`POST /curation/vlm/endpoints/{name}/probe`) checks reachability,
  vision, JSON mode, image limits and context size and records the result per
  `name@revision`. Activation can bypass only a missing or failed probe and a
  small context, and only with `force`.
- **Keys** are never served. An endpoint holds `api_key_ref` (`secret:<slug>`),
  read from `secrets/vlm/` on the host (`./openprocessor vlm key set <slug>`).
- **External endpoints.** A URL outside this deployment sends crops off the
  host. It needs `allow_external` on the endpoint and an acknowledgement at
  activation or per run. `OP_VLM_EXTERNAL_POLICY=deny` refuses them outright.
  This stack's own services, link-local and metadata addresses are always
  refused. See [SECURITY.md](../SECURITY.md#vlm-endpoints-and-server-side-requests).
- **Local catalog.** `GET /curation/vlm/catalog` lists the models in
  `examples/vlm/catalog.tsv` with whether each fits the GPU. The API records the
  wanted model (`POST /curation/vlm/local/select`) and reports
  `restart_required`; the host applies it with `./openprocessor vlm use <id>`.
- **Provenance.** Items record `vlm_endpoint` (`name@revision`), `vlm_model` and
  `vlm_prompt_pack` for the answer that wrote them.
- **Per-run selection.** Labeling and verification routes and the auto-label
  job take `?vlm=<name|name@revision>`; unknown names answer 422, a project with
  the VLM off answers 409.

---

## Datasets, reprocess and combine

**Dataset import** (`/datasets/*`): upload an archive
(`POST /curation/projects/{project}/datasets/uploads`) or name a server path,
preview it (`POST /curation/projects/{project}/datasets/preview`), then start a
job (`POST /curation/projects/{project}/datasets/imports`). The formats (YOLO, COCO,
the OpenProcessor export that round-trips labels, splits and the frozen test set,
and six more) are listed in [Dataset import](../docs-site/docs/guides/dataset-import.mdx). The mapping is a list of `{dataset_class, action}` with `action` one
of `map`, `create`, `skip`, `region`; matching is by name, and every class that
has boxes needs a decision. Options include `label_trust` (`validated` or
`suggestion`), `processing` (`none` or `propose`), `missing_label` and
`freeze_test_split`. An import can be cancelled, resumed and undone; undo keeps
anything a human has since edited. The import report counts images, items,
labels, regions, negatives and the labels a human lock refused.

**Reprocess** (`POST /curation/projects/{project}/reprocess`) re-runs pipeline
scopes over chosen targets (`image_ids`, `crop_ids` or a `filter`). Scopes:
`detect`, `open_vocab`, `region`, `vlm`, `embed`; region mode is `redetect` or `reverify`.
`dry_run` defaults to true. Locked items and boxes are never written and are
reported as `locked_skipped`. Large detect or embed runs become a job
(`GET /curation/projects/{project}/reprocess/jobs/{job_id}`); only one runs at a
time. Single-target forms:
`POST /curation/projects/{project}/crops/{crop_id}/reprocess` and
`POST /curation/projects/{project}/images/{image_id}/reprocess`.

---

## Embedding spaces

An encoder build is a **model version** (`embedding_model_versions`: model key, version,
modalities, native dimension, distance, precision, the Triton models that serve it, the engine
parity against the reference, licence). One vector field in one project index, written by one
model version for one target at one dimension, is a **space** (`embedding_spaces`). The registry
stores the physical field names (`vector_field`, `state_field`, and for a box the `nested_path`);
every reader takes them from a resolved space and never computes them (`src/domain/embedding_spaces.py`).
A project's existing PE-Core layout is its default space under the 0.6.0 names (`pe_embedding`,
`embedding_state`, `region_box_embeddings`), so nothing moves on upgrade.

- **Targets and roles.** A space targets `item`, `box`, `frame` or `item_text`. Per target one
  space is `primary` (always `ready`; search and clustering read it), others are `optional` or
  `candidate`, at most three live per target. The index role is `images` or `items`.
- **Lifecycle.** `POST /curation/projects/{project}/embedding/spaces` registers a space
  (202), `.../spaces/{space_key}/build` fills it with a reindex run (202; a PIT scan,
  an encoder pass, an OCC write guarded by box geometry, cancellable at
  `.../embedding/reindex-runs/{run_id}/cancel`), `.../promote` flips it to primary and
  `.../retire` removes it (202). The flip is one transaction fenced on the space `revision`,
  and bumps the project's config epoch so every worker re-resolves its spaces. A promote starts
  a fresh cluster run and a near-duplicate score on the new primary. Capacity (shards, disk) is
  checked before a build and refused with `embedding_insufficient_storage`.
- **Search and reads** take a `space` parameter; an item reports `embedding_states` per space.
  A space that is not `ready` answers `embedding_space_not_ready`.

---

## Cluster partitions

A clustering run does not overwrite cluster ids. It writes a **partition**: one versioned,
immutable result (`cluster_partitions`) for a target (`item`, `region_box`, `face`, `image`) and
a level (`class`, `identity`, `scene`), made by a method (IVF k-means, mini-batch k-means,
union-find, detector class, or a human `edit`) on a named embedding field, with its parameters,
seed, input digest, backend (`cpu` or `gpu`), counts and stage durations. Memberships
(`cluster_memberships`) carry a state per member: `assigned`, `review` or `unassigned`, with a
score and a margin from the calibration (`cluster_calibrations`) of that partition.

- **Stable identity.** A cluster has a `cluster_uid` (`cluster_identities`) that survives re-runs:
  a new run matches its clusters to the parent partition's by member overlap (Hungarian
  assignment on Jaccard) and a matched cluster keeps its uid, with a transition recorded
  (`new`, `survived`, `split`, `merged`, `absorbed`). A cluster a person named or anchored is
  pinned: its members stay, and a run that had to override pins is kept out of auto-activation.
- **The assigner.** New items are placed incrementally at the serving partition's nearest
  prototype with the calibrated bands (`cluster_assigned_by: incremental`); a periodic refresh
  (`op.periodic.cluster_refresh`) starts a full run when enough has changed. A running
  auto-label loop holds the refresh.
- **Activation.** A finished partition is pending; `PUT /curation/projects/{project}/clusters/activation`
  (202) moves it to serving in one OCC transaction (409 `partition_conflict` if the caller did not
  see the current serving partition) and a fenced projection writes the served assignment onto
  items and boxes (`cluster_partition_id`, `cluster_uid`, `cluster_state`, `cluster_score`,
  `cluster_margin`, `cluster_assigned_by`). `GET .../clusters`, `.../clusters/{cluster_uid}/facets`,
  `.../clusters/partitions` and `.../clusters/partitions/{partition_id}` read it.
  `POST .../clusters/runs` starts a run; profiles are `cluster_profiles` documents.
- The 0.6.0 sub-cluster refine, per-item `cluster_id`/`cluster_subid` and label-purity fields are gone.

---

## Near-duplicates

Exact and near duplicates are two levels of one service. **Exact**: ingest keys an image on its
sampled `imohash` and confirms by SHA-256 only when a hash is already stored (`image_contents`),
so the same bytes are one image. **Near**: a run (`near_dup_runs`, mode `full` or `incremental`,
level `image` or `item` per project, plus `core` for the global index) finds candidate edges
between vectors by k-NN above a cosine cut, groups them (connected components over the edges,
human decisions applied first, a group over the size guard split by complete linkage), picks the
most central member as the representative and carries group identity from the parent run.
Groups, members, edges and decisions are rows (`near_dup_groups`, `near_dup_members`,
`near_dup_edges`, `near_dup_decisions`); items carry a projection (`near_dup_group_id`,
`near_dup_leak_group_id`, `near_dup_is_representative`, `near_dup_group_size`, `near_dup_run_id`).

- **Pending rows.** Every ingest, vector change, box change or delete writes a `near_dup_pending`
  row; `op.periodic.near_dup_rescore` (30 s) drains them into incremental runs, and a weekly full
  build re-derives everything. Core ingest writes pending rows too; the request no longer groups.
- **Decisions.** `POST /curation/projects/{project}/near_duplicates/decisions` records `same` or
  `different` for a pair; a decision survives re-runs.
- **Holdout rule.** Training leakage is the reason near-duplicates exist. Before a test holdout is
  frozen nothing is refused. The first freeze needs a current item-level run; a freeze also holds
  out every member of a current item-level `leak` group that holds a selected item
  (`holdout_near_dup_moves`, counted as `moved_by_near_dup`); after a freeze, a re-freeze and an export refuse with `409 near_dup_stale`
  while the groups are stale. An export can collapse each group to its representative
  (`collapse_near_duplicates`).
- **Profiles** (`near_dup_profiles`) set the cuts, `k` and the group-size guard for the image and item levels.

---

## The auto-label loop record

The cluster-sampled auto-label loop keeps its whole record in PostgreSQL (`assist_loop_runs`,
`_iterations`, `_nodes`, `_selections`, `_proposals`, `_reviews`, `_budgets`,
`_budget_ledger`) instead of a state file, and runs as a job on the `assist_loop` queue.

- **Budgets** are reserved before every upstream call and settled from the endpoint's `usage`:
  images (crops), requests, completion tokens as the call's exact `max_tokens`, prompt tokens as an
  estimate. A call that reports no usage is charged what it reserved (fail closed). Wall time is
  charged per time slice, and the job re-enqueues itself between slices, so a loop is
  pausable between slices. `PATCH .../assist/loops/{loop_id}/budget` edits a live budget.
- **Control** is a pure function of the job state (`src/domain/assist_loop_control.py`): the wire
  carries `can_pause`, `can_resume`, `can_cancel`, `can_edit_budget`, `can_rollback` and the
  blocker of a resume (`project_paused` or `vlm_unavailable`); the client renders it. Pause,
  resume, cancel and rollback are `POST .../assist/loops/{loop_id}/pause|resume|cancel|rollback`.
- Presets are `assist_loop_presets` documents. Hourly VLM usage per endpoint is `vlm_usage_hourly`.

---

## Serving profile and the Triton router

Triton can run as one server or several. The deployment-wide **serving profile** (a
revisioned config document, `/curation/serving_profiles`: list, get, put, delete,
activate) names the servers (one per card: `name`, `grpc_url`, `metrics_url`, the card UUIDs),
the models each keeps resident (`load_list`), the instances per model, the GPU image-prep
settings and the interactive and bulk priorities. With no profile active the router is the single default server
(`TRITON_URL`), exactly as before.

`op.periodic.serving_inventory` (every 60 s) asks each server which models are ready and
writes `serving_reports`; only `ready` rows count, so a server the inventory could not
reach gets no call. `src/services/serving_router.py` builds the per-process
`TritonRouter` from the profile and those reports; every Triton caller (the async pool, the
sync inference, analyze, OCR, embed and visual-search paths, the detection worker) asks the
router for a server that holds every model it needs (`src/services/sync_triton.py` for the
synchronous callers). A change of the active profile is picked up on the next config poll:
the old router refuses calls, a new one is built, the pool opens channels to the new servers
and closes the dropped ones, and a rebuild that fails leaves the old router refusing until
the next tick (fail closed). Health probes stay on the default server; readiness per routed
server is `serving_reports`. Follow-up calls of one image are not pinned to one server yet.

---

## Workers, jobs and events

- **Jobs.** A job is a row in the `jobs` table (`src/db/repos/jobs.py`) with a kind
  (`src/config/job_kinds/`), a status (`queued`, `running`, `paused`, `completed`,
  `failed`, `cancelled`, `interrupted`, ...), attempts, items and progress. `start_job`
  inserts the row and enqueues one Celery message that carries only the job id; the
  task modules in `src/tasks/` load the row and run it. A kind may be *exclusive* per
  project or per scope (a second start answers 409 `job_busy`), *resumable* and
  *auto-resumed*. Each task route is in `src/config/task_routes/` (a decorator never sets
  `queue=`), so an operator can see every queue in one place: `jobs`, `periodic`,
  `open_vocab`, `cluster`, `autolabel`, `assist_loop`, `arbiter` and the GPU queues
  `gpu_train`, `gpu_eval` and `gpu_export`. Routes that used to run work in the request
  answer **202** with a job id: project create, delete and combine, batch relabel, move and
  label-operation undo, region `undo_batch` and new-class resolve over `OP_BATCH_SYNC_MAX`
  items, dataset upload and import, bulk ingest and source runs, promote, assist runs,
  scores, probe and projection rebuilds, cluster runs and activation, embedding space
  builds, near-duplicate runs and source backfill. `GET /curation/projects/{project}/jobs` lists a
  project's jobs and `GET /curation/jobs/{job_id}` reads one by id.
- **Claims, not locks.** The cross-process claim is the fenced claim of a job row or a
  PostgreSQL advisory lock (`src/config/advisory_locks.py`). A worker that cannot renew
  stops writing. No file lock guards cross-process state
  (`scripts/codegen/check_file_locks.py` fails on any), and no asyncio or threading lock
  stands in for one. Stream workers (detection, VLM) share projects through rows in
  `stream_worker_claims`.
- **The reaper** (`op.periodic.reaper`, every 30 s) recovers jobs whose worker died,
  resumes interrupted resumable jobs, fails the jobs of a queue whose profile never
  started, and prunes old events and finished jobs.
- **Cancel and pause are database writes.** Nothing calls Celery `revoke`; a worker reads the
  flag when it claims a message and again between slices of work, and refuses a message for
  a cancelled or paused job. A project pause (`POST /curation/projects/{project}/pause`) is a
  row every worker reads.
- **Per-process caches.** Each API worker process keeps its own caches, so a rule must not
  depend on another process noticing a change. The class-name cache
  (`src/utils/class_names.py`) is keyed by the `(st_mtime_ns, st_size)` of the model's
  `labels.txt`, one `stat` per lookup, so every worker sees a promote on its next request and a
  missing file is never cached. The settings document cache holds a value for 5 s and raises
  on a failed read instead of caching an empty document. The diverse-order and outlier caches
  are bounded `cachetools` TTL caches. The eval-dataset listing resolves class names at read
  time, and its cache key includes the mtimes of `data.yaml` and `images/test`. The VLM probe
  cap (two at a time) is per API worker process.
- **OCR detector input size.** `/ocr` (`src/clients/triton_ocr.py`) follows PaddleOCR
  `DetResizeForTest`: long side scaled into 640-960, each side rounded to a multiple of 32.
  The region cascade (`ocr_det_input_size`) sizes crops to 320-960 per side. The two stay
  separate on purpose: the default detector engine profile is 32x32 to 960x960
  (`scripts/export_paddleocr.sh`), so `/ocr` cannot send a thin image the engine refuses.
- **Detection worker** (package `scripts/curation/worker/`, started with
  `python -m scripts.curation.worker`): fetches pending items, runs the cascade, writes
  boxes in an OCC merge, and skips any document a human changed meanwhile.
- **Events.** `GET /curation/events` (global) and
  `GET /curation/projects/{project}/events` are Server-Sent Events. Producers
  insert a row in the PostgreSQL `events` table inside their own transaction and,
  after it commits, send a Valkey wake-up (`op:events:<project id | global>`).
  Each API process holds one Valkey subscription, reads new rows from its
  `(xid, id)` cursor on a wake-up or every 2 seconds, and fans them out to its
  clients. Every frame carries `id: <xid>-<id>`, so a reconnect with
  `Last-Event-ID` replays what was missed; a cursor older than the 24-hour
  retention (or from before a restore) gets a `reset` event first. Examples:
  project lifecycle, `combine.progress`, `config.changed`, `vlm.changed`.
- **GPU sharing during training.** The optional GPU arbiter
  (`docker-compose.gpu-arbiter.yml`, `OP_GPU_ARBITER_CONTAINERS`) stops
  configured sibling containers around a training run and restarts them after.
  Without it, `./openprocessor train-mode on|off` stops and starts the VLM (and
  a segmenter on the training GPU) by hand.

---

## Concurrency and the lock rule

The item has many writers: humans, ingest, the detection worker, the VLM worker,
auto-label, clustering, import and reprocess. All use optimistic concurrency
(`src/clients/occ.py`): read with the sequence number, write with
`if_seq_no` and `if_primary_term`, and on conflict re-read and re-merge. Human
routes surface a final conflict as 409; workers skip a conflicting document and
see the new state on the next pass, so a worker never overwrites a human.

**The lock rule** (`src/clients/occ_locks.py`): an automated writer never
changes

- a class a human set or confirmed, or a validated imported label;
- any item frozen into the test holdout;
- a box a human created, gave a verdict, or transcribed, or one that came from
  an import and is not a mere suggestion.

The item wire carries `label_locked`, and box elements carry `locked`.
Reprocess, reconcile, import undo and the delete paths re-check the lock at
write time.

Cluster projection (the served partition written onto items and boxes) and the
near-duplicate projection merge into the live document and retry a few times on
conflict; a run that finished after a human moved a box or set a class cannot
overwrite it. A cluster a person named or anchored keeps its members in a re-run.

---

## Export, training and promotion

- **Export.** `POST /curation/projects/{project}/export/yolo` writes a YOLO
  dataset with a deterministic split and a manifest checksum;
  `POST /curation/projects/{project}/export/single_class` exports one class or a
  subset. Frozen artifacts
  (`class_registry.json`, `data.yaml`, `manifest.json`, `label_stats.json`) are
  served by `GET /curation/projects/{project}/export/registry/{artifact}`.
- **Training.** `POST /curation/projects/{project}/train/preflight` validates
  the dataset and spec, `POST /curation/projects/{project}/train/start` submits a
  job, `GET /curation/projects/{project}/train/status/{job_id}` reads progress.
  The trainer writes checkpoints and a manifest with lineage (dataset hash, code
  versions, evaluation).
- **Promotion.** `POST /curation/projects/{project}/train/promote/{job_id}`
  installs the run into Triton under `<slug>__<name>`, with a `labels.txt` and,
  for a class-subset run, the class remap. A subset run with no resolvable
  remap is refused. Another project can use the model through
  `PUT /curation/projects/{project}/models/{model_name}/sharing`; classes map
  by name.
- **Bake-off.** `POST /curation/projects/{project}/bakeoff/run` compares models
  per class on an eval dataset's test split; the evaluator worker runs the job.

---

## Contracts

`contracts/openapi/curation.json` is generated from the FastAPI app and is the
source of truth for curation routes and schemas. `contracts/ts/` and
`contracts/json/` carry the item wire types and the keymap action table. One
serializer (`src/services/curation/wire.py`) maps storage names to the fixed
wire names, so a storage field override never changes the wire. After any route
or model change run `make contracts`; pre-commit rejects a stale contract.

---

## Security boundary

There is no authentication, authorization or rate limiting. The API is meant
for a trusted machine or network behind your own reverse proxy. Backend ports
bind to `OP_BIND_ADDRESS` (default `127.0.0.1`). Cropwright can be reachable on
the LAN; see [SECURITY.md](../SECURITY.md). The OpenSearch guard protects
project isolation against code bugs, not against a caller of the API.

---

## Per-process state

Any request may reach any worker of any replica, so the API holds no state that another process
would have to see. What is left is per process on purpose, and
`scripts/codegen/check_process_state.py` (baseline `scripts/codegen/baselines/process_state.txt`,
one reason per entry) fails on a new `asyncio` task spawned in a router, service or app module, on
a new module-level `asyncio`/`threading` lock, semaphore or event, and on a new module-level
mutable `dict`, `set` or `list`. Concurrency that must finish before the answer uses
`asyncio.TaskGroup`; work that outlives the request is a job.

| Kind | Examples | Why it is correct per process |
|---|---|---|
| Caches with a revision check | class names, class registry, config-store handles, embedding spaces, eval datasets, prompt packs | The stored revision (or file mtime) is compared on read, so a change made through another process is seen on the next request |
| Caches with a staleness window | pipeline stats, project disk stats, field coverage, URL locality, labeler precheck | A TTL bounds the stale answer, and the value is advisory |
| Local throttles | fetch gates, thread and CPU pools, concurrent config tests, segmenter latency average | They cap this process's own concurrency; the pools are sized per process from totals |
| Process handles | database engines, Valkey clients, object-store clients, tokenizer | A connection or client must not cross a fork or an event loop |
| Process loops | event listener, registry and config pollers, ops-metrics refresh | Every worker must hear every event and refresh its own caches |
| Request-scoped tasks | disconnect watcher, page prefetcher, ingest writer drain, reprocess read-ahead | Created and awaited or cancelled inside one request or job |

---

## Scaling notes

- One GPU runs Triton, the segmenter and a local VLM only when VRAM allows;
  `.env` GPU keys place each service. Instance counts per Triton model are in
  the `config.pbtxt` files; raise them for hot paths when VRAM is free.
- The API scales by uvicorn workers within one host: `API_WORKERS` in `.env` feeds
  `WEB_CONCURRENCY` (the installer writes `min(nproc, 32)`). The database, thread and Triton pools
  are split from deployment totals per process (`src/config/api_pools.py`, formulas in
  [INSTALLATION.md](../INSTALLATION.md#api-worker-count)), so the count is a tuning knob and
  several replicas need only `API_WORKERS=1` and `OP_API_REPLICAS`.
- OpenSearch heap is sized from RAM by the installer (half of RAM, clamped to 2-30 GB, `scripts/lib/opensearch_heap.sh`). Each
  project adds two indexes, so watch the shard budget
  ([INSTALLATION.md](../INSTALLATION.md#opensearch-heap-sizing)); creating a
  project past the capacity limit answers 409 with the capacity figures.
- Job state, events and claims live in PostgreSQL and the broker is Valkey, so API
  and worker replicas need no shared job directory. Several hosts also need a shared
  POSIX volume for uploads, exports and the crop cache; the GPU-arbiter overlay is
  single-host, and upgrades are stop-the-world (old and new processes never run
  together). See [INSTALLATION.md](../INSTALLATION.md).
- Workers: the Celery workers' concurrency is set per queue in `docker-compose.yml`
  (`OP_JOBS_CONCURRENCY`, replicas through `OP_JOBS_REPLICAS`); the connection total
  of the deployment is `API_WORKERS * OP_API_REPLICAS` API processes plus every
  worker process, and must stay under `OP_PG_MAX_CONNECTIONS` (default 400).

---

## Further reading

- [FastAPI concurrency](https://fastapi.tiangolo.com/async/)
- [Ultralytics thread-safe inference](https://docs.ultralytics.com/guides/yolo-thread-safe-inference/)
- [NVIDIA Triton optimization guide](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/optimization.html)
