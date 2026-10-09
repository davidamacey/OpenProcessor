# System Architecture

How OpenProcessor is built: the services, the inference path, the curation
data model (projects, items, region boxes, the config store) and the runtime
topology. For the user-facing guide see [CURATION.md](CURATION.md); for the
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
11. [Workers, jobs and events](#workers-jobs-and-events)
12. [Concurrency and the lock rule](#concurrency-and-the-lock-rule)
13. [Export, training and promotion](#export-training-and-promotion)
14. [Contracts](#contracts)
15. [Security boundary](#security-boundary)
16. [Scaling notes](#scaling-notes)

---

## Overview

```
 Client / Cropwright
        |
        v  :4603
  +-----------+  gRPC   +----------------+
  | yolo-api  |-------->| triton-server  |  TensorRT engines (GPU)
  | (FastAPI) |         +----------------+
  |           |  HTTP   +----------------+
  |           |-------->| opensearch     |  k-NN + curation datastore
  +-----------+         +----------------+
        ^   files (/jobs, state volume)
        |
  +---------------------------------------------+
  | curation workers (profile `curation`)       |
  |  detection, VLM, auto-label, cluster refresh|
  |  evaluator                                  |
  +---------------------------------------------+
        |  HTTP                      |  HTTP / files
        v                            v
  segmenter (profile `segmenter`)   vlm (profile `vlm`) or any remote
  trainer + MLflow (profile `training`)  OpenAI-compatible endpoint
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
| `yolo-api` | none | 4603 | FastAPI, uvicorn with 32 worker processes |
| `opensearch` | none | 4607 | k-NN indexes and every curation document |
| `curation-detection-worker` | `curation` | | Region cascade over pending items, every active project in turn |
| `curation-vlm-worker` | `curation` | | VLM class and verification loop |
| `curation-auto-label-worker` | `curation` | | Drives the auto-label job protocol |
| `curation-cluster-refresh` | `curation` | | Periodic residual-clustering refresh |
| `curation-evaluator` | `curation` | | Runs bake-off job specs |
| `segmenter` | `segmenter` | 4611 | Region proposals from a text prompt, `docker/segmenter/` |
| `curation-trainer` | `training` | | Runs training jobs from `job.json` files, `docker/trainer/` |
| `curation-mlflow` | `training` | 4609 | Experiment tracking, optional |
| `vlm` | `vlm` | 4612 | Local vLLM serving one catalog model |
| Prometheus, Grafana, Loki, Alloy, DCGM exporter, node exporter, OpenSearch Dashboards | `monitoring` | 4604, 4605, 4606, 4610, 4608 | Opt in only |
| `triton-sdk` | `benchmark` | | Benchmark client |

Named volumes: `openprocessor-state` (state dir, job files, events),
`openprocessor-jobs` (training and auto-label jobs), `openprocessor-crop-cache`.
Per-project class registries, exports and bake-off data live under
`./data/projects/<slug>/`.

`docker-compose.gpu-arbiter.yml` is an opt-in overlay that mounts the Docker
socket into `yolo-api` so the GPU arbiter can stop and restart sibling
containers around a training run. It gives that container control of the Docker
host; read its header before using it.

---

## Inference path

- **Triton models** (`models/`): `yolov11_small_trt_end2end` (GPU NMS),
  `yolo26_small_trt` (NMS-free), `scrfd_10g_bnkps`, `arcface_w600k_r50`,
  `mobileclip2_s2_image_encoder`, `mobileclip2_s2_text_encoder`,
  `paddleocr_det_trt`, `paddleocr_rec_trt`, `ocr_pipeline` (a BLS pipeline) and
  the curation encoders `pe_image_encoder` (TensorRT) and `pe_text_encoder`
  (CPU). Dynamic batching is on; preferred sizes and instance counts are in each
  `config.pbtxt`. The API reads a detector's output format from Triton
  metadata, so YOLO11 and YOLO26 are interchangeable at `/detect`.
- **One shared gRPC client pool** (`src/clients/triton_pool.py`) per process.
  Opening a connection per request defeats Triton's dynamic batching, so route
  code takes the shared client, never builds its own.
- **Preprocessing** (letterbox, normalization) runs in the API on CPU before
  the request. Keep the API layer to validation and decoding; the models own the
  rest.
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
| OpenSearch client | `src/clients/curation_opensearch.py` | Index bodies, class registry, item helpers |
| OCC | `src/clients/occ.py`, `occ_locks.py` | Optimistic-concurrency writes, the lock rule |
| Ingest and import | `src/services/curation/ingest*.py`, `dataset_import/` | Item creation, duplicate detection, labeled-dataset import |
| Regions | `src/services/curation/region_*.py`, `src/services/detection/` | Box list, edits, verification, text, embeddings, detector and segmenter cascade |
| Clustering | `src/services/curation/clustering/` | FAISS/IVF and AHC clustering, per-box clustering, auto-promote |
| Scoring, selection, search | `src/services/curation/item_scores/`, `selection/`, `semantic_search.py` | Mistakenness and uniqueness, diverse sampling, PE-Core text-to-image search |
| Review | `src/services/curation/review_*.py`, `holdout.py` | Queue queries, sorts, the frozen test set |
| VLM | `src/services/labeling/` | Transport, endpoints and probes, prompts, class and region labelers |
| Reprocess | `src/services/curation/reprocess*.py` | One selection and lock rule behind every re-run |
| Export and training | `src/services/curation/export*.py`, `src/services/training/` | YOLO export, job lifecycle, preflight, promotion |
| Routers | `src/routers/curation/`, `curation_images.py`, `curation_train/`, `curation_umap.py` | HTTP surface |
| Workers | `scripts/curation/` | Detection, VLM, auto-label, cluster refresh, evaluator |

The wire uses one vocabulary (`region_*`, `vlm_*`, ...) that does not change
with storage names; one serializer, `src/services/curation/wire.py`, maps
storage to wire.

---

## Projects and isolation

A project is a named, isolated dataset workspace. The registry is the
`op_projects` index (`OP_PROJECTS_INDEX`).

| Resource | Where |
|---|---|
| Indexes | `{OP_PROJECT_INDEX_PREFIX}{slug}__{role}`, default prefix `op_prj_`, roles `images`, `items`, `labels_confirmed`, `classes`, `umap_state`, `configs` |
| Settings and UMAP view state | folded into the project's `configs` index, by fixed document id |
| Class registry, exports, bake-off eval sets | `OP_PROJECTS_DATA_ROOT/<slug>/` (default `./data/projects/<slug>/`) |
| Uploads, bake-off jobs, pause flag | `OP_STATE_DIR/projects/<slug>/` |
| Training and auto-label jobs | `projects/<slug>/` under the job roots |
| MLflow experiment | `openprocessor-<slug>` |
| Promoted Triton models | prefixed `<slug>__` (the `default` project has no prefix) |

Names are computed once when the project is created and stored in the record;
a later env change never remaps a live project.

**Slugs** are 2-32 characters: lowercase letters, digits and single hyphens,
starting with a letter. `combine`, `new`, `all`, `none`, `projects`, `global`,
`settings`, `vlm` and `health` are reserved, and a deleted slug cannot be reused
(`slug_retired`).

**Lifecycle.** Statuses: `building`, `active`, `archived`, `deleting`,
`deleted`, `failed`.

| Action | Route | Notes |
|---|---|---|
| Create | `POST /curation/projects` | Optional `clone_settings_from` and `clone_axes` copy settings in one step |
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

| Index role | Holds | Key fields |
|---|---|---|
| `images` | One document per ingested source image | `image_id`, `image_path`, `imohash`, `phash`, `embedding` (512-d), `pe_embedding` (1024-d) |
| `items` | One document per crop (item) | see below |
| `labels_confirmed` | Mapping kept, nothing writes it in this release | `label_id`, `crop_id`, `class_id`, `class_name`, `confirmed_at` |
| `classes` | The class registry mirror | `class_id`, `class_name`, `group`, `deprecated`, `sample_count`, `validated_count` |
| `umap_state` | Fitted reducer cache for clustering | `state_id`, `reducer_b64` |
| `configs` | Config store plus folded settings and projection state | `doc_type`, `kind`, `name`, `revision`, `body` |

**Items** carry the class (`class_id`, `class_name`, `class_source`,
`class_validated`), the crop `bbox_norm` in the source frame, cluster placement
(`cluster_id`, `cluster_subid`, distances), quality and score fields, VLM
provenance (`vlm_endpoint`, `vlm_model`, `vlm_prompt_pack`), import and combine
provenance, and the region fields described next. The exact wire keys are in
[`contracts/json/item_wire.json`](../contracts/json/item_wire.json).

**Class identity is the name.** The class registry
(`class_registry.json` per project) assigns indexes, but every boundary maps by
name: dataset import and export, combine, training, promotion and sharing a
trained model with another project. An index is only a position inside one
registry at one moment.

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
stored per project in the `configs` index (VLM endpoints are deployment-wide in
`op_global_configs`).

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
job (`POST /curation/projects/{project}/datasets/imports`). Formats: YOLO, COCO
and the OpenProcessor export (which round-trips labels, splits and the frozen
test set). The mapping is a list of `{dataset_class, action}` with `action` one
of `map`, `create`, `skip`, `region`; matching is by name, and every class that
has boxes needs a decision. Options include `label_trust` (`validated` or
`suggestion`), `processing` (`none` or `propose`), `missing_label` and
`freeze_test_split`. An import can be cancelled, resumed and undone; undo keeps
anything a human has since edited. The import report counts images, items,
labels, regions, negatives and the labels a human lock refused.

**Reprocess** (`POST /curation/projects/{project}/reprocess`) re-runs pipeline
scopes over chosen targets (`image_ids`, `crop_ids` or a `filter`). Scopes:
`detect`, `region`, `vlm`, `embed`; region mode is `redetect` or `reverify`.
`dry_run` defaults to true. Locked items and boxes are never written and are
reported as `locked_skipped`. Large detect or embed runs become a job
(`GET /curation/projects/{project}/reprocess/jobs/{job_id}`); only one runs at a
time. Single-target forms:
`POST /curation/projects/{project}/crops/{crop_id}/reprocess` and
`POST /curation/projects/{project}/images/{image_id}/reprocess`.

---

## Workers, jobs and events

- **File protocols.** The API and the workers share the state volumes. Training
  and auto-label jobs are JSON files the API writes and the long-lived worker
  claims; status flows back as files. A heartbeat marks a live claim; a job
  whose heartbeat is stale is reported `interrupted` and can be resumed.
- **Detection worker** (`scripts/curation/region_worker_main.py`, package
  `scripts/curation/worker/`): fetches pending items, runs the cascade, writes
  boxes in an OCC merge, and skips any document a human changed meanwhile.
- **Pause and resume.** `POST /curation/projects/{project}/pause` and
  `.../resume` write and remove a flag file; workers skip a paused project and
  keep serving the others.
- **Events.** `GET /curation/events` (global) and
  `GET /curation/projects/{project}/events` are Server-Sent Events. Producers
  append to a shared JSONL log in the state dir and every uvicorn worker tails
  it, so a client on one worker sees events published by another
  (`OP_EVENT_BUS=process` restores in-process fan-out for single-worker
  setups). Events are advisory, not a durable feed. Examples: project lifecycle,
  `combine.progress`, `config.changed`, `vlm.changed`.
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

Cluster writes (partition, refine, false-positive sub-typing) merge into the
live box list and retry a few times on conflict; a model fit that finished
after a human moved a box cannot overwrite it.

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

## Scaling notes

- One GPU runs Triton, the segmenter and a local VLM only when VRAM allows;
  `.env` GPU keys place each service. Instance counts per Triton model are in
  the `config.pbtxt` files; raise them for hot paths when VRAM is free.
- The API scales by uvicorn workers (`--workers`, default 32) within one host.
  All workers share the Triton pool settings per process and the file-backed
  event bus.
- OpenSearch heap is sized from RAM by the installer (RAM/8, 1-8 GB). Each
  project adds six indexes, so watch the shard budget
  ([INSTALLATION.md](../INSTALLATION.md#opensearch-heap-sizing)); creating a
  project past the capacity limit answers 409 with the capacity figures.
- Scaling past one host is not a supported topology today: the workers and the
  API share local volumes and the file-backed job protocol.

---

## Further reading

- [FastAPI concurrency](https://fastapi.tiangolo.com/async/)
- [Ultralytics thread-safe inference](https://docs.ultralytics.com/guides/yolo-thread-safe-inference/)
- [NVIDIA Triton optimization guide](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/optimization.html)
