# Curation and Active Learning

This page is a map. The user guides live in the documentation site
(`docs-site/docs/`). Each section below says what the topic covers and links to the
page that explains it. Headings that code, compose files and scripts cite are
kept, so those references still resolve.

> **Status: experimental for v0.5.0.** The subsystem is working and tested, but its
> API can still change between releases. It ships opt-in, behind Docker Compose
> profiles.

## What this is

The curation subsystem is a domain-agnostic backend for building an image-labeling
dataset with active learning. You ingest or import images, detect items, optionally
find regions inside each item, cluster and browse, label by hand or with an
OpenAI-compatible vision-language model, review, export a dataset and train a
detector. What an "item" or a "region" is comes from data you own: a class registry,
a region profile, a prompt pack and a VLM endpoint, all stored per project.

Start with [the quick start](../docs-site/docs/getting-started/quick-start.mdx), or
[the path for your own domain](../docs-site/docs/guides/use-your-own-domain.mdx).

Companion documents:

- [`design/curation_api_contract.md`](design/curation_api_contract.md): the
  route-by-route wire contract.
- [`design/curation_design_rationale.md`](design/curation_design_rationale.md): why it
  is built this way.
- [`ARCHITECTURE.md`](ARCHITECTURE.md): services, data model, the config store and
  the lock rule in implementation terms.
- [`../contracts/openapi/curation.json`](../contracts/openapi/curation.json): the
  generated OpenAPI document. It is the source of truth for routes and schemas.

All curation routes sit under one prefix (`OP_API_PREFIX`, default `/curation`) and
have no `/v1` twin. Every route that touches data is scoped to a project:
`/curation/projects/{project}/...`.

## The workflow

1. Create a project and define its classes.
2. Activate a region profile and a prompt pack (optional), and pick a VLM endpoint
   (optional).
3. Ingest images, or import an already-labeled dataset.
4. Let the workers run: item detection, region detection, VLM labeling, embeddings,
   clustering.
5. Review and label in Cropwright or through the API.
6. Freeze a test holdout, export a YOLO dataset, run preflight, train.
7. Promote the model, compare it in the bake-off, and feed disagreements back into
   review.

Operational view of the same loop:
[curation workflow](../docs-site/docs/operations/curation-workflow.mdx).

## Use your own domain

The whole path for a new domain: project, classes, region profile, prompt pack, VLM
endpoint, ingest, review, export. See
[Use your own domain](../docs-site/docs/guides/use-your-own-domain.mdx) and the worked
example, [wheels on cars](../docs-site/docs/guides/wheel-example.mdx). The example
files are in [`examples/`](../examples/). The cost checks that keep the region stage
cheap (parent classes, the visibility check, the hit-rate gate) are in
[Region profiles](../docs-site/docs/guides/region-profiles.mdx).

## Projects

A project owns its OpenSearch indexes, files, uploads, class registry, settings and
config. Create, rename, archive, delete, clone settings, pause workers:
[Projects](../docs-site/docs/guides/projects.mdx). In Cropwright:
[Projects](../docs-site/docs/cropwright/user-guide/projects.md).

## Settings and the config store

Per-project settings are one shared document. Prompt packs, region profiles and VLM
endpoints are immutable revisions, and activation is a separate step.
[Settings and config](../docs-site/docs/guides/settings-and-config.mdx).

## Class registry and class identity

Class identity is the name, never the index. Anything that crosses a boundary (import,
combine, export, train, promote) pairs classes by name. Manage classes in
[Projects](../docs-site/docs/guides/projects.mdx) and the
[Cropwright classes page](../docs-site/docs/cropwright/user-guide/classes.md). The
rule is in [`ARCHITECTURE.md`](ARCHITECTURE.md) and
[`VISION_AND_GOALS.md`](VISION_AND_GOALS.md).

## The lock rule

Automated writers never overwrite a class or box that a person or a validated import
decided, or an item in the frozen test holdout. Skipped items are reported as
`locked_skipped`. See "Concurrency and the lock rule" in
[`ARCHITECTURE.md`](ARCHITECTURE.md).

## Region profiles

How regions are found inside an item. Data, stored per project, no built-in default.
[Region profiles](../docs-site/docs/guides/region-profiles.mdx).

## Multi-box regions

Every item holds its regions as one list, `region_boxes`. Edit routes, states, and
box-level browse and clustering:
[Multi-box regions](../docs-site/docs/guides/multi-box-regions.mdx).

## Prompt packs

The VLM prompts and vocabulary the labeler uses.
[Prompt packs](../docs-site/docs/guides/prompt-packs.mdx).

## Test on a crop

Try a draft pack or profile on stored crops before activating it. Read-only.
[Test on a crop](../docs-site/docs/guides/test-on-crop.mdx).

## Choosing a VLM

Endpoints are registered once per deployment and activated per project. The page
covers the registry workflow, the URL policy, local models, `./openprocessor vlm`, the
per-project policy and its scopes.
[VLM selection](../docs-site/docs/guides/vlm-selection.mdx). In Cropwright:
[VLM models](../docs-site/docs/cropwright/user-guide/vlm-models.md).

## Keymaps

Per-project keyboard bindings for the labeling frontend.
[Keymaps](../docs-site/docs/guides/keymaps.mdx).

## Ingest

Ingest does duplicate detection, a quality gate, crop-cache population, PE embedding
and bulk indexing. Routes for one image, a batch, an upload, and a folder or manifest
run are in the [API reference](../docs-site/docs/api-reference/curation.mdx).
Cropwright steps: [Ingest](../docs-site/docs/cropwright/user-guide/ingest.md).

### Bulk ingest

A folder or manifest of thousands of images runs as a background job that goes through
`POST /curation/projects/{project}/ingest/batch`. Use `scripts/curation/bulk_ingest.py`
inside the `api` container, which listens on port 8000 there:

```bash
docker compose -p <project> exec api python scripts/curation/bulk_ingest.py \
  --project <slug> --folder /data/source/<dir> --api-base http://localhost:8000/curation
```

Defaults are batch size 16 and concurrency 16
(`GET /curation/projects/{project}/ingest/bulk/config` shows the bounds). The measured
rate is in [PERFORMANCE.md](PERFORMANCE.md#bulk-ingest-runner). The full procedure for a
large share (mount, trial run, resume, time and disk estimate) is
[Ingest a large folder](../docs-site/docs/guides/ingest-large-folder.mdx).

### Ingest policy

Each project has a policy with two independent parts: a `detect` filter on the
detector output, and an `embedding.mode` (`all`, `selected` or `lazy`). It affects
future ingests only. [Ingest policy](../docs-site/docs/guides/ingest-policy.mdx).

### Items without an embedding

An item stored without a vector has `embedding_state` `not_selected`, `deferred` or
`failed`. `not_selected` comes from the `selected` mode. `deferred` comes from a
combine that dropped a vector, and from `lazy` ingest. Clustering, kNN search and the
VLM stage skip these items. Embed them with
`POST /curation/projects/{project}/reprocess`, scope `embed`, `only_missing: true`.
[Embedding state in Cropwright](../docs-site/docs/cropwright/user-guide/embedding-state.md).

### Embedding use cases

The ways an item gets a vector after ingest (moved box, failed embed, policy skip,
import, new embedding model) are covered by the same reprocess scope:
[Reprocess](../docs-site/docs/guides/reprocess.mdx).

### Ingest sources and S3

Every way of getting images in ends in the same pipeline. A request to `/ingest`,
`/ingest/batch` or `/ingest/upload`, a bulk run over a folder or manifest, and an S3
bucket all reach `ingest_batch`, so decode routing, the coalesced writer, the lock
rule and project isolation are the same for each.

**S3 and S3-compatible stores (MinIO and others).** Set `OP_S3_ALLOWED` to the
`bucket` or `bucket/prefix` entries the deployment may read; with it empty S3 is off
and every S3 route answers 503 `s3_not_configured`. A source or request outside the
list is refused 422 `s3_not_allowed`, at creation and again at use. `OP_S3_ENDPOINT_URL`,
`OP_S3_REGION` and `OP_S3_PATH_STYLE` point at a non-AWS store. Credentials are never
part of a request or a stored source: boto3 finds them in the `api` container's
environment (`AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`, or
`AWS_SHARED_CREDENTIALS_FILE` naming a mounted secret file) or from an instance or task
role. Give them read-only access to the listed prefixes.

| Purpose | Route |
|---|---|
| List sources, with trigger and queue status | `GET /curation/projects/{project}/ingest/sources` |
| Register a source (a bucket/prefix or a folder, options, triggers) | `POST /curation/projects/{project}/ingest/sources` |
| Read, change or remove one | `GET`, `PATCH`, `DELETE /curation/projects/{project}/ingest/sources/{source_id}` |
| Scan the source and ingest what is new, now | `POST /curation/projects/{project}/ingest/sources/{source_id}/run` |
| Put objects that gave up back in the queue | `POST /curation/projects/{project}/ingest/sources/{source_id}/retry-dead` |
| Receive S3 event notifications (webhook) | `POST /curation/projects/{project}/ingest/sources/{source_id}/events` |
| Fetch and ingest named objects (what the runner calls for S3) | `POST /curation/projects/{project}/ingest/objects` |
| A time-limited download URL of an original | `POST /curation/projects/{project}/ingest/objects/presign` |

A one-shot bucket ingest is the same mechanism: `POST .../ingest/bulk` with
`{"s3": {"bucket": "...", "prefix": "..."}}` (or `scripts/curation/bulk_ingest.py
--s3 s3://bucket/prefix`) finds or creates a trigger-less source for that bucket and
prefix and runs it, so starting it again resumes or ingests only what is new.

How it behaves:

- **One queue per source.** Whatever notices a new object (a scan, a webhook event,
  a manual run) inserts `(key, etag)` rows into the source's durable queue (SQLite on
  the shared jobs volume, under the project's bulk-ingest directory); a bulk run of
  kind `queue` drains the pending rows. A replayed event or a repeated scan queues
  nothing twice.
- **Staged, with the original referenced.** The API fetches an object into memory,
  writes it once, content-addressed, under the project's upload root (the same as
  `/ingest/upload`, so thumbnails, crops and the region worker keep working) and
  stores the reference of the original as the image's `source_identifier`:
  `s3://bucket/key#etag=...&version=...`. An object whose exact reference is already
  stored is answered `duplicate` without a download; the same bytes under another key
  are caught by the content hash as before. `.../objects/presign` turns a stored
  reference into a download URL of the original (inside the allow-list only).
- **Bounded.** Encoded bytes held by fetches are bounded per API process
  (`OP_S3_INFLIGHT_MB`, with `OP_S3_MAX_OBJECT_MB` as the per-object cap and
  `OP_S3_FETCH_WORKERS` threads), reserved per batch before any download so batches
  cannot deadlock each other; decoded pixels stay under the existing ingest gate.
- **Listing at scale.** A scan lists ObjectsV2 pages (1000 keys), one worker per
  top-level sub-prefix (`OP_S3_LIST_WORKERS`), and writes its position after every page,
  so a restart resumes instead of relisting. A finished pass moves a watermark (the
  pass start minus two minutes); the next pass queues objects modified since. A very
  large flat bucket is better served by the webhook than by relisting.
- **Webhook.** A source created with `triggers.webhook_enabled` accepts the S3 event
  notification JSON (`Records[].s3.bucket.name` and `.object.key`, `.eTag`, `.versionId`;
  S3 and MinIO send it) at its `.../events` route, which its status gives as
  `webhook.path`. Point the store's notification target at it (MinIO: `mc event add`
  with a webhook target). The route only writes queue rows and answers 202 with
  `{received, queued, duplicates, ignored}`; it validates the shape (422
  `invalid_s3_event`, naming the first bad field), deduplicates by bucket, key and etag
  so a redelivery queues nothing, and ignores records for other buckets, prefixes or file
  types and every event that is not a creation, so one bucket-wide notification can feed
  several sources. Keep the endpoint on loopback or a trusted network: like the rest of
  the API it has no authentication.
- **Watcher and schedule.** A source can poll (`triggers.watch`, `interval_s` at least 10)
  and run on a cron schedule (`triggers.schedule.cron`, five fields in UTC). A poll scans
  the bucket or folder into the queue on its own and starts a run only if something is
  pending, so a quiet source makes no runs; a schedule fire always starts a run (scan, then
  drain). A fire missed while the API was down is not replayed: the schedule resumes at its
  next time. The local-folder watcher is the same code (a modification-time watermark;
  a file changed in the last five seconds waits for the next pass so a copy in progress is
  not read half written). `GET .../ingest/sources/{source_id}` reports `watch`
  (`last_poll_at`, `next_poll_at`, `error`), `schedule` (`last_fire_at`, `next_fire_at`),
  the scan's progress and the last run.
- **The dispatcher.** One task in each API worker (`OP_INGEST_DISPATCH_S`, default 5
  seconds; 0 turns every trigger off) acts per project under a file lock, so one worker
  acts per tick. It also drains rows a webhook queued and resumes a run that a restart
  interrupted; a cancelled run is left cancelled, a failed one waits five minutes. A scan
  that fails (the store is unreachable, credentials are missing, the allow-list changed)
  records its error under `watch.error` and backs off, doubling to ten minutes, while the
  queue keeps what it has. Runs of the dispatcher send their batches to the API's own
  address (`OP_API_LOOPBACK_URL`, default `http://127.0.0.1:8000`, the listen address inside
  the container).
- **Failures are per object.** A missing object (`gone`) or one over the size cap
  (`too_large`), or one that is not an image, is `dead` at once; a permission or
  network error (`forbidden`, `fetch_error`) is retried by later runs up to three
  attempts. Each failure appears in the run's failure list with its reason and in the
  source's `errors`; `retry-dead` queues them again.

## Dataset import

Bring a labeled YOLO or COCO dataset, or an OpenProcessor export, into a project.
Preview, decide each class by name (`map`, `create`, `skip`, `region`), import, undo.
[Dataset import](../docs-site/docs/guides/dataset-import.mdx).

### Importing other formats

The importer reads YOLO, COCO and OpenProcessor exports. Pascal VOC, CVAT, LabelMe,
Label Studio and segmentation, OBB or keypoint-specific formats are not read directly.
Convert them to YOLO or COCO first, then check the result with
`POST /curation/projects/{project}/datasets/preview`.

- Pascal VOC, CVAT, LabelMe, Label Studio: convert with a general converter such as
  `fiftyone` (export as `fo.types.YOLOv5Dataset`) or `supervision`
  (`sv.DetectionDataset.from_pascal_voc(...)` then `.as_yolo(...)`). Keep one class
  name per class: the import decides by name.
- Polygon, OBB or keypoint labels: YOLO rows with more than five columns are imported
  as their bounding box. The extra geometry is not kept.
- A `format_undetected` error means no YOLO, COCO or export layout was found at the
  path.

## Reprocess

Re-run `detect`, `open_vocab`, `region`, `vlm` or `embed` over images, items or a
filter. Dry run by default, and it reports what the lock rule skips.
[Reprocess](../docs-site/docs/guides/reprocess.mdx).

## Open-vocabulary detection

Text prompts run through SAM 3 over the whole image. Needs the segmenter.
[Open-vocabulary detection](../docs-site/docs/guides/open-vocabulary.mdx).

## Combine projects

Build a new project from 1 to 8 existing ones, with class mapping by name.
[Combine projects](../docs-site/docs/guides/combine-projects.mdx).

## Review, search and clustering

Review queues, semantic search, item clustering, the auto-label job, the shared item
filter and bulk actions on a selection.
[Cropwright review](../docs-site/docs/cropwright/user-guide/review.md),
[clusters](../docs-site/docs/cropwright/user-guide/clusters.md),
[item filter](../docs-site/docs/cropwright/user-guide/item-filter.md). Which worker
schedules the cluster retrain, and when:
[Workers](../docs-site/docs/operations/workers.mdx).

## Export and training

Freeze a holdout, export, run preflight, train, promote, compare in the bake-off.
[Exports and retention](../docs-site/docs/operations/exports-and-retention.mdx),
[Training and promote](../docs-site/docs/operations/training-and-promote.mdx).

## Models you must supply

Nothing here ships a pretrained region detector, VLM or trainer weights.

- **PE-Core encoders.** The image tower `pe_image_encoder` is required; the text
  tower `pe_text_encoder` is needed for semantic search. Build both with
  `make export-pe`. If it fails with `ModuleNotFoundError: No module named 'core'`,
  the `api` image predates the source tree: rebuild it first.
- **An item detector for ingest**, named by `OP_INGEST_PRIMARY_DETECTOR_MODEL`.
  Ingest returns 503 until one is configured and loaded.
- **A segmenter and a region profile**, if you want regions (`OP_SEGMENTER_URL`).
- **A dual-head detector**, only if you want `backbone_embedding`.
- **An OCR model**, if your region has readable text.
- **A VLM**: see [Choosing a VLM](#choosing-a-vlm).
- **A dataset and base weights** for training.

Steps: [Model export](../docs-site/docs/getting-started/model-export.mdx) and
[`../export/README.md`](../export/README.md#pe-core-encoders-curation-embeddings).

## Workers and compose profiles

The HTTP API works with no workers. The asynchronous half is opt-in:

```bash
docker compose -p <project> --profile curation up -d
```

| Service | What it does |
|---|---|
| `curation-detection-worker` | Runs the region cascade and verifies boxes. Needs `triton-server`. |
| `curation-vlm-worker` | Labels classes through each project's active VLM endpoint. |
| `curation-auto-label-worker` | Drives the auto-label protocol. Clusters unless a caller passes `run_vlm=true`. |
| `curation-cluster-refresh` | Queues the full cluster retrain once 200 new items exist and ingest is quiet. |
| `curation-evaluator` | Bake-off runner. Run it on demand; it reserves a GPU. |

Details, GPU needs and the segmenter, trainer and GPU arbiter profiles:
[Workers](../docs-site/docs/operations/workers.mdx) and
[Compose profiles](../docs-site/docs/getting-started/compose-profiles.mdx).

## Mounting your image source

Ingest resolves paths under `OP_SOURCE_ROOT` (container default `/data/source`). Three
services mount the same host directory read-only at that path: `api`,
`curation-detection-worker` and `curation-auto-label-worker`.

```yaml
volumes:
  - ${OP_SOURCE_ROOT_HOST:-./data/source}:/data/source:ro
```

Set `OP_SOURCE_ROOT_HOST` in `.env`, then recreate all three services
(`./openprocessor start`, or `docker compose -p <project> up -d`). If a worker lacks
the mount, ingest succeeds but later worker reads fail with
`reason=image_unavailable`. `GET /curation/projects/{project}/ingest/config` lists the
accepted source roots. Because the mount is read-only, run dataset fetch scripts on the
host. More: [Basic configuration](../docs-site/docs/configuration/basic.mdx).

## Wiring up Cropwright

Cropwright, or any frontend that consumes `/curation`, needs `API_UPSTREAM`
(`http://op-api:8000`), `PUBLIC_API_PREFIX` (equal to `OP_API_PREFIX`) and the
`<COMPOSE_PROJECT_NAME>_triton_net` Docker network. See
[Cropwright deployment](../docs-site/docs/cropwright/operations/deployment.md) and
[Compose profiles](../docs-site/docs/getting-started/compose-profiles.mdx).

## Quick start for a fresh install

[Quick start](../docs-site/docs/getting-started/quick-start.mdx).

## Environment variables

All curation `OP_*` variables are optional. The authoritative list with defaults is
[`../env.template`](../env.template). Reference tables:
[Basic](../docs-site/docs/configuration/basic.mdx) and
[Advanced](../docs-site/docs/configuration/advanced.mdx) configuration. Index names
come from the project, not from the environment. Profiles, packs, VLM endpoints and
settings live in the config store and change at runtime.

## Wire naming

OpenSearch fields default to `region_*`, `vlm_*` and `classifier_*` names, and the wire
uses one generic vocabulary whatever the storage names are
([`design/curation_api_contract.md`](design/curation_api_contract.md)). Retired
environment-variable prefixes are rejected at startup by
`src/config/retired_env.py`.

## Known limits

- No authentication on the API. Do not expose it; see [`../SECURITY.md`](../SECURITY.md).
- You supply the encoders, item detector, segmenter, VLM and base weights.
- A region profile is per project, but one detection worker serves every project.
- Text reading and the OCR text hint are not previewed by the profile test route.
- The trainer needs internet access on first use of a base checkpoint unless you point
  `hyperparameters.model` at a local file.

## Resource links

`GET /curation/projects/{project}/settings` includes `resource_links`. See
[Monitoring](../docs-site/docs/operations/monitoring.mdx).
