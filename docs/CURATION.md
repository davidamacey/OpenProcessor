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
  --project <slug> --folder /data/source/<dir>
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

Bring images in from an S3 or S3-compatible bucket, and keep a project fed by a poll
watcher, a cron schedule or an S3 event webhook. Covers the allow-list
(`OP_S3_ALLOWED`), credentials, the source routes under
`/curation/projects/{project}/ingest/sources`, resume and troubleshooting:
[Ingest from S3 and watched sources](../docs-site/docs/guides/ingest-sources.mdx).

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
