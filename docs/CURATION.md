# Curation & Active Learning

> **Status: EXPERIMENTAL for v0.3.0.** This is a real, working, tested
> subsystem — not a stub — but it is new, still evolving, and not yet a
> stable API. Backward compatibility across releases is not guaranteed
> until it graduates out of experimental status. It ships opt-in, behind
> a Docker Compose profile, and is disabled by default.

## What this is

The curation subsystem is a generic, domain-agnostic backend for
building and maintaining an active-learning image-labeling dataset on
top of OpenProcessor: ingest images, detect and crop regions of
interest, cluster and browse the crops, label them (by hand or via an
OpenAI-compatible vision-language model), track class registries and
review queues, export labeled datasets, and drive a training loop.

It was ported and genericized from a private, domain-specific
(vehicle / license-plate) curation product. That history shows up in a
few frozen wire-level names described below (§ Naming you'll notice),
but the subsystem itself makes no assumption about what a "region of
interest" is — a license plate, a barcode, a defect on a manufactured
part, a tag on livestock — you configure your own domain via the
dataclasses in the next section. See
[`docs/design/curation_design_rationale.md`](design/curation_design_rationale.md)
for the deeper "why" behind the design (frozen wire contract, the
`RegionFields` indirection, the pre-commit ratchet exemptions), and
[`docs/design/curation_api_contract.md`](design/curation_api_contract.md)
for the full route-by-route wire contract.

All curation routes are mounted under a single configurable prefix
(`CurationConfig.api_prefix`, default `/curation`, override via
`OP_API_PREFIX`) with no `/v1` twin — see the API-versioning note in
[`CLAUDE.md`](../CLAUDE.md).

## The four configuration dataclasses

A new deployment configures the subsystem for its own domain through
four dataclasses instead of forking code. All four support
`from_env()` so most of a deployment can be configured purely through
environment variables (see the env var table below), including the
region `DetectionProfile` (`OP_REGION_PROFILE` / `OP_REGION_DETECTION_*`).
The `DetectionProfile` dataclass field defaults still describe the
reference license-plate domain's OCR/segmenter wiring, so review them for
a genuinely new region type.

| Dataclass | File | What it configures |
|---|---|---|
| `CurationConfig` | `src/config/curation.py` | OpenSearch index names (via `IndexRole` + `index_name()`), filesystem roots (class registry, exported datasets, crop cache, state dir), the API mount prefix, embedding-dimension/HNSW tuning. |
| `RegionFields` | `src/config/region_fields.py` | Per-attribute OpenSearch field-name overrides for the region-of-interest sub-annotation (e.g. rename `region_status` to `plate_status` if your existing data already uses that name) — lets storage field names diverge from the frozen HTTP wire-contract field names with zero reindex. |
| `DetectionProfile` | `src/config/detection_profile.py` | One detectable region-of-interest type as data: aspect-ratio/area heuristics, text-hint pattern and length range, which Triton models back detection/segmentation/OCR for it, their input sizes and confidence floors. One region profile is active per process (`OP_REGION_PROFILE` / `OP_REGION_DETECTION_*`; none by default). |
| `RegionStatus` | `src/config/region_state.py` | The canonical region-status state-machine enum (`pending_detection` → `detected`/`verify_rejected`/`no_region_box`; `pending_verification` → `detected`/`no_region_visible`; any path → `detection_failed`; plus a human-settable `false_positive` that preserves the box for hard-negative training). |

## Known gaps (read this before you rely on it)

Stated up front, honestly, rather than discovered in production:

- **Thinner ingest than a bespoke pipeline.** `POST /curation/ingest/image`
  and `/ingest/batch` create items with duplicate detection, a quality
  gate, crop-cache population, and bulk indexing; `POST
  /curation/import_labels(/batch)` imports pre-existing YOLO-format
  labels. What is *not* included: any domain-specific detector-ensemble
  policy, class allowlist, or region-status assignment heuristic tuned
  to one domain — you supply that via `DetectionProfile` and your own
  detector model(s).
- **One active region `DetectionProfile` per process.** You cannot run
  the region cascade for two region-of-interest types from one worker
  today. Prompt packs are selectable (several can be configured via
  `OP_PROMPT_PACK_PATHS` and chosen per auto-label run or via the
  settings default).
- **No authentication of any kind on the API.** See
  [`SECURITY.md`](../SECURITY.md) — do not expose this service directly
  to the internet.
- **You must supply your own models.** This is BYO-model territory, not
  a batteries-included product — see "Models you must supply" below.
- **Both the trainer and the segmenter ship as reference containers**,
  not just protocols: `docker/trainer/` and `docker/segmenter/` each
  implement the wire/file protocol the API already speaks, behind their
  own opt-in compose profiles (`training`, `segmenter`). You still bring
  your own dataset and base weights — see "Workers and the curation
  compose profile" below.
- **Coverage is uneven across the ported surface** — some routers carry
  thorough test suites, others were ported with comparatively thin
  coverage because the original implementation had thin coverage there
  too.

## Models you must supply

Nothing in this subsystem ships a pretrained region-detector, VLM, or
trainer. A deployment supplies:

- **An image-embedding model — `pe_image_encoder` (required, not
  optional).** Unlike everything else in this list, the Triton model
  *name* here is hardcoded, not configurable: `src/clients/pe_encoder.py`
  calls `pe_image_encoder` with a single FP32 input `images`
  `[B, 3, 336, 336]` and reads a single FP32 output `image_embeddings`
  `[B, 1024]`. The result is stored as the `pe_embedding` field and is
  what semantic search (`GET /curation/search/text`), near-duplicate
  detection, residual clustering and the embedding visualization all run
  on — without it, ingest cannot write an embedding and those features
  have nothing to query. This repo **does** ship the export chain for it:
  `export/export_pe_image_encoder.py` (PE-Core-L14-336 vision tower →
  ONNX), then `export/build_pe_trt.sh` (→ TensorRT plan) or
  `export/build_pe_ort_fallback.sh` (serve the ONNX directly when the
  TensorRT build fails on PE's attention-pool ops). See
  [`export/README.md`](../export/README.md#pe-core-image-encoder-curation-embeddings).
  Swapping in a different embedding model means keeping that same Triton
  model name and tensor contract, and matching the preprocessing in
  `src/services/detection/pe_preprocess.py`.
- **An item detector for ingest** — an end2end Triton model set via
  `OP_INGEST_PRIMARY_DETECTOR_MODEL` (plus any other
  `OP_INGEST_PRIMARY_<FIELD>`, read by `_get_detection_profile()` in
  `routers/curation/ingest.py`). It proposes the item crops in each
  image; `OP_INGEST_PRIMARY_CLASS_IDS` narrows which of its classes
  become items (unset = all). By default the primary is treated as a
  generic proposer (`OP_INGEST_PRIMARY_ASSIGNS_CLASS=false`): its
  detections are unlabeled `<name>_proposal` items carrying the model's
  own label (`OP_INGEST_PRIMARY_LABELS_PATH`), never a registry class
  looked up by its id. Set it `true` only when the primary was trained on
  your class registry. The `class_source` values the worker and
  clustering code filter on are derived from these profile names
  (`src/services/curation/ingest_class_sources.py`). Ingest returns `503` until one is
  configured and loaded. An optional raw-output secondary detector
  (`OP_INGEST_SECONDARY_DETECTOR_MODEL` + `OP_INGEST_SECONDARY_<FIELD>`)
  overrides the primary's class on IoU-matched boxes. The retired
  `OP_DETECTION_*` prefix is rejected at startup with a rename message.
- **Optionally, a region-of-interest profile** — the sub-region the
  detection worker's cascade looks for *inside* each item crop.
  **Neutral by default:** with nothing configured no region profile is
  active, `GET /methods` advertises an empty `detection_profile` axis,
  and the worker idles instead of running the cascade. Select one with
  `OP_REGION_PROFILE=<name>` (a profile your startup code registered via
  `src.services.detection.profile_registry.register_profile()`, or a
  built-in reference profile — today `license_plate`, which reproduces
  the original reference deployment's constants and is an example, not a
  suggested starting point), and/or override individual fields with
  `OP_REGION_DETECTION_<FIELD>` (e.g. `OP_REGION_DETECTION_SAM_TEXT_PROMPT`,
  `OP_REGION_DETECTION_SECONDARY_SHAPE_GROUPS`). The resolved profile is
  registered automatically, so it is exactly what `GET /methods`
  advertises. An unknown `OP_REGION_PROFILE` name fails at startup.
  While a region profile is active, ingest (and label import, for
  labels the detector missed) seeds every **newly created** item with
  region status `pending_detection` — the only way an item enters the
  worker's queue. An existing region status is never overwritten on
  re-ingest. The per-image `n_plates` count in the ingest response is
  the number of items seeded this way (`0` with no region profile), and
  `GET {prefix}/ingest/sam_drain` reports them under
  `pending_detection`. **Enabling a region profile on a deployment that
  already has ingested items:** those items have no region status and
  the worker will never see them; backfill them once with
  `python3 scripts/curation/requeue_regions.py --missing-status`
  (dry run: counts only) then `... --missing-status --apply`.
- **A dual-head detector, if you want the backbone embedding**
  (`v6_embedding`). Residual clustering, the embedding visualization,
  item scores and the OCC conflict handler all read that field, and it
  is produced by RoI-pooling a detector's backbone feature map over each
  detection box (`src.services.detection.geometry.roi_pool_sppf`,
  pooled to `CurationConfig.backbone_embedding_dim`). A stock detector
  export emits only the detection tensor, so the detector must be
  re-exported with a second output — use
  [`export/export_detector_dual_head.py`](../export/export_detector_dual_head.py)
  (`output0` + `sppf_feat`; see [`export/README.md`](../export/README.md)).
  Optional: by default residual clustering reduces `pe_embedding`
  instead (`OP_RESIDUAL_EMBEDDING_FIELD`), so a deployment that never
  populates `v6_embedding` still clusters — it just has one fewer
  embedding space to compare against. Ingest fills the field from the
  **secondary** detector (the `secondary_profile` passed to
  `CurationIngestService`): when Triton's model metadata lists
  `DetectionProfile.feature_output` (default `sppf_feat`) it is
  requested alongside `output0` and pooled over every item's bbox. A
  secondary model without that output is called exactly as before and
  the field is simply not written. A feature map with fewer channels
  than `backbone_embedding_dim` is zero-padded (e.g. a 768-channel map
  into the 1024-d default); one with *more* channels is skipped with a
  logged error rather than truncated. Note `OP_BACKBONE_EMBEDDING_DIM`
  sets the mapping only when the items index is created — changing it
  later does not resize an existing index's field.
- **An OCR/recognition model, if your region type has readable text**
  (`DetectionProfile.ocr_rec_model`) — optional, only used by the
  text-hint heuristics.
- **A segmenter, if you want the cascade's segmenter leg** — any
  service reachable at `SAM3_URL` (the env var name is legacy but the
  wire protocol is a generic segment-request/response; see
  `scripts/curation/worker/client.py`). This leg is optional: with
  `SAM3_URL` empty the cascade runs without it. A reference
  implementation **does** ship — `docker/segmenter/` wraps SAM 3 behind
  that wire protocol — but it is opt-in (its own compose profile: it
  needs a GPU and a HuggingFace token) and it is BYO-weights like
  everything else here.
- **A VLM for labeling assist and region verification** — any
  OpenAI-compatible `/v1/chat/completions` endpoint, configured via
  `OPENWEBUI_BASE_URL` / `OPENWEBUI_MODEL` / `OPENWEBUI_API_KEY`.
  `src/services/labeling/vlm_client.py` is the only thing that talks to
  it; nothing hardcodes a specific vendor or model.
- **A dataset and base weights for training.** `/curation/train/*` is a
  control plane over a shared-volume file protocol
  (`src/services/training/jobs.py`): the API writes `<job_id>.job.json`
  into `/jobs/` to start a run and a `<job_id>.cancel` sentinel to
  cancel one; the trainer watching that directory writes
  `<job_id>.status.json` every ~5s, `<job_id>.run.log`, and
  `<job_id>.manifest.json` at the end. A trainer container implementing
  that half **does** ship — `docker/trainer/`, compose service
  `curation-trainer` under `--profile training`. What you supply is the
  frozen dataset export (produced by `/curation/export/*`) and the base
  weights the job trains from; the trainer downloads the family/size
  checkpoint named by the job spec unless
  `hyperparameters.model` points at a local file or architecture YAML.
  You can still swap in your own trainer: it only has to speak the file
  protocol above.

## Class-registry schema

The class registry is a single JSON file at `OP_REGISTRY_PATH` (default
`./data/class_registry.json`), read/written atomically through
`src.clients.curation_opensearch.ClassRegistry`. A worked, **non-vehicle**
example ships at
[`data/class_registry.example.json`](../data/class_registry.example.json)
— a small warehouse/retail inventory set (`cardboard_box`,
`wooden_pallet`, `forklift`, ...). Copy it to `OP_REGISTRY_PATH` and edit
`classes` for your own domain:

```json
{
  "version": 1,
  "updated_at": "2026-01-01T00:00:00+00:00",
  "classes": [
    {
      "id": 0,
      "name": "cardboard_box",
      "group": "packaging",
      "sample_count": 0,
      "validated_count": 0,
      "added_at": "2026-01-01T00:00:00+00:00",
      "deprecated": false,
      "notes": "Generic corrugated shipping box, any size.",
      "merged_into": null,
      "hotkey_letter": "b"
    }
  ]
}
```

Each entry's `id` is the dense class id used everywhere in the wire
contract (`class_id` on crops, export label files, etc.). `group` and
`hotkey_letter` are UI conveniences (grouping/keyboard shortcuts in a
labeling frontend); `sample_count`/`validated_count` are maintained by
the backend, not hand-edited. `GET /curation/classes` reflects whatever
this file currently contains — starting the API against the example
file and calling that endpoint is a quick way to confirm your registry
loaded correctly.

## Workers and the curation compose profile

The synchronous HTTP API (browse, label, cluster, export) works
standalone with no workers running. The asynchronous half — automatic
detection, VLM labeling, clustering refresh — is a separate opt-in
layer, started with:

```bash
docker compose --profile curation up -d
```

This starts, in addition to the base services:

| Service | What it does |
|---|---|
| `curation-detection-worker` | Runs the detection cascade continuously over `pending_detection` items. |
| `curation-vlm-worker` | Verifies/labels items via the configured VLM. |
| `curation-auto-label-worker` | Drives the `/curation/pipeline/auto_label` protocol as a long-lived process. |
| `curation-cluster-refresh` | Periodically retrains/refreshes the residual clustering. |
| `curation-evaluator` (run on demand, not long-lived) | `docker compose --profile curation run --rm curation-evaluator` — the bake-off evaluation harness. |

None of these workers requires Triton or a GPU to *start* — they will
sit idle or error per-call until you've configured a real detector/VLM
endpoint. See `docker-compose.yml`'s `curation-*` service definitions
and `env.template` for every tunable.

The segmenter is a **second, separate profile** because unlike the
workers above it does need a GPU of its own and a HuggingFace token:

```bash
SAM3_URL=http://segmenter:8000 \
  docker compose --profile curation --profile segmenter up -d
```

| Service | What it does |
|---|---|
| `segmenter` | Promptable segmentation (SAM 3) serving the cascade's segmenter leg. See [`docker/segmenter/README.md`](../docker/segmenter/README.md). |

Without `SAM3_URL` the detection worker constructs a disabled client and
the segmenter leg is skipped entirely — no HTTP call, no failure.

## Seed / bootstrap path for a fresh install

1. Start the API (`docker compose up -d` or your own compose target).
   OpenSearch indexes are created automatically on startup via
   `create_curation_indexes` — there is no separate schema-migration
   step to run by hand.
2. Copy `data/class_registry.example.json` to wherever
   `OP_REGISTRY_PATH` points (default `./data/class_registry.json`) and
   edit `classes` for your domain, or start from an empty
   `{"version": 1, "updated_at": "...", "classes": []}` and add classes
   via `POST /curation/classes`.
3. Build and load the `pe_image_encoder` Triton model — see "Models you
   must supply" above and
   [`export/README.md`](../export/README.md#pe-core-image-encoder-curation-embeddings).
   Ingest writes no `pe_embedding` without it, and semantic search /
   near-dup / clustering then have nothing to operate on.
4. Configure at least an ingest detector model
   (`OP_INGEST_PRIMARY_DETECTOR_MODEL`) — ingest 503s until one is set.
5. Ingest images: `POST /curation/ingest/image` for one image at a
   time, or `scripts/curation/ingest_walker.py` for a bulk directory
   walk with a resumable progress file. If the images are not on storage
   the API container can mount, use `scripts/curation/ingest_upload.py`
   instead — it reads the files locally and uploads the bytes to
   `POST /curation/ingest/upload` (resume = server-side content dedup).
   To bring in an **already-labeled** YOLO dataset, use
   `scripts/curation/import_labeled_dataset.py`: it ingests each image
   with its `.txt` in one call, checks the dataset's class names against
   the registry first, and writes a disagreement report (where the
   detector missed a label, fired on a background image, or chose a
   different class). If the dataset's labels are *region* ground truth
   rather than item classes (whole frames labeled with, e.g., a single
   `license_plate` class plus background frames), add `--images-only`
   so the labels never touch the item registry, let the region cascade
   run, then score it with `scripts/curation/eval_regions_vs_gt.py
   --dataset <data.yaml> --state-dir <same state dir> --wait-pending 1800`
   (recall/precision/F1/mean IoU, background false-positive gate, and a
   worst-first list of misses). Seed the registry from the detector itself with
   `scripts/curation/seed_class_registry.py --model <detector.onnx>` so
   class ids cannot drift from the model's class order.
6. Optionally bring up the async workers (`--profile curation`) so
   detection/labeling/clustering keep running without you driving each
   step by hand.
7. Browse and label via `GET /curation/crops`, `PUT
   /curation/crops/{crop_id}/label`, etc., or point a labeling frontend
   (Cropwright is the first such consumer) at the API — see
   [`docs/design/curation_api_contract.md`](design/curation_api_contract.md).
8. Export a dataset with `POST /curation/export/yolo` once you have
   labeled data — or `POST /curation/export/single_class` to build a
   narrowed dataset for one class (or a class subset), which adds
   background/hard-negative frames the narrowed detector needs and a
   stronger integrity envelope (`dataset_sha` over the written label
   content, `frozen_test_sha` over the test split's identity, an
   atomically-flipped per-profile `current` symlink).

## Environment variables

All `OP_*` curation vars are optional; unset vars fall back to the
defaults in `CurationConfig.from_env()` / `RegionFields.from_env()` /
`DetectionProfile.from_env()`. The authoritative, always-current list
lives in [`env.template`](../env.template) — this table summarizes it
by area; consult `env.template`'s inline comments for full detail and
defaults.

**Import-time only:** curation routers build their mount prefix and
index names at *module import time*. Any `OP_*` var here must be set in
the process environment **before** `src.main` is imported — it cannot
be changed at runtime once the app has started.

| Area | Vars |
|---|---|
| OpenSearch index names | `OP_IMAGES_INDEX`, `OP_ITEMS_INDEX`, `OP_LABELS_CONFIRMED_INDEX`, `OP_CLASSES_INDEX`, `OP_CLUSTERS_INDEX`, `OP_SETTINGS_INDEX`, `OP_UMAP_STATE_INDEX`, `OP_UMAP_VIZ_STATE_INDEX` |
| Filesystem roots | `OP_REGISTRY_PATH`, `OP_SOURCE_ROOT`, `OP_SOURCE_PATH_ALIASES` (JSON object or `alias=path,...`), `OP_EXPORT_ROOT`, `OP_STATE_DIR`, `OP_CROP_CACHE_DIR` |
| VLM prompt pack | `OP_PROMPT_PACK_PATH` (default pack), `OP_PROMPT_PACK_PATHS` (extra selectable packs, comma-separated) |
| API surface | `OP_API_PREFIX`, `OP_API_TAG` |
| Embedding / HNSW tuning | `OP_EMBEDDING_DIM`, `OP_ENCODER_EMBEDDING_DIM`, `OP_BACKBONE_EMBEDDING_DIM`, `OP_HNSW_EF_CONSTRUCTION`, `OP_HNSW_M` |
| Region field-name overrides | `OP_REGION_FIELD_<ATTR>` (e.g. `OP_REGION_FIELD_STATUS`, `OP_REGION_FIELD_BBOX_NORM`) — see `RegionFields` for the full attribute list |
| Ingest item detectors | `OP_INGEST_PRIMARY_<FIELD>` (e.g. `OP_INGEST_PRIMARY_DETECTOR_MODEL`, `OP_INGEST_PRIMARY_INPUT_SIZE`, `OP_INGEST_PRIMARY_CLASS_IDS`), optional secondary `OP_INGEST_SECONDARY_<FIELD>` (e.g. `OP_INGEST_SECONDARY_DETECTOR_MODEL`, `OP_INGEST_SECONDARY_NAME`) — tuple/frozenset fields take a comma-separated value. Replaces the retired `OP_DETECTION_*` |
| Region detection profile (off by default) | `OP_REGION_PROFILE` (select by name, e.g. `license_plate`), `OP_REGION_DETECTION_<FIELD>` (per-field overrides, e.g. `OP_REGION_DETECTION_SAM_TEXT_PROMPT`, `OP_REGION_DETECTION_SECONDARY_SHAPE_GROUPS`) |
| Ingest | `OP_MAX_INGEST_CONCURRENCY` |
| Feature flags (off by default) | `OP_SEMANTIC_SEARCH_ENABLED`, `OP_VIZ_PROJECTION_ENABLED`, `OP_SELECT_DIVERSE_ENABLED`, `OP_SCORES_ENABLED`, `OP_SCORES_SHADOW` |
| Item-scores tuning | `OP_SCORES_KNN_K`, `OP_SCORES_NPROBE`, `OP_SCORES_STATE_DIR`, `OP_CROP_DUP_THRESHOLD`, `OP_FIELD_COVERAGE_TTL_S` |
| Diverse-selection tuning | `OP_SELECT_JOBS_DIR`, `OP_SELECT_JOB_MAX_N`, `OP_SELECT_MAX_N`, `OP_SELECT_SYNC_MAX_OPS`, `OP_SELECT_CACHE_TTL_S` |
| Clustering / IVF tuning | `OP_IVF_RETRAIN_CHECK_S`, `OP_IVF_RETRAIN_GROWTH`, `OP_IVF_RETRAIN_MIN_INTERVAL_S`, `OP_MAX_REFINE_MEMBERS`, `OP_OUTLIER_CACHE_TTL_S`, `OP_OUTLIER_MAX_MEMBERS`, `OP_RESIDUAL_EMBEDDING_FIELD`, `OP_REGION_CLUSTER_JOB_FILE`, `OP_REGION_FP_JOB_FILE`, `OP_REGION_PARTITION_MARKER`, `OP_REGION_REFINE_MARKER` |
| Training pipeline | `OP_TRAIN_JOBS_DIR`, `OP_TRAIN_RUNS_ROOT`, `OP_TRAIN_STAGING`, `OP_PREFLIGHT_SCAN_CAP` |
| GPU arbiter (`GpuArbiterConfig.from_env()`) | `OP_GPU_ALLOWED_IDS` (comma list; empty = unrestricted), `OP_GPU_ARBITER_CONTAINERS`, `OP_GPU_ARBITER_TRAINER_CONTAINER`, plus `OP_BAKEOFF_JOBS_DIR` |
| Export | `OP_BUILD_SHA` |
| Bake-off harness | `OP_BAKEOFF_JOBS_DIR`, `OP_BAKEOFF_OUT_DIR`, `OP_BAKEOFF_EVAL_ROOT`, `OP_BAKEOFF_CONCURRENCY`, `OP_BAKEOFF_GPUS`, `OP_BAKEOFF_BASELINES_PATH`, `OP_BAKEOFF_PROFILE`, `OP_BAKEOFF_PROFILE_<FIELD>` |
| Worker / pipeline flags | `OP_API`, `OP_AUTO_LABEL_STATE_DIR`, `OP_EVENT_API_URL`, `OP_ITEMS_INDEX_OVERRIDE`, `OP_PAUSE_SENTINEL`, `OP_WORKER_PAUSE_SENTINEL`, `OP_VIZ_JOBS_DIR`, `OP_VIZ_MAX_N` |
| VLM connection | `OPENWEBUI_BASE_URL`, `OPENWEBUI_MODEL`, `OPENWEBUI_API_KEY`, `OP_VLM_MAX_IMAGES_PER_CALL` (per-request image cap, default 8 — keep <= the engine's per-prompt image limit), `GEMMA_IMAGES_PER_CALL` (open-vocab chunk only, default 3), `GEMMA_HTTPX_MAX_CONNECTIONS`, `GEMMA_HTTPX_KEEPALIVE` |
| Segmenter connection | `SAM3_URL`, `SAM3_URLS`, `SAM3_HTTPX_MAX_CONNECTIONS`, `SAM3_HTTPX_KEEPALIVE` |

## Naming you'll notice

A handful of wire-level field and env-var names predate this
subsystem's generalization and are frozen (renaming them would be a
breaking wire-format change for zero functional benefit — see
`docs/design/curation_api_contract.md`'s "key invariant" section):

- The VLM connection vars (`OPENWEBUI_*`, `GEMMA_*`) and the
  `HealthResponse.gemma` field name predate the vendor-neutral `vlm_*`
  abstraction; they report whichever OpenAI-compatible backend you've
  actually configured, not literally Google's Gemma.
- The segmenter env vars (`SAM3_URL`, `SAM3_URLS`) and
  `DetectionProfile.segmenter_name` default describe the reference
  deployment's segmenter; any HTTP service speaking the same
  request/response shape works.
- A small number of OpenSearch document fields keep a `plate_*`
  prefix by default; `RegionFields` lets you rename them per-deployment
  with no reindex.

None of this affects correctness — it is purely a naming residue from
where the subsystem came from, called out here so it doesn't look like
an accident.
