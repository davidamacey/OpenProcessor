# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed (BREAKING)
- **One generic curation wire vocabulary.** Every region field is `region_<attr>`
  on the wire, fixed regardless of `OP_REGION_FIELD_*` storage overrides;
  `plate_thumbnail_url` → `region_thumbnail_url`; `gemma_*` → `vlm_*` and `v6_*` →
  `classifier_*` across item fields, `class_source` values, the `vlm_low_conf`
  review tab, auto_label params, `/health` and `/stats/dataset` (`plates` →
  `regions`); `coco_proposal_name` → `proposal_name`; ingest `n_plates` →
  `n_regions`. Region write bodies use `region_*` keys and reject unknown keys.
  Every item-returning endpoint (`/crops`, `/crops/{id}`, `/review/{tab}`,
  `/regions`, training candidates, `/search/text`) returns the same serialized
  item. Full old→new table: `docs/design/curation_api_contract.md` (B3).
- **Region detection is off by default.** The reference license-plate profile no
  longer self-registers; select it with `OP_REGION_PROFILE=license_plate` or
  configure one via `OP_REGION_DETECTION_*`.
- **`OP_DETECTION_*` is retired**; ingest detectors use `OP_INGEST_PRIMARY_*` and
  `OP_INGEST_SECONDARY_*` (leftover `OP_DETECTION_*` vars fail with a rename
  message). The secondary detector is now actually wired into ingest.
- **The ingest primary is a proposer by default** (`OP_INGEST_PRIMARY_ASSIGNS_CLASS=false`):
  its detections are unlabeled `<name>_proposal` items carrying the model's own
  label (`OP_INGEST_PRIMARY_LABELS_PATH`); the secondary assigns the class.
- **`detection_profile` is read-only**: `?detection_profile=` on
  `POST /pipeline/auto_label[/start]` and `PUT /settings` for that axis return
  422. `GET /methods` entries carry `settable: bool`.
- The detector bake-off harness is domain-neutral by default (`generic`
  `BakeoffProfile`; `--backend triton` requires a model); plate baselines moved
  to the `license_plate` example profile; paper-only scripts moved to
  `examples/bakeoff_lpr_paper/`.
- **Naming sweep, wave W1 — stored-data renames** (`docs/design/naming_sweep_plan.md`
  S1-S8; re-ingest required):
  - Items index kNN field `v6_embedding` → `backbone_embedding`
    (`CurationConfig.BACKBONE_EMBEDDING_FIELD`).
  - Images + items ingest-source field `hdd_source` → `source`; `GET
    /crops`'s `?hdd_source=` query param is removed (use the existing
    `?source=`).
  - Stored `region_source` / `candidate_source` provenance values:
    `sam3` → `segmenter`, `sam3_text_hint` → `segmenter_text_hint`, `lpr` →
    `detector`, `lpr_existing` → `detector_existing`.
  - `class_id_history[].writer` value `sam_worker` → `region_worker`.
  - `GET /curation/ingest/sam_drain` → `GET /curation/ingest/region_drain`;
    its response and `GET /stats/dataset`'s `in_progress.*` drop the legacy
    `pending`/`pending_verify` rollup keys (re-ingested data can never carry
    those short names).
  - Export manifest `dataset_kind` no longer accepts the alias
    `lpr_single_class`; only `single_class` is recognized.
  - No hardcoded model-id defaults: `OP_VLM_MODEL` has no default (was
    `gemma-4-e4b`) and `VlmLabeler` construction fails loudly when a VLM
    URL is configured without one; the reference license-plate profile's
    `detector_model` is the neutral example id `license_plate_detector`
    (was the proprietary Triton id `lpr_nanov11_640`).
  - `DELETE /curation/models/{name}`'s unload guard drops its hardcoded
    `lpr_` name prefix; a model is protected only via the active
    `DetectionProfile`'s configured model ids or the fixed `paddleocr_`
    prefix.
  - `OPENWEBUI_BASE_URL` / `OPENWEBUI_MODEL` / `OPENWEBUI_API_KEY` /
    `VLM_URL` / `GEMMA_URL` are retired; only `OP_VLM_URL` / `OP_VLM_MODEL`
    / `OP_VLM_API_KEY` are read now (the remaining `VLM_*`/`GEMMA_*` env
    vars — images-per-call, httpx pool sizing — are unchanged pending a
    later wave).

### Added
- **VLM class-attempt fields** `vlm_class_attempted_at` (date) and
  `vlm_class_empty_reason` (keyword: `no_answer` / `no_match` /
  `invalid_index` / `unparseable`; `null` when the attempt answered) on items
  (mapped, migrated on boot, class-guarded) and on the item wire. The `all`
  review tab surfaces items whose last attempt was empty; the VLM worker and
  the auto-label sweep skip them for 24 h. Every VLM class write (label
  batch, auto-label sweep, the region worker's combined call) now records a
  full, restorable `class_id_history` snapshot, including writes onto a
  proposal with no class yet and `class_source`-only writes.
- `scripts/curation/repair_empty_vlm_answers.py` (dry-run default,
  `--apply`, OCC): restores items stamped `vlm_unmatched` for an empty VLM
  answer to the class source they had before (VLM, ingest proposal or
  classifier, recovered from the untouched class provenance; class history
  as fallback) and records the empty attempt.
- **Naming sweep, wave W0 — served detector/segmenter/VLM vocabulary**
  (`docs/design/naming_sweep_plan.md`): `GET {prefix}/regions/vocabulary`
  serves `{detectors, region_sources, chain_actors}` (each entry `{id,
  label, role, filterable}`) built from the active `DetectionProfile` /
  ingest profiles / `OP_VLM_MODEL` — never a hardcoded model id — so the
  frontend stops keying a label/palette map on private ids
  (`lpr_nanov11_640`, `sam3`, `gemma-4-e4b`). `GET {prefix}/review/tabs`
  serves `{id, label, description}` for every review tab.
- `scripts/curation/backfill_region_embeddings.py` (dry-run default) and a
  shared region-embedding encode helper, so region false-positive clustering
  has embeddings to work with.
- `POST /curation/review/new_class_proposals/resolve` — bulk-resolves
  every pending `vlm_new_class_pending` item proposing a term in one call
  (map to an existing class or create one, `?dry_run=` to preview),
  instead of relabeling only the summary endpoint's capped
  `sample_crop_ids` one page at a time. Explicitly mapped
  `vlm_verify_completed_at` (`date`) on the items index — it was being
  written and range-queried but left to dynamic mapping.
- `GET /curation/train/gpus` — served training GPU picker (values, human
  labels, stop advisories, and the resolved default), so the frontend no
  longer hardcodes GPU ids/labels. `OP_GPU_ARBITER_CONTAINERS` entries may
  now carry a GPU scope (`name@2`, `name@0/2`); `OP_GPU_LABELS` and
  `OP_TRAIN_DEFAULT_GPUS` configure the option labels and default value.
- Curation operator tools: `run_probe.py` (probe-inference backfill),
  `reclassify_after_registry_growth.py`, `requeue_regions.py` (incl.
  `--missing-status` backfill), `seed_class_registry.py` (registry from ONNX
  `names`, `--check`), `cluster_raw_labels.py`, `ingest_upload.py` +
  `POST /curation/ingest/upload` (byte ingest with content dedup),
  `import_labeled_dataset.py` (incl. `--images-only`) and
  `eval_regions_vs_gt.py` (region cascade vs ground truth: recall, precision,
  IoU, background false-positive gate).
- `BakeoffProfile` + `GET /bakeoff/profiles` (with `default`/`default_profile`),
  optional `profile` on bake-off runs, a ported quantize leg.
- Ingest writes class provenance on every item, publishes `crop.created` SSE
  events, writes the backbone embedding from the secondary detector's feature
  map, and seeds `pending_detection` region status so the cascade picks new items up.
- `GpuArbiterConfig.from_env` (`OP_GPU_ALLOWED_IDS`, `OP_GPU_ARBITER_*`),
  `OP_VLM_MAX_IMAGES_PER_CALL`, `OP_SOURCE_PATH_ALIASES`, `OP_PROMPT_PACK_PATHS`
  (several selectable prompt packs), `OP_INGEST_PRIMARY_CLASS_IDS`.
- Per-run `?prompt_pack=` on auto_label (422 on unknown id, echoed in job args).
- `vlm_proposed_class_id` / `vlm_proposed_class_name` on every item;
  `GET /curation/class_sources`; `GET /curation/classes/{id}`; `GET /crops`
  `limit`/`sort`/`conf_min`/`conf_max`/`k`; `/export/datasets` `kind`/`profile_name`.
- Generated TypeScript `RegionStatus` contract (`contracts/ts/regionStatus.ts`)
  with a `--check` pre-commit drift hook.
- **Segmenter container (`docker/segmenter/`)**: a reference
  implementation of the detection cascade's segmenter leg — a FastAPI
  service wrapping Meta's SAM 3 that answers `POST
  /sam3/segment_plate` (alias `POST /segment`) and
  `/segment/batch` with candidate boxes in the submitted image's
  normalized frame. `text_prompt` is required per request with no
  server-side default, so the service carries no domain of its own.
  Ships behind its own `segmenter` compose profile (it needs a GPU and
  a HuggingFace token); the leg remains optional — an empty `SAM3_URL`
  still makes it a clean no-op. See
  [`docker/segmenter/README.md`](docker/segmenter/README.md).
- **Trainer container** (`docker/trainer/`, compose service
  `curation-trainer` behind the new `training` profile). Until now the
  API implemented only the control-plane half of the training file
  protocol and the repo shipped nothing that could answer it — a
  `/curation/train/start` had no counterparty outside the test harness's
  shell-script fake. The image watches `/jobs/` for `job.json`, runs
  each through Ultralytics, and writes `status.json` heartbeats,
  `run.log`, `best.pt` + `best.onnx`, and a `manifest.json` lineage
  envelope. Includes the subset/class-remap dataset rewrite,
  Albumentations stage-1 augmentation, cooperative cancel, CUDA-OOM
  batch backoff, multi-GPU AutoBatch, campaign auto-skip/auto-promote,
  and an optional side-by-side comparison against a served Triton
  model. Nothing domain-specific is baked in: dataset, class subset,
  hyperparameters, augmentation preset, orientation-sensitive class
  names and incumbent model all arrive via `job.json` or `OP_*` env.
- **`curation-mlflow`** compose service (port 4609) for optional
  experiment tracking of those runs.

### Changed
- The GPU arbiter now decides which containers to stop by **GPU scope**, not
  claim size: a single-GPU training claim that intersects a scoped
  container's GPU set stops that container (it no longer takes a multi-GPU
  claim to free a GPU that hosts a large sibling service). Unscoped
  containers keep the original "stopped only on a multi-GPU claim" behavior.
  `TrainJobSpec.cuda_visible_devices` / `TrainCampaignSpec.cuda_visible_devices`
  now default to the smallest `OP_GPU_ALLOWED_IDS` entry (or
  `OP_TRAIN_DEFAULT_GPUS` if set) instead of a hardcoded `'0'`, so a
  restricted allowlist that excludes GPU 0 no longer rejects the default spec.

### Fixed
- **Most VLM class answers were read as empty and recorded as `vlm_unmatched`**
  (live: 2,927 of 3,102 `vlm_unmatched` items had `vlm_raw_class=''`). The
  class calls sent no JSON-object `response_format`, so against a vLLM server
  with a reasoning parser the answer landed in the reasoning channel and
  `content` was empty or a lone `]`. Class calls now request JSON-object mode
  (with a `{"results": [...]}` envelope) and fall back to the reasoning
  channel; the combined reply accepts a class name / `"3=name"` / numeric
  string instead of rejecting the whole entry. An empty class answer (empty,
  `null`, `-1`, out-of-range, unparseable) no longer becomes `vlm_unmatched`:
  the item's class fields are left untouched and the attempt is recorded;
  `vlm_unmatched` is kept for a real, non-empty label (with `vlm_raw_class`).
  A VLM call that never completed writes nothing.
- Freshly ingested items never reached `/review/all`, the VLM worker or the
  pipeline VLM sweep (queues gated on a nonexistent `embedding` field).
- Region/label fields fell to dynamic `text` mapping on fresh indexes, breaking
  aggregations; every field is now explicitly mapped and queries no longer
  target `.keyword` subfields.
- A generic proposer's class ids were looked up in the domain class registry.
- Labels imported in the same `/ingest/batch` call were silently dropped.
- Secondary-detector NMS no longer depends on an external YOLOv5 checkout
  (native implementation following YOLOv5's documented semantics).
- Fresh `/jobs` volumes are writable by the app user; the evaluator image builds again.
- Bake-off quantize jobs silently scored nothing (missing module).
- `crop.region_verified` events carry `region_status`; `/export/datasets` lists
  single-class exports; server-built URLs and scripts follow `OP_API_PREFIX`.
- Removed host-specific paths and a LAN hostname from public source and docs.
- A subset-trained run now propagates its `class_remap.json` into the
  checkpoint's `weights/` directory *and* the run manifest, and reports
  `class_remap_copy_failed` on the job status when it cannot. This is
  the trainer half of the promote fix already present on the API side
  (`resolve_class_remap`): without it, promoting a subset run silently
  wrote a `labels.txt` from the full class registry, mislabeling every
  class the served model emits.
- The curation VLM worker never processed anything (bare-script import
  failure swallowed); it now runs as a module and exits loudly on an
  unhandled task exception.
- Training runs failed at the final step because MLflow's artifact root was
  a local path the trainer couldn't write; artifacts now proxy through the
  tracking server (`--serve-artifacts`), and the trainer's MLflow client is
  pinned to the server's major version.
- Promoted Triton models could land outside the mounted model repository;
  the repo path/URL are resolved from env at promoter construction, and
  promoted models are reloaded on API startup after a Triton restart.
- Auto-promote validated classifier labels via class clusters (purity 1.0 by
  construction); it now only considers candidate clusters, skips excluded
  items, and reports a correct dry-run count. Operator repair script:
  `scripts/curation/revert_class_cluster_promotions.py`.
- Region false-positive distances were squared L2 read as plain L2, loosening
  every FP-clustering threshold; region k-means centroids are re-normalized.
- The GPU-arbiter pause sentinel writer and readers used different paths.
- `id_normalize` pulled excluded items back into their class cluster.
- The GPU arbiter fell back to a sentinel-only pause when it could not stop a
  container sharing the claimed GPU; `/train/start` and
  `/train/start_campaign` now refuse with 409 and preflight blocks
  (`gpu_arbiter` check). The `docker` SDK is now a dependency.
- cuML kNN graph self-loops dropped (parity with sklearn).

- `POST /crops/move` into a candidate cluster wrote the cluster id as a
  validated class id; it now only sets placement (a human-owned class is
  cleared, nothing is validated), and unassigned or unregistered targets
  get 400. Export manifests record rows dropped for unregistered class ids
  (`dropped_unregistered_class_ids`), and preflight warns on them.
- `POST /crops/batch_unexclude` returns an unvalidated item to the
  candidate cluster it was excluded from while that cluster still has
  members, instead of leaving it outside every cluster until a recluster.

### Removed
- `DETECTION_YOLOV5_FORK`; the bake-off CoreML leg and `OP_COREML_HOST`
  (`quantize.coreml` returns 400).

## [0.3.0] - 2026-09-21

### Added
- **Curation subsystem (EXPERIMENTAL)**: a generic active-learning
  curation and labeling stack — class registry, item
  browse/label/move/exclude, clustering (AHC refinement +
  auto-promote), VLM-assisted region labeling/verification,
  review/active-learning queues, YOLO dataset export, and a
  training-job control API — mounted under `CurationConfig.api_prefix`
  (default `/curation`, 109 routes across 25 route groups as of this
  release). Configured through four dataclasses (`CurationConfig`,
  `RegionFields`, `DetectionProfile`, `RegionStatus`) rather than
  forked code. Ships opt-in behind the `curation` Docker Compose
  profile and disabled by default — see
  [`docs/CURATION.md`](docs/CURATION.md) for the user guide,
  `docs/design/curation_design_rationale.md` for the design rationale,
  and `docs/design/curation_api_contract.md` for the HTTP wire
  contract.
- **Curation ingest endpoints**: `POST /curation/ingest/image`,
  `/ingest/batch`, and `/import_labels(/batch)` — duplicate detection,
  a quality gate, crop-cache population, bulk OpenSearch indexing, and
  YOLO-format label import. Includes `scripts/curation/ingest_walker.py`
  for resumable bulk directory ingest.
- **Curation runtime companions**: `curation-detection-worker`,
  `curation-vlm-worker`, `curation-auto-label-worker`,
  `curation-cluster-refresh`, and an on-demand `curation-evaluator`
  bake-off container, all behind `docker compose --profile curation`.
  The detection cascade's segmenter leg is optional — a deployment may
  omit a segmenter entirely.
- **Class-selectable labeling assist**: the VLM prompt pack is
  deployment-supplied and loadable from a JSON file
  (`OP_PROMPT_PACK_PATH`); `GET /curation/methods` advertises the
  active detection profile and prompt pack, and dataset-export
  capability by kind.
- **Shared curation deployment settings**: `GET,PUT /curation/settings`
  — a durable, backend-stored shared default per strategy axis
  (cluster method, sort order, detection profile, prompt pack),
  replacing per-browser client defaults that reset on every reload.
- **Live write-path verification harness**: an isolated, disposable
  Compose stack (`docker/test/compose.yml`, project `op-live-verify`)
  plus a deterministic seed script
  (`scripts/curation/seed_live_harness.py`) and a `tests/live` suite
  (`-m live`) that exercises curation write endpoints against a real
  OpenSearch, a real file-based training protocol, and a fake
  OpenAI-compatible VLM/trainer — the first time any curation write
  endpoint has been run against a live stack rather than mocked
  clients. See `docker/test/README.md`.
- `.github/workflows/ci.yml` — runs the full pytest suite and
  `pre-commit run --all-files` on every PR and push to `main`.
  `requirements-test.txt` provides a CPU-installable dependency subset
  for the CI runner.
- `SECURITY.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, GitHub issue
  templates, a pull-request template, `CODEOWNERS`, and `dependabot.yml`.
- `docs/CURATION.md` — the previously-missing curation user guide.

### Fixed
- **Export/training artifact chain**: `class_registry.json` is now
  actually written by dataset export (it was declared but never
  produced), with a dense `export_id_map` so `include_classes`
  filtering on export is no longer inert. Export now does a
  stratified, group-aware split with image copy/resize, replacing an
  unstratified hash split that wrote labels but no pixels.
- **Persistence hardening**: orphaned running-job state is reconciled
  on API startup, and export-task tracking survives a restart instead
  of being lost.
- Restored CORS middleware and several `OP_*` env-var override paths
  (feature flags, crop-cache directory, region-field overrides) that
  had silently diverged between routers and services during the port.
- Removed private absolute host-path defaults from the bake-off
  harness.
- Renamed all `LEGACY_*` environment variables to `OP_*` (23 vars) and all
  `legacy_*` Prometheus metric names to `op_*`, closing the last
  reference-deployment naming residue in the config surface.
- **`region_*`/`plate_*` wire-contract leak**: `GET /curation/crops/{id}`
  returned the raw OpenSearch `_source` (RegionFields storage keys,
  `region_*` by default) instead of the frozen `ItemDoc` `plate_*` wire
  contract; `PATCH /crops/{id}/plate_meta`'s `updated_fields` echoed
  the same internal keys instead of the request's wire names; and
  `GET /review/{tab}` built its response dict using storage keys as
  literal JSON keys. All three now correctly emit `plate_*`. `ItemDoc`
  gained 11 previously-missing round-trip fields
  (`plate_status`/`plate_text`/`plate_detector`/`plate_verified`/etc)
  that `PATCH .../plate_meta` wrote but no `GET` ever returned. Found
  via a live cross-repo integration test against the Cropwright
  labeler frontend.
- **Shared settings could not be cleared**: `PUT /curation/settings`
  required every submitted value to be a currently-advertised strategy
  id, so once an axis was pinned there was no way back to "each
  endpoint uses its own tuned default" — a `null` value now clears
  that axis's override.

### Removed
- `docs/security/` and `docker/hardened/deepstream/` — an internal
  DeepStream CVE-hardening investigation unrelated to this product
  (the Triton half of that work is kept — see
  `docs/security/triton_cve_hardening.md` and
  `docker/hardened/triton/`).

### Changed
- **License: re-badged MIT → AGPL-3.0-or-later.** This project vendors
  an AGPL-3.0 Ultralytics fork (`src/ultralytics_patches/`); the whole
  repository is now correctly badged to match that copyleft obligation
  instead of the previous (incorrect) MIT badge. See `LICENSE`,
  `README.md`, `ATTRIBUTION.md`, and `pyproject.toml`.
- Test suite hardened: restored dropped OCC-invariant and
  write-guard tests, added coverage for previously-zero-coverage
  detection/clustering leaves, measured `scripts/` for coverage, and
  enforced a coverage floor.

## [0.2.1] - 2026-07-04

### Fixed
- Fresh-install path (`scripts/setup.sh`) on Triton 26.06: trtexec moved
  to `/usr/bin` and its `--fp16` flag was removed in TensorRT 11 — the
  PaddleOCR engine step now works out of the box.
- End2end `config.pbtxt` is written from the built engine's actual output
  dtypes (EfficientNMS_TRT precision varies across TRT releases/builds).
- Health checks in `setup.sh` accept the `/health` -> `/ready` status
  contract.
- CI: valid action pins (trivy-action v0.36.0, checkout v5,
  codeql-action v4) and a scan timeout suited to the image size.
- Endpoint suite: dual-family checks skip gracefully when the optional
  YOLO26 engine is not exported.

## [0.2.0] - 2026-07-04

### Added
- **YOLO26 support served alongside YOLO11** in the same Triton + API
  instance: native NMS-free export (`export/export_yolo26.py`), a
  detection-adapter registry that resolves each model's output contract
  from Triton metadata, `YOLO_MODEL` env for the default detector, and
  per-request selection via the existing `model_name` parameter.
- Dual export toolchains in one image: the proven YOLO11 EfficientNMS
  path keeps its exact pin (`ultralytics==8.3.253`) in an isolated
  `/opt/venv-y11`; `export_models.py` re-execs into it transparently.
- `/live` and `/ready` health endpoints with real per-dependency probes
  (Triton gRPC `is_server_live`, OpenSearch HTTP); `/health` is now an
  alias of `/ready`.
- Prometheus `/metrics` endpoint with an `http_request_duration_seconds`
  histogram labeled by route template.
- dcgm-exporter service + GPU Metrics Grafana dashboard (all host GPUs).
- `make scan` targets and a GitHub Actions Trivy workflow (filesystem +
  API image, SARIF upload, weekly cron).
- Integration test suites: 20 GPU-free pytest tests and a live endpoint
  suite (25 checks) including dynamic YOLO26 load/unload
  (`tests/test_endpoints.sh dual`).
- `docs/MIGRATION_TRITON_26.md` upgrade guide.

### Changed
- **BREAKING: Triton upgraded to 26.06 (CUDA 13.3, TensorRT 11.1)** —
  every existing TensorRT engine must be re-exported; see the migration
  guide. A build-time assertion keeps the server TRT and the
  `tensorrt-cu13` pip pin in lockstep.
- TensorRT 11 is strongly-typed: FP16 is baked into the ONNX at export
  (NVIDIA ModelOpt AutoCast / onnxconverter-common for EfficientNMS
  graphs). Text detection (`paddleocr_det`) defaults to FP32 for
  threshold robustness.
- Monitoring stack pinned (prometheus v3.12.0, grafana 13.1.0, loki
  3.6.12 non-root, node-exporter v1.10.2); Promtail (EOL 2026-03-02)
  replaced by Grafana Alloy v1.17.1.
- OpenSearch upgraded to 3.6.0 (3.0–3.2 carry known HIGH CVEs).
- Triton container runs as the non-root `triton-server` user; both built
  images apply apt security upgrades and scan clean of fixable
  HIGH/CRITICAL CVEs (Nsight Systems CLI removed from the runtime image).
- Triton batching: 25 ms max queue delay; instance counts are a 12 GB
  baseline with scale-up guidance in each `config.pbtxt`.
- gRPC message caps raised to 512 MB for large raw detector heads.
- `/detect` always applies its confidence filter (NMS-free engines emit
  all top-K candidates).
- Request-id context moved to `src.core.logging` (importable by worker
  processes); structlog `foreign_pre_chain` formatter bug fixed.

### Fixed
- `cluster_distance` sort no longer 400s on documents missing the field
  or on freshly created indices.
- Container HEALTHCHECK targets `/live` so a degraded downstream
  dependency cannot cascade restarts through `depends_on`.

## [0.1.0] - 2026-03-19

Initial public release: YOLO11 detection, SCRFD + ArcFace face
recognition, MobileCLIP embeddings, PP-OCRv5 OCR, OpenSearch visual
search, Triton 25.10 TensorRT serving, monitoring stack.
