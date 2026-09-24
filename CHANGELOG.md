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

### Added
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

### Fixed
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
