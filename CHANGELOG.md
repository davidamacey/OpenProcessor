# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Text-free region mode.** A region profile with `text_reader: "none"`
  stores region boxes and no region text: the region OCR reader never
  runs, a VLM reading is dropped, and `PATCH /crops/{id}/region_meta`
  answers 422 `{"error": "region_text_disabled"}` for a `region_text`
  edit. `GET /regions/vocabulary` then serves `text_rules: null` and
  `text_choices: []`, and the region-profile summary (also on `/health`)
  gains `reads_text` and `text_hint_enabled`. The `regions` review tab
  drops its `text` filter for such a profile, and the text-repair tools
  (`rederive_region_text.py`) exit cleanly with nothing to do.
- **Optional OCR text hint.** New profile fields `text_hint_enabled`
  (default `true`) and `text_hint_require_letters_and_digits` (default
  `false`). The text-hint re-pass after a segmenter miss runs only when it
  is enabled, an `ocr_pipeline_model` is set and the segmenter leg is on;
  otherwise the chain ends at `<segmenter>:miss`. The `OCR text hint`
  actor and the `segmenter_text_hint` region source are only listed in
  the vocabulary when the hint can run.
- **`parent_classes` region-profile field.** Restricts the region stage
  to items whose `class_name` or `proposal_name` matches (case-insensitive;
  empty = every item). Ingest seeds only matching items and the detection
  worker skips non-matching ones already pending.
- **Built-in text-free prompt pack `generic_region_v1`**, plus a public
  car -> wheel example: `examples/region_profiles/vehicle_wheel.json`
  (segmenter-only, text-free) and `examples/prompt_packs/vehicle_wheel.json`.

### Changed
- **Letters-and-digits text-hint rule is opt-in.** A text-hint candidate no
  longer has to mix letters and digits unless the profile sets
  `text_hint_require_letters_and_digits: true`
  (`examples/region_profiles/license_plate.json` does).
- **`examples/region_profiles/license_plate.json` is segmenter-only**
  (`detector_model: ""`) and sets its text-hint flags explicitly.
- Segmenter candidates carry the profile's `segmenter_name` as their
  source instead of a hardcoded `sam3`. New geometry rejects are recorded
  as `parent_bbox_unpack_failed` / `parent_bbox_degenerate` (were
  `vehicle_bbox_*`). `VlmLabeler.label_vehicle_batch` is renamed
  `label_item_batch`.

### Fixed
- **An empty `detector_model` no longer calls Triton.** It used to run
  inference against model `''` on every item, log `region_infer_failed`
  and append a `':miss'` trace tag with an empty actor; the detector leg
  is now skipped entirely.
- **Segmenter never became reachable on a stock install (F-75).** The
  `segmenter` service's `env_file: .env` loaded the host-port variable
  `SEGMENTER_PORT` (env.template default `4611`) straight into the
  container, and `docker/segmenter/main.py` read that same name as its
  uvicorn listen port -- so the container bound to `4611` while the port
  mapping, healthcheck and `OP_SEGMENTER_URL` all still targeted `8000`.
  The in-container variable is renamed `SEGMENTER_LISTEN_PORT` (default
  `8000`, also set explicitly under `environment:` so it beats
  `env_file`), and a new compose-contract test
  (`test_no_service_reads_a_host_port_var_as_its_own_container_config`)
  guards every other `env_file`-loading service against the same class of
  bug. An audit of the remaining host-port vars (`API_PORT`,
  `TRITON_*_PORT`, `PROMETHEUS_PORT`, `GRAFANA_PORT`, `LOKI_PORT`,
  `DCGM_PORT`, `OPENSEARCH_PORT`, `OPENSEARCH_DASHBOARDS_PORT`,
  `MLFLOW_PORT`, `VLM_PORT`) found no other container reading its own
  host-port var name. **Requires rebuilding the segmenter image.**
- **Region-dependency health check never saw a healthy segmenter (V-1
  follow-up).** `check_region_dependencies` looked up the profile's
  segmenter (e.g. `sam3`) in Triton's repository index, but SAM 3 runs as
  the separate HTTP segmenter service (`OP_SEGMENTER_URL`), not in
  Triton, so `stall_reason` never cleared even with a healthy segmenter.
  Triton-served detectors still go through the Triton repository index;
  the segmenter dependency now does a `GET {OP_SEGMENTER_URL}/health`
  with a short timeout, requiring `loaded: true`.
- **Training couldn't start on a stock install (F-72 regression).**
  `OP_GPU_ARBITER_TRAINER_CONTAINER` defaulting to
  `${COMPOSE_PROJECT_NAME}-trainer` (see the F-72 entry below) meant
  `/train/preflight`'s trainer probe now always ran -- but the stock
  `yolo-api` container has no docker socket/SDK, so the probe
  unconditionally reported `block` ("docker SDK/socket unavailable"),
  422ing `/train/start` even with a perfectly healthy trainer. The probe
  (moved to `src/services/training/trainer_reachability.py`) now reads
  the trainer's own heartbeat file (`.trainer_capabilities.json`, which
  the trainer's watch loop refreshes every ~30s) as its primary signal --
  no docker socket needed. A fresh heartbeat is `ok`; a missing or stale
  one is `warn`, never `block`. The docker SDK/socket path (only present
  behind the `docker-compose.gpu-arbiter.yml` overlay) is now a purely
  optional, confirming extra: it's only consulted when the heartbeat
  itself is missing/stale, and only then may it upgrade the warning to a
  definitive `block`.
- **API image builds again.** `perception_models` is installed with `--no-deps`
  at a pinned commit (its requirements exact-pin `timm==1.0.15`, which
  conflicts with `open-clip-torch>=3.2`'s `timm>=1.0.17`); the PE encoder's
  real runtime deps (`einops`, `regex`) are declared in `requirements.txt`.
- **Triton serves a partial model set.** `triton-server` now runs with
  `--exit-on-error=false --strict-readiness=false`, so one missing or failed
  engine (the minimal setup profile skips OCR; setup continues past a failed
  export) leaves only that model unloaded instead of stopping the server.
  The minimal profile's export now also builds the PE-Core image encoder that
  curation ingest needs. `TRITON_GPU_ID` (default `0`) selects Triton's GPU.
- **`vlm` profile image pinned by digest** to the vLLM Gemma 4 build this
  stack is tested against (`vllm/vllm-openai:gemma4-cu130@sha256:0d1525...`);
  the earlier `v0.11.0` default predates Gemma 4.

### Changed (BREAKING)
- **Compose/install portability (fresh-start gaps batch B).** `docker-compose.yml`
  no longer hardcodes `name: openprocessor` or any `container_name:` — both are
  now interpolated from `COMPOSE_PROJECT_NAME` (default `openprocessor`, so an
  existing single-stack deployment behaves identically). **Migration hint:**
  if you script against container names directly (e.g. `docker exec yolo-api
  ...`, `docker logs triton-server`), switch to `docker compose exec
  yolo-api ...` / `docker compose logs triton-server` — those already resolve
  by service name regardless of the interpolated container name, and keep
  working the same way after this change. Every host port
  (`API_PORT`, `TRITON_HTTP_PORT`, `TRITON_GRPC_PORT`, `TRITON_METRICS_PORT`,
  `PROMETHEUS_PORT`, `GRAFANA_PORT`, `LOKI_PORT`, `OPENSEARCH_PORT`,
  `OPENSEARCH_DASHBOARDS_PORT`, plus new `MLFLOW_PORT`, `DCGM_PORT`,
  `SEGMENTER_PORT`, `VLM_PORT`) is now interpolated from `.env`/the shell
  instead of hardcoded, so a second isolated stack on the same host only
  needs a `.env` with a different `COMPOSE_PROJECT_NAME` and remapped ports.
  `env.template`'s `TRITON_HTTP`/`TRITON_GRPC`/`TRITON_METRICS` were renamed to
  `TRITON_HTTP_PORT`/`TRITON_GRPC_PORT`/`TRITON_METRICS_PORT` to match the
  Makefile's existing names — update any script/CI reading the old names.
  `Makefile`'s port variables now use `?=` and load `.env` (`-include .env`),
  so both `.env` and `make API_PORT=... TRITON_HTTP_PORT=... <target>` work.
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
- **Region detection is off by default, and no profile ships built in.**
  `src/services/detection/reference_profiles.py` is removed; the
  license-plate example profile is a data file,
  `examples/region_profiles/license_plate.json`, loaded via
  `OP_REGION_PROFILE_PATH=<path>`. `OP_REGION_PROFILE=<name>` now only
  resolves a profile a deployment's own startup code registered.
  `DetectionProfile` gains `region_class_name`, `display_name` and
  `display_name_singular` fields, served on `GET {prefix}/regions/vocabulary`.
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
  to the `license_plate` example profile; paper-only scripts (a dedup-threshold
  sweep and a LaTeX-number generator that hardcoded a private model id and a
  live-deployment URL) removed from the public tree.
- **Model comparison (bake-off) API v2, generic and multi-class** (clean break,
  no compatibility fields; shapes in `docs/design/curation_api_contract.md`). Every
  `/curation/bakeoff/*` route is typed and result files carry
  `schema_version: 2` (older result files answer 409).
  `POST /bakeoff/run` takes `datasets: [{id}]` (`export:<path>`,
  `external:<group>/<name>`, or `run:<job_id>`) and `models[]` discriminated
  on `source` (`run` / `baseline` / `custom`), plus
  `quantize: {run_id, formats, n_calib, calib_split, throughput}`; removed:
  `dataset`, `datasets[].path/name`, `verify_frozen`, free-form model specs
  (`backend`/`profile`/`gt_class_id`/`gt_class_name`/`pred_class_id`/
  `lpdnet_variant`/`primary_classes`), `quantize.coreml`. Responses: eval
  datasets use `source` + `group` (no `n_test`/`frozen_sha`/`kind`);
  `trained_models` serves `trainer_map50` / `trainer_map50_split` (were
  `map50` / `map50_split`); comparison rows put metrics under `overall` /
  `common` with `per_class` and `coverage`; `results` takes `?dataset_id=`;
  matrix `best` values are lists of tied winners; job state adds `queued`.
- **`BakeoffProfile` loses `target_class_id` / `target_class_name`**: a
  profile scores every class in the eval split (`class_filter` narrows by
  name). `OP_BAKEOFF_PROFILE_TARGET_CLASS_ID` / `_NAME` are retired (startup
  fails with a pointer to `OP_BAKEOFF_PROFILE_CLASS_FILTER`). The
  license-plate profile, baselines, converters and the `lpdnet` /
  `open-image-models` backends moved to `examples/bakeoff/license_plate/`
  and load only by profile path; `GET /bakeoff/profiles` no longer lists
  example profiles and the default baseline registry is empty.
- The trainer's opt-in auto-quantize posts `POST /curation/bakeoff/run`
  (via `OP_API_BASE_URL` + `OP_API_PREFIX`) instead of writing a job file;
  `campaign.py` no longer reads `OP_BAKEOFF_JOBS_DIR` / `OP_BAKEOFF_OUT_DIR`.
- **Stored-data renames** (re-ingest required):
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
    / `OP_VLM_API_KEY` are read now.
- **Wire surface renames**: `GET /curation/methods`'
  operationId is `get_methods_curation_methods_get` (was a
  company-initialed operation id); its `flags` keys drop the same
  company-initialed prefix (`scores_enabled`, `scores_shadow`,
  `select_diverse_enabled`, `viz_projection_enabled`,
  `semantic_search_enabled`); the
  `coco_blind_spots` review tab id and its default-sort id are renamed
  to `classifier_blind_spots` / `classifier_blind_spots_default`.
- **Env vars, clean break, no aliases.** A
  startup guard (`src/config/retired_env.py`, called from `src/main.py`'s
  lifespan and both worker `main()` entry points) now fails loudly,
  naming the replacement, if any of these are still set:

  | Old | New |
  |---|---|
  | `VLM_URL`, `GEMMA_URL`, `OPENWEBUI_BASE_URL` | `OP_VLM_URL` |
  | `OPENWEBUI_MODEL` | `OP_VLM_MODEL` |
  | `OPENWEBUI_API_KEY` | `OP_VLM_API_KEY` |
  | `VLM_IMAGES_PER_CALL`, `GEMMA_IMAGES_PER_CALL` | `OP_VLM_OPEN_IMAGES_PER_CALL` |
  | `VLM_HTTPX_MAX_CONNECTIONS`, `GEMMA_HTTPX_MAX_CONNECTIONS` | `OP_VLM_HTTPX_MAX_CONNECTIONS` |
  | `VLM_HTTPX_KEEPALIVE`, `GEMMA_HTTPX_KEEPALIVE` | `OP_VLM_HTTPX_KEEPALIVE` |
  | `SAM3_URL` | `OP_SEGMENTER_URL` |
  | `SAM3_URLS` | `OP_SEGMENTER_URLS` |
  | `SAM3_HTTPX_MAX_CONNECTIONS` | `OP_SEGMENTER_HTTPX_MAX_CONNECTIONS` |
  | `SAM3_HTTPX_KEEPALIVE` | `OP_SEGMENTER_HTTPX_KEEPALIVE` |
  | `SAM3_SKIP_VLM_VERIFY_SCORE`, `SAM3_SKIP_GEMMA_VERIFY_SCORE` | `OP_SEGMENTER_SKIP_VERIFY_SCORE` |
  | `SAM_WORKER_VLM_CONCURRENCY`, `SAM_WORKER_GEMMA_CONCURRENCY` | `OP_REGION_WORKER_VLM_CONCURRENCY` |
  | `SAM_WORKER_VLM_VISIBLE_CONCURRENCY`, `SAM_WORKER_GEMMA_VISIBLE_CONCURRENCY` | `OP_REGION_WORKER_VLM_VISIBLE_CONCURRENCY` |
  | `SAM_WORKER_METRICS_PORT` | `OP_REGION_WORKER_METRICS_PORT` |
  | `OP_REGION_DETECTION_SAM_TEXT_PROMPT` | `OP_REGION_DETECTION_SEGMENTER_TEXT_PROMPT` |
  | `GEMMA_CROP_CACHE_DIR` | `OP_CROP_CACHE_DIR` |

  Also: the region worker's `--gemma-url` CLI flag is now `--vlm-url`;
  the segmenter service's `/sam3/segment_plate` and
  `/sam3/segment_plate_batch` path aliases are removed (`POST /segment`
  and `POST /segment/batch` are the only paths now; the shipped client
  posts to `/segment`).
- **Prometheus metric name cleanup.** Every metric
  constant and name in `src/services/curation/metrics.py` moved off the
  legacy metric prefix onto `OP_*`/`op_*`, and
  domain/vendor-named metrics were renamed alongside the prefix swap
  (for example, the combined/separate VLM call counters, the
  segmenter-leg duration and circuit-breaker metrics, and the
  region-detector stage duration). Metrics that carried no domain name
  (`occ_retry_count`, `worker_skip_human_won`, `shm_crop_cache_*`,
  `source_image_*`, `thumbnail_cache_*`, …) kept their name and only
  gained the `op_` prefix.
- **Structured log events use a `curation_` prefix** instead of the
  retired company-initialed one, across the OpenSearch client, ingest,
  index bootstrap, and job/status logging.
- **Training run status fields renamed**: `TrainJobStatus`'s
  `best_metric` / `last_metric` pair is replaced by two distinct rows,
  `last_epoch_metric` (the true last training epoch's metrics) and
  `best_checkpoint_metric` (the best checkpoint's own re-validation
  metrics) — see `docs/design/curation_api_contract.md`'s "Training run
  status" section for why two fields are needed. `Job.migrate_status`
  drops the retired keys from any pre-rename `status.json` on read
  rather than migrating their values, since the two were never the same
  measurement.

### Added
- **`GET /classes` exposes `merged_into`.** A class merged via `POST
  /classes/merge` has always tracked its `merged_into` target internally
  (`RegistryClassEntry.merged_into`), but the wire model never served it,
  so the frontend had no way to render "-> merged into X" without a
  second `GET /classes/{id}` round trip.
- **Fresh-start gaps batch B: compose and install portability.**
  - `yolo-api` and `curation-detection-worker` both mount a source-image root
    at the same container path (`${OP_SOURCE_ROOT_HOST:-./data/source}:/data/source:ro`,
    `OP_SOURCE_ROOT=/data/source`) and `./examples:/app/examples:ro`, so
    `OP_REGION_PROFILE_PATH=/app/examples/region_profiles/license_plate.json`
    (the env.template example) actually resolves in both containers.
  - `pe_image_encoder` is now in `triton-server`'s default `--load-model`
    list (CURATION.md already called it required, not optional). `make
    export-all`/`./scripts/setup.sh`'s automated export flow now builds it
    (weights download, image-tower ONNX, TensorRT with an ONNX Runtime
    fallback, text-tower ONNX) before Triton's first start — it needs this
    like every other listed model, since Triton's explicit
    `model-control-mode` exits at startup if a listed model fails to load.
  - New optional `vlm` compose profile (`docker compose --profile vlm up -d`):
    a pinned-tag vLLM service serving Gemma 4 E4B, configurable via
    `VLM_IMAGE`/`VLM_MODEL`/`VLM_SERVED_MODEL_NAME`/`VLM_DTYPE`/
    `VLM_MAX_MODEL_LEN`/`VLM_GPU_MEMORY_UTILIZATION`/`VLM_LIMIT_MM_IMAGES`/
    `VLM_GPU_ID`/`VLM_PORT`. `yolo-api` and `curation-vlm-worker` both carry
    `extra_hosts: ["host.docker.internal:host-gateway"]` so the
    `OP_VLM_URL=http://host.docker.internal:<port>/v1` (external VLM) example
    resolves on Linux, not just Docker Desktop.
  - New optional `docker-compose.gpu-arbiter.yml` overlay: mounts
    `/var/run/docker.sock` into `yolo-api` so `OP_GPU_ARBITER_CONTAINERS`
    coordination actually works, documented as an explicit opt-in with its
    security tradeoff spelled out. Without it, the arbiter now logs exactly
    one `arbiter_docker_unavailable` warning per outage (was: one per
    call) when it fails open.
  - `yolo-api` carries a network alias `op-api` so Cropwright's default
    `API_UPSTREAM=http://op-api:8000` resolves without an override; see
    `docs/CURATION.md` "Wiring up Cropwright" and the README's Cropwright
    paragraph for the exact env vars and docker network name.
  - `curation-trainer`'s `OP_TRAIN_GPU_ORDER` and `device_ids` are both
    interpolated from the same `OP_TRAIN_GPU_ORDER` env var (was hardcoded
    to `0` in `environment:`); a multi-GPU order still needs a compose
    override for `device_ids` (documented inline).
  - `scripts/setup.sh` gained `--force` (never overwrites an existing `.env`
    otherwise) and `--curation` (prints the curation-subsystem next steps:
    PE export, class registry, VLM, segmenter, `--profile`); its smoke tests
    and `unload_models_for_export`/`check_triton_container` now read the
    deployment's actual configured ports/compose-service state instead of
    hardcoded `4600`/`4603`/`4607` and a hardcoded `triton-server` container
    name.
  - `tests/test_full_system.py` reads `API_PORT`/`TRITON_HTTP_PORT`/
    `OPENSEARCH_PORT` from the environment; README's Testing section gained
    a Docker-only path (`docker compose exec yolo-api pytest tests/ -q`).
  - `tests/test_compose_contract.py` gained invariants pinning all of the
    above (no fixed project name, no hardcoded host ports, source root +
    examples mounted on both services, `pe_image_encoder` in the default
    load list).
- **S-2: heartbeat-based curation worker healthchecks.** The four
  curation background workers (detection, VLM, auto-label,
  cluster-refresh) now write a heartbeat file on their main loop —
  including while idle — checked by
  `python src/services/curation/worker_liveness.py check <name>
  --max-age 120`, replacing `pgrep -f <module>` (which can't see a
  deadlocked-but-still-running event loop). `yolo-api` gained its own
  `/health`-based healthcheck so `curation-vlm-worker` /
  `curation-cluster-refresh`'s `depends_on` can gate on
  `condition: service_healthy` instead of merely "container started."
  New `OP_HEARTBEAT_DIR` env var (container-local, no mount needed).
- **S-3: cross-process event bus.** `GET /events` SSE subscribers on any
  of `yolo-api`'s 8 uvicorn worker processes now see every published
  event, not just the ones published on the same process. Backed by a
  shared, bounded, rotated JSONL log
  (`{OP_STATE_DIR}/events/events.jsonl`) every process tails; new
  `OP_EVENT_BUS` (`file` default, `process` restores the old
  in-process-only behavior) and `OP_EVENT_LOG_MAX_BYTES` env vars.
  `curation-detection-worker` now sets `OP_EVENT_API_URL` so its
  `crop.region_verified` events reach every subscriber, not one
  arbitrarily-chosen worker; `bulk_writer.py`'s event-publish URL also
  falls back to `OP_API_BASE_URL`/`OP_API` when `OP_EVENT_API_URL` is
  unset. `GET /events/stats` now also reports `bus` and `log_path`.
- **Class deprecate/restore.** `POST /classes/{class_id}/deprecate` flips
  `deprecated` on a class nothing references (idempotent; refuses on a
  still-referenced class); `POST /classes/{class_id}/restore` undoes it
  (`404` unknown id, `409` if a non-deprecated class already uses the
  name) — a lighter-weight alternative to `POST /classes/merge` for a
  class that was never actually used.
- **Deployment-supplied training presets.** `OP_TRAIN_PRESETS_PATH` (a
  JSON list of the same shape as the built-in presets) appends
  deployment-specific `class_subset_presets` entries, served by
  `GET /train/presets` alongside the generic built-ins (`all`, and
  `all_except_region` / `region_only` when the active region profile
  sets `region_class_name`).
- **A background probe-inference job API**: `POST /probe/run` (resolves
  a finished training job's checkpoint, `409` if not `finished` or no
  checkpoint on disk; claims a GPU through the same arbiter
  `POST /train/start` uses), `GET /probe/status`, `POST /probe/cancel`
  — wraps `run_probe_inference` so a probe backfill runs as a tracked
  background job instead of blocking the request; one job at a time.
- **Review queues explain an empty result instead of just serving zero
  rows.** `GET /review/{tab}` computes `empty_reason` from live index
  state (for example, `"no probe predictions — run a probe"`,
  `"item scores never computed"`, `"no unclassified proposals"`, else
  `"no items match"`); `GET /review/tabs` gained
  `empty_state: {has_probe_predictions, has_item_scores}` so a client
  can word any tab's empty state without a per-tab round trip.
- **A naming-leak pre-commit guard** (`scripts/codegen/check_naming_leaks.py`,
  wired into `.pre-commit-config.yaml`): three `git grep` scans over the
  whole tracked tree catch a reintroduced company name, retired
  vendor/domain vocabulary, or private class-registry vocabulary before
  it ships, filtered through a reviewed, per-line allowlist
  (`scripts/codegen/naming_leak_allowlist.txt`) so a deliberate example
  or historical/negative-test mention doesn't need re-justifying on
  every commit.
- **Per-class model comparison.** Every export with a labelled test split is
  an eval dataset (`GET /bakeoff/eval_datasets`, class counts and
  test-split hashes computed from the files); finished training runs are
  contenders directly (`GET /bakeoff/trained_models?dataset_id=` with
  same-export / same-frozen-test / train-test overlap). The harness scores
  per class and overall (COCO mAP50, mAP50-95, micro P/R/F1), maps each
  model's classes onto the dataset's (run class remap, registry ids, names
  or an explicit map) and reports uncovered classes and unmapped
  predictions; rows rank on the classes every model covers.
- **Test-split identity and build identity in run lineage.** Export
  manifests record `frozen_test_sha` (which frames) and `test_label_sha`
  (which boxes); training job specs and run manifests record `dataset_sha`,
  `frozen_test_sha`, `test_label_sha` and `dataset_version_tag` separately,
  plus `code_versions.api_sha`, `trainer_sha` and `trainer_image_id`.
  `OP_BUILD_SHA` is baked into the API, trainer and evaluator images at
  build time (`build.args`, OCI revision label; `make build` passes it); a
  runtime value still overrides it.
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
- **Served detector/segmenter/VLM vocabulary**:
  `GET {prefix}/regions/vocabulary`
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
  `POST /curation/ingest/upload` (byte ingest with content dedup;
  persists content-addressed uploads server-side under
  `OP_UPLOAD_ROOT`), `GET /curation/ingest/config` (served upload/batch
  limits and accepted extensions so a client stops hardcoding them),
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
- Run lineage recorded the frozen test-split hash as `lineage.dataset_sha`
  and nothing for multi-class exports; `code_versions.api_sha` /
  `trainer_image` were always null (read from the trainer's own env, which
  nothing set); the trainer's MLflow dataset tags read keys the job spec
  does not have and were always empty.
- The GPU arbiter did not see queued bake-offs on the default config
  (`bakeoff_jobs_dir` was unset while the router wrote to
  `<state_dir>/bakeoff_jobs`), and the router claimed hardcoded GPUs
  `0,1` and continued when containers could not be stopped: a GPU-resident
  container could be restarted under a running bake-off. The arbiter now
  defaults to the router's dir, and enqueueing answers 409 (job removed)
  when it cannot stop them.
- The bake-off evaluator could not read exports (no mount); compose now
  mounts `./data` read-only on `curation-evaluator`.
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
- Renamed the reference deployment's company-initialed environment-variable
  prefix to `OP_*` (23 vars) and its matching Prometheus metric-name
  prefix to `op_*`, closing the last reference-deployment naming
  residue in the config surface.
- **Region wire-contract leak**: `GET /curation/crops/{id}`
  returned the raw OpenSearch `_source` (`RegionFields` storage keys,
  `region_*` by default) instead of the frozen `ItemDoc` wire contract;
  `PATCH /crops/{id}/region_meta`'s `updated_fields` echoed
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
