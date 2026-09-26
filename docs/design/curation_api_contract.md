# OpenProcessor `/curation` API contract

Status: **living reference doc**, owned by this backend. It
documents the *currently shipped* `/curation` route surface (mounted
under `CurationConfig.api_prefix`, default `/curation`) and the
Pydantic wire-model field names it serves, and states explicitly which
parts of that contract are frozen.

**This is OpenProcessor's generic curation/labeling API contract, not
"the labeler's API."** Cropwright (a SvelteKit active-learning labeling
frontend) is **one consumer** of this API, not the sole audience. Other
services are anticipated on the same backend: querying,
visualizing, and searching the same indexed dataset. None of them exist
yet, and none of them should have to learn any one frontend's historical
URL vocabulary to consume this API. Every recommendation and naming
choice in this doc follows from that: the canonical surface is the
generic one (`/curation`, `vlm`, `region_thumbnail`,
`RegionFields`-backed storage), and it does not move to accommodate any
one consumer.

**There is no alternate prefix and no vendor-named route alias, and
none ever will.** The backend serves `{prefix}` (default `/curation`);
where a consumer's own naming differs from the generic one, the
consumer migrates to it. A deployment-side proxy alias is a frontend
concern only, never a supported backend default and never dual-mounted.

## The key invariant: one generic wire vocabulary, independent of storage names

Every request and response on this API uses one generic vocabulary
(see "B3" under Coordination notes for the full rename summary). An
earlier rule that froze historical, domain- and vendor-named wire names
is **retired**: fresh deployments re-ingest, so there was no legacy data
to protect.

1. **Region attributes go out as `region_<attr>`** for every
   `RegionFields` attribute (`region_bbox_norm`, `region_status`,
   `region_verified`, `region_detector_chain`, `region_text`,
   `region_visible`, `region_cluster_id`, `region_cluster_subid`, …).
   These wire names are **fixed**: they are the stock `RegionFields()`
   default names, and they do not move when a deployment overrides its
   OpenSearch storage names via `OP_REGION_FIELD_*`. The translation
   storage→wire happens once, at the boundary, in
   `src/services/curation/wire.py`; with stock defaults it is the
   identity. Storage config never leaks onto the wire (enforced by
   `tests/curation/test_wire_contract.py::test_storage_override_does_not_change_wire_keys`).
2. **VLM and classifier names are vendor-neutral**: `vlm_*` (never a
   model vendor's name) and `classifier_*` (never a model version).
3. **Every item-returning endpoint emits the same item** (see "Item wire
   format" below), built by one serializer, so a client parses one shape.

Request bodies follow the same rule: a body key that writes a
`RegionFields` attribute is named `region_<attr>`.

## Route surface

Full route list (123 distinct paths / 128 method routes under
`/curation` as of this wave — the latest additions are the frontend
logic-move routes: `GET /regions/statuses`, `POST /crops/{crop_id}/discard`,
`POST /crops/discard_batch`, `POST /crops/{crop_id}/vlm_dismiss`,
`POST /crops/{crop_id}/review_undismiss`, `GET /crops/{crop_id}/history`,
`GET /crops/{crop_id}/context`, `GET /review/{tab}/locate`,
`GET /review/new_class_proposals/summary`, `POST /vlm/label_cluster/{cluster_id}`,
`GET /training_cohorts`), grouped
by router module; every path is relative to the configured
`api_prefix`:

| Router module | Routes |
|---|---|
| `classes.py` | `GET /class_sources`, `GET,POST /classes`, `POST /classes/merge`, `POST /classes/sync_to_opensearch`, `GET,PUT /classes/{class_id}`, `POST /classes/{class_id}/deprecate`, `POST /classes/{class_id}/restore`, `GET /classes/{class_id}/crops` |
| `crops.py` | `GET /crops`, `GET /crops/{crop_id}`, `PUT /crops/{crop_id}/label`, `PUT /crops/batch_label`, `POST /crops/move`, `POST /crops/flag_new_class`, `POST /crops/batch_exclude`, `POST /crops/batch_unexclude`, `POST /crops/{crop_id}/review_dismiss` |
| `label_undo.py` | `POST /crops/{crop_id}/label/undo`, `POST /crops/label/undo_batch`, `DELETE /crops/{crop_id}/label`, `POST /crops/{crop_id}/discard`, `POST /crops/discard_batch`, `POST /crops/{crop_id}/vlm_dismiss`, `POST /crops/{crop_id}/review_undismiss`, `GET /crops/{crop_id}/history` |
| `crop_context.py` | `GET /crops/{crop_id}/context` |
| `edit_undo.py` | `POST /crops/{crop_id}/region/undo`, `POST /crops/region/undo_batch`, `POST /crops/{crop_id}/vlm_dismiss/undo` |
| `cohorts.py` | `GET /training_cohorts` |
| `regions.py` / `regions_edit.py` / `regions_fp.py` | `GET /regions`, `GET /regions/statuses`, `PUT /crops/{crop_id}/region`, `PUT /crops/batch_region`, `PATCH /crops/{crop_id}/region_meta`, `POST /regions/batch_status`, `POST /regions/cluster`, `GET /regions/cluster/status`, `GET /regions/clusters`, `POST /regions/clusters/refine/{cluster_id}`, `POST /regions/fp_centroids/build`, `GET /regions/fp_centroids/status`, `GET /regions/suspected_false_positives`, `GET /regions/training_candidates`, `GET /crops/{crop_id}/region_thumbnail` |
| `events.py` | `GET /events`, `POST /events/publish`, `GET /events/stats` |
| `export.py` | `POST /export/yolo`, `GET /export/datasets`, `GET /export/status`, `GET /export/registry/{artifact}` |
| `export_single_class.py` | `POST /export/single_class`, `GET /export/single_class/status` |
| `ingest.py` | `POST /ingest/image`, `POST /ingest/batch`, `POST /ingest/upload`, `POST /import_labels`, `POST /import_labels/batch`, `GET /ingest/status`, `GET /ingest/region_drain`, `POST /ingest/path_lookup` |
| `models.py` | `GET /health`, `GET /models/status`, `DELETE /models/{model_name}` |
| `search.py` | `GET /search/text` |
| `stats.py` | `GET /stats/classes`, `GET /stats/dataset` |
| `pipeline.py` / `pipeline_control.py` / `pipeline_events.py` | `POST /pipeline/auto_label`, `POST /pipeline/auto_label/start`, `GET /pipeline/auto_label/status`, `GET /pipeline/auto_label/status/{job_id}`, `POST /pipeline/auto_label/cancel`, `POST /vlm/label_cluster/{cluster_id}`, `GET /pipeline/events` |
| `clusters.py` / `viz.py` | `GET /clusters`, `GET /clusters/representatives`, `POST /clusters/auto_promote`, `POST /clusters/refine/{cluster_id}`, `GET,POST /viz/projection*`, `POST /cluster/umap/rebuild` |
| `review.py` / `scores.py` / `select.py` / `methods.py` / `settings.py` | `GET /review/{tab}`, `GET /review/{tab}/locate`, `GET /review/new_class_proposals/summary`, `POST /review/new_class_proposals/resolve`, `GET /review/raw_label_clusters`, `GET /review/unmatched_terms`, `POST /test_holdout/freeze`, `GET /test_holdout/stats`, `POST,GET /scores/*`, `POST,GET /select/*`, `GET /methods`, `GET,PUT /settings` |
| `vlm.py` | `POST /vlm/label_batch`, `POST /vlm/verify_regions`, `POST /vlm/verify_region_batch`, `POST /vlm/region_visible_batch` |
| `bakeoff.py` | `GET /bakeoff/{eval_datasets,trained_models,profiles,baseline_models,runs}`, `POST /bakeoff/run`, `GET /bakeoff/{status,results,matrix}/{job_id}` (typed, schema v2; see "Model comparison" below) |
| `curation_images.py`, `curation_train.py`, `curation_umap.py` (outside the `curation` package, registered directly in `src/main.py`) | `GET /images/*`, `POST,GET /train/*`, `POST /cluster/umap/rebuild` |

The exact, always-current list is produced by:

```python
from src.main import app
routes = sorted(r.path for r in app.routes if r.path.startswith('/curation'))
print(len(routes)); print('\n'.join(routes))
```

Note the URL segment is `vlm`, not any one vendor's model name — the
labeling client is a pluggable `vlm_client`/`vlm_labeler`/`vlm_prompts`
abstraction over any OpenAI-compatible endpoint. No vendor-named route
is registered, and none will be added; see the VLM section below.

## Wire models

Model class names reflect `src/routers/curation/_common.py`; this table
is hand-maintained (see D3) — treat `_common.py` as authoritative if the
two disagree. `ItemDoc`'s field set is test-pinned to the serializer's
output (`test_item_doc_model_documents_exactly_the_serializer_keys`).

### Ingest

- `IngestImageRequest`: `path`, `source` (F-22: `extra='forbid'` -- an unknown key 422s instead of silently ingesting on defaults)
- `IngestImageResponse`: `status` (`success`/`duplicate`/`failed`), `image_id`, `image_path`, `imohash`, `n_crops`, `n_regions`, `error`, `secondary_detector_error` (F-43: set when a configured secondary detector call failed for this image -- the image still ingests successfully on the primary detector's output alone)
- `BatchIngestSummaryResponse`: `successful`, `duplicates`, `failed`, `mismatches`, `missed_labels`, `unmatched_detections`, `labels_imported`, `crops_indexed`, `secondary_detector_failures` (F-43: count of otherwise-successful images where the configured secondary detector call failed, e.g. a Triton `DEADLINE_EXCEEDED` -- previously only a `warning` log line, invisible on the wire)
- `BatchIngestResponse`: `status` (`success`/`partial`/`error`), `summary`, `results`, `disagreements` (with `detect_mismatches`: one record per model-vs-label disagreement, `kind` = `class_mismatch`/`missed_label`/`unmatched_detection`; also returned by `POST /import_labels/batch`)
- D3 (2026-09-25 F8 acceptance): dedup applies **within** a batch, not just against the index. Two byte-identical files (`imohash`) uploaded in the same request collapse onto one representative -- only the first ingests; every later same-hash upload in the batch reports `status: 'duplicate'` with `image_id` set to the representative's `image_id` (the same field the cross-batch duplicate path uses), and exactly one item is created. Previously `hash_to_existing` only resolved hashes already committed to the index before the batch started, so two identical files in one request both ingested `'success'` with `duplicates: 0`.
- `POST /ingest/upload` (multipart): `images` (files), `image_paths` (JSON list of identifiers, optional), `source` -> `BatchIngestResponse`
- `IngestBatchRequest`: `items` (F-22: required, non-empty; `extra='forbid'` -- a wrong key like `paths` used to 200 with all-zero counts instead of 422)
- `ImportLabelsRequest`: `image_path`, `label_txt_path`, `label_source` (F-22: `extra='forbid'`)
- `ImportLabelsBatchRequest`: `items` (F-22: required, non-empty; `extra='forbid'`)

### Crops

- `ItemDoc`: the shared wire item — see "Item wire format" below for the exact key list. Documentation/OpenAPI model only: handlers return the serializer's dict directly, so an unexpected stored value type never 500s a browse page.
- `CropsPageResponse`: `total`, `page`, `page_size`, `crops` (list of items), `method`, `version`, `n_pool`
- `CropLabelRequest`: `class_id`, `label_source` (`human` default or `human_confirmed` — any other value is a `422`; the server always writes `class_source: "human"` for this write, so a client can't make a human label look machine-written)
- `CropBatchLabelRequest`: `crop_ids`, `class_id`, `label_source` (same rule). Response: `updated`, `updated_ids` (exactly the crops written — the ids to pass to `undo_batch`), `conflicts` (`[{crop_id, current_source}]`, not written)
- `CropMoveRequest`: `crop_ids`, `cluster_id`. Response: same shape as `batch_label` (`updated`, `updated_ids`, `conflicts`)
- `CropExcludeRequest`: `crop_ids`, `reason`
- `CropUnexcludeRequest`: `crop_ids`
- `CropUndoBatchRequest` (`POST /crops/label/undo_batch`): `crop_ids`
- `ItemRegionRequest` (`PUT /crops/{crop_id}/region`): `region_bbox_norm` (`[x1,y1,x2,y2]` in `frame`, or `null` = "no region visible"), `region_label_source` (default `human`), `frame` (`source` default = source-image frame; `parent` = the item crop's own frame, projected server-side through the item's stored `bbox_norm`, `422` if the item has none). Stored boxes are always source-frame (`region_bbox_frame: "source"`). A box equal to the stored one (each coordinate within `1e-4`, after projection) is a **confirmation**: status/verified/validated/verifier are written and `region_detector`, `region_detector_version`, `region_score`, `region_detected_at` are kept; any other box is human geometry (`region_detector` = the human, `region_score` 1.0). Response: `crop_id`, `region_bbox_norm`, `region_status`, `item` (the post-write wire item).
- `ItemBatchRegionRequest` (`PUT /crops/batch_region`): `crop_ids`, `region_bbox_norm`, `region_label_source`, `frame` (`parent` projects through each item's own box; items without one land in `invalid`). Response: `updated`, `conflicts`, `invalid`, `items` (post-write wire items of the updated crops).
- `CropBatchStatusRequest` (`POST /regions/batch_status`): `crop_ids`, `region_status`, `region_label_source`; `region_status` must be human-writable (see "Region lifecycle" below). `region_verified` is still accepted but **ignored** (deprecated): the server derives it. Response: `updated`, `conflicts` (`[{crop_id, current_source}]`), `invalid` (`[{crop_id, detail}]`, e.g. `detected` on a crop with no box), `items` (post-write wire items).
- `ItemRegionMetaRequest` (`PATCH /crops/{crop_id}/region_meta`): `region_text`, `region_status`, `region_rejection_reason`, `region_label_source` (all optional; only provided fields are written). Response: `crop_id`, `updated_fields` (wire names, e.g. `["region_status", "region_text"]`), `item` (post-write wire item). `422` when the status write would break an invariant (`detected` with no box).
- All four region request models set `extra='forbid'`: a stale key (a pre-rename region-attribute name, `label_source`, …) is a `422`, never a silent no-op.
- `CropFlagNewClassRequest`: `crop_ids`, `note`

### Training cohorts — `GET /training_cohorts?class_id=`

`{cohorts: [{id, label, description, cutoffs, endpoint, params, row_kind}]}`
(source: `src/services/curation/training_cohorts.py`). Fetch a cohort's
rows with `GET {prefix}{endpoint}` + `params` (`class_id` already folded
in). Core cohorts (always): `validated`, `needs_labeling`,
`low_confidence` (`cutoffs: {classifier_conf_lt: 0.75}` — the backend's
review band; the frontend's `0.5` is gone), `model_disagreements`
(`row_kind: crop`). With a region profile configured, the
`/regions/training_candidates` modes follow (`row_kind: region`,
`params.mode`): `detector_blind_spots`, `low_conf_correct` (`cutoffs:
{region_score_lt: 0.6}`), `disagreement`, `human_corrected`,
`false_positives`; each `description` is exactly the `selection_reason`
that endpoint returns.

### Per-class dataset thresholds

One definition (`src/services/curation/dataset_thresholds.py`), enforced by
the training preflight and served wherever a client shows class counts:

```json
"thresholds": {"block_below": 20, "warn_below": 500, "min_test_per_class": 5,
               "min_train_per_class": 1, "min_val_per_class": 1,
               "aug_target_min": 500, "aug_target_max": 3000}
```

`min_train_per_class` / `min_val_per_class` are the per-class instance
minimums of the `export_class_split_coverage` preflight check (see
"Export" below).

- `adequacy` (`ok` / `warn` / `block`) of a class's validated count:
  `< block_below` → `block` (preflight refuses), `< warn_below` → `warn`,
  else `ok`. The frontend's old `100` "critical" line has no backend
  meaning and is gone.
- `aug_target` = validated count clamped to `[aug_target_min,
  aug_target_max]`; `aug_gap` = `aug_target - validated_count`.

| Endpoint | Adds |
|---|---|
| `POST /train/preflight` | `thresholds` |
| `GET /stats/classes` | `thresholds`; per row `adequacy`, `aug_target`, `aug_gap` |
| `GET /classes` | `thresholds`; per class `adequacy` |
| `GET /test_holdout/stats` | `min_test_per_class`; per `by_class` bucket `deficient` (`doc_count < min_test_per_class`) |

### Augmentation presets — `GET /train/augmentation_presets`

The one preset catalog is `src/services/training/augmentation_presets.py`;
the trainer image copies that file next to `docker/trainer/augment.py`,
which builds its `PRESETS` from it. Response
(`AugmentationPresetsResponse`): `presets[]` of `{id, label, description,
orientation_sensitive}` (in display order) and `default`
(`"balanced_default"`, used when a job omits `augmentation.preset`).
`orientation_sensitive: true` means horizontal flip is off for the whole
run. A client renders this list and never hardcodes ids.

An enabled `augmentation` block naming any other `preset`:

- `POST /train/preflight` → check `augmentation_preset`, severity
  `block`, message `unknown augmentation preset '<id>'; valid presets:
  none, balanced_default, …`, `detail: {preset, valid_presets}`
  (otherwise `ok`; a disabled block isn't judged);
- `POST /train/start` and `POST /train/start_campaign` → `422` with
  `detail: {message, field: "augmentation.preset", valid_presets}`,
  even with `force=true`, before any GPU claim or job write.

The other `AugmentationSpec` fields aren't enumerable here:
`albumentations` override keys are Albumentations transform names (the
trainer logs and skips unknown ones), and `multiplier` is range-checked
(`1..20`) by the model.

### Training `eval` block — `GET /train/status*`, `GET /train/manifest/{job_id}`

`status.json`'s (and the manifest's `results.eval`) `eval` block reports the
result of the trainer's post-training evaluation, and is deliberately
explicit about which split every number came from:

```json
{
  "map50": 0.62,
  "map50_95": 0.41,
  "precision": 0.71,
  "recall": 0.55,
  "split": "test",
  "val_last": {"map50": 0.9191, "map50_95": 0.742},
  "per_class": [{"class_id": 0, "name": "widget", "precision": 0.8,
                 "recall": 0.7, "f1": 0.75, "ap50": 0.79, "support": 12}],
  "confusion_matrix_url": "/curation/train/artifacts/<job_id>/confusion_matrix.png"
}
```

- `map50` / `map50_95` / `precision` / `recall` / `per_class` are the
  **frozen test-split** numbers (a fresh `model.val(..., split='test')` pass,
  Ultralytics' `DetMetrics.box.{map50,map,mp,mr}` + per-class rows) whenever
  that pass ran and produced usable metrics. `split: "test"` marks this case.
- When the test pass fails or the export has no `test` split, the same four
  overall keys instead carry the **training-time validation** numbers (the
  last row of `results.csv` — Ultralytics' per-epoch model-selection metric,
  recorded every epoch against the `val` split) and `per_class` is absent.
  `split: "val"` marks this case — a consumer MUST check `split` before
  treating `map50`/`map50_95` as "how the model does on unseen data."
- `val_last` (`{map50, map50_95}`) is **always** present when `results.csv`
  had a row, regardless of `split` — the training-time validation numbers,
  unambiguously named, for a consumer that specifically wants the training
  curve rather than the headline metric.
- `confusion_matrix_url` points at `GET
  {api_prefix}/train/artifacts/{job_id}/{name}` (whitelisted filenames only:
  `confusion_matrix.png`, `confusion_matrix_normalized.png`, `results.png`,
  `results.csv`, `BoxP_curve.png`, `BoxR_curve.png`, `BoxF1_curve.png`,
  `BoxPR_curve.png`) or `null` when the trainer never wrote one. The
  underlying server filesystem path is never on the wire.
- The promote gate (`POST /train/promote/{job_id}`, §15.2) reads this same
  block — a run whose test pass failed (`split: "val"`, no `per_class`)
  fails the gate's per-class check outright rather than silently passing on
  val-split numbers relabeled as test.

`mlflow_run_url` on the same payloads is rebuilt from `CurationConfig.
mlflow_public_url` (`OP_MLFLOW_PUBLIC_URL`) + `mlflow_run_id` +
`mlflow_experiment_id`; `null` when the public base isn't configured or
either id is missing (older run) — the trainer's internal tracking-server
hostname (`MLFLOW_TRACKING_URI`, a container name unreachable from a
browser) never reaches the wire. `mlflow_run_id` is served either way.

### VLM-label one cluster — `POST /vlm/label_cluster/{cluster_id}`

Queues the auto-label job (same job, same `GET /pipeline/auto_label/status`
/ `POST /pipeline/auto_label/cancel`, `409` while one runs) scoped to one
cluster with only the VLM stage: no re-clustering, no auto-promote, no cap.
The server selects **every** unvalidated, non-holdout, non-excluded member
(the global sweep's cost skips — classifier-confident, `vlm_unmatched`,
recently combined-classified, a recent empty class answer — don't apply to
an explicit request) and
chunks them itself. Optional `?prompt_pack=`. Response: the job state
(`args.cluster_id` echoes the scope). While running, `total` is the number
of members selected; on completion `result.stages.unvalidated_after_promote`
is that count, `result.stages.vlm` `{predicted, updated, …}`, and
`result.unvalidated_remaining` counts what is still unvalidated in the cluster.
The same scope is available as `?cluster_id=` on `POST
/pipeline/auto_label[/start]`. A cluster-scoped job writes **only** to the
members it selected: the index-wide stages (`cluster_id_normalize`,
`cluster_residuals`, `auto_promote`) are skipped even if requested
(`result.stages.<stage>` = `{skipped: true, reason: "cluster-scoped run: …"}`),
and the post-VLM `cluster_id = class_id` pass runs on the selected items
only (same for a `?class_id=`-scoped VLM stage).

### Auto-label job by id — `GET /pipeline/auto_label/status/{job_id}`

Poll the job a client started with the `job_id` its start response
returned. Same body as `GET /pipeline/auto_label/status` (`job_id`,
`status` `queued|running|completed|failed|cancelled|interrupted`, `stage`,
`processed`, `total`, `started_at`, `finished_at`, `error`,
`error_detail`, `result`, `args`, `pipeline`, backend/VRAM telemetry,
`stage_durations`, `eta_seconds`, `elapsed_seconds`). The current job is
read live; a job replaced by a later start answers with its final state
(the newest 50 are kept). `404` for an id no job had (or not a 32-hex id).

### Region shape warnings — deliberately none

The region cascade's geometry gate (`is_plausible_region_bbox`,
`src/services/detection/cascade_detect.py`) is geometry-only by design: an
aspect-ratio / size envelope built from one domain's assumptions rejects
legitimate regions from other domains. `DetectionProfile` carries no
review-time shape envelope (its `aspect_*` / `auto_confirm_*` bands drive
the OCR text-hint and the VLM-skip auto-confirm, not a verdict on a stored
box), so the API serves **no** `region_shape_warning`. A client should not
flag boxes by a shape prior of its own either.

### Region lifecycle — `GET /regions/statuses`

The single source is `REGION_STATUS_INFO` in `src/config/region_state.py`
(also emitted to `contracts/ts/regionStatus.ts` by the codegen:
`HUMAN_REGION_STATUSES`, `REGION_STATUS_ROLE`,
`CONFIRM_STATUS_VALUE`, `REJECT_STATUS_VALUE`, `FALSE_POSITIVE_STATUS_VALUE`).

```json
{"statuses": [{"value": "no_region_visible", "label": "no region visible", "role": "absent",
               "terminal": true, "human_writable": true, "clears_box": true, "wants_reason": true}, ...],
 "confirm_status": "detected", "reject_status": "no_region_visible",
 "false_positive_status": "false_positive"}
```

`statuses` lists every `RegionStatus` in enum order. `role` is one of
`pending`, `positive`, `rejected`, `absent`, `false_positive`, `failed`.
`wants_reason`: the UI may offer `region_rejection_reason` for it.

Every human region writer (`PUT /crops/{id}/region`, `PUT
/crops/batch_region`, `PATCH /crops/{id}/region_meta`, `POST
/regions/batch_status`) enforces, server-side:

- a status with `clears_box` (`no_region_visible`) clears `region_bbox_norm`
  and `region_score`, whichever writer set it;
- `region_verified` = (`region_status` == `confirm_status`), never taken
  from the request (`detected` → `true`, every other human status → `false`);
- `detected` on a crop with no box is refused (`422` single / `invalid[]` batch)
  — unless the crop carries a verifier-rejected candidate
  (`region_candidate_bbox_norm`, below): then `detected` (and
  `false_positive`) promotes the candidate into `region_bbox_norm`, taking
  `region_score` / `region_detector` / `region_detector_version` /
  `region_source` from the `region_candidate_*` fields, clearing them and
  `region_rejection_reason`. `PUT /crops/{id}/region` with the candidate's
  box is the same confirmation (provenance kept); any other box replaces
  the candidate;
- human writes set `region_validated=true`; `false_positive` parks the region
  in the FP cluster, any other status releases it;
- a write that re-asserts the stored status re-derives nothing:
  `region_verified` and the region-cluster placement stay as stored
  (confirming still sets `region_verified=true`);
- every write snapshots the pre-write region state for undo (see "Undo of
  region writes" below).

Each returns the post-write item, so a client adopts it rather than
re-deriving the result.

### Undo of region writes and VLM dismissals

Every human region writer snapshots the item's pre-write region state
(`region_bbox_norm`, `region_bbox_frame`, `region_status`, `region_score`,
`region_verified*`, `region_verifier*`, `region_validated`,
`region_label_source`, `region_detector*`, `region_detected_at`,
`region_rejection_reason`, `region_text*`, `region_cluster_*`) into the
item's `edit_history` (stored, not indexed, not on the wire; see
`src/services/curation/edit_history.py` for why it is a kind-tagged list
separate from `class_id_history`).

- `POST /crops/{crop_id}/region/undo` — restore the region to its state
  before the most recent not-yet-undone human region write (confirm,
  reject, false positive, box edit, status or text change). Repeated calls
  step back. Class fields are untouched. Response: the restored item.
  `404` unknown crop; `409` nothing left to undo.
- `POST /crops/region/undo_batch` (`CropRegionUndoBatchRequest`:
  `crop_ids`) — the same per crop; undo a `batch_status` / `batch_region`
  by passing the same `crop_ids`. Response: `items`, `undone`,
  `nothing_to_undo`, `conflicts`, `not_found`. `409` when no crop had
  anything to undo.
- `POST /crops/{crop_id}/vlm_dismiss/undo` — put the `vlm_dismissed_*`
  fields back to their state before the latest `vlm_dismiss`, so the
  dismissed suggestion is live again. Response: the restored item. `409`
  no dismissal to undo.

### Undo of human class writes

Every human class write — `PUT /crops/{crop_id}/label`,
`PUT /crops/batch_label`, `POST /crops/move` — appends a full snapshot of
the item's pre-write class state to `class_id_history` (`class_id`,
`class_name`, `class_source`, `label_source`, `confidence`,
`class_detector`, `class_detector_version`, `class_labeler`,
`class_labeled_at`, `class_validated`, `cluster_id`, `cluster_subid`,
with `restorable: true`). The undo routes restore that snapshot; the
frontend never decides between re-applying an earlier label and
reverting — it calls undo and renders the returned item.

- `POST /crops/{crop_id}/label/undo` — restores the crop to its state
  before its most recent not-yet-undone human class write, whatever that
  was (an earlier validated human label, a VLM suggestion, an ingest
  proposal, unlabeled). Repeated calls step back through successive human
  writes (each undo cancels one write). Response: the restored item
  (shared wire format). `404` unknown crop; `409` nothing left to undo.
- `POST /crops/label/undo_batch` (`CropUndoBatchRequest`) — the same,
  per crop, independently; undo a `batch_label` / `move` by passing the
  same `crop_ids`. Response: `items` (restored wire items), `undone`,
  `nothing_to_undo`, `conflicts`, `not_found` (crop id lists). `409` when
  no crop had anything to undo.
- `DELETE /crops/{crop_id}/label` — kept for compatibility; same restore,
  but with nothing on record it resets the crop to unlabeled (class and
  provenance cleared, nothing invented) instead of `409`. Response:
  `crop_id`, `reset`.

- `POST /crops/{crop_id}/discard` (`CropDiscardRequest`: `clear_class`
  default `true`, `dismiss_from_review` default `false`; `422` if both are
  false) — a **recorded** human write. `clear_class` clears class,
  provenance and validation and drops the item to the residual pool
  (`cluster_id: null`); `dismiss_from_review` stamps
  `review_dismissed_at`/`review_dismissed_by` so every `/review` tab hides
  it. Response: the post-write item. `POST /crops/discard_batch`
  (`CropDiscardBatchRequest`: `crop_ids` + the same flags) → `items`,
  `discarded`, `conflicts`, `not_found`.

Which one to call:

| Action | Route | Recorded (undoable)? |
|---|---|---|
| Label / confirm | `PUT /crops/{id}/label`, `PUT /crops/batch_label`, `POST /crops/move` | yes |
| Discard (clear the class and/or hide from review) | `POST /crops/{id}/discard`, `POST /crops/discard_batch` | yes — undo restores class, placement and review visibility |
| Undo the last recorded write | `POST /crops/{id}/label/undo`, `POST /crops/label/undo_batch` | is itself the undo; repeated calls step back |
| `DELETE /crops/{id}/label` | legacy undo (same restore; resets to unlabeled when nothing is on record) | no — it *is* an undo, so Z can't reverse it |
| `POST /crops/{id}/review_dismiss` | legacy one-way review hide | no — use `discard` with `clear_class: false, dismiss_from_review: true` instead |

- `POST /crops/{crop_id}/vlm_dismiss` — reject the VLM's class
  suggestion: stores `vlm_dismissed_class_id` / `vlm_dismissed_class_name`
  / `vlm_dismissed_at`; while the VLM's suggestion is that one, the
  suggestion keys are `null` and `proposed_class_*` no longer apply it
  (`null` / `""`). A different later VLM suggestion shows again. The class
  itself is untouched (label or discard separately). Response: the item.
  `409` no suggestion, `404` unknown crop.
- `GET /crops/{crop_id}/history` → `{crop_id, entries}`: `class_id_history`
  oldest first, each entry the class state *before* one write
  (`class_id`, `class_name`, `class_source`, `label_source`, `confidence`,
  `class_detector`, `class_detector_version`, `class_labeler`,
  `class_labeled_at`, `class_validated`, `cluster_id`, `cluster_subid`,
  `review_dismissed_at`, `review_dismissed_by`) + `writer` (e.g.
  `human:label_crop`, `human:discard_crop`, `vlm_pipeline`) + `at`;
  unrecorded keys are `null`.

- `POST /crops/{crop_id}/review_undismiss` — clear
  `review_dismissed_at`/`review_dismissed_by` (back into the review
  queues); for a `discard`-made dismissal `label/undo` does this *and*
  restores the class. Response: the item. List hidden items with
  `GET /crops?review_dismissed=true`; every item carries
  `review_dismissed_at`.
- `GET /crops/{crop_id}/context` → `{image: {image_id, image_path, width,
  height, source, indexed_at} | null, items: [...]}`: the source frame and
  every item detected in it (wire items, `crop_rank_in_image` ascending,
  max 500).

Cluster placement on restore: a restored validated class sits in its
class cluster (`cluster_id == class_id`, keeping the recorded
`cluster_subid` only if it belonged to that cluster); anything else goes
back to the cluster recorded before the write (`null` = residual pool).
Undo on an excluded crop keeps it excluded and stores the restored
validation/placement as the state `batch_unexclude` will apply.

Human writes made before snapshots were recorded carry only
`class_id`/`class_name`/`class_source`/`label_source`/`confidence`;
undo restores those and leaves the rest `null`/unvalidated.

### Exclude / un-exclude

`POST /crops/batch_exclude` sets `class_excluded`, moves the crop to
cluster `-2` and clears `class_validated`, recording the prior
validation and placement in `excluded_prior_class_validated`,
`excluded_prior_cluster_id`, `excluded_prior_cluster_subid` (re-excluding
keeps the first record). `POST /crops/batch_unexclude` restores them: a
validated crop returns to `cluster_id == class_id` (sub-cluster kept if
it was in that cluster); an unvalidated one drops to the residual pool
(`cluster_id: null`). Crops that aren't excluded are left untouched.
Response shapes unchanged (`excluded`/`unexcluded`, `errors`; a
concurrent-write conflict counts as an error).

### Cluster cards (`GET /clusters`)

`labelled_count` is the number of members with any `class_name`;
`dominant_count` / `label_purity` describe the top class among them
(`label_purity` = `dominant_count / labelled_count`), and
`labelled_share` = `labelled_count / size`. For a
candidate cluster (`cluster_id >= cluster_id_offset`)
`dominant_class_name` is set only for a unique top class with at least
3 members and at least half of the labelled members
(`CANDIDATE_DOMINANT_MIN_COUNT` / `CANDIDATE_DOMINANT_MIN_SHARE` in
`src/routers/curation/clusters.py`), else `null`; `dominant_class_id` is
always `null` for candidates. Class clusters report their top class as
before.

**`purity` (DQ-M2, changed meaning).** Label purity is 1.0 on every
class cluster by construction (`cluster_id == class_id`), so it used to
call visibly mixed class clusters "pure", and on candidates it covered only
the few labelled members. A card's `purity` is now a geometric signal
independent of the labels: the share of the cluster's members whose
nearest cluster centroid (among every cluster's member-mean centroid over
the item embedding) is their own cluster's. It is computed by the
cluster-geometry pass that follows every auto-label clustering stage
(`cluster_nearest_id` per item) and counts only members measured for
their current cluster. `purity_n` is how many members it was computed
over, `purity_basis` is `"nearest_centroid"`; `purity`, `purity_tier` are
`null` while `purity_n` is `0` (no pass since the members arrived).

Each card also carries `purity_tier` (`pure` / `mixed` / `noisy` from
`purity`) and `promotable` (the auto-promote gate — unchanged, on the
labels: at least `promote_min_members` members, at least
`promote_min_labelled_share` of them labelled, `label_purity` at least
`pure_min`; never true for a class cluster). The response serves the cut
points: `purity_thresholds: {pure_min: 0.85, mixed_min: 0.6,
promote_min_members: 4, promote_min_labelled_share: 0.5}` (source:
`src/services/curation/cluster_purity.py`; the same `pure_min` / `mixed_min`
cut both `purity` into tiers and `label_purity` at the gate) and
`core_similarity_min: 0.75` (the cut line for the items' `cluster_is_core`).
`POST /clusters/auto_promote` counts every labelled member in the purity
denominator (it used to count only the top-5 classes, overstating purity
on many-class clusters).

**Representatives are now paged (D-4, breaking change).**
`GET /clusters` used to attach representative crops to *every* returned
card via a `top_hits` sub-aggregation, which decompressed stored
`_source` for every representative across every bucket in the response
regardless of what the client actually displayed. It now returns every
card (still up to `max_clusters`, still carrying `size`/`purity`/etc.)
but only fills in `representatives` for cards in the
`[representatives_offset, representatives_offset + representatives_limit)`
window of the *returned, kind-filtered, `_count`-desc-ordered* card
list — new query params `offset` (default `0`) and `limit` (default
`50`, max `500`). Cards outside that window still carry the
`representatives` key, but as an empty list `[]` — the field never
disappears, so existing clients that only read `card.representatives`
degrade to "no thumbnails for this card" rather than a KeyError. The
response also now reports `representatives_offset` /
`representatives_limit` so the frontend knows which window was served.
Passing `per_cluster=0` (as before) skips representative computation
entirely — no `_msearch` is issued.

Representatives are computed by one `_msearch` (one query per cluster
in the window, each `{size: per_cluster, query: {bool: {filter:
[{term: {cluster_id}}], must_not: [{term: {class_excluded: true}}]}},
_source: [crop_id, cluster_distance, class_name, cluster_subid], sort:
[{cluster_distance: asc}, {crop_id: asc}]}`) instead of a per-bucket
`top_hits` sub-agg on the cards aggregation itself.

**Frontend action required:** paginate the cluster grid by requesting
successive `offset`/`limit` windows (matching whatever page of cards is
actually rendered) rather than assuming every card in one `GET
/clusters` response already carries thumbnails.

`GET /clusters/representatives` has the same shape change: the
`clusters` dict in the response now only contains keys for cluster ids
in the `[offset, offset + max_clusters)` window (ordered by member
count desc) — call again with a larger `offset` for the next page. New
`offset` query param (default `0`); response gains `offset` and
`max_clusters` fields. Previously this endpoint returned representatives
for every cluster (up to `max_clusters` total) in one response with no
paging concept at all.

`GET /crops` query parameters: `page` (≥1), `page_size` (1–500, default
50), `limit` (1–500; alias for `page_size`, wins when both are set),
`sort` (`'<field>[:asc|desc]'`, default `updated_at:desc`; fields
`updated_at`, `created_at`, `confidence`,
`crop_rank_in_image`, `crop_area_norm`, `blur_lap_ratio`,
`cluster_distance`, `mistakenness_score`, `uniqueness_score`; anything
else is a `400`; ignored by `order=outliers|diverse`), `class_id`,
`cluster_id`, `label_source`, `class_source`, `label_validated`,
`source` (ingest source tag; the retired short-name query param is
removed, S2),
`needs_new_class` (bool), `review_dismissed` (bool), `ids` (comma-separated, max 500: returns exactly
those items in that order, missing ids dropped, every other filter ignored —
use it to hydrate a `POST /select/diverse` page in one call),
`include_test`, `include_excluded`, `max_rank`,
`min_blur_ratio`, `classifier_conf_lt`, `conf_min` / `conf_max`
(inclusive band on `confidence`, `400` if min > max), `order`
(`default`/`outliers`/`core_first`/`diverse`; `outliers` and `core_first`
need `cluster_id`: members farthest from / nearest to the centroid of the
matched members first. Under `core_first` each served item's
`cluster_distance` / `cluster_similarity` / `cluster_is_core` is recomputed
against that same live centroid, so the cluster view's cut line — the
first item with `cluster_is_core: false` — always matches the order;
`method` reports the order that ran, and a pool too large to rank falls
back to `sort`), `k` (1–10000, `order=diverse` only:
rank just the first `k` k-center-greedy picks; `total` is then `k`),
`item_text` (≤200 chars; text read on the item crop — every letter/digit
word of the query must be a case-insensitive prefix of one of the item's
`item_text_tokens`, e.g. `smith mot` matches an item whose OCR read
`Smith Motors`, `abc1234` matches `ABC-1234`; a query with no letter or
digit is a `400`).

### Classes

- `ClassEntry`: `class_id`, `class_name`, `group`, `sample_count`, `validated_count`, `cluster_size`, `deprecated`, `hotkey_letter`, `adequacy`, `added_at` (from the registry)
- `ClassListResponse`: `classes`, `thresholds`, `reserved_hotkeys` (sorted single keys no class may bind: `/ a b d e f g m n u x z` — the labeling actions, the class picker and the region-review keys)
- `ClassCreateRequest`: `name`, `group`, `notes`, `hotkey_letter` (optional)
- `ClassUpdateRequest`: `name`, `group`, `hotkey_letter`
- Class names must match `^[a-z0-9_]+$` on create and rename (`422` otherwise; they become export/training class names). Hotkeys, on create and update: one character (`400`), not reserved (`422`), not bound to another active class (`409`); `""` on update clears. A create that fails any hotkey rule writes nothing.
- `ClassMergeRequest`: `source_id`, `target_id`. `POST /classes/merge?dry_run=true` writes nothing and returns `{dry_run: true, source_id, target_id, would_relabel, validations_carried_over, holdout_blocking, blocked}` — `would_relabel` counts every non-holdout item of the source class (validated or not); `validations_carried_over` the validated ones among them (F-56 follow-up, 2026-09-25: a merge relabels with `class_source: class_merge` and KEEPS `class_validated` — a human-validated crop of the source class stays validated under the target, so this field counts who carries validation over, not who loses it, and this is the field's replacement for the retired `would_unvalidate`); `blocked` = the real merge would `409` on frozen test-holdout items. `400` for an unknown id or a self-merge. `class_id_history` records the merge (`writer: 'class_merge'`) with a snapshot of the prior `class_id`/`class_source`/`class_validated`, so the pre-merge validation state is on the audit trail either way. `POST /classes/{id}/restore` still `409`s on a merged class (`merged_into` set) — un-merging isn't supported; relabel the crops back manually instead.
- `GET /class_sources` -> `{"class_sources": [{"id", "label", "role", "short_label"}, ...]}` — see "`class_source` values" below
- `POST /classes/{class_id}/deprecate` -> `RegistryClassEntry` (`class_id`, `class_name`, `group`, `sample_count`, `validated_count`, `added_at`, `deprecated`, `notes`, `merged_into`, `hotkey_letter`). Retires a class with no target and no data — the direct counterpart to `POST /classes/merge`, which needs both. Counts items-index docs (`term: class_id`) and confirmed-labels-index docs (`term: class_id`, same field the merge relabel touches); `409` while either count is nonzero, body `{error: "class_still_referenced", message, class_id, item_count, confirmed_label_count}` naming merge as the alternative. `404` unknown id. Idempotent — re-deprecating an already-deprecated class returns it unchanged without re-checking references. Clears `hotkey_letter` so the freed letter can't collide with a class bound to it later.
- `POST /classes/{class_id}/restore` -> `RegistryClassEntry`. Clears `deprecated`. `404` unknown id; `409` (plain-string detail) if a non-deprecated class already holds this class's `class_name` — the same name-uniqueness rule `rename_class`/`add_class` enforce. `409` (structured detail `{error: "class_merged", message, class_id, merged_into: {class_id, class_name}, hint}`) if the class was merged via `POST /classes/merge` (`merged_into` set) — its crops already live on the merge target, so restoring it directly would resurrect an empty class while the data stays put; relabel the crops back manually instead. A plainly deprecated class (no `merged_into`) restores as before.

### VLM labeling/verification

Registered at `POST {prefix}/vlm/*` (`src/routers/curation/vlm.py`).
Every VLM-related name on the wire is `vlm_*` — URL segment, stored and
returned fields (`vlm_confidence`, `vlm_raw_label`, …), `class_source`
values (`vlm`, `vlm_unmatched`, …), the review tab `vlm_low_conf`, the
auto-label params and the stats keys (see B3).

- `VlmLabelBatchRequest` (`POST /vlm/label_batch`): `crop_ids`. A reply that
  resolves to a registry class also sets `cluster_id = class_id` (and clears
  `cluster_subid`) unless the item is excluded, as the worker's combined call
  and the pipeline's normalize do (DQ-m3) — the item no longer waits in its
  candidate or old class cluster for the next clustering run. Undo restores
  the prior placement.
- `VlmVerifyRegionsRequest` (`POST /vlm/verify_regions`): `crop_ids`. A
  crop_id the VLM gave no usable answer for (upstream failure, empty or
  unparseable reply) is skipped — its verify state is left untouched for
  a retry rather than written as `verified=False`.
- `VlmVerifyRegionBatchItem`: `crop_id`, `region_image_b64` (base64 JPEG of the region crop, no `data:` prefix), `candidate_text` (optional, upstream OCR hint, echoed back not consumed)
- `VlmVerifyRegionBatchRequest` (`POST /vlm/verify_region_batch`): `items: list[VlmVerifyRegionBatchItem]`
- `VlmVerifyRegionBatchResult`: `crop_id`, `is_region`, `confidence`, `reason`, `candidate_text`
- `VlmVerifyRegionBatchResponse`: `results` — a `crop_id` the VLM gave no
  verdict for (whole-chunk upstream failure, empty/unparseable/misaligned
  reply, or an individual crop missing from an otherwise-aligned reply)
  is absent from `results` entirely, the same "omit, don't reject"
  contract `/vlm/region_visible_batch` uses for its `visible` map.
- `VlmRegionVisibleBatchItem`: `crop_id`, `image_b64`
- `VlmRegionVisibleBatchRequest` (`POST /vlm/region_visible_batch`): `items`
- `VlmRegionVisibleBatchResponse`: `visible` (`dict[str, bool]`, keyed by `crop_id`)

### Review / holdout

On the `mismatches` tab each item's `reason` says why it is there
(DQ-m4): the default "VLM's reply did not match any registry class";
`VLM named registry class '<answer>' at <vlm_confidence> confidence; not
applied` when the VLM's answer (`vlm_raw_class`, else `vlm_raw_label`) is an
active registry class name (a low-confidence answer the label path routes
to review); `VLM gave no class answer` when none is stored.

`vlm_low_conf` selects items whose label came from the VLM (a VLM
`class_source`) and whose `vlm_confidence` is `medium` or `low`. It no
longer also requires `confidence < 0.80` (that is the detector/classifier
score — DQ-M8).

`GET /review/{tab}` tabs: `all`, `mismatches`, `vlm_low_conf`, `outliers`,
`uncertainty`, `model_disagreements`, `regions`, `primary_low_conf`,
`classifier_blind_spots`, **`new_class_proposals`** (items flagged
`needs_new_class` by a human, or `class_source: vlm_new_class_pending`).
Filters (every tab): `include_test`, `max_rank` (`crop_rank_in_image <=
max_rank`; omitted = no limit, except `primary_low_conf` /
`classifier_blind_spots`, which default to `2`), `min_blur_ratio`,
`min_mistakenness`, `hide_near_duplicates`, **`class_id`**, **`source`**,
**`conf_min` / `conf_max`** (inclusive band on `confidence`, `400` if
min > max), `sort`; `text` and `region_status` on the `regions` tab only
(ignored elsewhere).

`region_status` (`regions` tab; DQ-B2 follow-up) selects which region
boxes the queue serves: `'all'` (default) — today's accepted-but-
unvalidated boxes (`region_bbox_norm` present) plus a verifier-rejected
candidate that still has a box to show (`region_status=verify_rejected`
AND `region_candidate_bbox_norm` present); `'detected'` — accepted boxes
only, same as `'all'` before this existed; `'verify_rejected'` — only the
rejected candidates. `false_positive` and `no_region_visible` items never
appear in any mode; a `verify_rejected` row with no candidate box (rejected
before the candidate was kept) never appears either. `400` for an
unrecognized value. A rejected candidate's per-item `reason` names the
rejection (`region_rejection_reason` when the worker recorded one) instead
of the generic "needs human confirmation" string. `locate` honours the
same parameter.

`GET /review/tabs` → `{tabs: [{id, label, description, filters,
filter_defaults, filter_specs}]}` (typed: `ReviewTabsResponse`): `filters` is the list of query
parameters the tab honours (a parameter not listed is accepted and
ignored), `filter_defaults` the values it applies when one is omitted
(`{"max_rank": 2}` for the two primary-subject tabs, `{"region_status":
"all"}` for `regions`, else `{}`). `filter_specs` is a self-describing
entry for each honoured filter with a fixed value set — `{param, kind:
"enum", label, options: [{value, label}]}` (today only `regions`:
`{"param": "region_status", "kind": "enum", "label": "Status", "options":
[{"value": "all", ...}, {"value": "detected", ...}, {"value":
"verify_rejected", ...}]}`), `[]` for a tab with none — so the frontend
renders any enum filter generically instead of hardcoding its values. The queue query reads the same
`filters`/`filter_defaults` table, so the catalog can't advertise a filter
a tab ignores (DQ-M6: `max_rank` used to be honoured only by the two
primary tabs). Response: `total`, `page`, `page_size`,
`items` (item + `reason`), `sort_applied` (the sort id that actually ran),
`sort_fallback_reason` (`null`, or a human-readable string when the
resolved default was replaced — see below).

Sort: an explicit `sort` wins (honored even if its field has no coverage);
omitted (or `default`) → the **tab's own default** (a deployment `sort`
default from `PUT /settings` never overrides it — it only applies to a tab
without one, e.g. `new_class_proposals`, else `recent`). If that resolved
default orders by a field **no item in the index has** (0% coverage), the
queue falls back to the tab's next covered sort (`all`: `mistakenness`;
`uncertainty`: `mistakenness`, then `atypicality`; every tab ends at
`recent`), `sort_applied` names the sort that ran and
`sort_fallback_reason` says which default was skipped and why. Unknown
coverage (count failed) never triggers a fallback. `PUT /settings` refuses
(`422`) a `sort` default whose field has 0% coverage. Every queue ends in a `crop_id` ascending tiebreak so pages are
stable and positions are exact.

`GET /review/{tab}/locate?crop_id=…` (same filters + `sort`, plus
`page_size`) → `{crop_id, in_queue, rank, page, page_size, total, reason,
sort_applied, sort_fallback_reason}`: `rank` is 0-based, `page` the 1-based page holding it;
out of the queue `rank`/`page` are `null` and `reason` is `not_found` or
`filtered_out`. It counts the items sorting before the crop (one count, any
queue depth) — use it for `/review?crop_id=` deep links instead of paging.

`GET /review/new_class_proposals/summary?size=&samples=` →
`{total_pending, without_term, top_terms, flagged_terms, term_rules}`
(DQ-M11). The summary, the `new_class_proposals` queue and the resolve
below share one selection (`src/services/curation/new_class_terms.py`
`proposal_query`), so `total_pending` equals the queue's `total`, and each
term's `count` equals what a resolve for that `label` matches.
`without_term` counts queue items with no proposed name (a human flag).
Each term is `{label, count, sample_crop_ids, flag, class_id}`, most
common first; `top_terms` holds only terms worth creating (`flag: null`),
`flagged_terms` the rest:

| `flag` | Rule | Suggested action |
|---|---|---|
| `existing_class` | the name (normalized: lowercase, spaces/hyphens → `_`) is an active registry class; `class_id` is set | resolve with `class_id` |
| `generic_parent` | the whole name is in `OP_NEW_CLASS_GENERIC_TERMS`, or is a registry `group` name or one `-`-separated part of one | assign a specific class, don't create |
| `non_object` | the name, or one `_`-separated token of it, is in `OP_NEW_CLASS_NON_OBJECT_TERMS` | discard / exclude |

`term_rules` serves the active rule: `{generic_terms, non_object_terms,
registry_groups_are_generic: true, existing_classes_flagged: true,
generic_terms_env, non_object_terms_env}`. Both env lists are
comma-separated and empty by default — no vocabulary is built in.
Generic terms match whole names only (`sports_car` is not flagged by a
generic `car`).

`POST /review/new_class_proposals/resolve?dry_run=` (`ResolveNewClassRequest`
→ `ResolveNewClassResponse`): bulk-resolves **every** item of the
new-class queue proposing `label` (`vlm_new_class_pending` rows and
`needs_new_class` flags carrying that `vlm_proposed_class`; never a
validated, review-dismissed, excluded or test-holdout item), not just the summary's
capped `sample_crop_ids`. Exactly one of `class_id` (map to an existing
registry class) / `create` (`{class_name, group, notes}`, registered
through the same path as `POST /classes`) — else `422`; unknown `class_id`
→ `400`; duplicate `create.class_name` → `409` with no item writes.
`create` runs before any item write (a zero-match resolve still creates
the class and reports `matched: 0`). Each item is written independently
(bounded concurrency), re-checked at write time to still be pending this
exact proposal — one that changed state in between lands in `skipped`,
not `updated_ids`. Response: `{class_id, class_name, created, label,
matched, matched_ids, updated, updated_ids, conflicts:
[{crop_id, current_source}], skipped}`. `dry_run=true` reports the match
(`matched`, `matched_ids`) without writing or creating anything
(`class_id` is `null` when `create` was given). Writes are undoable via
`POST /crops/label/undo_batch` on `updated_ids`, same as
`PUT /crops/batch_label`.

- `TestHoldoutFreezeRequest`: `percent` only (`1`–`50`, default `10`). Unknown fields are
  rejected (`extra='forbid'`): there is no seed — selection is deterministic — so a request
  carrying `seed` is a `422` instead of being silently ignored.
- `TestHoldoutFreezeResponse`: `n_frozen`, `n_classes_covered`, `test_holdout_sha`,
  `per_class_counts`, `selection` (always `"sha1_per_class"`: per class, the crops with the
  smallest `sha1(crop_id)`, `max(min_per_class, round(n * percent / 100))` of them, capped at the
  class size), `percent` (echoed), `min_per_class` (`5`). A client shows the method, not a
  Seed input.

### Shared curation-strategy defaults

- `CurationSettingsResponse` (`GET,PUT /settings`): `defaults` (`dict[str, str]`, open map keyed by axis id), `updated_at` (ISO 8601 or `null`), `updated_by` (always `null` today — no user-account system)
- `CurationSettingsUpdateRequest` (`PUT /settings` body): `defaults` (`dict[str, str]`, partial — only the axes being changed)

### Health / status

- `HealthResponse`: `status` (`ok`/`degraded`/`down`), `triton`, `opensearch`, `vlm`, `registry` — `vlm` reports the configured VLM backend's reachability regardless of which model it is.
- `StatusResponse`: `status`, `detail`, `extra`

### Export

- `ExportYoloRequest`: `export_dir`, `version_tag`, `seed`, `max_images`, `dedup_threshold`,
  `require_fully_labeled_images` (default `false`)
- `ExportSingleClassRequest`: `export_dir`, `version_tag`, `class_ids`,
  `box_source` (`item`/`region`), `region_class_name`, `profile_name`,
  `seed`, `skip_test_split`, `empty_bg_ratio`, `max_positive_images`,
  `dedup_threshold`, `image_mode` (`whole_frame`/`item_crop`),
  `img_max_side`, `copy_images`

**Multi-class layout (`POST /export/yolo`): one image file and one label
file per source image.** Validated items (one object each) are grouped
by `image_id`. Each exported image is written once as
`images/<split>/<image_id>.<ext>`, next to `labels/<split>/<image_id>.txt`
with one `cls cx cy w h` line per validated, non-excluded, non-dismissed
object on it. `cls` is the dense `export_id`; `cx cy w h` come from the
item's `bbox_norm` (`[x1, y1, x2, y2]`, normalized to the full source
frame), clamped to `[0, 1]`, so they are relative to the full source
image. Objects in a file are ordered by item id, so re-runs are
byte-identical. An item with no `image_id`, no usable box or no class is
left out and counted in the manifest's `skipped_items`. `resize_mode` on
the service accepts only `null` (copy as-is) or `aspect`; `letterbox` is
refused because its padding would shift every box. (Earlier exports
wrote one full-frame copy and one single-line label file per *item*, so
an image with three validated objects became three copies each labeled
with one object, and the detector learned the other two as background.)

**Partial frames.** An exported image can also hold *unlabeled* objects:
items on it that the export does not write — not validated yet,
validated on a class with no dense id (unregistered or deprecated), or
validated without a usable box. They are still in the pixels, so
training learns them as background. `class_excluded` and
review-dismissed items are not objects to label and never count.

- Default (`require_fully_labeled_images: false`): the image is exported
  with its validated objects labeled. The manifest records
  `unlabeled_items_on_exported_images` and `images_with_unlabeled_items`,
  and training preflight warns (`export_unlabeled_objects`).
- `require_fully_labeled_images: true`: every image with at least one
  unlabeled object is left out; the manifest records how many as
  `images_dropped_not_fully_labeled`. If no image is fully labeled the
  export is refused with `422` and nothing is written.

The partial-frame policy runs first, then `dedup_threshold` (which
collapses near-duplicate *images*; a kept image keeps all its objects,
a frozen-holdout image is preferred as the survivor, and the manifest's
`dedup.n_input_rows` / `n_output_rows` count images), then `max_images`
(a cap on images: an even round-robin over each image's rarest class, so
a rare class survives the cap).

**Multi-class manifest counts.**

| Field | Counts |
|---|---|
| `image_count` | exported images (= label files) |
| `object_count` | exported objects (= label lines) |
| `split_counts` | images per split, `{train, val, test}` |
| `split_object_counts` | objects per split, `{train, val, test}` |
| `class_split_counts` | objects per class per split (rows below) |
| `unlabeled_items_on_exported_images` | unlabeled objects on the exported images |
| `images_with_unlabeled_items` | exported images holding at least one unlabeled object |
| `require_fully_labeled_images` | the request flag |
| `images_dropped_not_fully_labeled` | images left out by that flag (`0` when off) |
| `skipped_items` | `{no_image_id, no_usable_box_or_class}` validated items left out |

`label_stats.json` (`{class_name: objects}`) sums `class_split_counts`
per class.

`POST /export/yolo` returns `status`, `export_dir`, `version_tag`,
`manifest_path`, `dataset_sha`, `image_count`, `object_count`,
`split_counts`, `split_object_counts`, `require_fully_labeled_images`,
`unlabeled_items_on_exported_images`, `images_with_unlabeled_items`,
`images_dropped_not_fully_labeled`, `skipped_items`
(`{no_image_id, no_usable_box_or_class}`), `dedup` (the requested
threshold), `started_at`, `finished_at`.

**`dataset_sha`** is a hash of the export's actual on-disk *content*, not
of which item ids were selected — two exports of the same items with
different splits, a corrected box, or a different class map (even with
byte-identical label files, e.g. after a pure registry rename) always get
different `dataset_sha`s. Concretely it hashes, over every written
`labels/<split>/*.txt` file sorted by relative path: the relative path
(so a split reassignment changes the digest even when the label bytes
don't) and the sha256 of the file's bytes, then folds in the export's
ordered `names:` list (`data.yaml` / dense export id → class name) so a
class rename with no id change still changes the digest. The multi-class
export records the full 64-hex sha256 digest; `POST /export/single_class`
records the same digest truncated to 16 hex chars. The shared
implementation is `label_content_sha` in
`src/services/curation/export_support.py`.

Example manifest excerpt (`img-a` with three objects of two classes,
`img-b` with one, and `img-c` with one validated object next to one
unreviewed item). It writes `labels/train/img-a.txt` (three lines, e.g.
`0 0.200000 0.400000 0.200000 0.400000` for a `bbox_norm` of
`[0.1, 0.2, 0.3, 0.6]`), `labels/train/img-b.txt` and
`labels/val/img-c.txt`:

```json
{
  "group_key": "image_id",
  "image_count": 3,
  "object_count": 5,
  "split_counts": {"train": 2, "val": 1, "test": 0},
  "split_object_counts": {"train": 4, "val": 1, "test": 0},
  "class_split_counts": [
    {"class_id": 1, "export_id": 0, "class_name": "alpha", "train": 2, "val": 1, "test": 0},
    {"class_id": 2, "export_id": 1, "class_name": "beta", "train": 2, "val": 0, "test": 0}
  ],
  "require_fully_labeled_images": false,
  "unlabeled_items_on_exported_images": 1,
  "images_with_unlabeled_items": 1,
  "images_dropped_not_fully_labeled": 0,
  "skipped_items": {"no_image_id": 0, "no_usable_box_or_class": 0}
}
```

`POST /export/single_class` builds a narrowed dataset for a single class
or a class subset, with an extra integrity field the multi-class export
doesn't need: `frozen_test_sha` hashes the test split's *identity*
(which frames, not their content — a label correction inside the test
set must not trip it) so "the held-out set never changed between two
runs" is checkable. `dataset_sha` uses the same content-hash mechanism as
the multi-class export (see above). The profile's own `current` symlink
is flipped atomically. `GET
/export/single_class/status?profile_name=...` reports the last run for
one profile, with the same `idle`/`unknown`/`success` contract as
`GET /export/status`. Each `profile_name` gets its own output root and
its own `current` symlink, so narrowed exports never clobber each other
or the multi-class dataset.

**Readiness (DQ-M9).** Both exports refuse with `422`
(`detail: "nothing to export: <reason>"`) when nothing is exportable —
`POST /export/yolo`: no item is `class_validated` (and not excluded or
review-dismissed), none has an image id, a box and a class, every one is
on a class id missing from (or deprecated in) the registry, or
`require_fully_labeled_images` left no image; `POST
/export/single_class`: no item matches the profile. Nothing is written
and `current` keeps pointing at the previous export. Every manifest
records `items_index: {index, uuid, created_at}` — the items index it was
read from (`null` if it could not be read).

`POST /train/preflight` adds these export checks (see
`src/services/curation/export_readiness.py`):

| Check | `block` when | `unknown` when |
|---|---|---|
| `export_not_empty` | the manifest's `image_count` (else the sum of `split_counts`) is `0` images | no readable manifest / no count |
| `export_splits_nonempty` | `split_counts.train` or `split_counts.val` (images) is `0` (message names the empty split(s); `detail.empty_splits`) | the manifest records no train/val counts |
| `export_class_split_coverage` | a class the run trains on (`include_classes`, else every class in `class_split_counts`) has fewer than `min_train_per_class` (`1`) train or `min_val_per_class` (`1`) val objects — message lists each as `name (class id): train=N, val=N`; `detail.classes[]` carries `class_id`, `class_name`, `train`, `val`, `test`, `missing_splits`. Always `ok` ("not applicable") for a single-class export, which `export_splits_nonempty` already covers | the manifest has no `class_split_counts` (exported before they were recorded — re-export) |
| `export_unlabeled_objects` | never blocks: `warn` when `unlabeled_items_on_exported_images > 0` — message gives `images_with_unlabeled_items/image_count` and the object count, says training learns unlabeled objects as background, and points at `require_fully_labeled_images`; `detail` carries `image_count`, `unlabeled_items_on_exported_images`, `images_with_unlabeled_items`, `require_fully_labeled_images`, `images_dropped_not_fully_labeled`. `ok` when `0`, and always `ok` ("not applicable") for a single-class export | the manifest has no unlabeled counts (exported before per-image labels — re-export) |
| `export_generation` | the manifest's `items_index.uuid` differs from the live items index's (the index was rebuilt since the export); for an unstamped export, its `exported_at` is before the live index's creation | the live index can't be read, or the manifest has neither a stamp nor `exported_at` |

The index `uuid` is the staleness signal because it changes on every
index creation and is immune to clock skew; label edits after an export
are deliberately not "stale" (exports are snapshots, and retraining on a
past one is supported).

**Split assignment** (`stratified_split` in
`src/services/curation/export_support.py`; the manifest records
`group_key` and `seed`):

- **Group = source image** (`group_key: "image_id"`). Items cut from
  one image never straddle train/val/test; the multi-class export writes
  each image once, so the image and all its objects share one split. `cluster_id` is not a leakage unit — class clusters
  have `cluster_id == class_id`, so grouping on it made each class one
  group. Crop-level `dup_group_id` is not used either: it is written only
  by an opt-in scorer run, only for items in a multi-member group, and
  its ids (`dup_<n>`) are numbered per run, so two runs can reuse an id
  for unrelated items. Whole-frame near-duplicate bursts are handled
  before the split by the export's `dedup_threshold`.
- **Frozen holdout**: an image carrying a `test_holdout` item goes to
  `test` with every object on it (any class).
- **Strata**: each remaining group counts toward its most common class.
  Within a class, groups are ordered by `sha256(seed:class:group)`, so
  the same data and seed always give the same split.
- **Per-class allocation of the `n` remaining groups**: a class with at
  least one frozen holdout item uses the holdout as its test set and
  splits the rest train : val = `train_ratio : val_ratio` (0.8 : 0.1); a
  class with no holdout item splits train / val / test at 0.8 / 0.1 /
  0.1. Every split with a positive ratio gets one group before any
  gets a second (priority train → val → test); the rest follow the
  ratio. So `n = 0` → the class appears only in test (its holdout);
  `n = 1` → train; `n = 2` → one train + one val; `n >= 3` → at least
  one train and one val (and, with no holdout, at least one test).

The multi-class manifest's `class_split_counts` lists every class in the
export (`class_id` registry id, `export_id` dense id, `class_name`,
`train`, `val`, `test` object counts), including classes with no
objects. `label_stats.json` keeps its flat `{class_name: count}` shape
(objects per class).

**`GET /export/status`** (`ExportStatusResponse`) serves the last
completed multi-class export — the `current` symlink's manifest:
`status` (`idle` / `unknown` / `success`), `path` (resolved export dir;
`export_dir` is the same value), `last_run` (finish, else start time),
`version_tag`, `dataset_sha`, `seed`, `group_key`, `image_count`
(images), `object_count` (objects), `class_count`, `split_counts`
(images per split, `{train, val, test}`), `split_object_counts` (objects
per split), `class_split_counts` (objects per class per split, rows as
in the manifest), `require_fully_labeled_images`,
`unlabeled_items_on_exported_images`, `images_with_unlabeled_items`,
`images_dropped_not_fully_labeled`, `skipped_items`
(`{no_image_id, no_usable_box_or_class}`). A field the manifest does not
record (an export written before it existed) is `null` — this is the
common case for `skipped_items` against an export from before it was
added to the manifest. `idle` sets every other field to `null`; `unknown`
(manifest missing/unreadable) sets only `path` / `export_dir`.

```json
{
  "status": "success",
  "path": "/exports/20260924T120000Z",
  "export_dir": "/exports/20260924T120000Z",
  "last_run": "2026-09-24T12:00:04+00:00",
  "version_tag": "v1",
  "dataset_sha": "4c1f...",
  "seed": 42,
  "group_key": "image_id",
  "image_count": 3,
  "object_count": 5,
  "class_count": 2,
  "split_counts": {"train": 2, "val": 1, "test": 0},
  "split_object_counts": {"train": 4, "val": 1, "test": 0},
  "class_split_counts": [
    {"train": 2, "val": 1, "test": 0, "class_id": 1, "export_id": 0, "class_name": "alpha"},
    {"train": 2, "val": 0, "test": 0, "class_id": 2, "export_id": 1, "class_name": "beta"}
  ],
  "require_fully_labeled_images": false,
  "unlabeled_items_on_exported_images": 1,
  "images_with_unlabeled_items": 1,
  "images_dropped_not_fully_labeled": 0,
  "skipped_items": {"no_image_id": 0, "no_usable_box_or_class": 0}
}
```

### Training run status — `last_epoch_metric` / `best_checkpoint_metric`

`GET /train/status`, `GET /train/status/{job_id}` and `GET /train/runs`
(`TrainJobStatus`) serve two distinct per-run metric rows instead of the
former `best_metric`/`last_metric` pair:

- `last_epoch_metric`: `{"epoch": <int>, "map50": <float>, "map50_95": <float>}`
  — the true LAST TRAINING epoch's metrics.
- `best_checkpoint_metric`: same shape — the best checkpoint's
  (`best.pt`) own re-validation metrics, as one coherent row (both
  `map50` and `map50_95` from the same validation pass).

Why two fields: Ultralytics fires its `on_fit_epoch_end` callback once
more after training completes, re-validating `best.pt` — but without
advancing its internal epoch counter, so that call is otherwise
indistinguishable from a repeated epoch. The trainer
(`docker/trainer/trainer.py::_make_ultralytics_callbacks`) detects the
repeat and routes it to `best_checkpoint_metric` instead of clobbering
`last_epoch_metric` with the wrong (best-checkpoint, not last-epoch)
values — and `best_checkpoint_metric` is a single row rather than the
former per-key running max, which could otherwise report `map50` from
one epoch and `map50_95` from another. A status written before these
fields existed serves both as `null`; the retired `best_metric` /
`last_metric` keys are dropped, never mapped onto the new fields, and
`eval` is never copied into them (it may be test-split numbers).

`/bakeoff/trained_models`'s `trainer_map50` column and the promote gate
(`src/routers/curation_train.py::_evaluate_promote_gate`) read from
`eval.map50` (the fresh test-split re-validation, `state.eval` /
`populate_eval_block`), not from either of these two fields — they are
diagnostic epoch-level metrics, not the run's scored comparison metric.
`/bakeoff/trained_models` also serves `trainer_map50_split` (`eval.split`:
`"test"`, or `"val"` when the test pass fell back).

`GET /train/manifest/{job_id}`'s `results` block mirrors the same two
field names (`last_epoch_metric`, `best_checkpoint_metric`) in place of
the old `results.best_metric`.

**`eval.head`.** YOLO26 exports/serves the NMS-free one-to-one head
(`nms=False` at export, since Ultralytics forces `nms=False` on any
`end2end` model). The trainer's own post-training test-split
re-validation (`_finalize_run`, ~`docker/trainer/trainer.py:585`)
explicitly forces that same head (`model.end2end = True`) before calling
`.val(split='test', ...)` when the checkpoint is a genuine dual-head
(one-to-one + one-to-many) YOLO26 build — `.val()` has no `end2end=`
keyword; the only real toggle is the loaded model's own `.end2end`
property (`ultralytics.nn.tasks.DetectionModel.end2end`, a setter
delegating to `set_head_attr`). `state.eval.head` records `"end2end"`
when this was applied, so a comparison result is explicit about which
head it scored rather than silently depending on Ultralytics' own
`.val()` default for the loaded checkpoint.

### Capability discovery — `GET /methods`

`GET {prefix}/methods` (`src/routers/curation/methods.py`) is the
capability-discovery endpoint every consumer should gate optional UI on
instead of feature-probing a write endpoint with a throwaway request.
It returns `{'strategies': [...], 'flags': {...}}`; each `strategies`
entry carries an `axis` of `cluster` / `score` / `sort` / `overlay` /
**`export`**.

**`export` axis** — which dataset-export *kinds*
`POST {prefix}/export/{kind}` can actually produce on this deployment:

| `id` | `status` | Notes |
|---|---|---|
| `yolo` | `stable` | Backed by `GenericYoloExportService`; always advertised. |
| `single_class` | `stable` | Backed by `SingleClassExportService`; single-class or class-subset export. |

There is deliberately **no** domain-named export id. An earlier
single-class export is covered by
`single_class`, which takes its target class ids from the request
instead of hardcoding a domain vocabulary — a domain-named export kind
would be exactly the
hardcoding this axis exists to avoid.

### Shared curation-strategy defaults — `GET,PUT /settings`

Cropwright's StrategyBar/AssistScopeBar (cluster method, sort order,
detection profile, prompt pack dropdowns) previously reset to a
hardcoded client default on every reload. There is no user-account
system (single shared instance), so the shared default per axis is now
stored once, backend-side, instead of per-browser.

`GET {prefix}/settings`:

```json
{"defaults": {"cluster": "ivf"}, "updated_at": "2026-09-20T12:00:00+00:00", "updated_by": null}
```

`defaults` is an **open map** keyed by axis id — deliberately not a
fixed set of named fields (`cluster`/`sort`/`detection_profile`/
`prompt_pack`) — so a future axis never requires a wire-format change.
A missing key means "no shared override for that axis." No document has
ever been written yet (nothing has been `PUT`) is not an error: this
still returns `200` with `defaults: {}`, `updated_at: null`,
`updated_by: null`. `updated_by` is always `null` today (no
user-account system); the field exists on the wire for when one does.

`PUT {prefix}/settings` (partial body — only the axes being changed):

```json
{"defaults": {"cluster": "ahc"}}
```

Merges into the stored document; axes already set and not mentioned in
the body are left untouched. Returns the full updated record, same
shape as the `GET`. Each `axis` key must be one of
`src.services.curation.strategy_defaults.SETTABLE_DEFAULT_AXES`
(`cluster` / `sort` / `prompt_pack` today —
`score`/`overlay`/`export` have no single-selectable-id "default"
concept a shared override could apply to, and `detection_profile` is
read-only (the region cascade runs on the process's `OP_REGION_PROFILE`),
so they 422 rather than
silently accepting a value nothing will ever honor), and each `id` must
be a currently-advertised id for that axis per `GET /methods` — either
violation returns `422` with a message listing the valid axes/ids.

**The consistency guarantee (the actual point of this endpoint):**
`GET /methods`'s per-axis `default: true/false` flag is *derived* from
this settings document via
`src.services.curation.strategy_defaults.resolve_effective_default(axis)`
— it looks up `defaults.get(axis)`; if present and still a
currently-advertised id for that axis, that id is the effective
default; otherwise it falls back to the axis's pre-existing hardcoded
default constant (`DEFAULT_METHOD` for `cluster`,
`get_default_profile_name()` for `detection_profile`,
`resolve_prompt_pack().name` for `prompt_pack`; `sort` has no single
hardcoded default — only a per-tab mapping,
`review_sorts.default_sort_for_tab` — so a `sort` override is an
*additional*, opt-in global choice layered on top of the untouched
per-tab defaults, not a replacement for them). This exact function is
also called by every real endpoint that applies a hardcoded default
when a request omits that axis's param, so setting a shared default
changes actual server behavior, not just what `GET /methods` displays:

| Axis | Real endpoint call site |
|---|---|
| `cluster` | `src.services.curation.clustering.orchestrator.cluster_residuals` — resolves the effective cluster method when `?clustering_method` is omitted (feeds `POST /pipeline/auto_label*` and `POST /clusters/*`'s residual-clustering stage). |
| `sort` | `src.services.curation.review_sorts.build_sort` — when `GET /review/{tab}`'s `?sort` is omitted or `'default'`, a valid shared override is tried before falling back to that tab's own hardcoded default. |
| `detection_profile` | **Read-only.** `GET /methods` lists the registered region profiles with the active one (`OP_REGION_PROFILE` / `OP_REGION_DETECTION_*`) as `default: true` and `settable: false`; a stored settings override is ignored and `PUT /settings` with this axis is a `422`. `POST /pipeline/auto_label*` rejects `?detection_profile=` with a `422` (no auto-label stage runs region detection) rather than silently ignoring it. |
| `prompt_pack` | `POST /pipeline/auto_label*` — `?prompt_pack=<id>` selects the pack for that job's VLM labeling stage (same override/`422`/echo semantics); omitted resolves via this function. Every VLM endpoint (`POST /vlm/label_batch`, `/vlm/verify_regions`, `/vlm/verify_region_batch`, `/vlm/region_visible_batch`) also uses the effective default. Selectable ids: the built-in generic pack, every `OP_PROMPT_PACK_PATHS` pack, and the `OP_PROMPT_PACK_PATH` pack (the fallback default). |

Storage: a single OpenSearch document (not a full index of many rows),
in its own small index (`IndexRole.SETTINGS`, default `op_curation_settings`,
override via `OP_SETTINGS_INDEX`) addressed by the fixed doc id
`CURATION_SETTINGS_DOC_ID = 'default'` — following the exact same
`IndexRole` + `INDEX_BODIES` convention every other curation index uses
(`src/clients/curation_opensearch.py`), wired into the same
`create_curation_indexes` startup bootstrap automatically. The
`defaults` field is mapped `{'type': 'object', 'enabled': False}` (never
queried, so never indexed) — OpenSearch's partial-update `doc` merge
still recursively merges into it regardless of `enabled`, which is what
lets a partial `PUT` avoid clobbering other axes without a
read-modify-write round trip in application code.

### Model comparison (bake-off) — `/bakeoff/*` (BREAKING, schema v2)

Models in `src/routers/curation/_bakeoff_models.py`; every
route has a `response_model`. Evaluator-written files (`status.json`,
`comparison.json`, `matrix.json`) are validated on read; a file that is not
schema v2 answers `409 "bake-off result <file> has an unsupported schema
(schema_version != 2)"`. No compatibility fields for the v1 wire.

- `GET /bakeoff/eval_datasets?source=export|external` -> `EvalDatasetList`
  `{datasets: EvalDataset[], count}`. `EvalDataset`: `id`
  (`export:<path under export_root>` | `external:<group>/<name>`), `source`,
  `group`, `name`, `path`, `is_current`, `dataset_kind`
  (`multi_class|single_class|external`), `nc`, `classes`
  (`EvalDatasetClass`: `eval_class_id`, `name`, `registry_class_id`,
  `n_objects`, `n_images`; only classes with objects in test), `n_images`,
  `n_objects`, `n_background_images`, `frozen_test_sha` (identity),
  `test_label_sha` (content), `sha_source` (`manifest|computed`),
  `dataset_sha`, `exported_at`, `unlabeled_items_on_exported_images`,
  `frozen_ok` (external only).
- `GET /bakeoff/trained_models?dataset_id=&limit=` -> `TrainedModelList`
  `{models: TrainedModel[], count}`. `TrainedModel`: `run_id`,
  `display_name`, `model_family`, `model_size`, `imgsz`, `checkpoint_path`,
  `finished_at`, `campaign_id`, `train_export_id`, `dataset_sha`,
  `frozen_test_sha`, `class_names`, `single_cls`, `trainer_map50`,
  `trainer_map50_split`, `for_dataset` (only with `?dataset_id`:
  `dataset_id`, `same_export`, `same_frozen_test`, `n_classes_mapped`,
  `train_test_overlap {n_images, fraction}`).
- `GET /bakeoff/profiles` -> `BakeoffProfileList` `{profiles, count,
  default_profile, default_error}`; rows (`BakeoffProfileRow`): `name`,
  `description`, `kind` (`registered|configured`), `default`,
  `class_filter`, `imgsz`, `conf_floor`, `nms_iou`, `op_conf`, `op_iou`,
  `rank_metric`, `default_backend`, `triton_model`, `context_class_ids`,
  `baselines_path`. Example profiles are not listed.
- `GET /bakeoff/baseline_models?profile=` -> `BaselineModelList`
  `{baselines: BaselineModel[], count}` (default registry empty).
  `BaselineModel`: `name`, `backend`, `weights`, `imgsz`, `mode`,
  `class_map` (`{"<model class id>": "<eval class name>"}` | null),
  `backend_options`, `training_data`, `triton_model`.
- `POST /bakeoff/run` body `BakeoffRunRequest` (`extra='forbid'`): `job_id?`,
  `profile?`, `datasets: [{id}]` (`id` may be `run:<job_id>`), `models[]`
  discriminated on `source`: `RunModelRef {source:"run", run_id,
  display_name?, backend?: ultralytics|onnxruntime, mode?}`,
  `BaselineModelRef {source:"baseline", name, display_name?}`,
  `CustomModelRef {source:"custom", name, backend, weights?, triton_model?,
  imgsz?, mode?, class_map?, backend_options?, display_name?}`;
  `quantize?: {run_id, formats?, n_calib?, calib_split?, throughput?}`.
  Answers `BakeoffRunAccepted` `{status:"enqueued", job_id, profile,
  datasets: [{id, path, frozen_test_sha, test_label_sha, n_eval_classes}],
  models: [{model, display_name, source, class_mapping: {<dataset id>:
  ClassMapping}, train_test_overlap: {<dataset id>: {n_images, fraction} |
  null}}], warnings}`. `ClassMapping`: `method`, `model_to_eval`,
  `unmapped_model_classes`, `not_covered_eval_classes`, `warnings`. Errors:
  400 (no models/quantize, no datasets, unknown/invalid dataset, run or
  baseline id, duplicate model keys, unknown profile), 422
  (`single_cls` run over several classes on a multi-class dataset; unknown
  fields), 409 (job id exists; GPU-resident containers could not be
  stopped — the job file is removed and the status set to `error`).
- `GET /bakeoff/status/{job_id}` -> `BakeoffStatus` (`schema_version`,
  `job_id`, `state` `queued|running|done|error`, `profile`, `datasets`,
  `models`, `started_at`, `finished_at`, `progress {done,total}`,
  `completed [{dataset, model}]`, `failed [{stage, dataset, model, error}]`,
  `error`).
- `GET /bakeoff/runs` -> `BakeoffRunList` `{runs: [{job_id, state, profile,
  datasets, models, started_at, finished_at}]}`; non-v2 dirs skipped.
- `GET /bakeoff/results/{job_id}?dataset_id=` -> `BakeoffComparison`
  (default: the job's first dataset): `schema_version`, `job_id`, `profile`,
  `thresholds`, `dataset`, `eval_classes`, `common_classes`, `rank_by`,
  `rank_scope` (`common|overall`), `models: ComparisonRow[]` (`rank`,
  `model`, `display_name`, `source`, `run_id`, `runtime`, `imgsz`,
  `training_data`, `overall: MetricBlock`, `common: CommonMetricBlock`,
  `per_class: PerClassRow[]`, `coverage`, `class_mapping {method,
  warnings}`, `train_test_overlap`, `latency_ms {mean,p50,p90,p99}`, `fps`,
  `size_mb`, `per_stratum`), `failed`, `warnings`, `n_models`.
- `GET /bakeoff/matrix/{job_id}` -> `BakeoffMatrix`: `datasets[]` (`id`,
  shas, `rank_scope`, `n_common_classes`), `models[]`, `metrics`,
  `cells[model][dataset]` (`map_50`, `map_50_95`, `precision`, `recall`,
  `f1`, `latency_ms`, `size_mb`, `coverage`, `rank`), `best[dataset][metric]`
  = list of every tied winner.

### Internal / worker-facing

- `_PathLookupRequest`: `image_paths` (max 10,000)
- `_PathLookupResponse`: `known_paths` (`dict[image_path, image_id]`)
- `_PublishEvent` (`POST /events/publish`, used by the SAM worker): `type`, `crop_id`, `class_id`, `class_name`, `class_source`, `region_status`, `region_text`, `image_path`, `topic`, `extra`. `extra='forbid'`: an unknown key is a `422` (a mismatched status key used to be silently dropped, so worker-published `crop.region_verified` events arrived with no status — audit S7).

## Item wire format

Built by `serialize_item()` in `src/services/curation/wire.py`. 97 keys,
always all present (a value is `null` when the stored doc has no value;
`bbox_norm` defaults to `[]`, `class_name`/`class_source`/
`label_source`/`updated_at`/`source`/`proposed_class_name` to `""`,
`confidence` to `0.0`, `label_validated`/`class_validated`/`test_holdout`/
`needs_new_class`/`class_excluded` to `false`, `item_text_lines` to `[]`).

Item keys (67): `id`, `crop_id`, `image_id`, `image_path`, `source_image_path`, `bbox_norm`, `class_id`, `class_name`, `class_source`, `confidence`, `class_confidence`, `class_confidence_source`, `label_source`, `label_validated`, `class_validated`, `class_detector`, `class_detector_version`, `class_labeled_at`, `class_labeler`, `vlm_confidence`, `vlm_class_attempted_at`, `vlm_class_empty_reason`, `vlm_raw_class`, `vlm_proposed_class_id`, `vlm_proposed_class_name`, `proposed_class_id`, `proposed_class_name`, `needs_new_class`, `needs_new_class_note`, `cluster_id`, `cluster_kind`, `cluster_distance`, `cluster_similarity`, `cluster_is_core`, `cluster_nearest_id`, `cluster_subid`, `class_excluded`, `excluded_reason`, `excluded_at`, `review_dismissed_at`, `source`, `test_holdout`, `crop_rank_in_image`, `crop_area_norm`, `blur_lap_ratio`, `proposal_name`, `probe_pred_class`, `probe_pred_class_id`, `probe_pred_entropy`, `probe_disagreement`, `probe_in_scope`, `probe_model_version`, `probe_actionable`, `mistakenness_score`, `mistakenness_method`, `mistakenness_version`, `mistakenness_scored_at`, `uniqueness_score`, `dup_group_id`, `dup_group_size`, `dup_is_representative`, `updated_at`, `thumbnail_url`, `region_thumbnail_url`, `item_text_lines`, `region_bbox_in_parent`, `region_candidate_bbox_in_parent`.

`vlm_class_attempted_at` / `vlm_class_empty_reason`: when a VLM was last
asked for the item's class, and why that attempt gave no class — `no_answer`
(empty / `null`), `no_match` (`-1`, or `__new__` with no proposed name),
`invalid_index`, `unparseable` (no usable reply entry); `null` when it
answered. An empty answer leaves every class field as it was (it is **not**
`vlm_unmatched`, which means the VLM named a label outside the registry and
carries it in `vlm_raw_class`). Such items appear in the `all` review tab and
stay out of the VLM selectors for 24 h.

`vlm_raw_class`: the VLM's class answer verbatim, `null` when none is
stored. On a `vlm_unmatched` item it is the label the VLM named that is not
in the registry (the item's `class_name` is whatever it already carried),
so a reviewer sees what the VLM actually said.

Region keys (42, one per `RegionFields` attribute except `embedding`,
`prefix` and the `*_legacy` rollback columns): `region_bbox_norm`, `region_bbox_frame`, `region_bbox_correct`, `region_status`, `region_score`, `region_confidence`, `region_reason`, `region_rejection_reason`, `region_text`, `region_text_raw`, `region_text_confidence`, `region_text_source`, `region_text_engine_version`, `region_text_vlm`, `region_text_ocr`, `region_text_disagreement`, `region_text_choice`, `region_text_vlm_invalid`, `region_validated`, `region_auto_confirmed`, `region_verified`, `region_verified_at`, `region_verifier`, `region_verifier_version`, `region_visible`, `region_detector`, `region_detector_version`, `region_detector_chain`, `region_detected_at`, `region_candidate_bbox_norm`, `region_candidate_score`, `region_candidate_detector`, `region_candidate_detector_version`, `region_candidate_source`, `region_cluster_id`, `region_cluster_subid`, `region_cluster_distance`, `region_class_id`, `region_label_source`, `region_source`, `region_pairing`, `region_skip_verify`.

Derived keys (computed by the serializer, never stored):

- `region_bbox_in_parent` — the region box in the item-crop frame
  (`[x1,y1,x2,y2]`, clamped to `[0, 1]`); `null` when there is no region or
  the item has no usable `bbox_norm`. Draw it on the item thumbnail as-is.
- `region_candidate_bbox_in_parent` — the same projection of
  `region_candidate_bbox_norm` (`null` without a candidate).
- `proposed_class_id` / `proposed_class_name` — the class a one-key confirm
  applies, on **every** item endpoint (was `/review`-only): the VLM
  suggestion when there is one, else `class_id` and `vlm_raw_class` or
  `class_name` or `""` (see "VLM class suggestion").
- `confidence` is always the **detector/classifier score** stored at
  ingest, whatever wrote the current label — never the VLM's. Label it as
  such. `class_confidence` / `class_confidence_source` (DQ-M8) are the
  confidence of the writer that set the label: for a VLM `class_source`
  (`vlm`, `vlm_unmatched`, `vlm_new_class_pending`, `vlm_reclassified`)
  the VLM's category (`vlm_confidence`) mapped high `0.92` / medium `0.70`
  / low `0.40` with source `vlm` (`null` for a missing/unknown category);
  for a classifier source (`<profile>_model`) the stored score with
  source `model`; `null`/`null` for human, move, merge, import,
  cluster-vote and unclassified-proposal labels.
- `cluster_kind` — `class` / `candidate` / `unassigned` from `cluster_id`
  (`null` without one); same rule as the cluster cards.
- `cluster_similarity` — `1 - cluster_distance` clamped to `[0, 1]` (`null`
  without a distance); `cluster_is_core` — `cluster_similarity >=
  core_similarity_min` (served on `GET /clusters`, `0.75`).
  `cluster_distance` is the cosine distance to the item's cluster
  centroid. Candidate clusters: written by every residual clustering run
  whatever the method (IVF's own centroids; otherwise the cluster's
  member-mean centroid). Class clusters (DQ-M3): written by the
  cluster-geometry pass that follows every auto-label clustering stage
  (`stages.cluster_residuals.cluster_geometry` in the job summary), as the
  distance to the class cluster's member-mean centroid. Every writer also
  stores the stored-only `cluster_distance_cluster_id` (the cluster it was
  measured against); when that differs from the item's current
  `cluster_id` (the item moved since), `cluster_distance`,
  `cluster_similarity` and `cluster_is_core` are served `null` rather than
  describing a cluster the item has left. `null` also for noise and for
  items not yet measured. `cluster_nearest_id` — the cluster whose
  centroid is nearest the item (equal to `cluster_id` when the item sits
  best where it is; the per-item input to a card's `purity`), from the same
  pass and gated the same way.
- Pass-throughs: `needs_new_class` (bool), `needs_new_class_note`,
  `class_excluded` (bool), `excluded_reason`, `excluded_at`,
  `probe_pred_class_id` (registry id of `probe_pred_class`, written by the
  probe pass), `source` (ingest source tag; stored under the `source`
  key — the retired short-name storage key is gone, S2).
- `probe_disagreement` / `probe_in_scope` / `probe_model_version` (D1,
  2026-09-25 F8 acceptance): the item wire's opinion on whether the
  active-learning probe's top-1 prediction agrees with the item's
  current class, made explicit so the frontend never has to infer scope
  from a null. `probe_in_scope` is `null` until the probe has scored this
  item at all; once scored, `true` when the item's class is one the probe
  was trained on (`probe_disagreement` is then a real `true`/`false`) and
  `false` when the item's class is outside the probe's class set
  (`probe_disagreement` is then `null` — the probe structurally has no
  opinion, which is NOT the same as agreement, and the UI must not offer
  an "accept model's class" action for it). `probe_model_version` is the
  probe checkpoint's version tag — the closest thing to a "probe run id"
  this system persists today (see
  `src.services.curation.probe_predictions`).
- `probe_actionable` (D1 follow-up, 2026-09-25): the backend's own
  accept/no-accept decision, so the frontend never re-derives a
  threshold. `null` mirrors `probe_in_scope`/`probe_disagreement` (the
  probe hasn't scored this item). Once scored: `true` only when
  `probe_in_scope` is `true` AND `probe_disagreement` is `true` AND the
  probe's top-1 posterior (`probe_pred_confidence`) is at least
  `CurationConfig.probe_actionable_min_confidence` (env
  `OP_PROBE_ACTIONABLE_MIN_CONFIDENCE`, default `0.5`, echoed read-only
  as `actionable_min_confidence` on `GET /probe/status`); `false` for
  every other case, including in-scope-and-agreeing, out-of-scope, and
  disagreeing-but-unsure. **Confidence gates this, not
  `probe_pred_entropy`** — entropy is a raw Shannon value in nats bounded
  by `log(nc)` (`nc` = the probe checkpoint's class count), which varies
  across probe versions/class-subsets and isn't stored per item, so a
  fixed threshold against it would silently drift as `nc` changes;
  `probe_pred_confidence` is always in `[0, 1]` by construction (a
  sum-to-1 posterior's top value) and is written in the same bulk update
  as `probe_pred_class`, so it's reliably present whenever the probe has
  scored an item.

  **UI contract:** offer "Accept model's class" only when
  `probe_actionable` is `true`. When `probe_disagreement` is `true` but
  `probe_actionable` is `false` (the probe disagrees but isn't confident
  enough), show `"model unsure: <probe_pred_class>"` with no Accept
  action — never let a client infer this from `probe_pred_entropy` or
  `probe_pred_confidence` directly; the threshold decision lives only in
  `probe_actionable`.

`label_validated` is derived (`class_validated` OR `region_validated`).

`region_validated` is **human** validation only: a human confirmed, drew
or rejected the region. The detection worker never sets it. When the
worker's auto-confirm policy accepts a box (detector and verifier agree
strongly enough) it sets `region_auto_confirmed=true` instead: the region
is accepted (`detected`, exported as a positive) but unreviewed, so it stays
in the `regions` review tab. Rows an older worker stamped
`region_validated=true` without a human are re-labelled by
`scripts/curation/repair_region_validation.py` (dry run by default).
`thumbnail_url` / `region_thumbnail_url` are built from the configured
`api_prefix` (`{prefix}/crops/{crop_id}/thumbnail` and
`…/region_thumbnail`), so `OP_API_PREFIX` and the frontend's proxy prefix
must match.

| Endpoint | Items at | Keys |
|---|---|---|
| `GET /crops`, `GET /classes/{class_id}/crops` | `crops[]` | item |
| `GET /crops/{crop_id}` | body | item |
| `GET /review/{tab}` | `items[]` | item + `reason` |
| `GET /regions` | `items[]` | item |
| `GET /regions/training_candidates` | `items[]` | item + `selection_reason` |
| `GET /search/text` | `items[]` | item + `semantic_score` |

### Region text — `region_text*`

`region_text` is the chosen reading of the region's text. Which reader
fills it is the region profile's `text_reader`
(`OP_REGION_DETECTION_TEXT_READER`):

| `text_reader` | Region OCR runs | `region_text` |
|---|---|---|
| `vlm` | only when no VLM is configured | the VLM's reading |
| `ocr` | always | the OCR reading (VLM's if OCR read nothing) |
| `vlm_then_ocr` (generic default) | when the VLM read nothing | VLM's, else OCR's |
| `both` (reference `license_plate` profile) | always | VLM's, else OCR's |

- `region_text_source`: `vlm` or `ocr` (a human edit writes `human`).
- `region_text_engine_version`: the VLM model id, or the OCR det + rec
  model ids for an OCR reading (`<det>:<ver>+<rec>:<ver>`).
- `region_text_confidence`: VLM category mapped to 0.92/0.70/0.40, or the
  minimum recognition score of the kept OCR lines.
- `region_text_raw`: every line the OCR read on the region crop,
  unfiltered, in reading order, joined by a space; the VLM's verbatim
  reading when OCR did not run.
- `region_text_vlm` / `region_text_ocr`: each reader's own reading
  whenever it produced one (keyword).
- `region_text_disagreement`: `true`/`false` when both *valid* readings
  exist, compared after the profile's normalization; `null` otherwise
  (boolean).
- `region_text_choice`: why the chosen reading won — `readers_agree`,
  `vlm_preferred` (both valid, they differ, the mode prefers the VLM),
  `vlm_only`, `ocr_only`, `ocr_mode` (`text_reader=ocr`), `vlm_invalid`
  (the VLM reading was rejected, the OCR reading won), `no_valid_reading`
  (every reading was rejected; `region_text` is `null`), `human` (typed by
  a human).
- `region_text_vlm_invalid`: why the VLM's reading (still kept in
  `region_text_vlm`) is not text — `placeholder`, `no_reading`, `sequence`,
  `charset`, `too_short`, `too_long`, `format`; `null` when it is valid.

Before a reading is chosen, every reader's reading is checked by the
region-text rules (`src/services/detection/region_text_rules.py`), served
as `text_rules` on `GET /regions/vocabulary` (`null` without a region
profile) with the choice values as `text_choices`. A reading is not text
when it is a generic "no reading" word (`NOT_READABLE`, `N/A`, …); a
placeholder — one of the active prompt pack's quoted example values, or a
truncation of one at least 3 characters long, or a profile
`text_placeholders` entry (`OP_REGION_DETECTION_TEXT_PLACEHOLDERS`); with
`text_reject_sequences` (reference `license_plate` profile), one repeated
character or one ascending / descending run (`999`, `123456`, `XYZ`); or
outside the profile's normalization, `text_len_min`..`text_len_max`, or the
optional `text_format` regex. A rejected VLM reading counts as no reading,
so the OCR reader's valid reading is chosen (and `vlm_then_ocr` runs OCR).
`scripts/curation/rederive_region_text.py` re-applies these rules to stored
rows (dry run by default).

The OCR reader keeps the region's dominant text: lines at least
`text_min_height_ratio` × the tallest line's height, not centered in the
outer `text_border_margin` band of the crop, minus `text_stopwords`,
ordered in rows top-to-bottom / left-to-right, normalized
(`text_uppercase`, `text_charset`), joined with `text_join`, and accepted
only within `text_len_min`..`text_len_max` and above
`text_min_confidence`. With no VLM configured (`OP_VLM_URL` unset) the
worker never calls a VLM:
detector regions are written `detected` with `region_verified=false`
(`<src>:accepted_unverified` on the chain) and their text is read by OCR.

### Item text — `item_text_lines`

Every OCR line read on the item crop by the detection worker (gated by
`OP_ITEM_TEXT_ENABLED`, default on when the region profile names an OCR
pipeline; lines below `OP_ITEM_TEXT_MIN_CONFIDENCE`, default 0.5, are not
stored): a list of `{text, box_norm, confidence, rel_height}` —
`box_norm` is `[x1, y1, x2, y2]` normalized to the item crop,
`rel_height` the line height over the crop height. Always present on the
wire (`[]` when none or not yet read). The normalized search tokens
(`item_text_tokens`, keyword array: each letter/digit word uppercased,
plus each multi-word line with separators removed) are storage-only and
back `GET /crops?item_text=`.

### `region_detector_chain` entries

A list of strings, oldest first, each exactly `<actor>:<event>` — one
colon after the actor, no version, no timestamp (`region_detected_at` /
`region_verified_at` carry the times). Entries are unique within a doc
and capped at 16 (oldest dropped). `<det>` is the profile's primary
detector model, `<seg>` its segmenter, `<ocr>` its OCR recognizer model;
`<src>` is whichever of those produced the candidate box.

| Entry | Meaning |
|---|---|
| `<det>:hit` / `<det>:miss` | primary detector found / found no candidate |
| `<seg>:hit` / `<seg>:miss` | segmenter found / found no candidate |
| `vlm_visible:yes` / `vlm_visible:no` | VLM pre-filter: a region is / isn't visible in the item |
| `<src>:combined_verify_ok` | VLM confirmed the candidate box (region written `detected`) |
| `<src>:combined_verify_reject` | VLM rejected the candidate box |
| `<src>:combined_verify_reject:region_visible_elsewhere` | VLM sees a region and answered `region_bbox_correct=false` for the candidate box |
| `<src>:combined_verify_reject:verifier_no_verdict` | VLM gave no box verdict on every allowed attempt (see below) |
| `<src>:vlm_reject:verifier_no_verdict` | same, from the per-crop cascade's region-only verify call |
| `vlm_visible:no_verdict` | visibility pre-filter gave no verdict on every allowed attempt; sent on to detection (fail open) |
| `<src>:combined_no_region_visible` | VLM sees no region at all |
| `<src>:sanity_reject:<reason>` | box failed the geometry gate (`<reason>` e.g. `aspect`) |
| `<seg>:skip_vlm_verify` | high-score segmenter box written without a VLM call |
| `<src>:accepted_unverified` | no VLM configured: box written `detected` with `region_verified=false` (text from OCR) |
| `<ocr>:text_hint:hit` / `:miss` / `:no_region_shape`, `<seg>:text_hint:miss` | OCR-hinted segmenter re-pass |

A VLM reply that sees a region but gives no box verdict
(`region_bbox_correct` `null`, absent, or a quoted null) is not a reject:
nothing is written and the item stays pending for a retry, so no chain
entry is stored for it. An unparseable / missing combined entry is
treated the same way.
Likewise an empty reply to the visibility pre-filter is no verdict (never
`vlm_visible:no`): the item stays pending and is retried. `POST
/vlm/region_visible_batch` leaves such crops out of its `visible` map.
`POST /vlm/verify_region_batch` and the single-crop `verify_regions`
path have the same contract: a crop the VLM gave no verdict for is
omitted (batch) or left untouched (single) rather than written as
`is_region=False` / `verified=False`.

A combined-reply entry that nests its answer fields one level down under
an invented key (some reasoning-model replies do this instead of the flat
shape the prompt asks for) is unwrapped when there is exactly one
dict-valued key carrying the expected fields; two or more such candidates
is ambiguous and the entry is left as a no-verdict.

The VLM runs at temperature 0, so a no-verdict reply is often
deterministic. Retries are bounded per item and stage by
`OP_REGION_WORKER_MAX_NO_VERDICT_ATTEMPTS` (default 3, counted in the
worker process, reset by a restart). At the cap the combined stage writes
`verify_rejected` with `region_rejection_reason=verifier_no_verdict`, the
candidate kept in `region_candidate_*` and `region_bbox_correct=null` (no
verdict was given), so a human can confirm it or it can be retried with
`requeue_regions.py --status verify_rejected --reason verifier_no_verdict`;
the visibility stage sends the item on to detection (fail open). The
per-crop cascade (`_process_crop`: region-only `verify_plate` and its
combined cohort path) parks its candidate the same way after the same
number of no-verdict passes. A VLM transport failure (no reply at all) is
not a no-verdict reply: it is retried and never counted
(`label_combined_batch` raises `CombinedTransportError`; the worker
calls `verify_plate(..., raise_on_transport=True)`, which raises
`VlmTransportError`; without the flag it still returns `None`).

The combined call marks the candidate box with a red rectangle drawn just
*outside* the box (so it never covers the region's own pixels) and its
prompt says so.

Readers match whole entries with `term` queries — e.g. `GET
/regions/training_candidates?mode=detector_blind_spots` requires
`<det>:miss`, `mode=disagreement` requires both `<det>:hit` and
`<seg>:hit`. Builds before 2026-09-24 wrote `<actor>::<event>@<iso>`;
the worker rewrites such a chain to this form the next time it writes
the doc.

### VLM class suggestion — `vlm_proposed_class_id` / `vlm_proposed_class_name`

On every item, always present, derived from the stored doc by
`vlm_suggestion()` (`src/services/curation/class_sources.py`):

| Stored state | `vlm_proposed_class_id` | `vlm_proposed_class_name` |
|---|---|---|
| `class_source` is `vlm` or `vlm_reclassified`, `class_validated` false, `class_id` set | `class_id` | `class_name` |
| `class_source` is `vlm_new_class_pending`, `class_validated` false | `null` | the proposed new class name (stored `vlm_proposed_class`; `null` if absent) |
| anything else (incl. `vlm_unmatched`, any validated class, non-VLM sources) | `null` | `null` |

When the VLM's answer resolves to a registry class it is **applied**:
`class_id`/`class_name` are already that class, `class_validated` stays
false. The suggestion keys just mark "this class is the VLM's, not yet
confirmed". A `vlm_new_class_pending` item keeps whatever class it had
before (often none); only the name is suggested. The stored
`vlm_proposed_class` field can go stale after a later relabel — the wire
keys are keyed off `class_source`, so a stale value never leaks.

**Accepting a suggestion** (no dedicated endpoint):

- Registry class (`vlm_proposed_class_id` not null): `PUT /crops/{crop_id}/label`
  `{"class_id": <vlm_proposed_class_id>}` (bulk: `PUT /crops/batch_label`
  `{"crop_ids": [...], "class_id": ...}`). Sets `class_validated=true`,
  `class_source` = `human`, `label_source` = the body's `label_source`
  (`human` default, or `human_confirmed` for an accepted suggestion); both suggestion keys become `null`.
- New class (`vlm_proposed_class_id` null, name set): `POST /classes`
  `{"name": <vlm_proposed_class_name>}` -> `{"class_id": N, ...}` (`409` if
  the name exists — then use `GET /classes` to find its id), then
  `PUT /crops/{crop_id}/label` / `PUT /crops/batch_label` with `class_id: N`.
  The label call does not clear the stored `needs_new_class` flag.

`GET /review/{tab}`'s `proposed_class_id` / `proposed_class_name` use the
same derivation: when `vlm_proposed_class_name` is not null they equal
the two suggestion keys; otherwise `proposed_class_id` = `class_id` and
`proposed_class_name` = `vlm_raw_class` (the raw unmatched VLM answer) or
`class_name` or `""`. Changes vs before this key existed: a
`vlm_new_class_pending` item's `proposed_class_id` is now `null` (was the
item's unrelated current `class_id`); a VLM-applied class reports the
resolved registry `class_name` (was the raw VLM slug `vlm_raw_class` when
the VLM's new-class answer matched a synonym); a stale `vlm_proposed_class`
on an item that is no longer pending no longer overrides the name.

`GET /regions/suspected_false_positives`: `threshold` is optional — omit it
and the server applies `default_threshold` (`0.35`, served on every
response next to the `threshold` actually used).

`GET /regions` filter params: `page`, `page_size`, `class_id`,
`cluster_id`, `region_cluster_id`, `region_cluster_subid`,
`sort_by_subid`, `max_rank`, `min_score`, `max_score`, `verified`,
`detector`, `text`, `status`, `include_test`. Without `status` only items
carrying a region box are listed; `status=<region_status>` (any value from
`GET /regions/statuses`, else `400`) lists every item in that status
instead, box or not — e.g. `status=verify_rejected` for the verifier
rejections, whose box is `region_candidate_bbox_norm`.

### Verifier-rejected candidates — `region_candidate_*`

When the verifier rejects a detector's box (`region_status=verify_rejected`)
the worker keeps it for review instead of discarding it:
`region_candidate_bbox_norm` (source frame), `region_candidate_score`,
`region_candidate_detector`, `region_candidate_detector_version`,
`region_candidate_source`, plus `region_rejection_reason`
(`region_visible_elsewhere` for a verifier `region_bbox_correct=false`,
`sanity_reject:<gate reason>` for the geometry gate, `verifier_no_verdict`
when the verifier never gave a box verdict) and `region_bbox_correct` for
the verifier verdict (`false`, or `null` for `verifier_no_verdict`).
`GET /regions/vocabulary` serves these reasons as `rejection_reasons`:
`[{id, label, kind, match, label_template}]`, `kind` one of
`model_verdict` / `automatic` / `needs_human`, `match` `exact` or
`prefix` (`sanity_reject:` -- the rest of the stored value is the gate's
reason, substituted for `{detail}` in `label_template`). A human-written
reason is free text and not listed. The candidate is
never an accepted region: `region_bbox_norm` stays `null`, so browse,
clustering and export ignore it. A human reverses the rejection with the
confirm write (see "Region lifecycle"); region undo restores it. An
accepted worker write clears any stale candidate. Items rejected before
this existed carry no candidate (re-queue them to get one).

**Review-queue reachability (DQ-B2 follow-up):** a rejected candidate is
now reachable from `GET /review/regions` — see the `region_status` filter
above. Its default ('all') `region_score` sort falls back to
`region_candidate_score` (a second sort key) for items with no
`region_score`, so rejected candidates sort by their own score instead of
tying on `missing: '_last'` and falling back to shard order among
themselves. `GET /crops/{crop_id}/region_thumbnail` renders the candidate
box when there is no accepted `region_bbox_norm` (404 only when neither
exists) — the thumbnail cache key includes the box's own coordinates, so
a later promotion or re-detection that changes the box is never served
stale.

### SSE — `GET /events`

`crop.region_verified` data: `type`, `topic` (`region_status`),
`crop_id`, `region_status`, `region_text`, `ts`. The data keys other than
`type`/`topic`/`ts` are item keys with the same meaning. The same payload
is produced in-process (`publish_region_verified`) and by the SAM worker
via `POST /events/publish`.

S-3: the hub is cross-process by default (`OP_EVENT_BUS=file`) — every
uvicorn worker process tails the same shared JSONL log
(`{OP_STATE_DIR}/events/events.jsonl`, bounded and rotated at
`OP_EVENT_LOG_MAX_BYTES`) so an SSE client connected to any one worker
sees events published by any other, and by the out-of-process detection
worker's `POST /events/publish` calls. `GET /events/stats` reports the
active `bus` (`file`/`process`) and `log_path` alongside the existing
`subscribers`/`events_published`/`events_dropped` counters.

### `GET /stats/dataset`

- `labeled`: `by_human`, `by_vlm`, `by_classifier`, `other` (F-23: `by_proposal`
  moved to `unlabeled` -- those class_source values never carry a
  `class_id`, so it was structurally always 0 here)
- `regions`: `boxed`, `confirmed`, `total_detected`, `by_detector`,
  `by_segmenter`, `by_human`, `by_human_drew`, `verified_by_human`,
  `verified_by_vlm`, `validated_by_human` (`by_detector` /
  `by_segmenter` / `by_human_drew` are matched against the active
  `DetectionProfile`'s `detector_model` / `segmenter_name` /
  `human_detector_name`; `verified_by_vlm` counts every non-human
  verifier, because the VLM stamps its own model id)
- unchanged: `as_of`, `total_crops`, `validated`, `test_holdout`,
  `by_source`, `unlabeled`, `in_progress`
- `clusters`: `cluster_count` is the number of distinct non-noise
  `cluster_id`s in the index now; `last_run_cluster_count` is the last
  auto-label run's own count (`null` if none recorded — a residual-only
  pass reports just the clusters it made); `last_run_at`, `method`,
  `residual_count`, `noise_count` describe that run

### `GET /export/datasets`

Query: `kind` (`yolo` | `single_class` — the same ids the `/methods`
export axis advertises), `profile_name`. Rows: `kind`, `profile_name`
(`null` for multi-class), `export_dir`, `version_tag`, `image_count`,
`object_count` (`null` when the manifest does not record it, e.g. every
single-class export), `split_counts`, `dataset_sha`, `exported_at`, `class_count`,
`is_current`. Multi-class versions live directly under the export root;
single-class versions under `<export_root>/<profile_name>/<version>/`, and
`is_current` is judged against that profile's own `current` symlink.

### `class_source` values — `GET /class_sources`

`GET {prefix}/class_sources` returns
`{"class_sources": [{"id": str, "label": str, "role": str, "short_label": str}, ...]}` (`short_label`: 1-2 words for badges; `label`: full text for menus/tooltips): every
`class_source` value this deployment can write, built by
`class_source_catalog()` (`src/services/curation/class_sources.py`).
Ingest values come first, derived from the configured ingest profiles
(`OP_INGEST_PRIMARY_*` / `OP_INGEST_SECONDARY_*`, label uses the
profile's `DETECTOR_MODEL`, falling back to its `NAME`):

| `id` | `role` | Present when |
|---|---|---|
| `{primary}_proposal` | `proposal` | always |
| `{primary}_low_conf` | `low_conf` | primary `ASSIGNS_CLASS=true` |
| `{primary}_model` | `model` | primary `ASSIGNS_CLASS=true` |
| `{secondary}_model` | `model` | `OP_INGEST_SECONDARY_DETECTOR_MODEL` set (a secondary `NAME` alone configures nothing) |
| `unlabeled_proposal` | `proposal` | always (item-doc default before a detector stamps a source) |
| `vlm` | `vlm` | always |
| `vlm_unmatched` | `vlm_unmatched` | always |
| `vlm_new_class_pending` | `vlm_new_class_pending` | always |
| `vlm_reclassified` | `vlm_reclassified` | always |
| `cluster_majority_agreement` | `cluster` | always |
| `human` | `human` | always |
| `human_move` | `human` | always (`POST /crops/move`) |
| `class_merge` | `merge` | always (`POST /classes/merge`) |
| `external_label` | `label_import` | always (label-import default) |

`role` enum: `proposal`, `low_conf`, `model`, `vlm`, `vlm_unmatched`,
`vlm_new_class_pending`, `vlm_reclassified`, `cluster`, `human`, `merge`,
`label_import`. Not listed because nothing writes them any more:
`classifier_vlm_agreement`, `vlm_human_confirmed` (still recognised by
queries/rollups; may appear on older docs). The label endpoints and label
import take a caller-chosen `label_source` (defaults `human` /
`external_label`) that is stored as `class_source`, so a client that
passes its own value can see ids outside the catalog — render unknown
ids verbatim. `tests/curation/test_class_sources.py` scans every
`class_source` write in `src/` and `scripts/` and fails if a written
value is missing from the catalog. The
`classifier_confidence_skip_vlm` skip, auto-promote, the
`primary_low_conf` / `classifier_blind_spots` review tabs and the
`/stats/dataset` rollup all filter on these derived sets, never on one
deployment's detector names.

## Errors: read endpoints fail closed

A backend outage is a `503`, never an empty or zero answer that reads as
real data. `GET /ingest/region_drain` (its `total_unfinished: 0` is the
"worker caught up" signal), `GET /ingest/status`, `GET /classes` (live
counts) and `GET /stats/classes` (registry join) used to answer zeros /
empty lists on failure and now `503`. Single-item reads added in this
wave (`GET /crops/{id}/history`, `GET /review/{tab}/locate`) answer `404` /
`not_found` only when the item doesn't exist and `503` on an outage.

## What is explicitly NOT on the wire

- **Backend OpenSearch field names** for region attributes — governed by
  `RegionFields` (`src/config/region_fields.py`), overridable per
  deployment via `OP_REGION_FIELD_*` (see `env.template`). They pick
  where a value is read from and written to; the wire name is fixed.
- **`RegionStatus` enum values** in `src/config/region_state.py` are
  values, not field names (B2 renamed `no_plate_box` /
  `no_plate_visible` to `no_region_box` / `no_region_visible`).
- **The `/curation` URL prefix itself** — a config field
  (`CurationConfig.api_prefix`, env override `OP_API_PREFIX`) that
  defaults to `/curation`. A deployment may run behind a different
  prefix; consumers should not hardcode `/curation` or any other fixed
  prefix.

## H3/H4 — historical decisions

**H4 — superseded by B3.** An earlier ruling kept historical,
domain-named field names on the wire and closed a storage reindex as
WONTFIX. B3 moves the *wire* to generic names; storage names stay
configurable via `RegionFields` exactly as before, so no reindex is
required of any deployment. The wire contract above is already fully
decoupled from OpenSearch storage field names via `RegionFields`; a
storage rename is invisible to any consumer by construction, so its
cost (a full reindex against a live deployment) would buy nothing a
client can observe.

**H3 — open items, recorded here for consumers:**

- **D1** (`annotation_slots` on `GET /classes`): not added. A tier-2
  static-profile loading path isn't wired into any consumer yet; adding
  a server field with zero consumers would freeze a wire commitment
  before the design is exercised. Publish a consumer's slot-spec draft
  first.
- **D2** (a client-side status-enum codegen's ownership): recommendation
  is to retire that codegen and let a consumer's own slot profile be
  the source of truth. Not yet actioned.
- **D3** (field-mapping table ownership): this doc's per-model field
  lists above are hand-maintained and can drift from
  `src/routers/curation/_common.py` (see the caveat at the top of the
  Pydantic-models section). Recommendation is a generated,
  test-enforced table here rather than a hand-written one. Not yet
  built.
- **D4** (`POST /train/candidates`): concur with not scheduling it —
  the current tier-1 cohorts cover the common case; only a second
  capable slot would justify it. No action planned.

## Coordination notes for consumers

### B2 — domain-named routes removed from the public surface (BREAKING)

Pure 1:1 renames — no handler, filter, or semantic change, and no
statuses were merged:

| Kind | Change |
|---|---|
| Routes | Every domain-named `/plates*` route (list, training-candidates, batch-status, clustering, false-positive centroids) moved to the equivalent `/regions*` route; the three `/crops/{crop_id}/...` region routes lost their domain-named segment |
| Status values | The two domain-named "no box"/"no detection visible" status strings became `no_region_box` / `no_region_visible` |
| Cohort `mode=` | The two domain-detector-named cohort modes became generic `detector_blind_spots` / `low_conf_correct` |
| `GET /review/{tab}` tab | The domain-named tab id became `regions` |

The three `/crops/{crop_id}/...` renames bring those routes in line with
their already-generic sibling `GET /crops/{crop_id}/region_thumbnail`.

B2 deliberately left every domain-named item JSON key alone; B3 below
renames all of them. The `disagreement` / `human_corrected` /
`false_positives` cohort modes were already generic.

Deployments carrying documents written before B2 need a one-off
`update_by_query` rewriting the two status strings; nothing else in
storage changes.

### B3 — one generic wire vocabulary (BREAKING)

Fresh deployments re-ingest, so no data migration is provided.

Every previously domain- or vendor-named wire surface moved to the
generic vocabulary used throughout this doc:

- **Item keys** (all item endpoints): every domain-prefixed region
  attribute became `region_<attr>` (`region_bbox_norm`,
  `region_status`, …; full list under "Item wire format"), including
  the thumbnail URL key.
- **VLM- and classifier-named fields**: every vendor-model-prefixed
  confidence/label/cluster field on the wire and in storage moved to
  the generic `vlm_*` / `classifier_*` prefix (e.g. `vlm_confidence`,
  `vlm_raw_label`, `vlm_proposed_class`, `vlm_item_make`,
  `classifier_raw_confidence`).
- **`class_source` / `label_source` values**: every VLM-named value
  (unmatched, new-class-pending, reclassified, human-confirmed) moved
  to the `vlm_*` prefix; the classifier/VLM-agreement values became
  `classifier_vlm_agreement` / `cluster_majority_agreement`; hardcoded
  detector-model-named source values were replaced by the configured
  ingest profiles' own values (`{secondary}_model`, `{primary}_proposal`,
  `{primary}_low_conf`, …).
- **Proposal naming**: the detector-family-prefixed proposal-name key
  became the generic `proposal_name`.
- **`region_detector_chain` entries**: every VLM-named verify/visible
  action moved to `vlm_*` / `combined_verify_*` naming (full vocabulary
  under "`region_detector_chain` entries").
- **Writer id, review tab, review reason text**: the VLM-named history
  writer id, the `GET /review/{tab}` low-confidence VLM tab, and the
  human-readable `reason` strings all moved off vendor/model names onto
  generic wording ("VLM's reply did not match…", "region detected —
  needs human confirmation", etc).
- **Query params**: `GET /crops` dropped its classifier-vendor-prefixed
  confidence filter for `classifier_conf_lt`, and gained `limit`, `sort`,
  `conf_min`, `conf_max`, `k`; `GET /regions` moved its cluster filters
  to `region_cluster_id` / `region_cluster_subid`; `POST
  /pipeline/auto_label[/start]` moved every VLM-vendor-prefixed
  parameter to `vlm_*` naming; `GET /export/datasets` gained `kind` and
  `profile_name` (rows gain the same two fields).
- **Request/response bodies**: `PUT /crops/{id}/region`, `PUT
  /crops/batch_region`, `PATCH /crops/{id}/region_meta`, `POST
  /regions/batch_status`, and `POST /events/publish` all moved their
  domain-prefixed keys (`bbox_norm`, status/text/rejection-reason,
  `label_source`) onto the fixed `region_*` wire names; `POST
  /events/publish` now rejects unknown keys with `422`.
- **SSE**: `crop.region_verified` data and its `topic` moved from
  storage field names to the fixed `region_status` / `region_text`
  wire names.
- **Stats**: `GET /stats/dataset`'s classifier-vendor-named labeled
  bucket became `labeled.by_classifier`; its
  domain-named `plates` block became `regions` with
  `regions.by_detector` / `regions.by_segmenter` /
  `regions.verified_by_vlm`.
- **Health**: `GET /health`'s vendor-named field became `vlm`.
- **Ingest response**: the domain-named region count became
  `n_regions`.
- **Cluster cards**: `GET /regions/clusters`' `dominant_class_name` no
  longer hardcodes the example domain's class name — it reports the
  active region profile's own `region_class_name`.
- **New route**: `GET /classes/{class_id}`.
- **Env vars**: every vendor-prefixed VLM/segmenter connection
  variable moved to the `OP_VLM_*` / `OP_SEGMENTER_*` prefix (see
  section 3 of the naming sweep for the full old→new table; the old
  names are retired with no fallback — see `src/config/retired_env.py`).

Not renamed, deliberately: Prometheus metric names (a later wave), an
internal-only GPU-arbiter Python alias (not wire), and the review tab
id `classifier_blind_spots` (a proposer-named tab id, left for the
owners to decide; it now filters on the configured proposal sources —
`GET /review/tabs` serves it a generic "Classifier blind spots" label).
An internal-only classifier-embedding storage field (never on the
wire) was renamed to `backbone_embedding` in the naming sweep's stored-
data wave — the `CurationConfig.BACKBONE_EMBEDDING_FIELD` constant, not
a wire key.

- This doc is the shared source of truth for the `/curation` API. Point
  any consumer's docs here instead of duplicating the field list.
- No wire-format change ships without a corresponding update to this
  doc; `tests/curation/test_wire_contract.py` pins the item key set.
- A route-parity CI guard
  (`tests/integration/test_labeler_route_parity.py`) keeps this doc's
  route table honest against `app.routes` for any consumer migrating
  onto this contract.

### 2026-09-24/25 visual + ingest audit batch (X2, D1, R5, R10, L3, M2, R4, E2, K6, K3, BA-1..7, C1, C3)

Fixes and new routes from a 2026-09-24 frontend visual audit and an
ingest hardening pass. Grouped by area; each item's wire shape is exact
JSON, not illustrative.

**`GET /classes` / `GET /classes/{id}` (X2)** — `ClassEntry` gained
`kind: 'item' | 'region'` and `trainable` / `trainable_gap`. A class
whose name equals the active region profile's `region_class_name` is
marked `kind: 'region'`; its `sample_count` / `validated_count` /
`cluster_size` are the real (usually 0) item-class-aggregation numbers,
**no longer** overridden with the region inventory total — that
inflated `/train`'s class picker and `/export`'s per-class table.
`trainable = validated_count - test_holdout - class_excluded`,
`trainable_gap = max(0, block_below - trainable)`. Same two fields added
to `GET /stats/classes`'s `classes[]` rows.

```json
{"class_id": 80, "class_name": "defect", "kind": "region",
 "sample_count": 0, "validated_count": 0, "cluster_size": 0,
 "trainable": 0, "trainable_gap": 20}
```

**`GET /stats/dataset` (D1)** — `labeled.*` is now built only from docs
that carry a `class_id`; a `class_source` alone (e.g. `vlm_unmatched` /
`vlm_new_class_pending` with no class) no longer counts as
`labeled.by_vlm`. New `unlabeled.vlm_no_class`: the subset of
`no_label_source` where a VLM answered/proposed but never landed a
class. F-23: `unlabeled.by_proposal` -- the fixed accounting for
'detector proposed it, nothing has classified it yet' (moved from the
always-0 `labeled.by_proposal` above) -- is a second, disjoint subset
of `no_label_source`. `by_proposal + vlm_no_class` can equal
`no_label_source` exactly (every unclassified crop happens to be one
or the other) without either counting the other's docs.

```json
{"labeled": {"by_human": 174, "by_vlm": 3200, "by_classifier": 3551, "other": 0},
 "unlabeled": {"pending_detection": 0, "pending_verification": 0, "no_label_source": 1252, "vlm_no_class": 1036, "by_proposal": 216}}
```

**`GET /review/new_class_proposals` + `/summary` (R5)** — the queue
(and everything built on `build_tab_query`'s `new_class_proposals`
branch) now excludes: an item whose last VLM attempt was
`vlm_class_empty_reason=no_answer` (nothing was proposed), and an item
that already carries a resolved `class_id` (a later write settled it;
`needs_new_class` was stale). No field removed; fewer rows.

**`GET /review/regions` / any tab's per-item `reason` on a
`verify_rejected` region (R10)** — reworded from the served
rejection-reason vocabulary (`region_rejection_reason` +
`GET /regions/vocabulary`'s `kind`: `model_verdict` / `automatic` /
`needs_human`) instead of embedding the raw id. Exactly one verb per
reason: `needs human review: ...` for `needs_human`, `rejected: ...`
otherwise.

```json
{"reason": "needs human review: verifier gave no verdict"}
{"reason": "rejected: the detection is wrong (region is elsewhere)"}
```

**`GET /review/new_class_proposals/summary`'s `term_rules` (L3)** — the
served non-object term rules now support a prefix (`unidentifiable_*`)
and suffix (`*_scene`) form, matched against the whole term or any of
its `_`-separated tokens — one configured rule now flags a family of
terms instead of needing every literal enumerated. Shape unchanged
(`non_object_terms: string[]`); only the matching semantics of entries
ending/starting with `*` changed.

**`GET /models/status` (M2)** — the fixed roster now also includes the
configured primary item proposer (`OP_INGEST_PRIMARY_*`), the optional
secondary classifier (`OP_INGEST_SECONDARY_*`, omitted when
unconfigured) and the segmenter (`DetectionProfile.segmenter_name`), in
addition to the region detector and OCR det/rec already served. Same
entry shape as every other model (`name`, `friendly_name`, `role`,
`kind`, ...).

**`GET /methods` sort catalog (R4)** — `classifier_blind_spots_default`'s
served `label` changed from `"Largest COCO blind spot"` to `"Largest
classifier blind spot"`.

**Export manifest / `GET /export/datasets` / `GET /export/status`
(E2)** — new `classes_with_objects` alongside `class_count`.
`class_count` stays the registry size written into `data.yaml`'s `nc`;
`classes_with_objects` is how many of those classes have at least one
labeled object in this export.

```json
{"class_count": 84, "classes_with_objects": 5}
```

**`GET /curation/health` (T1)** — new `mlflow_public_url: string | null`
(`CurationConfig.mlflow_public_url` / `OP_MLFLOW_PUBLIC_URL`), the
browser-reachable MLflow base a client should build run links from
instead of guessing a port.

**`GET /crops/{id}/image` (K6, BREAKING)** — no longer draws any
overlay. Always the clean source render (EXIF-transpose + RGB-convert +
optional `max_dim` downscale), byte-identical regardless of the item's
`bbox_norm` / region box. A client draws every box itself from
`GET /crops/{id}/context`, which already serves `bbox_norm`,
`region_bbox_norm`, `region_candidate_bbox_norm` (all source-image
normalized, per-item `region_bbox_frame='source'`), `class_id` /
`class_name`, `region_status`, `region_rejection_reason` and validation
flags for every item on the frame — plus the image's `width`/`height`,
now filled from the file header when the images-index doc doesn't carry
them and the image is servable.

**K3 (data hygiene, no wire change)** — writers audited: an empty VLM
class answer already leaves `label_source` untouched (records only the
attempt, per `vlm_class_attempt.py`). A new operator script,
`scripts/curation/repair_stale_label_source.py`, clears a stale
`label_source` on any class-less item (`class_id` missing) regardless of
`class_source` — dry-run by default.

**`POST /probe/run`, `GET /probe/status`, `POST /probe/cancel` (C1,
new)** — wraps `run_probe_inference` as a background job instead of
blocking the request. `POST /probe/run` resolves `job_id`'s
`checkpoint_path` from the training job's status (409 if the run isn't
`finished` or has no checkpoint on disk); `probe_model_version` is
stamped as the job id. `gpu` (a `cuda_visible_devices` string) claims
through the same arbiter `POST /train/start` uses — a claim failure is
`409`, never silent. One job at a time (`409` otherwise).

`GET /probe/status` additionally serves `actionable_min_confidence`
(read-only, always present, independent of job state) — a direct echo of
`CurationConfig.probe_actionable_min_confidence`, so a client can render
"model unsure" copy without hardcoding the threshold.

```json
// POST /probe/run {"job_id": "2026-09-25T00-46-20_yolo26n", "gpu": null}
{"job_id": "2026-09-25T00-46-20_yolo26n", "status": "running",
 "train_job_id": "2026-09-25T00-46-20_yolo26n",
 "model_path": "/jobs/.../weights/best.onnx", "gpu": null,
 "started_at": "2026-09-25T00:00:00+00:00", "finished_at": null,
 "updated_count": null, "error": null, "actionable_min_confidence": 0.5}
```

**`GET /review/{tab}` + `GET /review/tabs` (C3, new field)** — a
zero-result page now carries `empty_reason`, computed from live index
state: `"no probe predictions — run a probe"` (uncertainty /
model_disagreements with no probe-scored item), `"item scores never
computed"` (a `min_mistakenness` filter was set but no item has
`mistakenness_score`), `"no unclassified proposals"`
(`new_class_proposals`), else `"no items match"`. `GET /review/tabs`
gained `empty_state: {has_probe_predictions, has_item_scores}` so a
client can word ANY tab's empty state without a per-tab round trip.
`GET /review/{tab}=all` when items exist: `empty_reason: null`.

### BA-1..BA-7 — ingest hardening

**BA-1 (blocking, breaking wire shape for uploads)** —
`POST /ingest/upload` now persists uploaded bytes server-side, content
addressed, under `CurationConfig.upload_root` /
`OP_UPLOAD_ROOT` (default under the state dir):
`<upload_root>/<imohash[:2]>/<imohash><ext>`, written atomically
(temp file + `os.replace`), so the same bytes are only ever stored
once. `image_path` on the images doc and in every `IngestImageResponse`
is now that persisted, servable path — **not** the client's identifier.
The client's identifier moves to a new field, `source_identifier`
(images-index mapping migration `ensure_images_upload_fields`, wired
into the existing startup migration sequence).
`POST /ingest/path_lookup` matches on `image_path` **or**
`source_identifier`, keying its result by whichever field matched.

```json
// POST /ingest/upload response row
{"status": "success", "image_id": "...", 
 "image_path": "/var/lib/openprocessor/uploads/ab/ab12.../ab12....jpg",
 "source_identifier": "remote://shoot1/a.jpg",
 "imohash": "ab12...", "n_crops": 3, "n_regions": 0,
 "error": null, "error_kind": null}
```

**BA-2** — `GET /ingest/config` (new), typed:

```json
{"upload": {"enabled": true, "max_images_per_request": 128,
            "max_bytes_per_request": 536870912,
            "accepted_extensions": [".jpg", ".jpeg", ".png"],
            "persists_bytes": true},
 "batch": {"enabled": true, "max_items": 512, "source_roots": ["/data/images", "/var/lib/openprocessor/uploads"]},
 "region_drain": {"poll_interval_s": 10.0, "stable_polls": 3}}
```

All three limits are real `CurationConfig` fields
(`OP_UPLOAD_MAX_IMAGES_PER_REQUEST`, `OP_UPLOAD_MAX_BYTES_PER_REQUEST`,
`OP_UPLOAD_ACCEPTED_EXTENSIONS`, `OP_BATCH_MAX_ITEMS_PER_REQUEST`) and
enforced: `POST /ingest/upload` 413s over the image-count or
total-request-byte limit, and fails an individual item
`error_kind: 'unsupported_type'` for an extension outside
`accepted_extensions`.

**BA-3** — `GET /ingest/region_drain` gained a server-computed
stability verdict (`src/services/curation/region_drain.py`):

```json
{"pending_detection": 0, "pending_verification": 0, "total_unfinished": 0,
 "drained": true, "stable_for_s": 32.4, "observed_at": "2026-09-25T00:00:32+00:00"}
```

`drained` is true only after `total_unfinished` has read `0` for
`region_drain.stable_polls` (`OP_REGION_DRAIN_STABLE_POLLS`, default 3)
consecutive polls of this endpoint — single-process, in-memory; a
multi-worker deployment polling from different processes tracks
independent streaks (each worker's own view of "stable", never a
correctness issue for the raw counts).

**V-1** — the response also carries `region_dependencies` and
`stall_reason` (`src/services/curation/region_dependency_health.py`),
so a queue that isn't shrinking has a visible cause instead of reading
as a flat, unexplained pending count:

```json
{"pending_detection": 3516, "pending_verification": 0, "total_unfinished": 3516,
 "drained": false, "stable_for_s": 0.0, "observed_at": "2026-09-25T14:05:00+00:00",
 "region_dependencies": [
   {"role": "detector", "model": "region_detector_v1", "ready": true, "unavailable_since": null, "detail": "READY"},
   {"role": "segmenter", "model": "sam3", "ready": false, "unavailable_since": "2026-09-25T14:02:11+00:00", "detail": "not in Triton repository index (never loaded)"}
 ],
 "stall_reason": "3516 item(s) awaiting region detection; segmenter (sam3) unavailable since 2026-09-25T14:02:11+00:00"}
```

`region_dependencies` is checked directly against Triton's own
repository index from the API process (which can always reach Triton
over the network, unlike probing the detection worker container, whose
heartbeat is written to a container-local path the API can't see) —
empty when no region profile is configured at all (the neutral/off
case, not a stall). `stall_reason` is null whenever nothing is pending
or every dependency is READY (the worker just hasn't caught up to a
backlog yet, which is not a stall). `GET /stats/dataset`'s
`in_progress.region_stall_reason` mirrors the same computation for the
dashboard's "In-flight pipeline" panel.

**Item behavior when a dependency is down (design decision, not a code
change):** items simply stay in `pending_detection` — by design, the
detection worker never writes a terminal region status on an infra
failure (see `scripts/curation/worker/cascade.py`'s `_process_crop`
docstring), so they're already retryable the moment the dependency
recovers, with no dequeue/requeue logic needed. A new `region_unavailable`
terminal-ish status was considered and rejected for this pass: it would
touch the worker's state machine (`RegionStatus`, `region_state.py`'s
writable-status set, the cascade's retry path) with no way to exercise
that change against a live worker in this pass (no GPU/compose
available) — the observability fix above (surface *why* it's stalled)
covers the operator-facing gap without that risk.

**BA-4** — `POST /ingest/upload` gained an optional `run_id` form
field, recorded as `ingest_run_id` on every image doc from that call.
`GET /ingest/status?run_id=` scopes `total`/`by_source`/`by_day` to it.

**BA-5** — `label_txt_path` on `IngestBatchItem` (`POST /ingest/batch`)
now gets the same root guard `image_path` already had — a
label file path outside the configured source roots fails that item
(`error_kind: 'unservable_path'`) before any read. Batch item cap (`CurationConfig.batch_max_items_per_request`) served on
`GET /ingest/config` and enforced on `POST /ingest/batch` (413 over the
cap).

**BA-6** — `GET /ingest/status` is now a typed `IngestStatusResponse`
(`total`, `by_source`, `by_day`); shape unchanged, just declared.

**BA-7** — every `IngestImageResponse` / batch result row gained
`error_kind: string | null` — one of `empty`, `unservable_path`,
`unsupported_type`, `decode_failed`, `detector_infer`, `bulk_index`
(non-exhaustive; always present alongside `error` when
`status == 'failed'`). `unidentified_image` and `decode_error` were
merged into `decode_failed`.

**`GET /models/status` (M3, new fields)** — every roster entry now
carries `optional: bool` (default `false`). It's `true` only for the
active profile's region detector, and only when a segmenter is also
configured as its fallback — mirroring
`region_dependency_health.stall_reason`'s "a ready segmenter means a
down detector isn't a stall" semantics. When that detector is entirely
absent from Triton's `/v2/repository/index` (never shipped/installed —
e.g. the public `license_plate` example profile's
`license_plate_detector`, which has no public model), `status` is a new
value, `not_installed`, instead of the generic `not_ready`. A model
present in the index but not `READY` (unloaded, failed) keeps the
unchanged `not_ready` status regardless of `optional` — this only
changes the "entirely missing from the index" case. A client renders
`not_installed` as "optional, not installed" rather than a red NOT
READY.

```json
{"name": "license_plate_detector", "friendly_name": "Region Detector",
 "kind": "triton", "status": "not_installed", "optional": true, ...}
```
