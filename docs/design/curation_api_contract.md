# OpenProcessor `/curation` API contract

Status: **living reference doc**, owned by this backend. Originated
alongside the generic `curation` subsystem's initial port; rescoped by
`docs/design/cropwright_backend_integration_plan.md` §0.1/T-E1. It
documents the *currently shipped* `/curation` route surface (mounted
under `CurationConfig.api_prefix`, default `/curation` — landed on
`main` in commit `1079933`) and the Pydantic wire-model field names it
serves, and states explicitly which parts of that contract are frozen.

**This is OpenProcessor's generic curation/labeling API contract, not
"the labeler's API."** Cropwright (a SvelteKit active-learning labeling
frontend) is **one consumer** of this API — the first one, and the one
this contract was originally drafted against. Other services are
anticipated on the same backend: querying,
visualizing, and searching the same indexed dataset. None of them exist
yet, and none of them should have to learn Cropwright's historical URL
vocabulary to consume this API. Every recommendation and naming choice
in this doc follows from that: the canonical surface is the generic one
(`/curation`, `vlm`, `region_thumbnail`, `RegionFields`-backed storage),
and it does not move to accommodate any one consumer.

**No `/legacy` or `/gemma` compatibility alias exists in this backend, and
none ever will.** Where a historical consumer's naming differs from the
generic one, the consumer migrates. See
`docs/design/cropwright_backend_integration_plan.md` §0.1 and §11
acceptance criterion 8. A transitional `/legacy` prefix may appear
*temporarily, on the frontend side only*, as a deployment convenience
during that migration (same doc, §2) — it is never a supported backend
default and never dual-mounted.

## The key invariant: one generic wire vocabulary, independent of storage names

Every request and response on this API uses one generic vocabulary
(decided 2026-09-23 by both teams' owners; see "B3" under Coordination
notes for the full old→new table). The earlier rule that froze the
historical `plate_*` / `gemma_*` wire names is **retired**: fresh
deployments re-ingest, so there was no legacy data to protect.

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
| `classes.py` | `GET /class_sources`, `GET,POST /classes`, `POST /classes/merge`, `POST /classes/sync_to_opensearch`, `GET,PUT /classes/{class_id}`, `GET /classes/{class_id}/crops` |
| `crops.py` | `GET /crops`, `GET /crops/{crop_id}`, `PUT /crops/{crop_id}/label`, `PUT /crops/batch_label`, `POST /crops/move`, `POST /crops/flag_new_class`, `POST /crops/batch_exclude`, `POST /crops/batch_unexclude`, `POST /crops/{crop_id}/review_dismiss` |
| `label_undo.py` | `POST /crops/{crop_id}/label/undo`, `POST /crops/label/undo_batch`, `DELETE /crops/{crop_id}/label`, `POST /crops/{crop_id}/discard`, `POST /crops/discard_batch`, `POST /crops/{crop_id}/vlm_dismiss`, `POST /crops/{crop_id}/review_undismiss`, `GET /crops/{crop_id}/history` |
| `crop_context.py` | `GET /crops/{crop_id}/context` |
| `cohorts.py` | `GET /training_cohorts` |
| `regions.py` / `regions_fp.py` | `GET /regions`, `GET /regions/statuses`, `PUT /crops/{crop_id}/region`, `PUT /crops/batch_region`, `PATCH /crops/{crop_id}/region_meta`, `POST /regions/batch_status`, `POST /regions/cluster`, `GET /regions/cluster/status`, `GET /regions/clusters`, `POST /regions/clusters/refine/{cluster_id}`, `POST /regions/fp_centroids/build`, `GET /regions/fp_centroids/status`, `GET /regions/suspected_false_positives`, `GET /regions/training_candidates`, `GET /crops/{crop_id}/region_thumbnail` |
| `events.py` | `GET /events`, `POST /events/publish`, `GET /events/stats` |
| `export.py` | `POST /export/yolo`, `GET /export/datasets`, `GET /export/status`, `GET /export/registry/{artifact}` |
| `export_single_class.py` | `POST /export/single_class`, `GET /export/single_class/status` |
| `ingest.py` | `POST /ingest/image`, `POST /ingest/batch`, `POST /ingest/upload`, `POST /import_labels`, `POST /import_labels/batch`, `GET /ingest/status`, `GET /ingest/sam_drain`, `POST /ingest/path_lookup` |
| `models.py` | `GET /health`, `GET /models/status`, `DELETE /models/{model_name}` |
| `search.py` | `GET /search/text` |
| `stats.py` | `GET /stats/classes`, `GET /stats/dataset` |
| `pipeline.py` / `pipeline_control.py` / `pipeline_events.py` | `POST /pipeline/auto_label`, `POST /pipeline/auto_label/start`, `GET /pipeline/auto_label/status`, `GET /pipeline/auto_label/status/{job_id}`, `POST /pipeline/auto_label/cancel`, `POST /vlm/label_cluster/{cluster_id}`, `GET /pipeline/events` |
| `clusters.py` / `viz.py` | `GET /clusters`, `GET /clusters/representatives`, `POST /clusters/auto_promote`, `POST /clusters/refine/{cluster_id}`, `GET,POST /viz/projection*`, `POST /cluster/umap/rebuild` |
| `review.py` / `scores.py` / `select.py` / `methods.py` / `settings.py` | `GET /review/{tab}`, `GET /review/{tab}/locate`, `GET /review/new_class_proposals/summary`, `POST /review/new_class_proposals/resolve`, `GET /review/raw_label_clusters`, `GET /review/unmatched_terms`, `POST /test_holdout/freeze`, `GET /test_holdout/stats`, `POST,GET /scores/*`, `POST,GET /select/*`, `GET /methods`, `GET,PUT /settings` |
| `vlm.py` | `POST /vlm/label_batch`, `POST /vlm/verify_regions`, `POST /vlm/verify_region_batch`, `POST /vlm/region_visible_batch` |
| `bakeoff.py` | `GET,POST /bakeoff/*` |
| `curation_images.py`, `curation_train.py`, `curation_umap.py` (outside the `curation` package, registered directly in `src/main.py`) | `GET /images/*`, `POST,GET /train/*`, `POST /cluster/umap/rebuild` |

The exact, always-current list is produced by:

```python
from src.main import app
routes = sorted(r.path for r in app.routes if r.path.startswith('/curation'))
print(len(routes)); print('\n'.join(routes))
```

Note the URL segment is `vlm`, not `gemma` — `gemma_labeler.py` was
generalized into a pluggable `vlm_client`/`vlm_labeler`/`vlm_prompts`
abstraction (a deployment need not run Google's Gemma at all). No
`/gemma/*` route is registered, and none will be added; see the VLM
section below.

## Wire models

Model class names reflect `src/routers/curation/_common.py`; this table
is hand-maintained (see D3) — treat `_common.py` as authoritative if the
two disagree. `ItemDoc`'s field set is test-pinned to the serializer's
output (`test_item_doc_model_documents_exactly_the_serializer_keys`).

### Ingest

- `IngestImageRequest`: `path`, `source`
- `IngestImageResponse`: `status` (`success`/`duplicate`/`failed`), `image_id`, `image_path`, `imohash`, `n_crops`, `n_regions`, `error`
- `BatchIngestSummaryResponse`: `successful`, `duplicates`, `failed`, `mismatches`, `missed_labels`, `unmatched_detections`, `labels_imported`, `crops_indexed`
- `BatchIngestResponse`: `status` (`success`/`partial`/`error`), `summary`, `results`, `disagreements` (with `detect_mismatches`: one record per model-vs-label disagreement, `kind` = `class_mismatch`/`missed_label`/`unmatched_detection`; also returned by `POST /import_labels/batch`)
- `POST /ingest/upload` (multipart): `images` (files), `image_paths` (JSON list of identifiers, optional), `source` -> `BatchIngestResponse`
- `ImportLabelsRequest`: `image_path`, `label_txt_path`, `label_source`
- `ImportLabelsBatchRequest`: `items`

### Crops

- `ItemDoc`: the shared wire item — see "Item wire format" below for the exact key list. Documentation/OpenAPI model only: handlers return the serializer's dict directly, so an unexpected stored value type never 500s a browse page.
- `CropsPageResponse`: `total`, `page`, `page_size`, `crops` (list of items), `method`, `version`, `n_pool`
- `CropLabelRequest`: `class_id`, `label_source` (`human` default or `human_confirmed` — any other value is a `422`; the server always writes `class_source: "human"` for this write, so a client can't make a human label look machine-written)
- `CropBatchLabelRequest`: `crop_ids`, `class_id`, `label_source` (same rule). Response: `updated`, `updated_ids` (exactly the crops written — the ids to pass to `undo_batch`), `conflicts` (`[{crop_id, current_source}]`, not written)
- `CropMoveRequest`: `crop_ids`, `cluster_id`. Response: same shape as `batch_label` (`updated`, `updated_ids`, `conflicts`)
- `CropExcludeRequest`: `crop_ids`, `reason`
- `CropUnexcludeRequest`: `crop_ids`
- `CropUndoBatchRequest` (`POST /crops/label/undo_batch`): `crop_ids`
- `ItemRegionRequest` (`PUT /crops/{crop_id}/region`): `region_bbox_norm` (`[x1,y1,x2,y2]` in `frame`, or `null` = "no region visible"), `region_label_source` (default `human`), `frame` (`source` default = source-image frame; `parent` = the item crop's own frame, projected server-side through the item's stored `bbox_norm`, `422` if the item has none). Stored boxes are always source-frame (`region_bbox_frame: "source"`). Response: `crop_id`, `region_bbox_norm`, `region_status`, `item` (the post-write wire item).
- `ItemBatchRegionRequest` (`PUT /crops/batch_region`): `crop_ids`, `region_bbox_norm`, `region_label_source`, `frame` (`parent` projects through each item's own box; items without one land in `invalid`). Response: `updated`, `conflicts`, `invalid`, `items` (post-write wire items of the updated crops).
- `CropBatchStatusRequest` (`POST /regions/batch_status`): `crop_ids`, `region_status`, `region_label_source`; `region_status` must be human-writable (see "Region lifecycle" below). `region_verified` is still accepted but **ignored** (deprecated): the server derives it. Response: `updated`, `conflicts` (`[{crop_id, current_source}]`), `invalid` (`[{crop_id, detail}]`, e.g. `detected` on a crop with no box), `items` (post-write wire items).
- `ItemRegionMetaRequest` (`PATCH /crops/{crop_id}/region_meta`): `region_text`, `region_status`, `region_rejection_reason`, `region_label_source` (all optional; only provided fields are written). Response: `crop_id`, `updated_fields` (wire names, e.g. `["region_status", "region_text"]`), `item` (post-write wire item). `422` when the status write would break an invariant (`detected` with no box).
- All four region request models set `extra='forbid'`: a stale key (`bbox_norm`, `plate_status`, `label_source`, …) is a `422`, never a silent no-op.
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
               "aug_target_min": 500, "aug_target_max": 3000}
```

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

### VLM-label one cluster — `POST /vlm/label_cluster/{cluster_id}`

Queues the auto-label job (same job, same `GET /pipeline/auto_label/status`
/ `POST /pipeline/auto_label/cancel`, `409` while one runs) scoped to one
cluster with only the VLM stage: no re-clustering, no auto-promote, no cap.
The server selects **every** unvalidated, non-holdout, non-excluded member
(the global sweep's cost skips — classifier-confident, `vlm_unmatched`,
recently combined-classified — don't apply to an explicit request) and
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
- `detected` on a crop with no box is refused (`422` single / `invalid[]` batch);
- human writes set `region_validated=true`; `false_positive` parks the region
  in the FP cluster, any other status releases it.

Each returns the post-write item, so a client adopts it rather than
re-deriving the result.

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
`dominant_count` / `purity` describe the top class among them. For a
candidate cluster (`cluster_id >= cluster_id_offset`)
`dominant_class_name` is set only for a unique top class with at least
3 members and at least half of the labelled members
(`CANDIDATE_DOMINANT_MIN_COUNT` / `CANDIDATE_DOMINANT_MIN_SHARE` in
`src/routers/curation/clusters.py`), else `null`; `dominant_class_id` is
always `null` for candidates. Class clusters report their top class as
before.

Each card also carries `purity_tier` (`pure` / `mixed` / `noisy`, `null`
with no labelled member) and `promotable` (the auto-promote gate: at least
`promote_min_members` members, at least `promote_min_labelled_share` of
them labelled, purity at least `pure_min`). The response serves the cut
points: `purity_thresholds: {pure_min: 0.85, mixed_min: 0.6,
promote_min_members: 4, promote_min_labelled_share: 0.5}` (source:
`src/services/curation/cluster_purity.py`; `pure_min` *is* the gate, so a
"pure" card is always one the gate would promote on purity) and
`core_similarity_min: 0.75` (the cut line for the items' `cluster_is_core`).
`POST /clusters/auto_promote` counts every labelled member in the purity
denominator (it used to count only the top-5 classes, overstating purity
on many-class clusters).

`GET /crops` query parameters: `page` (≥1), `page_size` (1–500, default
50), `limit` (1–500; alias for `page_size`, wins when both are set),
`sort` (`'<field>[:asc|desc]'`, default `updated_at:desc`; fields
`updated_at`, `created_at`, `confidence`, `classifier_raw_confidence`,
`crop_rank_in_image`, `crop_area_norm`, `blur_lap_ratio`,
`cluster_distance`, `mistakenness_score`, `uniqueness_score`; anything
else is a `400`; ignored by `order=outliers|diverse`), `class_id`,
`cluster_id`, `label_source`, `class_source`, `label_validated`,
`hdd_source` / `source` (same filter; `source` is the wire name),
`needs_new_class` (bool), `review_dismissed` (bool), `ids` (comma-separated, max 500: returns exactly
those items in that order, missing ids dropped, every other filter ignored —
use it to hydrate a `POST /select/diverse` page in one call),
`include_test`, `include_excluded`, `max_rank`,
`min_blur_ratio`, `classifier_conf_lt`, `conf_min` / `conf_max`
(inclusive band on `confidence`, `400` if min > max), `order`
(`default`/`outliers`/`diverse`), `k` (1–10000, `order=diverse` only:
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
- `ClassMergeRequest`: `source_id`, `target_id`. `POST /classes/merge?dry_run=true` writes nothing and returns `{dry_run: true, source_id, target_id, would_relabel, would_unvalidate, holdout_blocking, blocked}` — `would_relabel` counts every non-holdout item of the source class (validated or not), `would_unvalidate` the validated ones among them (a merge relabels with `class_source: class_merge` and clears validation), `blocked` = the real merge would `409` on frozen test-holdout items. `400` for an unknown id or a self-merge.
- `GET /class_sources` -> `{"class_sources": [{"id", "label", "role", "short_label"}, ...]}` — see "`class_source` values" below

### VLM labeling/verification

Registered at `POST {prefix}/vlm/*` (`src/routers/curation/vlm.py`).
Every VLM-related name on the wire is `vlm_*` — URL segment, stored and
returned fields (`vlm_confidence`, `vlm_raw_label`, …), `class_source`
values (`vlm`, `vlm_unmatched`, …), the review tab `vlm_low_conf`, the
auto-label params and the stats keys (see B3).

- `VlmLabelBatchRequest` (`POST /vlm/label_batch`): `crop_ids`
- `VlmVerifyRegionsRequest` (`POST /vlm/verify_regions`): `crop_ids`
- `VlmVerifyRegionBatchItem`: `crop_id`, `region_image_b64` (base64 JPEG of the region crop, no `data:` prefix), `candidate_text` (optional, upstream OCR hint, echoed back not consumed)
- `VlmVerifyRegionBatchRequest` (`POST /vlm/verify_region_batch`): `items: list[VlmVerifyRegionBatchItem]`
- `VlmVerifyRegionBatchResult`: `crop_id`, `is_region`, `confidence`, `reason`, `candidate_text`
- `VlmVerifyRegionBatchResponse`: `results`
- `VlmRegionVisibleBatchItem`: `crop_id`, `image_b64`
- `VlmRegionVisibleBatchRequest` (`POST /vlm/region_visible_batch`): `items`
- `VlmRegionVisibleBatchResponse`: `visible` (`dict[str, bool]`, keyed by `crop_id`)

### Review / holdout

`GET /review/{tab}` tabs: `all`, `mismatches`, `vlm_low_conf`, `outliers`,
`uncertainty`, `model_disagreements`, `regions`, `primary_low_conf`,
`coco_blind_spots`, **`new_class_proposals`** (items flagged
`needs_new_class` by a human, or `class_source: vlm_new_class_pending`).
Filters (every tab): `include_test`, `text` (regions tab), `max_rank`,
`min_blur_ratio`, `min_mistakenness`, `hide_near_duplicates`, **`class_id`**,
**`source`**, **`conf_min` / `conf_max`** (inclusive band on `confidence`,
`400` if min > max), `sort`. Response: `total`, `page`, `page_size`,
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
`{total_pending, top_terms: [{label, count, sample_crop_ids}]}`: the VLM's
proposed new-class names over unvalidated `vlm_new_class_pending` items,
most common first.

`POST /review/new_class_proposals/resolve?dry_run=` (`ResolveNewClassRequest`
→ `ResolveNewClassResponse`): bulk-resolves **every** unvalidated
`vlm_new_class_pending` item proposing `label`, not just the summary's
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

- `TestHoldoutFreezeRequest`: `percent`, `seed` (accepted but ignored — selection is deterministic, SHA1-of-crop_id)
- `TestHoldoutFreezeResponse`: `n_frozen`, `n_classes_covered`, `test_holdout_sha`, `per_class_counts`

### Shared curation-strategy defaults

- `CurationSettingsResponse` (`GET,PUT /settings`): `defaults` (`dict[str, str]`, open map keyed by axis id), `updated_at` (ISO 8601 or `null`), `updated_by` (always `null` today — no user-account system)
- `CurationSettingsUpdateRequest` (`PUT /settings` body): `defaults` (`dict[str, str]`, partial — only the axes being changed)

### Health / status

- `HealthResponse`: `status` (`ok`/`degraded`/`down`), `triton`, `opensearch`, `vlm`, `registry` — `vlm` reports the configured VLM backend's reachability regardless of which model it is.
- `StatusResponse`: `status`, `detail`, `extra`

### Export

- `ExportYoloRequest`: `export_dir`, `version_tag`, `seed`, `max_images`, `dedup_threshold`
- `ExportSingleClassRequest`: `export_dir`, `version_tag`, `class_ids`,
  `box_source` (`item`/`region`), `region_class_name`, `profile_name`,
  `seed`, `skip_test_split`, `empty_bg_ratio`, `max_positive_images`,
  `dedup_threshold`, `image_mode` (`whole_frame`/`item_crop`),
  `img_max_side`, `copy_images`

`POST /export/single_class` builds a narrowed dataset for a single class
or a class subset, with a stronger integrity envelope than the
multi-class export: `dataset_sha` hashes the written label *content*,
`frozen_test_sha` hashes the test split's identity, and the profile's
own `current` symlink is flipped atomically. `GET
/export/single_class/status?profile_name=...` reports the last run for
one profile, with the same `idle`/`unknown`/`success` contract as
`GET /export/status`. Each `profile_name` gets its own output root and
its own `current` symlink, so narrowed exports never clobber each other
or the multi-class dataset.

### Capability discovery — `GET /methods`

`GET {prefix}/methods` (`src/routers/curation/methods.py`) is the
capability-discovery endpoint every consumer should gate optional UI on
instead of feature-probing a write endpoint with a throwaway request.
It returns `{'strategies': [...], 'flags': {...}}`; each `strategies`
entry carries an `axis` of `cluster` / `score` / `sort` / `overlay` /
**`export`** (added by T-C2, cropwright_backend_integration_plan.md
§4.3).

**`export` axis** — which dataset-export *kinds*
`POST {prefix}/export/{kind}` can actually produce on this deployment:

| `id` | `status` | Notes |
|---|---|---|
| `yolo` | `stable` | Backed by `GenericYoloExportService`; always advertised. |
| `single_class` | `stable` | Backed by `SingleClassExportService`; single-class or class-subset export. |

There is deliberately **no** `lpr` id. The reference implementation's
single-class license-plate export is covered by `single_class`, which
takes its target class ids from the request instead of hardcoding a
domain vocabulary — a domain-named export kind would be exactly the
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

### Internal / worker-facing

- `_PathLookupRequest`: `image_paths` (max 10,000)
- `_PathLookupResponse`: `known_paths` (`dict[image_path, image_id]`)
- `_PublishEvent` (`POST /events/publish`, used by the SAM worker): `type`, `crop_id`, `class_id`, `class_name`, `class_source`, `region_status`, `region_text`, `image_path`, `topic`, `extra`. `extra='forbid'`: an unknown key is a `422` (a mismatched status key used to be silently dropped, so worker-published `crop.region_verified` events arrived with no status — audit S7).

## Item wire format

Built by `serialize_item()` in `src/services/curation/wire.py`. 91 keys,
always all present (a value is `null` when the stored doc has no value;
`bbox_norm` defaults to `[]`, `class_name`/`class_source`/
`label_source`/`updated_at`/`source`/`proposed_class_name` to `""`,
`confidence` to `0.0`, `label_validated`/`class_validated`/`test_holdout`/
`needs_new_class`/`class_excluded` to `false`, `item_text_lines` to `[]`).

Item keys (57): `id`, `crop_id`, `image_id`, `image_path`, `source_image_path`, `bbox_norm`, `class_id`, `class_name`, `class_source`, `confidence`, `classifier_raw_confidence`, `label_source`, `label_validated`, `class_validated`, `class_detector`, `class_detector_version`, `class_labeled_at`, `class_labeler`, `vlm_confidence`, `vlm_proposed_class_id`, `vlm_proposed_class_name`, `proposed_class_id`, `proposed_class_name`, `needs_new_class`, `needs_new_class_note`, `cluster_id`, `cluster_kind`, `cluster_distance`, `cluster_similarity`, `cluster_is_core`, `cluster_subid`, `class_excluded`, `excluded_reason`, `excluded_at`, `review_dismissed_at`, `source`, `test_holdout`, `crop_rank_in_image`, `crop_area_norm`, `blur_lap_ratio`, `proposal_name`, `probe_pred_class`, `probe_pred_class_id`, `probe_pred_entropy`, `mistakenness_score`, `mistakenness_method`, `mistakenness_version`, `mistakenness_scored_at`, `uniqueness_score`, `dup_group_id`, `dup_group_size`, `dup_is_representative`, `updated_at`, `thumbnail_url`, `region_thumbnail_url`, `item_text_lines`, `region_bbox_in_parent`.

Region keys (34, one per `RegionFields` attribute except `embedding`,
`prefix` and the `*_legacy` rollback columns): `region_bbox_norm`, `region_bbox_frame`, `region_bbox_correct`, `region_status`, `region_score`, `region_confidence`, `region_reason`, `region_rejection_reason`, `region_text`, `region_text_raw`, `region_text_confidence`, `region_text_source`, `region_text_engine_version`, `region_text_vlm`, `region_text_ocr`, `region_text_disagreement`, `region_validated`, `region_verified`, `region_verified_at`, `region_verifier`, `region_verifier_version`, `region_visible`, `region_detector`, `region_detector_version`, `region_detector_chain`, `region_detected_at`, `region_cluster_id`, `region_cluster_subid`, `region_cluster_distance`, `region_class_id`, `region_label_source`, `region_source`, `region_pairing`, `region_skip_verify`.

Derived keys (computed by the serializer, never stored):

- `region_bbox_in_parent` — the region box in the item-crop frame
  (`[x1,y1,x2,y2]`, clamped to `[0, 1]`); `null` when there is no region or
  the item has no usable `bbox_norm`. Draw it on the item thumbnail as-is.
- `proposed_class_id` / `proposed_class_name` — the class a one-key confirm
  applies, on **every** item endpoint (was `/review`-only): the VLM
  suggestion when there is one, else `class_id` and `vlm_raw_class` or
  `class_name` or `""` (see "VLM class suggestion").
- `cluster_kind` — `class` / `candidate` / `unassigned` from `cluster_id`
  (`null` without one); same rule as the cluster cards.
- `cluster_similarity` — `1 - cluster_distance` clamped to `[0, 1]` (`null`
  without a distance); `cluster_is_core` — `cluster_similarity >=
  core_similarity_min` (served on `GET /clusters`, `0.75`).
- Pass-throughs: `needs_new_class` (bool), `needs_new_class_note`,
  `class_excluded` (bool), `excluded_reason`, `excluded_at`,
  `probe_pred_class_id` (registry id of `probe_pred_class`, written by the
  probe pass), `source` (ingest source tag; stored under the legacy
  `hdd_source` key — the wire name is `source`).

`label_validated` is derived (`class_validated` OR `region_validated`).
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
- `region_text_disagreement`: `true`/`false` when both readings exist,
  compared after the profile's normalization; `null` otherwise (boolean).

The OCR reader keeps the region's dominant text: lines at least
`text_min_height_ratio` × the tallest line's height, not centered in the
outer `text_border_margin` band of the crop, minus `text_stopwords`,
ordered in rows top-to-bottom / left-to-right, normalized
(`text_uppercase`, `text_charset`), joined with `text_join`, and accepted
only within `text_len_min`..`text_len_max` and above
`text_min_confidence`. With no VLM configured (no `VLM_URL` /
`GEMMA_URL` / `OPENWEBUI_BASE_URL`) the worker never calls a VLM:
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
| `<src>:combined_verify_reject:region_visible_elsewhere` | VLM sees a region, but not in the candidate box |
| `<src>:combined_no_region_visible` | VLM sees no region at all |
| `<src>:sanity_reject:<reason>` | box failed the geometry gate (`<reason>` e.g. `aspect`) |
| `<seg>:skip_vlm_verify` | high-score segmenter box written without a VLM call |
| `<src>:accepted_unverified` | no VLM configured: box written `detected` with `region_verified=false` (text from OCR) |
| `<ocr>:text_hint:hit` / `:miss` / `:no_region_shape`, `<seg>:text_hint:miss` | OCR-hinted segmenter re-pass |

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
`detector`, `text`, `include_test`.

### SSE — `GET /events`

`crop.region_verified` data: `type`, `topic` (`region_status`),
`crop_id`, `region_status`, `region_text`, `ts`. The data keys other than
`type`/`topic`/`ts` are item keys with the same meaning. The same payload
is produced in-process (`publish_region_verified`) and by the SAM worker
via `POST /events/publish`.

### `GET /stats/dataset`

- `labeled`: `by_human`, `by_vlm`, `by_classifier`, `by_proposal`, `other`
- `regions`: `boxed`, `confirmed`, `total_detected`, `by_detector`,
  `by_segmenter`, `by_human`, `by_human_drew`, `verified_by_human`,
  `verified_by_vlm`, `validated_by_human` (`by_detector` /
  `by_segmenter` / `by_human_drew` are matched against the active
  `DetectionProfile`'s `detector_model` / `segmenter_name` /
  `human_detector_name`; `verified_by_vlm` counts every non-human
  verifier, because the VLM stamps its own model id)
- unchanged: `as_of`, `total_crops`, `validated`, `test_holdout`,
  `by_source`, `unlabeled`, `in_progress`, `clusters`

### `GET /export/datasets`

Query: `kind` (`yolo` | `single_class` — the same ids the `/methods`
export axis advertises), `profile_name`. Rows: `kind`, `profile_name`
(`null` for multi-class), `export_dir`, `version_tag`, `image_count`,
`split_counts`, `dataset_sha`, `exported_at`, `class_count`,
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
`primary_low_conf` / `coco_blind_spots` review tabs and the
`/stats/dataset` rollup all filter on these derived sets, never on one
deployment's detector names.

## Errors: read endpoints fail closed

A backend outage is a `503`, never an empty or zero answer that reads as
real data. `GET /ingest/sam_drain` (its `total_unfinished: 0` is the
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
  prefix; consumers should not hardcode `/curation` any more than they
  should hardcode `/legacy`.

## H3/H4 — cross-repo decisions (cropwright_backend_integration_plan.md §6/§7)

**H4 — superseded by B3 (2026-09-23).** H4 kept the historical
`plate_*` names on the wire and closed the storage reindex as WONTFIX.
B3 moves the *wire* to generic names; storage names stay configurable
via `RegionFields` exactly as before, so no reindex is required of any
deployment. Original ruling, for the record: **formally closed as
WONTFIX.** Agreed by both the backend and Cropwright independently. The
wire contract above is already fully decoupled from OpenSearch storage
field names via `RegionFields`; a storage rename is invisible to any
consumer by construction, so its cost (a 347k-document reindex against
a live deployment) would buy nothing a client can observe. See
`cropwright_backend_integration_plan.md` §7 for the full rationale,
including the caveat (now fixed) that `get_region_fields()` previously
ignored `OP_REGION_FIELD_*` overrides.

**H3 — not yet ruled; recorded here as open, per
`cropwright_backend_integration_plan.md` §6:**

- **D1** (`annotation_slots` on `GET /classes`): not added. The
  frontend's tier-2 static-profile loading isn't wired yet; adding a
  server field with zero consumers would freeze a wire commitment
  before the design is exercised. Publish the frontend's slot-spec
  draft first.
- **D2** (`plate_state.py` / `plateStatus.ts` codegen ownership):
  recommendation is to retire the codegen (its `DEFAULT_TARGET` points
  at a renamed sibling repo by absolute path and has zero importers)
  and let the frontend's slot profile be the source of truth. Not yet
  actioned.
- **D3** (field-mapping table ownership): this doc's per-model field
  lists above are hand-maintained and can drift from
  `src/routers/curation/_common.py` (see the caveat at the top of the
  Pydantic-models section). Recommendation is a generated,
  test-enforced table here rather than a hand-written one. Not yet
  built.
- **D4** (`POST /train/candidates`, H5): concur with not scheduling it
  — tier-1 cohorts cover the current deployment; only a second capable
  slot would justify it. No action planned.

## Coordination notes for consumers

### B2 — LPR vocabulary removed from the public surface (BREAKING)

Agreed with the Cropwright team before landing; ship both sides
together. Pure 1:1 renames — no handler, filter, or semantic change,
and no statuses were merged:

| Kind | Before | After |
|---|---|---|
| Route | `GET /plates` | `GET /regions` |
| Route | `GET /plates/training_candidates` | `GET /regions/training_candidates` |
| Route | `POST /plates/batch_status` | `POST /regions/batch_status` |
| Route | `POST /plates/cluster` | `POST /regions/cluster` |
| Route | `GET /plates/cluster/status` | `GET /regions/cluster/status` |
| Route | `POST /plates/clusters/refine/{cluster_id}` | `POST /regions/clusters/refine/{cluster_id}` |
| Route | `GET /plates/clusters` | `GET /regions/clusters` |
| Route | `POST /plates/fp_centroids/build` | `POST /regions/fp_centroids/build` |
| Route | `GET /plates/fp_centroids/status` | `GET /regions/fp_centroids/status` |
| Route | `GET /plates/suspected_false_positives` | `GET /regions/suspected_false_positives` |
| Route | `PUT /crops/{crop_id}/plate` | `PUT /crops/{crop_id}/region` |
| Route | `PATCH /crops/{crop_id}/plate_meta` | `PATCH /crops/{crop_id}/region_meta` |
| Route | `PUT /crops/batch_plate` | `PUT /crops/batch_region` |
| Status value | `no_plate_box` | `no_region_box` |
| Status value | `no_plate_visible` | `no_region_visible` |
| Cohort `mode=` | `lpr_blind_spots` | `detector_blind_spots` |
| Cohort `mode=` | `lpr_low_conf_correct` | `low_conf_correct` |
| `GET /review/{tab}` tab | `plates` | `regions` |

The three `/crops/{crop_id}/...` renames bring those routes in line with
their already-generic sibling `GET /crops/{crop_id}/region_thumbnail`.

B2 deliberately left every `plate_*` JSON key, `plate_thumbnail_url`,
`n_plates` and the `plates` stats block alone; B3 below renames all of
them. The `disagreement` / `human_corrected` / `false_positives` cohort
modes were already generic.

Deployments carrying documents written before B2 need a one-off
`update_by_query` rewriting the two status strings; nothing else in
storage changes.

### B3 — one generic wire vocabulary (BREAKING, 2026-09-23)

Agreed by both teams' owners; ship both sides together. Fresh
deployments re-ingest, so no data migration is provided. Also fixes the
backend rows of the frontend's contract audit
(`e2e-contract-audit-2026-09-23`): S4, S7, S8, S9, S10, the hardcoded
`/curation` thumbnail URL, and the MISSING `GET /classes/{class_id}`.

| Kind | Before | After |
|---|---|---|
| Item key (all item endpoints) | `plate_<attr>` (e.g. `plate_bbox_norm`, `plate_status`, `plate_verified`, `plate_validated`, `plate_detector_chain`, `plate_bbox_frame`, `plate_text`, `plate_text_raw`, `plate_visible`, `plate_cluster_id`, `plate_cluster_subid`, `plate_cluster_distance`, `plate_label_source`, …) | `region_<attr>` (`region_bbox_norm`, `region_status`, …) — full list under "Item wire format" |
| Item key | `plate_thumbnail_url` | `region_thumbnail_url` |
| Item key | `gemma_confidence` | `vlm_confidence` |
| Item key | `v6_raw_confidence` | `classifier_raw_confidence` |
| Item key (semantic search) | `region_bbox_norm` / `region_score` read under the storage names | fixed wire names (same values) |
| Item keys added to `/crops`, `/crops/{id}`, `/regions`, search | — | the full item (provenance chain, bbox frame, scores, class provenance, `updated_at`, …) — was stripped by `ItemDoc` (S9) |
| Stored doc field | `gemma_confidence`, `gemma_raw_label`, `gemma_raw_label_conf`, `gemma_raw_class`, `gemma_proposed_class`, `gemma_proposed_class_id`, `gemma_label_cluster_id`, `gemma_label_cluster_name`, `gemma_label_cluster_distance`, `gemma_verify_completed_at` | `vlm_confidence`, `vlm_raw_label`, `vlm_raw_label_conf`, `vlm_raw_class`, `vlm_proposed_class`, `vlm_proposed_class_id`, `vlm_label_cluster_id`, `vlm_label_cluster_name`, `vlm_label_cluster_distance`, `vlm_verify_completed_at` |
| Stored doc field | `gemma_vehicle_make`, `gemma_vehicle_model` | `vlm_item_make`, `vlm_item_model` |
| Stored doc field | `v6_raw_confidence` | `classifier_raw_confidence` |
| `class_source` / `label_source` value | `gemma`, `gemma_unmatched`, `gemma_new_class_pending`, `gemma_reclassified`, `gemma_human_confirmed` | `vlm`, `vlm_unmatched`, `vlm_new_class_pending`, `vlm_reclassified`, `vlm_human_confirmed` |
| `class_source` value | `v6_gemma_agreement`, `cluster_v6_majority_agreement` | `classifier_vlm_agreement`, `cluster_majority_agreement` |
| `class_source` value (queried) | hardcoded `v6_model`, `v6_low_conf`, `coco_yolo11_proposal` | the configured ingest profiles' values (`{secondary}_model`, `{primary}_proposal`, `{primary}_low_conf`, …) |
| Item key + stored doc field | `coco_proposal_name` | `proposal_name` |
| `region_detector_chain` entry | `<det>:gemma_verify_ok`, `<det>:gemma_reject`, `gemma_visible:yes` / `gemma_visible:no`, `<seg>:skip_gemma_verify` | `<det>:combined_verify_ok`, `<det>:combined_verify_reject`, `vlm_visible:yes` / `vlm_visible:no`, `<seg>:skip_vlm_verify` — full vocabulary under "`region_detector_chain` entries" |
| Writer id (`class_id_history`) | `gemma_pipeline` | `vlm_pipeline` |
| Review tab (`GET /review/{tab}`) | `gemma_low_conf` | `vlm_low_conf` |
| Review `reason` text | "gemma's reply did not match…", "gemma confidence below high", "…v6 unsure or missed", "COCO found a vehicle v6 missed…", "plate detected — needs human confirmation" | "VLM's reply did not match…", "VLM confidence below high", "…classifier unsure or missed", "detector proposed an item the classifier missed (blind spot)", "region detected — needs human confirmation" |
| Query param `GET /crops` | `v6_conf_lt` | `classifier_conf_lt` |
| Query params `GET /crops` (new) | — | `limit`, `sort`, `conf_min`, `conf_max`, `k` (S10) |
| Query param `GET /regions` | `plate_cluster_id`, `plate_cluster_subid` | `region_cluster_id`, `region_cluster_subid` |
| Query params `POST /pipeline/auto_label[/start]` | `gemma_batch_size`, `gemma_concurrency`, `max_gemma_crops`, `run_gemma`, `v6_confidence_skip_gemma` | `vlm_batch_size`, `vlm_concurrency`, `max_vlm_crops`, `run_vlm`, `classifier_confidence_skip_vlm` |
| Auto-label job stage / summary key | `gemma`, `cluster_id_normalize_post_gemma` | `vlm`, `cluster_id_normalize_post_vlm` |
| Query params `GET /export/datasets` (new) | — | `kind`, `profile_name`; rows gain `kind`, `profile_name` (S4) |
| Body `PUT /crops/{id}/region`, `PUT /crops/batch_region` | `bbox_norm`, `label_source` | `region_bbox_norm`, `region_label_source` |
| Body `PATCH /crops/{id}/region_meta` | `plate_text`, `plate_status`, `plate_rejection_reason`, `label_source` | `region_text`, `region_status`, `region_rejection_reason`, `region_label_source` |
| Body `POST /regions/batch_status` | `plate_status`, `plate_verified`, `label_source` | `region_status`, `region_verified`, `region_label_source` |
| Response `PUT /crops/{id}/region` | `plate_bbox_norm`, `plate_status` | `region_bbox_norm`, `region_status` |
| Response `PATCH …/region_meta` `updated_fields` | `plate_text`, `plate_status`, `plate_rejection_reason` | `region_text`, `region_status`, `region_rejection_reason` |
| Body `POST /events/publish` | `plate_status`, `plate_text` | `region_status`, `region_text` (unknown keys now 422) |
| SSE `crop.region_verified` data | status/text under the storage field names; `topic` = storage status name; worker events carried no status (S7) | `region_status`, `region_text`; `topic` = `region_status` |
| Stats `GET /stats/dataset` | `labeled.by_v6`, `labeled.by_yolo11_proposal` | `labeled.by_classifier`, `labeled.by_proposal` |
| Stats `GET /stats/dataset` | `plates` block; `plates.by_lpr`, `plates.by_sam3`, `plates.verified_by_gemma` | `regions` block; `regions.by_detector`, `regions.by_segmenter`, `regions.verified_by_vlm` |
| `GET /health` | `gemma` | `vlm` |
| Ingest response | `n_plates` | `n_regions` |
| `GET /regions/clusters` card | `dominant_class_name: "license_plate"` | `dominant_class_name: "region"` |
| Route (new) | — | `GET /classes/{class_id}` |
| Env var | `GEMMA_URL`, `GEMMA_IMAGES_PER_CALL`, `GEMMA_HTTPX_MAX_CONNECTIONS`, `GEMMA_HTTPX_KEEPALIVE`, `SAM_WORKER_GEMMA_CONCURRENCY`, `SAM_WORKER_GEMMA_VISIBLE_CONCURRENCY`, `SAM3_SKIP_GEMMA_VERIFY_SCORE` | `VLM_URL`, `VLM_IMAGES_PER_CALL`, `VLM_HTTPX_MAX_CONNECTIONS`, `VLM_HTTPX_KEEPALIVE`, `SAM_WORKER_VLM_CONCURRENCY`, `SAM_WORKER_VLM_VISIBLE_CONCURRENCY`, `SAM3_SKIP_VLM_VERIFY_SCORE` (old names still read as fallbacks) |

Not renamed, deliberately: the internal-only `v6_embedding` storage
field (never on the wire), Prometheus metric names, the
`needs_gemma_stop` Python alias in the GPU arbiter (not wire), and the
review tab id `coco_blind_spots` (a proposer-named tab id; left for the
owners to decide, it now filters on the configured proposal sources).

- This doc is the shared source of truth for the `/curation` API. Point
  any consumer's docs here instead of duplicating the field list.
- No wire-format change ships without a corresponding update to this
  doc; `tests/curation/test_wire_contract.py` pins the item key set.
- Cropwright is migrating onto this contract per
  `docs/design/cropwright_backend_integration_plan.md` — see that doc
  for the prefix-migration sequencing (`/legacy` → `/curation`, frontend-side
  only) and the route-parity CI guard
  (`tests/integration/test_labeler_route_parity.py`) that keeps this
  doc's route table honest against `app.routes`.
