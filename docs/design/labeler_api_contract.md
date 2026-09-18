# `/kb` labeler HTTP API contract (frozen for Phase 2)

Status: **living reference doc**, requested by the current labeler
frontend team as part of the OSS genericization work
(`docs/design/oss_genericization_phase2_plan.md`, decision log #4). This
documents the *current* `/kb` route surface and the Pydantic wire-model
field names it serves, and states explicitly which parts of that
contract are frozen.

This repo does not (yet) ship an implementation of these routes on
`origin/main` — they exist today only on a private, read-only reference
line. This doc is written from that reference and is the shared source
of truth for the frontend team while the backend curation subsystem is
genericized behind the scenes.

## The key invariant: HTTP JSON field names are independent of backend storage field names

The genericization plan (§3.2 of the design doc) introduces a
`RegionFields` config layer that lets the *backend* read/write its
OpenSearch documents under configurable field names (defaulting to
generic `region_*` names, with the existing production deployment
keeping its current `plate_*` names — no data migration, no reindex).

**This is purely a backend/OpenSearch concern. It does not touch the
HTTP JSON contract documented below.** Pydantic model attribute names
(`CropDoc.plate_bbox_norm`, `CropBatchStatusRequest.plate_status`, etc.)
are class-level static declarations that define the wire format the
current labeler frontend already speaks. They are **frozen** for
Phase 2: no field is renamed, added meaning changed, or removed on the
wire, regardless of what OpenSearch field name the backend reads or
writes internally to satisfy that JSON key.

Concretely: a router handler may change from
`doc['plate_status']` to `doc[region_fields.status]` internally, while
the Pydantic response model it returns keeps the literal attribute name
`plate_status`. The frontend sees zero change either way.

## Route surface (`/kb` prefix)

| Method | Path | Purpose |
|---|---|---|
| POST | `/kb/ingest` | Ingest a single image |
| POST | `/kb/ingest/batch` | Batch ingest |
| POST | `/kb/import_labels` | Import a YOLO-format label file for one image |
| POST | `/kb/import_labels/batch` | Batch label import |
| GET | `/kb/crops` | Paginated crop listing (supports outlier/diverse ordering overlays) |
| POST | `/kb/crops/label` | Label a single crop |
| POST | `/kb/crops/label/batch` | Batch label crops |
| POST | `/kb/crops/move` | Move crops to a different cluster |
| POST | `/kb/crops/exclude` | Exclude crops from training/clustering (reversible) |
| POST | `/kb/crops/unexclude` | Reverse an exclusion |
| POST | `/kb/crops/flag_new_class` | Flag crops as needing a class not yet in the registry |
| PUT | `/kb/crops/{crop_id}/plate` | Set/clear the plate sub-bbox on one crop (owns bbox geometry) |
| POST | `/kb/crops/plate/batch` | Bulk set/clear plate sub-bbox |
| POST | `/kb/crops/plate_status/batch` | Bulk-set `plate_status` (cluster-view triage) |
| PATCH | `/kb/crops/{crop_id}/plate_meta` | Patch plate metadata fields without touching the bbox |
| GET | `/kb/classes` | List classes |
| POST | `/kb/classes` | Create a class |
| PATCH | `/kb/classes/{class_id}` | Update a class |
| POST | `/kb/classes/merge` | Merge two classes |
| POST | `/kb/gemma/label_batch` | VLM class-labeling batch |
| POST | `/kb/gemma/verify_plate` | VLM plate-verify (single) |
| POST | `/kb/gemma/verify_plate_batch` | VLM plate-verify (batch) |
| POST | `/kb/gemma/plate_visible_batch` | VLM plate-visibility pre-filter (batch) |
| POST | `/kb/review/test_holdout/freeze` | Freeze the deterministic test-holdout split |
| GET | `/kb/health` | Component health (Triton, OpenSearch, VLM, registry) |
| POST | `/kb/export/yolo` | Export vehicle dataset in YOLO format |
| POST | `/kb/export/lpr` | Export standalone single-class LPR dataset |
| POST | `/kb/events/publish` | Publish a pipeline event (used by the SAM worker) |
| — | *(full router-by-router endpoint list)* | See `CLAUDE.md` in this repo for the up-to-date `/kb/*` table by sub-router (clusters, review, select, stats, models, train, plates, semantic, viz, umap, images, crops). |

## Frozen Pydantic wire models (attribute names are the JSON contract)

These are transcribed from the reference `_common.py` (shared router
foundations). Field names below are **frozen** — do not rename, even
when the corresponding backend OpenSearch field is renamed via
`RegionFields`.

### Ingest

- `IngestImageRequest`: `path`, `source`
- `IngestImageResponse`: `status` (`success`/`duplicate`/`failed`), `image_id`, `image_path`, `imohash`, `n_crops`, `n_plates`, `error`
- `BatchIngestSummaryResponse`: `successful`, `duplicates`, `failed`, `mismatches`, `labels_imported`, `crops_indexed`
- `BatchIngestResponse`: `status` (`success`/`partial`/`error`), `summary`, `results`
- `ImportLabelsRequest`: `image_path`, `label_txt_path`, `label_source`
- `ImportLabelsBatchRequest`: `items`

### Crops

- `CropDoc`: `crop_id`, `image_id`, `image_path`, `bbox_norm`, `class_id`, `class_name`, `class_source`, `confidence`, `cluster_id`, `cluster_distance`, `cluster_subid`, `label_validated`, `label_source`, `plate_bbox_norm`, `plate_score`, `test_holdout`, `crop_rank_in_image`, `crop_area_norm`, `blur_lap_ratio`, `coco_proposal_name`, `thumbnail_url`
- `CropsPageResponse`: `total`, `page`, `page_size`, `crops`, `method`, `version`, `n_pool`
- `CropLabelRequest`: `class_id`, `label_source`
- `CropBatchLabelRequest`: `crop_ids`, `class_id`, `label_source`
- `CropMoveRequest`: `crop_ids`, `cluster_id`
- `CropExcludeRequest`: `crop_ids`, `reason`
- `CropUnexcludeRequest`: `crop_ids`
- `CropPlateRequest`: `bbox_norm` (source-image frame), `label_source`
- `CropBatchPlateRequest`: `crop_ids`, `bbox_norm`, `label_source`
- `CropBatchStatusRequest`: `crop_ids`, `plate_status`, `plate_verified`, `label_source` — `plate_status` must be one of `HUMAN_PLATE_STATUS_VALUES` = `{'detected', 'no_plate_visible', 'verify_rejected', 'false_positive'}` (transient pipeline states like `pending_detection` are never set by hand)
- `CropPlateMetaRequest`: `plate_text`, `plate_status`, `plate_rejection_reason`, `label_source` (all optional; only provided fields are written; `extra='forbid'`)
- `CropFlagNewClassRequest`: `crop_ids`, `note`

### Classes

- `ClassEntry`: `class_id`, `class_name`, `group`, `sample_count`, `validated_count`, `cluster_size`, `deprecated`, `hotkey_letter`
- `ClassListResponse`: `classes`
- `ClassCreateRequest`: `name`, `group`, `notes`
- `ClassUpdateRequest`: `name`, `group`, `hotkey_letter`
- `ClassMergeRequest`: `source_id`, `target_id`

### VLM (Gemma) labeling/verification

- `GemmaLabelBatchRequest`: `crop_ids`
- `GemmaVerifyPlatesRequest`: `crop_ids`
- `GemmaVerifyPlateBatchItem`: `crop_id`, `plate_image_b64`, `candidate_text`
- `GemmaVerifyPlateBatchRequest`: `items`
- `GemmaVerifyPlateBatchResult`: `crop_id`, `is_plate`, `confidence`, `reason`, `candidate_text`
- `GemmaVerifyPlateBatchResponse`: `results`
- `GemmaPlateVisibleBatchItem`: `crop_id`, `image_b64`
- `GemmaPlateVisibleBatchRequest`: `items`
- `GemmaPlateVisibleBatchResponse`: `visible` (`dict[str, bool]`, keyed by `crop_id`)

### Review / holdout

- `TestHoldoutFreezeRequest`: `percent`, `seed` (accepted but ignored — selection is deterministic, SHA1-of-crop_id)
- `TestHoldoutFreezeResponse`: `n_frozen`, `n_classes_covered`, `test_holdout_sha`, `per_class_counts`

### Health / status

- `HealthResponse`: `status` (`ok`/`degraded`/`down`), `triton`, `opensearch`, `gemma`, `registry`
- `StatusResponse`: `status`, `detail`, `extra`

### Export

- `ExportYoloRequest`: `export_dir`, `version_tag`, `seed`, `max_images`, `dedup_threshold`
- `ExportLprRequest`: `export_dir`, `version_tag`, `skip_test_split`, `empty_bg_ratio`, `max_positive_images`, `dedup_threshold`, `image_mode` (`whole_frame`/`vehicle_crop`), `img_max_side`

### Internal / worker-facing

- `_PathLookupRequest`: `image_paths` (max 10,000)
- `_PathLookupResponse`: `known_paths` (`dict[image_path, image_id]`)
- `_PublishEvent` (POST `/kb/events/publish`, used by the SAM worker): `type`, `crop_id`, `class_id`, `class_name`, `class_source`, `plate_status`, `plate_text`, `image_path`, `topic`, `extra`

## What is explicitly NOT frozen

- **Backend OpenSearch field names** (`plate_status`, `plate_bbox_norm`,
  etc. as document keys) — governed by `RegionFields`
  (`src/config/region_fields.py`), overridable per deployment, and
  expected to diverge from the JSON names above over time as the
  generic curation subsystem is built out.
- **`PlateStatus` enum values** in `src/config/plate_state.py` — these
  are values, not field names, and are a separate codegen contract with
  the labeler frontend's TypeScript status enum
  (`scripts/codegen/export_plate_status_to_ts.py`). Untouched here.
- **The `/kb` URL prefix itself** — a config field (`CurationConfig.api_prefix`)
  on the generic side; the existing production deployment keeps serving
  `/kb` unchanged.

## Coordination notes for the labeler frontend team

- This doc is the shared source of truth referenced by decision log #4
  in `docs/design/oss_genericization_phase2_plan.md`. Point frontend
  docs here instead of duplicating the field list.
- Because the JSON contract above is frozen for Phase 2, a
  field-mapping adapter on the frontend side is **optional**, not a
  prerequisite — no wire-format changes are shipping from this work.
- A proposed `annotation_slots` field on `GET /kb/classes` was raised by
  the frontend team but has not yet been received/approved by the
  backend as of this writing; it is not reflected above.

---

## Chunk 9 update — the generic `/curation` implementation now exists

Everything above this line was written in Chunk 0 from the read-only
reference tree, before `origin/main` had any implementation at all. As
of Chunk 9 (the final content wave of
`docs/design/oss_genericization_phase2_plan.md`), the generic port is
complete and mounted under `CurationConfig.api_prefix` (default
`/curation`). This section is the update the frontend team asked for:
does the frozen-attribute-names statement above still hold against the
*real* implementation, and what does the full route list actually look
like.

**The frozen-attribute-names statement holds.** Every Pydantic
wire model referenced above (`CropDoc` → `ItemDoc`, `CropBatchStatusRequest`,
`CropPlateMetaRequest`, etc. — see `src/routers/curation/_common.py`)
kept its `plate_*` attribute names verbatim through the port, exactly as
promised: `ItemDoc.plate_bbox_norm`, `ItemDoc.plate_score`,
`CropBatchStatusRequest.plate_status`, `_PublishEvent.plate_status`, and
so on are unchanged Python identifiers on the response/request models.
Only the *OpenSearch document field* each handler reads/writes
internally moved onto `RegionFields` (defaulting to `region_*` names);
the JSON a client sends/receives is byte-for-byte what this doc already
described. `scripts/codegen/check_no_literal_region_fields.py`'s ratchet
is the enforcement mechanism — a `'plate_...'` string literal in a
ported module (as an OpenSearch field reference) fails pre-commit, while
the same string as a Pydantic attribute declaration
(`^\s*plate_[a-z_]+\s*:`) is explicitly exempted for exactly this reason.

**Full route list (103 routes under `/curation` as of this wave).**
Grouped by router module; every path is relative to the configured
`api_prefix`:

| Router module | Routes |
|---|---|
| `classes.py` | `GET,POST /classes`, `POST /classes/merge`, `POST /classes/sync_to_opensearch`, `PUT /classes/{class_id}`, `GET /classes/{class_id}/crops` |
| `crops.py` | `GET /crops`, `GET /crops/{crop_id}`, `PUT /crops/{crop_id}/label`, `DELETE /crops/{crop_id}/label`, `PUT /crops/batch_label`, `POST /crops/move`, `POST /crops/flag_new_class`, `POST /crops/batch_exclude`, `POST /crops/batch_unexclude`, `POST /crops/{crop_id}/review_dismiss` |
| `regions.py` / `regions_fp.py` | `GET /plates`, `PUT /crops/{crop_id}/plate`, `PUT /crops/batch_plate`, `PATCH /crops/{crop_id}/plate_meta`, `POST /plates/batch_status`, `POST /plates/cluster`, `GET /plates/cluster/status`, `GET /plates/clusters`, `POST /plates/clusters/refine/{cluster_id}`, `POST /plates/fp_centroids/build`, `GET /plates/fp_centroids/status`, `GET /plates/suspected_false_positives`, `GET /plates/training_candidates`, `GET /crops/{crop_id}/region_thumbnail` |
| `events.py` | `GET /events`, `POST /events/publish`, `GET /events/stats` |
| `export.py` | `POST /export/yolo`, `GET /export/datasets`, `GET /export/status`, `GET /export/registry/{artifact}` |
| `ingest.py` | `GET /ingest/status`, `GET /ingest/sam_drain`, `POST /ingest/path_lookup` |
| `models.py` | `GET /health`, `GET /models/status`, `DELETE /models/{model_name}` |
| `search.py` | `GET /search/text` |
| `stats.py` | `GET /stats/classes`, `GET /stats/dataset` |
| `pipeline.py` / `pipeline_control.py` / `pipeline_events.py` | `POST /pipeline/auto_label`, `POST /pipeline/auto_label/start`, `GET /pipeline/auto_label/status`, `POST /pipeline/auto_label/cancel`, `GET /pipeline/events` |
| `clusters.py` / `viz.py` | `GET /clusters`, `GET /clusters/representatives`, `POST /clusters/auto_promote`, `POST /clusters/refine/{cluster_id}`, `GET,POST /viz/projection*`, `POST /cluster/umap/rebuild` |
| `review.py` / `scores.py` / `select.py` / `methods.py` | `GET /review/{tab}`, `GET /review/raw_label_clusters`, `GET /review/unmatched_terms`, `POST /test_holdout/freeze`, `GET /test_holdout/stats`, `POST,GET /scores/*`, `POST,GET /select/*`, `GET /methods` |
| `vlm.py` | `POST /vlm/label_batch`, `POST /vlm/verify_regions`, `POST /vlm/verify_region_batch`, `POST /vlm/region_visible_batch` |
| `bakeoff.py` | `GET,POST /bakeoff/*` |
| `curation_images.py`, `curation_train.py`, `curation_umap.py` (outside the `curation` package, registered directly in `src/main.py`) | `GET /images/*`, `POST,GET /train/*`, `POST /cluster/umap/rebuild` |

The exact, always-current list is produced by the plan's own §6.0
verification snippet:

```python
from src.main import app
routes = sorted(r.path for r in app.routes if r.path.startswith('/curation'))
print(len(routes)); print('\n'.join(routes))
```

**What changed vs. the Chunk 0 description above, and why it's not a
wire-contract break:** the reference's `/kb/import_labels`,
`/kb/import_labels/batch`, `/kb/ingest` (single + batch) and
`/kb/export/lpr` have no equivalent under `/curation` — their only real
implementation lives in Bucket B services (proprietary bulk-ingest,
label-import, and single-class dataset-export logic) that this plan
never ports anywhere (§1, §7 R5). This is a scope gap in
the generic offering, not a contract change to any route that *does*
exist — every route this doc documents above is present and unchanged
on the wire. `POST /export/yolo` is backed by a new generic
`GenericYoloExportService` (`src/services/curation/export.py`) rather
than the reference's domain-specific exporter, but its request/response
JSON shape (`ExportYoloRequest` in, the same `status/export_dir/
version_tag/manifest_path/dataset_sha/split_counts/...` envelope out)
is unchanged.
