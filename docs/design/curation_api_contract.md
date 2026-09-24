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

## The key invariant: HTTP JSON field names are independent of backend storage field names

`RegionFields` (`src/config/region_fields.py`) lets the *backend*
read/write its OpenSearch documents under configurable field names
(defaulting to generic `region_*` names; a deployment with pre-existing
data under other names, e.g. `plate_*`, constructs its own instance —
no reindex). This is purely a backend/OpenSearch storage concern and
does **not** touch the HTTP JSON contract documented below. Pydantic
model attribute names (`ItemDoc.plate_bbox_norm`,
`CropBatchStatusRequest.plate_status`, etc.) are class-level static
declarations that define the wire format every consumer speaks. They
are **frozen**: no field is renamed, has its meaning changed, or is
removed on the wire, regardless of what OpenSearch field name the
backend reads or writes internally to satisfy that JSON key.

Concretely: a router handler may read `doc[region_fields.status]`
internally while the Pydantic response model it returns keeps the
literal attribute name `plate_status`. A consumer sees zero change
either way. See the `RegionFields` module docstring
(`src/config/region_fields.py`) for the full design rationale, and H4
below for why a matching rename of the wire names themselves was
formally closed as **WONTFIX**.

## Route surface

Full route list (109 routes under `/curation` as of this wave — the
curation deployment-settings plan added `GET,PUT /settings`), grouped
by router module; every path is relative to the configured
`api_prefix`:

| Router module | Routes |
|---|---|
| `classes.py` | `GET,POST /classes`, `POST /classes/merge`, `POST /classes/sync_to_opensearch`, `PUT /classes/{class_id}`, `GET /classes/{class_id}/crops` |
| `crops.py` | `GET /crops`, `GET /crops/{crop_id}`, `PUT /crops/{crop_id}/label`, `DELETE /crops/{crop_id}/label`, `PUT /crops/batch_label`, `POST /crops/move`, `POST /crops/flag_new_class`, `POST /crops/batch_exclude`, `POST /crops/batch_unexclude`, `POST /crops/{crop_id}/review_dismiss` |
| `regions.py` / `regions_fp.py` | `GET /regions`, `PUT /crops/{crop_id}/region`, `PUT /crops/batch_region`, `PATCH /crops/{crop_id}/region_meta`, `POST /regions/batch_status`, `POST /regions/cluster`, `GET /regions/cluster/status`, `GET /regions/clusters`, `POST /regions/clusters/refine/{cluster_id}`, `POST /regions/fp_centroids/build`, `GET /regions/fp_centroids/status`, `GET /regions/suspected_false_positives`, `GET /regions/training_candidates`, `GET /crops/{crop_id}/region_thumbnail` |
| `events.py` | `GET /events`, `POST /events/publish`, `GET /events/stats` |
| `export.py` | `POST /export/yolo`, `GET /export/datasets`, `GET /export/status`, `GET /export/registry/{artifact}` |
| `export_single_class.py` | `POST /export/single_class`, `GET /export/single_class/status` |
| `ingest.py` | `POST /ingest/image`, `POST /ingest/batch`, `POST /import_labels`, `POST /import_labels/batch`, `GET /ingest/status`, `GET /ingest/sam_drain`, `POST /ingest/path_lookup` |
| `models.py` | `GET /health`, `GET /models/status`, `DELETE /models/{model_name}` |
| `search.py` | `GET /search/text` |
| `stats.py` | `GET /stats/classes`, `GET /stats/dataset` |
| `pipeline.py` / `pipeline_control.py` / `pipeline_events.py` | `POST /pipeline/auto_label`, `POST /pipeline/auto_label/start`, `GET /pipeline/auto_label/status`, `POST /pipeline/auto_label/cancel`, `GET /pipeline/events` |
| `clusters.py` / `viz.py` | `GET /clusters`, `GET /clusters/representatives`, `POST /clusters/auto_promote`, `POST /clusters/refine/{cluster_id}`, `GET,POST /viz/projection*`, `POST /cluster/umap/rebuild` |
| `review.py` / `scores.py` / `select.py` / `methods.py` / `settings.py` | `GET /review/{tab}`, `GET /review/raw_label_clusters`, `GET /review/unmatched_terms`, `POST /test_holdout/freeze`, `GET /test_holdout/stats`, `POST,GET /scores/*`, `POST,GET /select/*`, `GET /methods`, `GET,PUT /settings` |
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

## Frozen Pydantic wire models (attribute names are the JSON contract)

Field names below are **frozen** — do not rename, even when the
corresponding backend OpenSearch field is renamed via `RegionFields`.
Model class names reflect `src/routers/curation/_common.py` as of this
writing; per D3 below, this table is hand-maintained today and can
drift from the source — treat `_common.py` as authoritative if the two
disagree, and see D3 for the plan to close that gap.

### Ingest

- `IngestImageRequest`: `path`, `source`
- `IngestImageResponse`: `status` (`success`/`duplicate`/`failed`), `image_id`, `image_path`, `imohash`, `n_crops`, `n_plates`, `error`
- `BatchIngestSummaryResponse`: `successful`, `duplicates`, `failed`, `mismatches`, `labels_imported`, `crops_indexed`
- `BatchIngestResponse`: `status` (`success`/`partial`/`error`), `summary`, `results`
- `ImportLabelsRequest`: `image_path`, `label_txt_path`, `label_source`
- `ImportLabelsBatchRequest`: `items`

### Crops

- `ItemDoc`: `crop_id`, `image_id`, `image_path`, `bbox_norm`, `class_id`, `class_name`, `class_source`, `confidence`, `cluster_id`, `cluster_distance`, `cluster_subid`, `label_validated`, `label_source`, `plate_bbox_norm`, `plate_score`, `test_holdout`, `crop_rank_in_image`, `crop_area_norm`, `blur_lap_ratio`, `coco_proposal_name`, `thumbnail_url`, `plate_status`, `plate_text`, `plate_text_source`, `plate_text_confidence`, `plate_rejection_reason`, `plate_detector`, `plate_detector_version`, `plate_verified`, `plate_verified_at`, `plate_verifier`, `plate_label_source` — the last eleven are round-trip counterparts of what `PATCH /crops/{id}/region_meta` and `PUT /crops/{id}/region` write (via `RegionFields` on the storage side), added so a `GET` after either write actually reflects the region metadata instead of silently dropping it.
- `CropsPageResponse`: `total`, `page`, `page_size`, `crops`, `method`, `version`, `n_pool`
- `CropLabelRequest`: `class_id`, `label_source`
- `CropBatchLabelRequest`: `crop_ids`, `class_id`, `label_source`
- `CropMoveRequest`: `crop_ids`, `cluster_id`
- `CropExcludeRequest`: `crop_ids`, `reason`
- `CropUnexcludeRequest`: `crop_ids`
- `CropPlateRequest`: `bbox_norm` (source-image frame), `label_source`
- `CropBatchPlateRequest`: `crop_ids`, `bbox_norm`, `label_source`
- `CropBatchStatusRequest`: `crop_ids`, `plate_status`, `plate_verified`, `label_source` — `plate_status` must be one of `HUMAN_REGION_STATUS_VALUES` = `{'detected', 'no_region_visible', 'verify_rejected', 'false_positive'}` (transient pipeline states like `pending_detection` are never set by hand)
- `CropPlateMetaRequest`: `plate_text`, `plate_status`, `plate_rejection_reason`, `label_source` (all optional; only provided fields are written; `extra='forbid'`)
- `CropFlagNewClassRequest`: `crop_ids`, `note`
- **Region thumbnail URLs**: `ItemDoc`/`/regions` responses carry `thumbnail_url` and `plate_thumbnail_url` fields whose *values* point at `GET {prefix}/crops/{crop_id}/region_thumbnail` — the JSON key `plate_thumbnail_url` is frozen (do not rename), but the URL path segment it contains is the generic `region_thumbnail`, not `plate_thumbnail` (no such route is registered; see `cropwright_backend_integration_plan.md` §1.3 for the bug this fixed).

### Classes

- `ClassEntry`: `class_id`, `class_name`, `group`, `sample_count`, `validated_count`, `cluster_size`, `deprecated`, `hotkey_letter`
- `ClassListResponse`: `classes`
- `ClassCreateRequest`: `name`, `group`, `notes`
- `ClassUpdateRequest`: `name`, `group`, `hotkey_letter`
- `ClassMergeRequest`: `source_id`, `target_id`

### VLM labeling/verification

Registered at `POST {prefix}/vlm/*` (`src/routers/curation/vlm.py`).
The vendor-neutral name is the URL segment and the Python model/class
names; the frontend's local review-tab ids and OpenSearch field names
(`gemma_suggested_class_id`, `gemma_low_conf`, `by_gemma`,
`class_source='gemma'`, etc.) are a separate, frozen wire/storage
naming that predates this generalization and is untouched here — see
the key-invariant section above.

- `VlmLabelBatchRequest` (`POST /vlm/label_batch`): `crop_ids`
- `VlmVerifyRegionsRequest` (`POST /vlm/verify_regions`): `crop_ids`
- `VlmVerifyRegionBatchItem`: `crop_id`, `plate_image_b64` (base64 JPEG of the region crop, no `data:` prefix), `candidate_text` (optional, upstream OCR hint, echoed back not consumed)
- `VlmVerifyRegionBatchRequest` (`POST /vlm/verify_region_batch`): `items: list[VlmVerifyRegionBatchItem]`
- `VlmVerifyRegionBatchResult`: `crop_id`, `is_region`, `confidence`, `reason`, `candidate_text`
- `VlmVerifyRegionBatchResponse`: `results`
- `VlmRegionVisibleBatchItem`: `crop_id`, `image_b64`
- `VlmRegionVisibleBatchRequest` (`POST /vlm/region_visible_batch`): `items`
- `VlmRegionVisibleBatchResponse`: `visible` (`dict[str, bool]`, keyed by `crop_id`)

Note `plate_image_b64` is itself a frozen wire field name carried over
unchanged from the reference implementation. `is_region` (the verdict
boolean on `VlmVerifyRegionBatchResult`) is **not** `is_plate` — an
earlier draft of this class in `src/routers/curation/_common.py` used
`is_plate` and was never imported by the actual route
(`src/routers/curation/vlm.py` defines and uses its own, wired,
`is_region`-bearing class); that dead duplicate has been removed from
`_common.py` so the code has exactly one definition, matching this
table.

### Review / holdout

- `TestHoldoutFreezeRequest`: `percent`, `seed` (accepted but ignored — selection is deterministic, SHA1-of-crop_id)
- `TestHoldoutFreezeResponse`: `n_frozen`, `n_classes_covered`, `test_holdout_sha`, `per_class_counts`

### Shared curation-strategy defaults

- `CurationSettingsResponse` (`GET,PUT /settings`): `defaults` (`dict[str, str]`, open map keyed by axis id), `updated_at` (ISO 8601 or `null`), `updated_by` (always `null` today — no user-account system)
- `CurationSettingsUpdateRequest` (`PUT /settings` body): `defaults` (`dict[str, str]`, partial — only the axes being changed)

### Health / status

- `HealthResponse`: `status` (`ok`/`degraded`/`down`), `triton`, `opensearch`, `gemma`, `registry` — the `gemma` key name is itself frozen wire naming (predates the VLM generalization) and reports the configured VLM backend's reachability regardless of which model it actually is.
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
(`cluster` / `sort` / `detection_profile` / `prompt_pack` today —
`score`/`overlay`/`export` have no single-selectable-id "default"
concept a shared override could apply to, so they 422 rather than
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
| `detection_profile` | `POST /pipeline/auto_label` and `/pipeline/auto_label/start` — `?detection_profile=<id>` overrides the default for that one job (never written to settings; unknown id → `422` with `valid_ids`), omitted resolves via this function; the resolved id is echoed in the job `args` and the run summary. No auto-label stage runs region detection, so today it is validated and recorded, not consumed; the region cascade (detection worker) uses the process's active profile (`OP_REGION_PROFILE` / `OP_REGION_DETECTION_*`). Neutral default: no profile registered → the axis is empty and the id resolves to `null`. |
| `prompt_pack` | `POST /pipeline/auto_label*` — `?prompt_pack=<id>` selects the pack for that job's VLM labeling stage (same override/`422`/echo semantics); omitted resolves via this function. `POST /vlm/label_batch` and `/vlm/verify_regions` also use the effective default. Selectable ids: the built-in generic pack, every `OP_PROMPT_PACK_PATHS` pack, and the `OP_PROMPT_PACK_PATH` pack (the fallback default). `POST /vlm/verify_region_batch` and `/vlm/region_visible_batch` take no OpenSearch dependency and use the `OP_PROMPT_PACK_PATH` pack. |

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
- `_PublishEvent` (`POST /events/publish`, used by the SAM worker): `type`, `crop_id`, `class_id`, `class_name`, `class_source`, `plate_status`, `plate_text`, `image_path`, `topic`, `extra`

## What is explicitly NOT frozen

- **Backend OpenSearch field names** (`plate_status`, `plate_bbox_norm`,
  etc. as document keys) — governed by `RegionFields`
  (`src/config/region_fields.py`), overridable per deployment via
  `OP_REGION_FIELD_*` (see `env.template`).
- **`RegionStatus` enum values** in `src/config/region_state.py` — these
  are values, not field names. See D2 below for the codegen contract's
  status. Work item B2 exercised exactly that freedom: `no_plate_box` /
  `no_plate_visible` became `no_region_box` / `no_region_visible` (see
  the B2 note under "Coordination notes" below).
- **The `/curation` URL prefix itself** — a config field
  (`CurationConfig.api_prefix`, env override `OP_API_PREFIX`) that
  defaults to `/curation`. A deployment may run behind a different
  prefix; consumers should not hardcode `/curation` any more than they
  should hardcode `/legacy`.

## H3/H4 — cross-repo decisions (cropwright_backend_integration_plan.md §6/§7)

**H4 — `plate_*` → generic storage-field rename: formally closed as
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

Not renamed, deliberately: `GET /crops/{crop_id}/region_thumbnail`
(already generic), every `plate_*` JSON key including
`plate_thumbnail_url` and the `n_plates` ingest counter (frozen — see
the invariant at the top of this doc), every OpenSearch document field
name (`plate_bbox_norm` etc. — the reindex is WONTFIX per H4 below),
the `plates` key in `GET /stats/dataset`'s response body (a wire field,
not a path), and the `disagreement` / `human_corrected` /
`false_positives` cohort modes.

Deployments carrying documents written before B2 need a one-off
`update_by_query` rewriting the two status strings; nothing else in
storage changes.

- This doc is the shared source of truth for the `/curation` API. Point
  any consumer's docs here instead of duplicating the field list.
- The JSON contract above is frozen: a consumer-side field-mapping
  adapter is a convenience, not a prerequisite — no wire-format changes
  ship without a corresponding update to this doc.
- Cropwright is migrating onto this contract per
  `docs/design/cropwright_backend_integration_plan.md` — see that doc
  for the prefix-migration sequencing (`/legacy` → `/curation`, frontend-side
  only) and the route-parity CI guard
  (`tests/integration/test_labeler_route_parity.py`) that keeps this
  doc's route table honest against `app.routes`.
