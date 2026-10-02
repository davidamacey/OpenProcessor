# OpenProcessor `/curation` API contract

Reference for the `/curation` HTTP API as shipped in v0.1.0. The generated
OpenAPI document `contracts/openapi/curation.json` is the source of truth for
routes and schemas. The JSON/TypeScript helpers in `contracts/json/` and
`contracts/ts/` are generated from the same code. This document adds the
semantics that a schema cannot carry: ordering, locking, error codes,
concurrency tokens and which route to call for which job.

The API is generic. Cropwright is one consumer. Nothing here is specific to a
domain, a model vendor or a frontend.

- [Conventions](#conventions)
- [Error model](#error-model)
- [Projects and lifecycle](#projects-and-lifecycle)
- [Settings, config store and capability discovery](#settings-config-store-and-capability-discovery)
- [Keymap](#keymap)
- [Prompt packs](#prompt-packs)
- [Region profiles](#region-profiles)
- [VLM endpoints, catalog and activation](#vlm-endpoints-catalog-and-activation)
- [Items and the item wire format](#items-and-the-item-wire-format)
- [Crops: browse, label, undo](#crops-browse-label-undo)
- [Regions: multi-box edits and queues](#regions-multi-box-edits-and-queues)
- [Review tabs](#review-tabs)
- [Classes](#classes)
- [Clusters](#clusters)
- [VLM labeling and the auto-label pipeline](#vlm-labeling-and-the-auto-label-pipeline)
- [Ingest](#ingest)
- [Dataset import](#dataset-import)
- [Reprocess](#reprocess)
- [Export](#export)
- [Training, promote and models](#training-promote-and-models)
- [Bake-off](#bake-off)
- [Scores, selection, projection, probe, search, stats](#scores-selection-projection-probe-search-stats)
- [Images](#images)
- [Events](#events)
- [The lock rule](#the-lock-rule)
- [Class identity](#class-identity)
- [Breaking wire changes](#breaking-wire-changes)

## Conventions

### Prefix and project scoping

- Every path is relative to `CurationConfig.api_prefix` (`OP_API_PREFIX`,
  default `/curation`). Clients should not hardcode the prefix.
- Data and per-project configuration live under
  `/curation/projects/{project}/...`. In the tables below these paths are
  written without that prefix. For example `/crops` means
  `/curation/projects/{project}/crops`.
- Deployment-wide routes are written in full: `/curation/health`,
  `/curation/events`, `/curation/projects`, `/curation/projects/combine*`,
  `/curation/vlm/*`.
- There is no unscoped alias of a project route and no vendor-named alias.
- `{project}` is a slug (see [Projects and lifecycle](#projects-and-lifecycle)).
  A slug that does not name a project is `404 project_not_found`, never
  `422`.
- A project is bound before the handler runs. A project that is `building`,
  `deleting` or `failed` cannot be bound (`409`). An `archived` project, or a
  project bound while the project registry is stale, is bound read-only: every
  non-GET method answers `409 project_archived` or `409 project_read_only`.

### Concurrency tokens

Writes that replace a stored document carry the revision that the client
loaded. A stale token is `409`.

| Token | Where | Stale answer |
|---|---|---|
| `expected_revision` (int) | project PATCH/archive/unarchive/clone_settings, pack/profile/VLM-endpoint PUT, `DELETE ...?expected_revision=`, keymap PUT/reset (also `If-Match: "keymap:N"`), model sharing | `409 revision_conflict` (`current_revision` in the body) |
| `expected_active` (`{name, revision}`) | activate, rollback, deactivate of a pack, profile or VLM endpoint | `409 active_conflict` (`current` in the body) |
| `expected_region_revision` (int) | per-item box writes (`PUT /crops/{crop_id}/regions`, `PATCH /crops/{crop_id}/regions/{box_id}`, `POST /regions/batch_box_state` via `expected_region_revisions`) | `409 region_conflict` (`current_region_revision`, `current_box_ids`, `item`) |
| `expected_preview_sha` | `POST /curation/projects/combine` | `409 preview_stale` |
| `expected_import_key` | `POST /datasets/imports` | `409 dataset_changed` |

Every stored document also serves an `etag`. Single-document `GET` routes
send it as the `ETag` header.

### Revisioned documents

Prompt packs, region profiles and VLM endpoints are immutable revisions:
every save is a new revision number, numbers are never reused, and
`GET .../{name}/revisions[/{revision}]` reads history. Documents have a
`source`: `builtin`, `file`, `template`, `env`, `registered` or `stored`. Only
`stored` documents can be saved or deleted; the others are `read_only`
(`403 read_only`). Clone one to get an editable copy.

### Paging and ordering

- Lists page with `page` (from 1) and `page_size`. A page past the 10,000th
  result is `422`.
- Every queue ends in a `crop_id` ascending tie-break, so pages are stable.
- Read endpoints fail closed: a backend outage is `503`, never an empty or
  zero answer.

### Write conventions

- Request models that write region state use `extra='forbid'`: an unknown key
  is `422`.
- A human class or region write never takes its provenance from the client
  beyond the documented `label_source` / `region_label_source` values. The
  server stamps `class_source: "human"` itself.
- Class names, slugs and pack/profile names follow the patterns served by the
  relevant `schema`/`limits` field; the pattern is part of the response, not a
  client constant.

## Error model

Routes of the project, config-store, dataset, reprocess, combine, VLM and test
surfaces raise a typed body:

```json
{"detail": {"error": "revision_conflict", "message": "...", "current_revision": 4}}
```

`error` is one of the codes below. Extra fields depend on the code (`project`,
`current_revision`, `current`, `report`, `issues`, `unmapped`, `valid_ids`,
`projects`, `jobs`, `crop_ids`, `endpoint`, `activate_via`, `limit`,
`requested`). `report` is a `ValidationReport` (`ok`, `errors[]`, `warnings[]`,
`force_allowed`; an issue has `code`, `severity`, `message`, `field`, `detail`,
`bypassable`).

Older routes (crops, classes, review, export, training, clusters) raise
`{"detail": "<string>"}` or `{"detail": {"error": "<code>", ...}}`. Validation
failures of a request body are the standard FastAPI `422` list.

| Status | Codes |
|---|---|
| `404` | `project_not_found`, `not_found`, `unknown_revision`, `model_not_found`, `crop_not_found`, `image_not_found`, `import_not_found`, `combine_not_found`, an export artifact that is not whitelisted or not present, `unknown_box_id` |
| `403` | `read_only` (a non-stored pack, profile or endpoint), region-detector models in `DELETE /models/{model_name}` |
| `409` | `project_archived`, `project_read_only`, `project_building`, `project_deleting`, `project_failed`, `project_busy`, `project_protected`, `slug_taken`, `slug_retired`, `last_active_project`, `shard_budget_exceeded`, `invalid_transition`, `target_not_empty`, `clone_source_not_ready`, `in_use`, `name_conflict`, `revision_conflict`, `active_conflict`, `no_previous`, `previous_deleted`, `no_active_profile`, `preview_stale`, `combine_not_resumable`, `import_busy`, `import_resumable`, `import_not_resumable`, `import_not_undoable`, `dataset_changed`, `reprocess_busy`, `class_hotkey_conflict`, `hotkey_taken`, `region_conflict`, `vlm_not_configured`, `vlm_endpoint_unavailable`, `no_local_vlm`, `finish_in_progress`, `no_classes` |
| `413` | `upload_too_large`, an upload or batch over its per-request limit |
| `422` | `validation_failed`, `slug_invalid`, `confirm_mismatch`, `combine_invalid`, `class_mapping_incomplete`, `class_mapping_invalid`, `import_blocked`, `format_undetected`, `dataset_path_not_allowed`, `archive_invalid`, `reprocess_targets_invalid`, `region_profile_required`, `unknown_pack`, `unknown_profile`, `unknown_vlm`, `vlm_external_not_acknowledged`, `unknown_catalog_id`, `vlm_catalog_does_not_fit`, `export_outside_project`, `pack_invalid`, `profile_invalid`, `no_box_to_verify`, `too_many_crops`, `too_many_crop_ids`, `too_many_boxes`, `region_text_disabled`, `box_id_in_batch`, `box_id_required` |
| `429` | `probe_busy` (VLM endpoint probe), `test_busy` (test-on-crop) |
| `500` | `internal_isolation_error` (a cross-project access was refused), `path_escape` |
| `502` | `vlm_transport_error`, `segmenter_error`, `detector_error`, a Triton load refusal on promote |
| `503` | `config_store_unavailable`, an OpenSearch outage on a read, ingest unavailable before the encoder has loaded |
| `504` | `test_timeout` |

The full code enumeration is `ConfigErrorDetail.error` in the OpenAPI
document. The validation issue codes are `ValidationIssue.code`.

## Projects and lifecycle

A project is an isolated dataset: its own OpenSearch indexes, class
registry, files, config documents, jobs and promoted models. Nothing is
shared between projects except the deployment-wide VLM endpoint registry and
models a project explicitly shares. The `default` project always exists and
cannot be deleted.

### Slugs

Pattern `^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$`, 2 to 32 characters. Reserved:
`combine`, `new`, `all`, `none`, `projects`, `global`, `settings`, `vlm`,
`health`. A deleted project's slug is retired and cannot be reused
(`409 slug_retired`). `GET /curation/projects` serves the pattern, the
bounds, the reserved and retired slugs and the cloneable axes under `limits`.

### Status

| `status` | Meaning | `writable` | `selectable` |
|---|---|---|---|
| `building` | being created or filled (a combine target) | no | no |
| `active` | normal | yes | yes |
| `archived` | read-only | no | yes |
| `deleting` | removal running in the background | no | no |
| `failed` | creation failed, `error` has `code` and `message`; delete it | no | no |
| `deleted` | tombstone, not served | no | no |

`GET /curation/projects` lists `active`, `building`, `failed` and `deleting`
projects, plus `archived` ones with `include_archived=true`.

### Routes

| Method | Path | Body or query | Response | Errors |
|---|---|---|---|---|
| GET | `/curation/projects` | `include_archived` | `ProjectsResponse` (`default_slug`, `projects[]` of `ProjectSummary`, `capacity`, `limits`, `labels`) | |
| POST | `/curation/projects` | `CreateProjectRequest` (`slug`, `display_name`, `description`, `clone_settings_from`, `clone_axes`) | `201 ProjectLifecycleResponse` | `422 slug_invalid`, `409 slug_taken`, `409 slug_retired`, `409 shard_budget_exceeded`, `409 clone_source_not_ready`, `422 validation_failed` |
| GET | `/curation/projects/{project}` | | `ProjectRecordResponse` (summary plus `resources`, `error`) | `404 project_not_found` |
| PATCH | `/curation/projects/{project}` | `PatchProjectRequest` (`display_name`, `description`, `expected_revision`). The slug is immutable | `ProjectLifecycleResponse` | `409 revision_conflict` |
| POST | `/curation/projects/{project}/archive` | `{expected_revision}` | `ProjectLifecycleResponse` | `409 invalid_transition`, `409 project_busy`, `409 revision_conflict` |
| POST | `/curation/projects/{project}/unarchive` | `{expected_revision}` | `ProjectLifecycleResponse` | `409 invalid_transition`, `409 revision_conflict` |
| DELETE | `/curation/projects/{project}` | `dry_run`, `confirm`, `force` | `200 DeleteDryRunResponse` for a dry run, otherwise `202 ProjectLifecycleResponse` | `422 confirm_mismatch`, `409 project_protected`, `409 project_busy`, `409 last_active_project`, `409 in_use` |
| POST | `/curation/projects/{project}/clone_settings` | `CloneSettingsRequest` (`from`, `axes`, `expected_revision`) | `ProjectLifecycleResponse` | `409 target_not_empty`, `409 clone_source_not_ready`, `422 validation_failed` |
| GET | `/stats` | | `ProjectStatsResponse` (`counts`, `indexes`, `disk`, `jobs`, `last_ingest_at`) | |
| GET | `/pause` | | `PipelinePauseState` (`paused`, `paused_by`, `reason`) | |
| POST | `/pause` | | `PipelinePauseState` | |
| POST | `/resume` | | `PipelinePauseState` | |

The `ProjectSummary` carries `slug`, `display_name`, `description`, `prefix`
(the project's API base), `status`, `writable`, `selectable`, `is_default`,
`deletable`, `archivable`, `unarchivable`, `revision`, `created_at`,
`updated_at`, `counts` (`images`, `items`, `validated`), `origin` (set for a
combine target) and `paused`. A client enables buttons from the boolean flags
and does not recompute them from `status`.

Lifecycle responses carry `warnings[]` (for example `shard_budget_high`) and
`keymap_clone_conflicts[]` (keymap actions that a `keymap` clone dropped
because their combo collides with a class hotkey of the target).

### Create and clone

Creation runs: validate, check shard capacity, write the record as
`building`, create indexes and directories, optionally clone, then flip to
`active`. A failure leaves the record `failed`. Every refusal that can be
known up front (unknown axis, clone into itself, source not ready) is raised
before the first write, so a refused clone burns no slug.

`clone_settings_from` / `POST .../clone_settings` copy configuration, never
data. The cloneable axes (`limits.cloneable_axes`) are `settings_defaults`,
`classes`, `activations`, `keymap`, `prompt_packs` and `vlm_activation`. A
clone never copies an external-endpoint acknowledgement; an external source
endpoint refuses the clone (`422 vlm_external_not_acknowledged`). `409 target_not_empty` when the target already has items (`classes`), stored
packs (`prompt_packs`) or an activation or pack name that the clone would
overwrite.

### Delete

`DELETE ...?dry_run=true` writes nothing and reports `indexes[]`, `dirs[]`,
`promoted_models[]`, `mlflow_experiment`, `running_jobs[]`, `referenced_by[]`
and `blocking[]` with `blocking_detail[]`. A real delete needs
`confirm=<slug>` and answers `202` with the `deleting` record. Index and
directory removal, promoted-model unload and the tombstone run in the
background. Follow the `project.deleted` event on `/curation/events`, or poll
`GET /curation/projects/{project}` until it answers `404`.

A delete is refused while the project has running jobs (`project_busy`), is
the last active project, or has promoted models shared with other projects
(`in_use`; `force=true` bypasses only this check). Repeating a `DELETE` for a
project that is already `deleting` retries the background work.

### Pause

`POST /pause` writes a per-project pipeline pause flag. Workers stop picking
up new work for that project at their next quiesce point. `paused_by` lists
`project` and `gpu_training` (a training run holds the GPU for the whole
deployment). `POST /resume` clears the project flag and is idempotent.

### Combine projects

A combine builds a new project from 1 to 8 source projects. Sources are only
read. The target is `building` while it fills, then `active` (or `failed`).
Deleting the target undoes the combine. A running combine marks its sources
and its target busy (`project_busy`).

| Method | Path | Body | Response | Errors |
|---|---|---|---|---|
| POST | `/curation/projects/combine/preview` | `CombineRequest` | `CombinePreview` | |
| POST | `/curation/projects/combine` | `CombineStartRequest` (`CombineRequest` plus `expected_preview_sha`) | `202 {job_id, target}` | `409 preview_stale`, `422 combine_invalid` (with `report`), `409 slug_taken` |
| GET | `/curation/projects/combine/{job_id}` | | `CombineJobResponse` | `404 combine_not_found` |
| POST | `/curation/projects/combine/{job_id}/cancel` | | `CombineJobResponse` | `404 combine_not_found` |
| POST | `/curation/projects/combine/{job_id}/resume` | | `CombineJobResponse` | `409 combine_not_resumable` (only `interrupted` or `cancelled`), `404 combine_not_found` |

`CombineRequest`:

| Field | Meaning |
|---|---|
| `target` | `{slug, display_name, description}` of the new project |
| `sources[]` | `{project, include: {label_states: all or validated_only}}`. Order is priority: the first source wins ties |
| `class_mapping` | `{<source project>: [ClassMappingEntry]}`, one row per source class (see [Class identity](#class-identity)) |
| `target_classes` | optional explicit target class names |
| `dedup` | `content_hash` (default) or `none`. Byte-identical images are copied once from the first-listed source; boxes of the duplicate merge by IoU (`dedup_iou`, default 0.9) and target class, with human over import over VLM over model |
| `holdout` | `preserve_union` (default), `recompute` (warns), `none` |
| `settings_from` | optional source project to copy settings from |

`CombinePreview` writes nothing and returns `ok`, `errors[]`, `warnings[]`
(`CombineIssue`: `code`, `severity`, `message`, `project`, `detail`),
`suggested_mapping`, `sources[]`, `target`, `dedup`, `bytes` and
`preview_sha`. Starting with a different body than the one previewed is
`409 preview_stale`.

`CombineJobResponse`: `job_id`, `status` (`queued`, `running`, `completed`,
`failed`, `cancelled`, `interrupted`), `phase`, `done`, `total`, `report`,
`next_steps[]`, `error`, `target`, `sources`, `started_at`, `finished_at`.
Jobs are file-backed, chunked, marked `interrupted` after a restart and
resumable from the persisted plan.

Combined items carry `origin_project`, `origin_item_id`, `origin_image_id`,
`origin_split`, `import_ids` (the job), `combine_conflict`,
`combine_conflict_origins` and `combine_merged_origins`. A box that disagrees
on class keeps the priority label and is flagged `combine_conflict`; review
those with the `combine_conflict` filter. Nothing numbered in a source (class
ids, cluster ids, class history) crosses into the target.

### Health

| Method | Path | Response |
|---|---|---|
| GET | `/curation/health` | `GlobalHealthResponse`: `status` (`ok`, `degraded`, `down`), `triton`, `opensearch`, `vlm` (the active endpoint), `mlflow_public_url`, `version`, `api_version` |
| GET | `/health` | `HealthResponse`: the same facts plus `project`, `registry` and `region_profile` (`name`, `display_name`, `display_name_singular`, `region_class_name`, `text_reader`, `reads_text`, `text_hint_enabled`, `limits.max_boxes_per_write`; `null` when no profile is active) |

`mlflow_public_url` (`OP_MLFLOW_PUBLIC_URL`) is the browser-reachable MLflow
base to build run links from.

## Settings, config store and capability discovery

Per-project configuration is stored as documents in the project's config
index. Three axes are activated through the config store: the prompt pack
(`prompt_pack`), the region profile (`detection_profile`) and the VLM
endpoint (`vlm`). Activations are served to workers and applied at their next
quiesce point. A worker reports what it applies as `applied[]` on the
activation response. Each `applied[]` row always carries `profile` and `pack`
(`{name, revision}`); `vlm` is `null` when the worker never reported a VLM axis
and `{name: null, revision: null}` when it reported no VLM configured.

### Project settings

| Method | Path | Body | Response |
|---|---|---|---|
| GET | `/settings` | | `CurationSettingsResponse`: `defaults` (map axis id to id), `updated_at`, `updated_by` (always `null`) |
| PUT | `/settings` | `{defaults: {<axis>: <id or null>}}`, partial | the updated `CurationSettingsResponse` |

`defaults` is an open map. A missing key means no override for the axis.
A project with no settings document answers `200` with `defaults: {}`.

Settable axes: `cluster`, `sort`, `prompt_pack`, `detection_profile`, `vlm`.
An unknown axis or an id that `GET /methods` does not advertise is `422`.
`null` clears an override. A `sort` default that orders by a field no item
has is refused (`422`).

`prompt_pack`, `detection_profile` and `vlm` are not stored in the settings
document. A `PUT` activates them through the config store, in one call with
the same activation gate as the `/activate` routes. `detection_profile`
accepts `off`. `vlm` accepts an endpoint name, `off`, or `null` (the
deployment's `env` endpoint). The server resolves and gates every axis of one
request against the pending values of the others before writing any of them,
so a combined request cannot leave a half-applied pairing. Errors:
`422 unknown_pack`, `422 unknown_profile`, `422 unknown_vlm`,
`422 validation_failed` (with `report`), `409 active_conflict`.

`GET /methods` derives each axis's `default` flag from the same resolver that
the endpoints use when a request omits the axis parameter. Setting a default
changes server behavior, not only the display.

### Config vocabulary

`GET /config/vocabulary` (`include_other_projects` optional) is the one place
that serves every model choice a form needs:

| Field | Content |
|---|---|
| `detectors[]` | Triton detector models (`name`, `source`, `ready`, `state`, `versions`, `job_id`, `promoted_at`, `choice`) |
| `segmenters[]` | segmenter services (`name`, `endpoint`, `status`, `masks`, `default_min_score`, `max_candidates`) |
| `ocr` | `available`, `det_models[]`, `rec_models[]`, `pipeline_models[]` |
| `vlm` | `active` and `endpoints[]` from the registry |
| `text_reader_modes[]` | `none`, `vlm`, `ocr`, `vlm_then_ocr`, `both`, each with `reads_text`, `needs_vlm`, `needs_ocr` |
| `registry_classes[]` | the project's classes |
| `prompt_pack_calls[]` | the VLM calls a pack defines |
| `model_choices[]` | fixed roles with `role`, `label`, `scope` (`per_request`, `per_run`, `region_profile`, `config_store`, `deployment`), `current`, `choices[]`, `settable`, `settable_via`, `reason`, `dims` |
| `labels` | display copy for served enums |

### Capability discovery

`GET /methods` returns `{axes[], strategies[], flags}`. Gate optional UI on
it instead of probing a write endpoint. `strategies[]` rows have `axis`
(`cluster`, `score`, `sort`, `overlay`, `export`, `prompt_pack`,
`detection_profile`, `vlm`), `id`, `label`, `status` (`stable`,
`experimental`, `shadow`, `disabled`), `default`, `settable`,
`requires_field`, `field_coverage` and `field_coverage_total`. A client shows
`stable` and `experimental` rows and hides a row whose `field_coverage` is
exactly `0`.

The `export` axis lists the kinds `POST /export/yolo` and
`POST /export/single_class` can produce. The `vlm` axis lists every endpoint
plus `off`. Each `vlm` row adds `endpoint_status` (`ready`, `unprobed`,
`probe_failed`, `unreachable`), `endpoint_status_label`,
`sends_images_externally`, `warning`, `default_ack_recorded` and
`per_run_ack_required` (what a run enforces; see
[VLM endpoints](#vlm-endpoints-catalog-and-activation)).

## Keymap

Per-project, configurable keyboard map. Action ids, contexts, the key grammar
and the locked and browser-reserved keys are served; a client does not
hardcode them. `contracts/json/keymap_actions.json` is the generated action
catalog.

| Method | Path | Body | Response | Errors |
|---|---|---|---|---|
| GET | `/keymap` | | `KeymapGetResponse`: `project`, `revision`, `etag` (`keymap:N`), `is_default`, `updated_at`, `grammar`, `contexts[]`, `actions[]`, `overrides`, `reserved_hotkeys`, `issues[]` | |
| PUT | `/keymap` | `KeymapPutRequest`: `overrides` (action id to combos), `expected_revision`, `unbind_conflicting_class_hotkeys` | `KeymapPutResponse` (`KeymapGetResponse` plus `unbound_class_hotkeys[]`) | `409 revision_conflict`, `409 class_hotkey_conflict`, `422 validation_failed` |
| POST | `/keymap/validate` | `{overrides}` | `KeymapValidateResponse`: `ok`, `errors[]`, `warnings[]`, `force_allowed`, `resolved`, `reserved_hotkeys`, `class_conflicts[]` | |
| POST | `/keymap/reset` | `KeymapResetRequest`: `action_ids` (all when omitted), `expected_revision`, `unbind_conflicting_class_hotkeys` | `KeymapPutResponse` | `409 revision_conflict`, `409 class_hotkey_conflict` |

- `PUT` replaces the whole override map. An action absent from `overrides`
  takes its default.
- The revision comes from `expected_revision` or an `If-Match: "keymap:N"`
  header. If both are given they must agree. If neither is given the request
  is `422`.
- A combo that collides with a class hotkey of the project is
  `409 class_hotkey_conflict` with `class_conflicts[]`. With
  `unbind_conflicting_class_hotkeys: true` the server unbinds those class
  hotkeys and saves the keymap in one step, and rolls the registry back if
  the save fails. `unbound_class_hotkeys[]` reports what was unbound.
- A write publishes `config.changed` with `axis: "keymap"` on the project's
  event stream.

## Prompt packs

A prompt pack holds the prompts and reply contracts of every VLM call:
classification, open-vocabulary classification, combined classify and verify,
region verification, region visibility, per-class descriptions and synonyms.
The built-in generic pack is read-only. Packs are per project.

| Method | Path | Body or query | Response | Errors |
|---|---|---|---|---|
| GET | `/prompt_packs` | | `PromptPackList`: `packs[]`, `templates[]`, `active`, `config_revision`, `stale` | |
| POST | `/prompt_packs` | `{name, body, description}` | `201 PromptPackDoc` | `409 name_conflict`, `422 validation_failed` |
| GET | `/prompt_packs/schema` | | `PromptPackSchema`: `calls[]`, `fields[]` (placeholders, expected reply keys), `placeholders[]`, `reply_key_contract` | |
| POST | `/prompt_packs/validate` | `{body, name}`, query `profile` | `ValidationReport` | |
| POST | `/prompt_packs/test` | `PackTestRequest` | `PackTestResponse` | see [Test on crops](#test-on-crops) |
| GET | `/prompt_packs/active` | | `ActiveConfigResponse`: `axis`, `active`, `source` (`stored`, `env`, `off`; with `env`, `active.name` is the env/file default in effect), `activated_at`, `previous`, `applied[]`, `config_revision`, `stale` | |
| POST | `/prompt_packs/active/rollback` | `{expected_active}` | `ActiveConfigResponse` | `409 no_previous`, `409 previous_deleted`, `409 active_conflict` |
| GET | `/prompt_packs/{name}` | | `PromptPackDoc` (`name`, `body`, `description`, `revision`, `source`, `read_only`, `active`, `cloned_from`, `etag`) | `404 not_found` |
| PUT | `/prompt_packs/{name}` | `{body, description, expected_revision}` | `PromptPackDoc` (a new revision) | `403 read_only`, `409 revision_conflict`, `422 validation_failed` |
| DELETE | `/prompt_packs/{name}` | query `expected_revision` (required) | `204` | `403 read_only`, `409 in_use` (the active pack), `409 revision_conflict` |
| POST | `/prompt_packs/{name}/activate` | `{revision, expected_active, force}` | `ActiveConfigResponse` plus `validation` | `404 not_found`, `409 active_conflict`, `422 validation_failed` |
| POST | `/prompt_packs/{name}/clone` | `{new_name, source, revision, from_project, description}` | `201 PromptPackDoc` | `409 name_conflict`, `404 not_found` |
| GET | `/prompt_packs/{name}/revisions` | | `{name, revisions[]}` | `404 not_found` |
| GET | `/prompt_packs/{name}/revisions/{revision}` | | `PromptPackDoc` | `404 unknown_revision` |

- `clone` copies from the built-in pack, a template file, a stored pack
  (`source`), or a stored pack of another project (`from_project`, read-only).
- Validation checks names, required fields, placeholders, reply keys,
  synonym and description targets, and example values. A pack that asks for
  text with a text-free profile, or a single-box pack with a multi-box
  profile (`max_regions_per_item > 1`), is reported. Pairing errors that make
  the combination unusable are refused at activation and cannot be bypassed;
  `force` bypasses only the bypassable ones.
- Activating a pack validates it against the active region profile and VLM
  endpoint. A per-run override is available as `?prompt_pack=<name>` or
  `<name>@<revision>` on the labeling routes.

## Region profiles

A region profile describes how the sub-regions of an item are found and read:
the optional detector leg, the optional segmenter leg (a text prompt), the
OCR and text reader, the parent classes it applies to, and
`max_regions_per_item` (1 to 64; `1` is the default and a single box).

| Method | Path | Body or query | Response | Errors |
|---|---|---|---|---|
| GET | `/region_profiles` | `include_templates` | `RegionProfileList`: `profiles[]`, `templates[]`, `active`, `config_revision`, `stale` | |
| POST | `/region_profiles` | `{name, body, description}` | `201 RegionProfileDoc` | `409 name_conflict`, `422 validation_failed` |
| GET | `/region_profiles/schema` | | `RegionProfileSchema`: `fields[]`, `groups[]`. A field `type` is one of `string`, `int`, `float`, `bool`, `enum`, `string_list`, `int_list`, `float_pair`, `rgb`. For `type: enum` the choices are in the sibling `enum[]` (static list); a dynamic list names its source in `choices_from` instead (`detectors`, `segmenters`, `ocr_pipeline_models`, `ocr_det_models`, `ocr_rec_models`, `registry_classes`, `text_reader_modes`) | |
| POST | `/region_profiles/validate` | `{body, name}`, query `for_activation` | `ValidationReport` | |
| POST | `/region_profiles/validate_segmenter_prompt` | `{text_prompt, sole_leg}` | `ValidationReport` | |
| POST | `/region_profiles/test` | `RegionTestRequest` | `RegionTestResponse` | see [Test on crops](#test-on-crops) |
| GET | `/region_profiles/active` | | `ActiveConfigResponse` (`axis: detection_profile`) | |
| GET | `/region_profiles/active/impact` | | `ActivationImpact` | |
| POST | `/region_profiles/active/rollback` | `{expected_active}` | `ActiveConfigResponse` | `409 no_previous`, `409 previous_deleted`, `409 active_conflict` |
| POST | `/region_profiles/deactivate` | `{expected_active}` | `ActiveConfigResponse` (`source: off`) | `409 active_conflict` |
| GET | `/region_profiles/{name}` | | `RegionProfileDoc` (`body`, `effective`, `source`, `read_only`, `revision`, `active`, `etag`) | `404 not_found` |
| PUT | `/region_profiles/{name}` | `{body, description, expected_revision}` | `RegionProfileDoc` | `403 read_only`, `404 not_found`, `409 revision_conflict`, `422 validation_failed` |
| DELETE | `/region_profiles/{name}` | query `expected_revision` (required) | `204` | `403 read_only`, `409 in_use` (the active profile), `409 revision_conflict` |
| POST | `/region_profiles/{name}/activate` | `{revision, expected_active, force}` | `ActiveConfigResponse` plus `impact` and `validation` | `404 not_found`, `409 active_conflict`, `422 validation_failed` |
| POST | `/region_profiles/{name}/clone` | `{new_name, ...}` | `201 RegionProfileDoc` | `409 name_conflict`, `404 not_found` |
| GET | `/region_profiles/{name}/revisions` | | `{name, revisions[]}` | `404 not_found` |
| GET | `/region_profiles/{name}/revisions/{revision}` | | `RegionProfileDoc` | `404 unknown_revision` |

`RegionProfileDoc.effective` reports what the profile actually does:
`legs[]`, `reads_text`, `segmenter_enabled`, `text_hint_active`.

### Activation impact and rollback

Activating a profile does not rewrite any item. `ActivationImpact` (returned
by `/activate` and by `GET /region_profiles/active/impact`) reports how many
items exist per profile and revision (`by_profile[]`), `items_total`,
`validated_items`, `unseeded_items`, `pending_items`, `pending_not_matching`
and `stale_items`. `suggested_reprocess` is a dry-run `ReprocessRequest`
that selects every unlocked, machine-written item that the active
profile and revision did not produce. Post it to `POST /reprocess` as is, or
change `dry_run` to apply it.

`POST /region_profiles/active/rollback` re-activates the previous activation.
The previous target must still exist (`previous_deleted` otherwise).
`/region_profiles/deactivate` turns the profile axis off. Rollback and
deactivate are activations, so they publish the same events and are subject
to `expected_active`.

Validation covers the detector model (it must exist, be ready, be owned by or
shared with this project, and its classes must map by name), the segmenter
prompt, OCR models and the text reader, `parent_classes` (known classes) and
the pairing with the active pack and endpoint.

### Test on crops

`POST /prompt_packs/test` and `POST /region_profiles/test` run a draft or a
saved pack, profile or VLM endpoint against crops already in the project and
return what the worker would decide. They write nothing: no index update and no
enqueue.

| | `POST /prompt_packs/test` | `POST /region_profiles/test` |
|---|---|---|
| Subject | `pack_name`, `pack_revision` or `draft`; `call` (`combined`, `classify`, `open_classify`, `region_verify`, `region_visible`); `crop_ids[]`; `use_region_box`; `class_names`; `profile_name` | one `crop_id`; `profile_name`, `profile_revision` or `draft`; `segmenter_text_prompt`; `verify`; `prompt_pack_name`, `prompt_pack_revision` or `prompt_pack_draft` |
| VLM source | `vlm_name`, `vlm_revision` or `vlm_draft`; `acknowledge_external` | the same |
| Response | `call`, `pack`, `vlm`, `prompt` (exact request text), `raw_reply`, `reasoning`, `latency_ms`, `parse_ok`, `parse_error`, `results[]` (`crop_id`, `box_id`, `parsed`, `preview_item`, `skipped`), `validation` | `crop_id`, `profile`, `item_eligible`, `legs[]` (`leg`: `detector` or `segmenter`; `status`; `candidates[]`), optional `verify`, `preview_basis`, `preview_item`, `validation` |

A test candidate (`RegionTestCandidate`) is a box wire plus `candidate_index`,
`selected`, `drop_reason` (`below_min_score`, `nms`, `over_max`), `mask_iou`,
`bbox_in_parent`, `mask_polygon` and `mask_polygon_in_parent` (at most 256
points). `preview_basis` is `selection_accepted` without a verify leg (every
selected box is shown as accepted) and `vlm_verdicts` with one.

The profile test does not run the OCR text-hint leg or text reading: a
profile that reads text previews boxes only.

Limits: a crop id from another project is `404 crop_not_found`; at most 4
concurrent segmenter calls and 2 concurrent VLM runs (`429 test_busy`); a 60
second bound (`504 test_timeout`); a crop cap (`422 too_many_crops`,
`422 too_many_crop_ids`); `422 no_box_to_verify`; `422 pack_invalid` and
`422 profile_invalid` carry `report`; upstream failures are
`502 vlm_transport_error`, `502 segmenter_error` and `502 detector_error`.

## VLM endpoints, catalog and activation

A VLM endpoint is an OpenAI-compatible chat endpoint with vision. The
endpoint registry is deployment-wide: an endpoint created in one project is
visible in all of them. Which endpoint a project uses is that project's
activation. The built-in `env` endpoint (`OP_VLM_URL`, `OP_VLM_MODEL`) is
listed first and is read-only. Clone it to get an editable copy.

### Registry (global routes)

| Method | Path | Body or query | Response | Errors |
|---|---|---|---|---|
| GET | `/curation/vlm/endpoints` | | `VlmEndpointList`: `endpoints[]` (`VlmEndpointSummary`), `external_policy` (`ack` or `deny`), `secret_refs[]`, `labels`, `config_revision`, `stale` | |
| POST | `/curation/vlm/endpoints` | `{name, body, description}` | `201 VlmEndpointDoc` | `409 name_conflict`, `422 validation_failed` |
| GET | `/curation/vlm/endpoints/schema` | | `VlmEndpointSchema`: form `fields[]` and `groups[]` (`choices_from`: `secret_refs` or `vlm_catalog`) | |
| POST | `/curation/vlm/endpoints/validate` | `{body, name}`, query `probe` | `VlmValidateResponse`: `validation`, `locality`, `sends_images_externally`, optional `probe`. Always `200` | |
| GET | `/curation/vlm/endpoints/{name}` | | `VlmEndpointDoc` | `404 not_found` |
| PUT | `/curation/vlm/endpoints/{name}` | `{body, description, expected_revision}` | `VlmEndpointDoc` (a new revision) | `403 read_only`, `409 revision_conflict`, `422 validation_failed` |
| DELETE | `/curation/vlm/endpoints/{name}` | query `expected_revision` (required) | `204` | `403 read_only`, `409 in_use` (`projects[]`), `409 revision_conflict` |
| POST | `/curation/vlm/endpoints/{name}/clone` | `{new_name, revision, description}` | `201 VlmEndpointDoc` | `409 name_conflict` |
| POST | `/curation/vlm/endpoints/{name}/probe` | | `VlmProbeResult` | `429 probe_busy` (more than 2 concurrent probes) |
| GET | `/curation/vlm/endpoints/{name}/revisions` | | `{name, revisions[]}` | `404 not_found` |
| GET | `/curation/vlm/endpoints/{name}/revisions/{revision}` | | `VlmEndpointDoc` | `404 unknown_revision` |

Names match `^[a-z0-9][a-z0-9_.-]{1,63}$`. Reserved: `env`, `off`, `none`,
`default`, `local`, `active`, `schema`, `validate`, `deactivate`.

`VlmEndpointBody`: `base_url`, `model`, `api_key_ref` (`secret:<slug>`),
`timeout_s`, `requests_per_second`, `max_images_per_call`,
`open_images_per_call`, `json_mode` (`auto`, `on`, `off`), `allow_external`,
`catalog_id`.

- Keys are references only. The API serves `api_key_ref` and
  `api_key_present`, never a key. Write a key on the host with
  `openprocessor vlm key set <slug>`.
- A probe sends synthetic images only. It records the served model root,
  context length, image token cost, the server's image cap and JSON mode
  support. A probe belongs to the revision and body it tested. A new revision
  is `unprobed` until probed.
- URL policy: the stack's own services, link-local, metadata and unspecified
  addresses (in any notation, for the literal host and every resolved
  address) are refused. No request to an endpoint follows a redirect.
- `locality` is `compose`, `host`, `private`, `external` or `unknown`. An
  endpoint outside this deployment (`sends_images_externally`) needs an
  acknowledgement. `OP_VLM_EXTERNAL_POLICY=deny` refuses such endpoints
  outright. An acknowledgement is recorded per project and per
  `name@revision`.
- Every registry write publishes the global event `vlm.changed`.

### Activation (project routes)

| Method | Path | Body | Response | Errors |
|---|---|---|---|---|
| GET | `/vlm/endpoints/active` | | `VlmActiveResponse` (`axis: vlm`, `active`, `source`: `stored`, `env` or `off`, `previous`, `applied[]`, `config_revision`, `stale`) | |
| POST | `/vlm/endpoints/{name}/activate` | `{revision, expected_active, force, acknowledge_external}` | `VlmActiveResponse` plus `validation` | `404 not_found`, `409 active_conflict`, `422 validation_failed`, `422 vlm_external_not_acknowledged` (`endpoint`, `activate_via`) |
| POST | `/vlm/endpoints/active/rollback` | `{expected_active}` | `VlmActiveResponse` | `409 no_previous`, `409 previous_deleted`, `409 active_conflict` |
| POST | `/vlm/endpoints/deactivate` | `{expected_active}` | `VlmActiveResponse` (`source: off`) | `409 active_conflict` |

`PUT /settings` with `defaults.vlm` activates through the same gate. The
detection worker follows a project's VLM at its quiesce points. An
activation, a re-probe or a rollback swaps the labeler. A new revision of the
active endpoint changes nothing until it is activated.

### One gate, every path

Every path that selects or switches an endpoint calls one gate: activate,
rollback, `PUT /settings`, a clone's activation copy, a per-run `?vlm=`, and
the draft tests. The gate checks:

- endpoint validation (URL, secret reference, external policy, and for an
  activation that a probe exists and did not fail);
- the external-images acknowledgement. Activation needs
  `acknowledge_external` or a recorded acknowledgement of that exact
  `name@revision`. Rollback, settings and clone need the recorded one. A
  one-off run needs `acknowledge_external` unless the endpoint is the
  project's acknowledged default;
- pairing of endpoint, prompt pack and region profile: an estimate of the
  context size of a labeler call, the server's own image cap, multi-box and
  text-reading verification, JSON mode.

`force` applies to `activate` only and bypasses only `vlm_not_probed`,
`vlm_probe_failed` and `vlm_context_too_small`. `vlm_max_images_exceeds_server`
is never bypassable.

### Per-run selection

The routes below take `?vlm=<name>` or `?vlm=<name>@<revision>` and
`?acknowledge_external=true`: `POST /vlm/label_batch`,
`POST /vlm/verify_regions`, `POST /vlm/verify_region_batch`,
`POST /vlm/region_visible_batch`, `POST /vlm/label_cluster/{cluster_id}`,
`POST /pipeline/auto_label` and `POST /pipeline/auto_label/start`. An unknown
endpoint is `422 unknown_vlm` (`valid_ids`). A project whose VLM is off is
`409 vlm_not_configured`. An endpoint that cannot be used is
`409 vlm_endpoint_unavailable`. `/pipeline/auto_label/start` pins the
resolved `(name, revision)` into the job.

Items record which endpoint answered: `vlm_endpoint` (`name@revision`),
`vlm_model` (the resolved model) and `vlm_prompt_pack`. `region_verifier`,
`text_engine_version` and the class `detector`/`labeler` provenance carry the
resolved model, not the configured name.

### Local model catalog (global routes)

The catalog (`examples/vlm/catalog.tsv`) lists models that the in-compose
vLLM service can serve, with license, VRAM and disk needs, context and image
limits, and whether each is `tested` or `to_verify`.

| Method | Path | Body | Response | Errors |
|---|---|---|---|---|
| GET | `/curation/vlm/catalog` | | `VlmCatalogResponse`: `entries[]` (`fits`, `serving`, `desired`, `vram_gb`, `disk_gb`, `max_model_len`, `max_images`, `license`, `gated`, `status`), `labels`, `local` | |
| GET | `/curation/vlm/local` | | `VlmLocalStatus`: `configured`, `served`, `desired`, `restart_required`, `can_restart_from_api`, `gpu_total_gb`, `reason`, `poll_after_s` | |
| POST | `/curation/vlm/local/select` | `{catalog_id, force}` | `202 VlmLocalStatus` | `409 no_local_vlm`, `422 unknown_catalog_id`, `422 vlm_catalog_does_not_fit` |
| DELETE | `/curation/vlm/local/select` | | `VlmLocalStatus` | |

The API never restarts the VLM service. `POST /curation/vlm/local/select`
records the desired model and returns `restart_required` with the host
command. `served` changes only after that command ran and a probe recorded
the new model.

Host CLI: `openprocessor vlm list`, `openprocessor vlm status`,
`openprocessor vlm use <id> [--force] [--yes]`, `openprocessor vlm apply`,
`openprocessor vlm probe` and `openprocessor vlm key set <slug>`. `use` checks
fit, handles the training lock and pause, rewrites the `.env` file (restoring
it on failure), waits for the new model, probes and unpauses.

## Items and the item wire format

An item is one detected object (a crop) in a source image. Every route that
returns an item returns the same shape, built by one serializer
(`serialize_item` in `src/services/curation/wire.py`). The wire names are
fixed. They do not change when a deployment overrides its storage field names
with `OP_REGION_FIELD_*`; the translation happens once at the boundary.
`contracts/json/item_wire.json` lists the exact key set (106 keys, all always
present, nullable where the schema says so) and `contracts/ts/itemWire.ts` is
the TypeScript form. `tests/curation/test_wire_contract.py` pins the key set.

### Item keys by group

| Group | Keys |
|---|---|
| Identity and geometry | `id`, `crop_id`, `image_id`, `image_path`, `source_image_path`, `bbox_norm` (`[x1, y1, x2, y2]` normalized to the source frame), `thumbnail_url`, `source`, `updated_at`, `crop_rank_in_image`, `crop_area_norm`, `blur_lap_ratio` |
| Class | `class_id`, `class_name`, `class_source`, `confidence`, `class_confidence`, `class_confidence_source`, `label_source`, `label_validated`, `class_validated`, `class_detector`, `class_detector_version`, `class_labeled_at`, `class_labeler`, `label_locked`, `test_holdout`, `proposal_name`, `proposal_chain` |
| Exclusion and review | `class_excluded`, `excluded_reason`, `excluded_at`, `review_dismissed_at`, `needs_new_class`, `needs_new_class_note` |
| VLM | `vlm_confidence`, `vlm_prompt_pack`, `vlm_endpoint`, `vlm_model`, `vlm_class_attempted_at`, `vlm_class_empty_reason`, `vlm_raw_class`, `vlm_proposed_class_id`, `vlm_proposed_class_name`, `proposed_class_id`, `proposed_class_name` |
| Cluster | `cluster_id`, `cluster_kind`, `cluster_distance`, `cluster_similarity`, `cluster_is_core`, `cluster_nearest_id`, `cluster_subid` |
| Probe and scores | `probe_pred_class`, `probe_pred_class_id`, `probe_pred_entropy`, `probe_disagreement`, `probe_in_scope`, `probe_model_version`, `probe_actionable`, `mistakenness_score`, `mistakenness_method`, `mistakenness_version`, `mistakenness_scored_at`, `uniqueness_score`, `dup_group_id`, `dup_group_size`, `dup_is_representative` |
| Region (item level) | `region_status`, `region_reason`, `region_rejection_reason`, `region_validated`, `region_auto_confirmed`, `region_verified`, `region_verified_at`, `region_verifier`, `region_verifier_version`, `region_visible`, `region_detector_chain`, `region_detected_at`, `region_profile`, `region_profile_revision`, `region_class_id`, `region_label_source`, `region_pairing`, `region_skip_verify` |
| Region boxes | `region_boxes`, `region_count`, `region_rejected_count`, `region_max_score`, `region_set_complete`, `region_revision` |
| Import and combine | `import_ids`, `dataset_split`, `imported_at`, `proposed_by_import`, `on_negative_frame`, `import_standalone_region`, `origin_project`, `origin_item_id`, `origin_image_id`, `origin_split`, `combine_conflict`, `combine_conflict_origins`, `combine_merged_origins` |
| Text | `item_text_lines` |

Endpoint-specific extra keys on top of the item:

| Endpoint | Items at | Extra keys |
|---|---|---|
| `GET /crops`, `GET /classes/{class_id}/crops` | `crops[]` | none |
| `GET /crops/{crop_id}` | body | none |
| `GET /review/{tab}` | `items[]` | `reason` |
| `GET /regions` | `items[]` | `region_box_id` |
| `GET /regions/training_candidates` | `items[]` | `region_box_id`, `selection_reason` |
| `GET /regions/suspected_false_positives` | `items[]` | `region_box_id`, `suspected_fp_distance`, `nearest_fp_subid` |
| `GET /search/text` | `items[]` | `semantic_score` |

### Derived and special keys

- `label_validated` is `class_validated` or `region_validated`.
- `label_locked` is the server's evaluation of [the lock rule](#the-lock-rule)
  for the whole item. A client disables nothing on its own reading of
  provenance fields; it reads this flag.
- `confidence` is the detector or classifier score stored at ingest,
  whatever wrote the current label. `class_confidence` and
  `class_confidence_source` are the confidence of the writer that set the
  label. For a VLM source the VLM category (`vlm_confidence`) maps high
  `0.92`, medium `0.70`, low `0.40` with source `vlm`. For a classifier
  source it is the stored score with source `model`. Human, move, merge,
  import, cluster-vote and unclassified-proposal labels have `null`.
- `cluster_kind` is `class`, `candidate` or `unassigned`, from `cluster_id`
  (see [Clusters](#clusters)). `cluster_similarity` is `1 - cluster_distance`.
  `cluster_is_core` is similarity at or above `core_similarity_min` (served
  on `GET /clusters`, default `0.75`). When the distance was measured against
  a cluster the item has since left, the three distance keys are `null`.
- `proposed_class_id` and `proposed_class_name` are the class a one-key
  confirm applies, on every item endpoint: the VLM suggestion when there is
  one, else `class_id` and `vlm_raw_class` or `class_name` or `""`.
- `probe_actionable` is the server's accept decision for the probe's top-1
  class. It is `true` only when `probe_in_scope` and `probe_disagreement` are
  `true` and the probe's confidence is at least
  `OP_PROBE_ACTIONABLE_MIN_CONFIDENCE` (default 0.5, echoed as
  `actionable_min_confidence` on `GET /probe/status`). Offer "accept model
  class" only when it is `true`. `probe_disagreement` is `null` when the
  item's class is outside the probe's class set.
- `region_validated` is human validation only. The detection worker's
  auto-confirm sets `region_auto_confirmed` instead: the region is accepted
  but unreviewed and stays in the `regions` review tab.
- `thumbnail_url` and each box's `thumbnail_url` are built from the
  configured prefix, so `OP_API_PREFIX` and a frontend proxy prefix must
  match.
- `item_text_lines` is every OCR line read on the item crop
  (`{text, box_norm, confidence, rel_height}`), `[]` when none. It is
  filled only when item text is enabled (`OP_ITEM_TEXT_ENABLED`; lines below
  `OP_ITEM_TEXT_MIN_CONFIDENCE` are not stored). The normalized search tokens
  are storage-only and back the `item_text` filter of `GET /crops`.

### VLM class suggestion

`vlm_proposed_class_id` and `vlm_proposed_class_name` are derived from the
stored document:

| Stored state | `vlm_proposed_class_id` | `vlm_proposed_class_name` |
|---|---|---|
| `class_source` is `vlm` or `vlm_reclassified`, `class_validated` false, `class_id` set | `class_id` | `class_name` |
| `class_source` is `vlm_new_class_pending`, `class_validated` false | `null` | the proposed new class name (`null` if absent) |
| anything else, including `vlm_unmatched` and every validated class | `null` | `null` |

`vlm_class_attempted_at` and `vlm_class_empty_reason` record when a VLM was
last asked and why it gave no class: `no_answer`, `no_match`, `invalid_index`
or `unparseable`; `null` when it answered. An empty answer leaves every class
field as it was. It is not `vlm_unmatched`, which means the VLM named a label
outside the registry (carried in `vlm_raw_class`). Such items stay out of the
VLM selectors for 24 hours.

Accepting a suggestion has no dedicated route:

- a registry class: `PUT /crops/{crop_id}/label` or `PUT /crops/batch_label`
  with `class_id` set to `vlm_proposed_class_id` (`label_source:
  "human_confirmed"` marks an accepted suggestion);
- a new class: `POST /classes` with `{name}` set to
  `vlm_proposed_class_name` (`409` if it exists), then label with the new
  `class_id`. To resolve every item that proposes one name at once, use
  `POST /review/new_class_proposals/resolve`.

### `class_source` and `label_source`

`GET /class_sources` returns `{class_sources: [{id, label, role,
short_label}]}`, every `class_source` value this deployment can write.
Ingest values come first and derive from the configured ingest profiles
(`OP_INGEST_PRIMARY_*`, `OP_INGEST_SECONDARY_*`). `role` is one of
`proposal`, `low_conf`, `model`, `vlm`, `vlm_unmatched`,
`vlm_new_class_pending`, `vlm_reclassified`, `cluster`, `human`, `merge`,
`label_import`. Label routes take a caller-chosen `label_source`, so a
client renders an id it does not know verbatim. `contracts/ts/classSources.ts`
is generated from the catalog.

Fixed ids include `vlm`, `vlm_unmatched`, `vlm_new_class_pending`,
`vlm_reclassified`, `cluster_majority_agreement`, `human`, `human_move`,
`class_merge` and `external_label` (a validated dataset import).

### `region_boxes`

Regions are a list per item. One region is a list of one; there is no
separate single-box shape. Each element:

| Key | Meaning |
|---|---|
| `box_id` | `b<N>`, stable within the item and never reused after a delete |
| `bbox_norm` | `[x1, y1, x2, y2]` in the source-image frame, always |
| `state` | `proposed`, `accepted`, `rejected` or `false_positive` |
| `score`, `detector`, `detector_version`, `source`, `detected_at` | provenance. A box a human draws or moves has `detector` set to the profile's human detector name, `score` 1.0 and `source` `human` |
| `bbox_correct`, `confidence` | the verifier's verdict on the geometry and its category; `bbox_correct` is `null` when no verdict was given |
| `rejection_reason` | why the box is `rejected` (see below) |
| `text`, `text_raw`, `text_source`, `text_engine_version`, `text_confidence`, `text_vlm`, `text_ocr`, `text_choice`, `text_vlm_invalid`, `text_disagreement` | per-box text, only when the region profile reads text |
| `cluster_id`, `cluster_subid`, `cluster_distance` | per-box region cluster placement |
| `locked` | server-derived: [the lock rule](#the-lock-rule) for this box |
| `bbox_in_parent` | server-derived: the box in the item-crop frame, clamped to `[0, 1]`; `null` when the item has no usable `bbox_norm`. Draw it on the item thumbnail as is |
| `thumbnail_url` | server-derived: `GET /crops/{crop_id}/region_thumbnail?box_id=<box_id>` |

Item-level summaries: `region_count` counts `accepted` boxes,
`region_rejected_count` counts `rejected` ones, `region_max_score` is the best
score, `region_set_complete` says the worker finished the set, and
`region_revision` is the token for `expected_region_revision`. It advances on
every write that changes what an editor sees.

Embeddings per box live in a separate nested field (`region_box_embeddings`)
that is not on the wire. A box a human moved is recognized as stale and
re-embedded.

The item `region_status` is derived from the boxes in a fixed order:
`detected` if any box is `accepted`, else `false_positive` if any box is
`false_positive`, else `pending_verification` if any is `proposed`, else
`verify_rejected` if any is `rejected`, else the empty status. The item-level
`region_rejection_reason` mirrors the best rejected box only when the item
has no `accepted` or `false_positive` box.

A verifier-rejected box is kept with `state: rejected`. Its
`rejection_reason` is `region_visible_elsewhere` (the verifier says the
region is elsewhere), `sanity_reject:<gate reason>` (the geometry gate),
`verifier_no_verdict` (no verdict after the allowed attempts), or the
human reason. `GET /regions/vocabulary` serves these as `rejection_reasons[]`
(`id`, `label`, `kind` of `model_verdict`, `automatic`, `needs_human` or
`human`, `match` of `exact` or `prefix`, `label_template`). A rejected box is
never an accepted region: browse, clustering and export ignore it. A human
reverses a rejection with a confirm write; region undo restores it.

### Text reading

A box's `text` is the chosen reading. Which reader fills it is the region
profile's `text_reader`:

| `text_reader` | Region OCR runs | `text` |
|---|---|---|
| `none` | never | no text on the box |
| `vlm` | only when no VLM is configured | the VLM's reading |
| `ocr` | always | the OCR reading (the VLM's if OCR read nothing) |
| `vlm_then_ocr` (default) | when the VLM read nothing | the VLM's, else the OCR's |
| `both` | always | the VLM's, else the OCR's |

- `text_source` is `vlm`, `ocr` or `human`. `text_engine_version` is the VLM
  model id or the OCR detector and recognizer ids. `text_confidence` is the
  VLM category mapped to 0.92, 0.70 or 0.40, or the minimum recognition score
  of the kept OCR lines. `text_raw` is every OCR line, unfiltered. `text_vlm`
  and `text_ocr` are each reader's own reading. `text_disagreement` is
  `true` or `false` when both valid readings exist, else `null`.
- `text_choice` is why the chosen reading won: `readers_agree`,
  `vlm_preferred`, `vlm_only`, `ocr_only`, `ocr_mode`, `vlm_invalid`,
  `no_valid_reading` (`text` is `null`) or `human`.
- `text_vlm_invalid` says why a VLM reading is not text: `placeholder`,
  `no_reading`, `sequence`, `charset`, `too_short`, `too_long`, `format`.
- The rules and the choice values are served as `text_rules` and
  `text_choices` on `GET /regions/vocabulary` (`null` rules without a
  profile). A reading that is a "no reading" word, a placeholder from the
  active pack, a repeated or ascending sequence (when the profile rejects
  them), or outside the profile's charset, length or format is not text.
- With no VLM configured the worker calls no VLM. Detector regions are
  written `detected` with `region_verified` false (chain entry
  `<src>:accepted_unverified`) and their text comes from OCR.
- The VLM's reply keys `region_bbox_correct`, `region_confidence` and
  `region_text` are a fixed protocol of the prompt packs. They are stored as
  the box's `bbox_correct`, `confidence` and `text`.
- `scripts/curation/rederive_region_text.py` re-applies the rules to stored
  box text (dry run by default).

### `region_detector_chain`

A list of strings, oldest first, each `<actor>:<event>` with no version or
timestamp, unique per document and capped at 16. `<det>` is the profile's
detector, `<seg>` its segmenter, `<ocr>` its OCR recognizer and `<src>`
whichever produced the candidate.

| Entry | Meaning |
|---|---|
| `<det>:hit`, `<det>:miss` | the detector found, or found no, candidate |
| `<seg>:hit`, `<seg>:miss` | the segmenter found, or found no, candidate |
| `vlm_visible:yes`, `vlm_visible:no` | the VLM pre-filter saw, or did not see, a region |
| `vlm_visible:no_verdict` | no verdict after the allowed attempts; the item goes on to detection |
| `<src>:combined_verify_ok` | the VLM confirmed the candidate (written `detected`) |
| `<src>:combined_verify_reject`, `...:region_visible_elsewhere`, `...:verifier_no_verdict` | the VLM rejected the candidate, with the reason |
| `<src>:vlm_reject:verifier_no_verdict` | the same, from the per-crop cascade |
| `<src>:combined_no_region_visible` | the VLM sees no region at all |
| `<src>:sanity_reject:<reason>` | the geometry gate refused the box |
| `<seg>:skip_vlm_verify` | a high-score segmenter box written without a VLM call |
| `<src>:accepted_unverified` | no VLM configured |
| `<ocr>:text_hint:hit`, `:miss`, `:no_region_shape`, `<seg>:text_hint:miss` | the OCR-hinted segmenter re-pass |

A VLM reply that sees a region but gives no box verdict is not a reject.
Nothing is written and the item stays pending for a retry. Retries are
bounded per item and stage by `OP_REGION_WORKER_MAX_NO_VERDICT_ATTEMPTS`
(default 3). At the cap the combined stage writes `verify_rejected` with
`verifier_no_verdict` and keeps the box rejected so a human can confirm it.
A transport failure (no reply) is retried and never counted.

Readers match whole entries with `term` queries. For example
`GET /regions/training_candidates?mode=detector_blind_spots` requires
`<det>:miss`.

### Region lifecycle vocabulary

`GET /regions/statuses` serves the lifecycle. The source is
`src/config/region_state.py`, also emitted to `contracts/ts/regionStatus.ts`.

```json
{"statuses": [{"value": "no_region_visible", "label": "no region visible", "role": "absent",
               "terminal": true, "human_writable": true, "clears_box": true, "wants_reason": true}],
 "confirm_status": "detected", "reject_status": "no_region_visible",
 "false_positive_status": "false_positive",
 "box_states": [{"value": "accepted", "label": "accepted", "role": "accepted", "tone": "accepted",
                 "human_writable": true, "exported": true, "dashed": false, "dim": false, "badge": null}],
 "box_state_routes": [{"route": "PUT /crops/{crop_id}/regions",
                       "states": ["proposed", "accepted", "rejected", "false_positive"],
                       "new_box_default": "accepted"}]}
```

Item statuses: `pending_detection`, `pending_verification`, `detected`,
`verify_rejected`, `no_region_box`, `no_region_visible`, `detection_failed`,
`false_positive`. `role` is `pending`, `positive`, `rejected`, `absent`,
`false_positive` or `failed`. `tone` is a semantic meaning, not a color.
`box_state_routes` lists which box states each write route accepts and the
default state of a new box.

## Crops: browse, label, undo

### Browse

| Method | Path | Response |
|---|---|---|
| GET | `/crops` | `CropsPageResponse`: `crops[]`, `total`, `page`, `page_size`, `method`, `version`, `n_pool` |
| GET | `/crops/{crop_id}` | item (`404` when unknown) |
| GET | `/crops/{crop_id}/context` | `CropContextResponse`: `image` (`image_id`, `image_path`, `width`, `height`, `source`, `indexed_at`; `null` when unknown) and `items[]`: every item on the frame, `crop_rank_in_image` ascending, at most 500 |
| GET | `/crops/{crop_id}/history` | `{crop_id, entries[]}`: `class_id_history` oldest first |
| GET | `/classes/{class_id}/crops` | `CropsPageResponse` (`page`, `page_size`, `include_test`) |

`GET /crops` query parameters:

| Parameter | Meaning |
|---|---|
| `page`, `page_size` (1 to 500, default 50), `limit` | paging. `limit` is an alias of `page_size` and wins when both are set |
| `sort` | `<field>[:asc\|desc]`, default `updated_at:desc`. Fields: `updated_at`, `created_at`, `confidence`, `crop_rank_in_image`, `crop_area_norm`, `blur_lap_ratio`, `cluster_distance`, `mistakenness_score`, `uniqueness_score`. Anything else is `400`. Ignored by `order=outliers` and `order=diverse` |
| `class_id`, `cluster_id`, `label_source`, `class_source`, `label_validated`, `source` | exact filters |
| `needs_new_class`, `review_dismissed` | boolean filters |
| `ids` | comma-separated, at most 500. Returns exactly those items in that order, drops missing ids and ignores every other filter |
| `include_test`, `include_excluded` | include frozen test-holdout and excluded items |
| `max_rank`, `min_blur_ratio`, `classifier_conf_lt`, `conf_min`, `conf_max` | numeric bands (`400` if `conf_min` is above `conf_max`) |
| `order` | `default`, `outliers`, `core_first`, `diverse`. `outliers` and `core_first` need `cluster_id` and order by distance to the live centroid of the matched members. Under `core_first` each item's cluster distance keys are recomputed against that centroid. `method` reports the order that ran |
| `k` | 1 to 10000, `order=diverse` only: the first `k` k-center-greedy picks; `total` is then `k` |
| `item_text` | at most 200 characters. Every letter or digit word must be a case-insensitive prefix of one of the item's text tokens. A query with no letter or digit is `400` |
| `import_id` | items whose label or proposal came from that dataset import |
| `dataset_split` | `train`, `val` or `test` as imported |
| `on_negative_frame` | `true` keeps only items on a reviewed-negative frame, `false` hides them |
| `proposed_by_import` | `true` keeps items an import proposed, `false` hides them |

### Human class writes

Every write below is a recorded human write: it snapshots the item's
pre-write class state into `class_id_history`, so the undo routes can restore
it. Human writes are not blocked by [the lock rule](#the-lock-rule). The lock
applies to automated writers.

| Method | Path | Body | Response | Notes |
|---|---|---|---|---|
| PUT | `/crops/{crop_id}/label` | `CropLabelRequest`: `class_id`, `label_source` (`human` default, `human_confirmed`, `new_class_proposal`) | `{crop_id, class_id, class_name}` | `400` for an unknown `class_id`, `404` for an unknown crop. The server always writes `class_source: "human"` and `class_validated: true` |
| PUT | `/crops/batch_label` | `CropBatchLabelRequest`: `crop_ids`, `class_id`, `label_source` | `{updated, updated_ids, conflicts[]}` | `conflicts` is `[{crop_id, current_source}]`, not written. `updated_ids` are the ids to pass to `undo_batch` |
| POST | `/crops/move` | `CropMoveRequest`: `crop_ids`, `cluster_id` | `{updated, updated_ids, conflicts[]}` | a class cluster is also a relabel (`class_source: human_move`, validated); a candidate cluster is placement only (a human-owned class is cleared, a machine suggestion is kept, nothing is validated). `400` for an unassigned (negative) target, an unknown class id or a candidate with no members |
| POST | `/crops/flag_new_class` | `{crop_ids, note}` | `{flagged, errors}` | sets `needs_new_class` |
| POST | `/crops/batch_exclude` | `{crop_ids, reason}` | `{excluded, errors}` | sets `class_excluded`, moves the item to cluster `-2`, clears `class_validated` and records the prior state |
| POST | `/crops/batch_unexclude` | `{crop_ids}` | `{unexcluded, errors}` | restores the recorded state: a validated item returns to `cluster_id == class_id`, an unvalidated one to the residual pool. Items that are not excluded are untouched |
| POST | `/crops/{crop_id}/discard` | `{clear_class: true, dismiss_from_review: false}`, `422` if both false | the item | clears class, provenance and validation and drops the item to the residual pool, and/or hides it from every review tab. Recorded and undoable |
| POST | `/crops/discard_batch` | `{crop_ids, clear_class, dismiss_from_review}` | `{items[], discarded, conflicts, not_found}` | |
| POST | `/crops/{crop_id}/review_dismiss` | | `{crop_id, dismissed}` | one-way review hide, not recorded. Prefer `discard` with `clear_class: false` and `dismiss_from_review: true` |
| POST | `/crops/{crop_id}/review_undismiss` | | the item | clears `review_dismissed_at` |
| POST | `/crops/{crop_id}/vlm_dismiss` | | the item | rejects the VLM's class suggestion. While the VLM's suggestion is that one the suggestion keys are `null`. A different later suggestion shows again. `409` when there is no suggestion |

### Undo

| Method | Path | Body | Response | Errors |
|---|---|---|---|---|
| POST | `/crops/{crop_id}/label/undo` | | the restored item | `404` unknown crop, `409` nothing to undo |
| POST | `/crops/label/undo_batch` | `{crop_ids}` | `{items[], undone, nothing_to_undo, conflicts, not_found}` | `409` when no crop had anything to undo |
| DELETE | `/crops/{crop_id}/label` | | `{crop_id, reset}` | the same restore. With nothing on record it resets the crop to unlabeled instead of `409`. It is itself an undo, so it is not recorded |
| POST | `/crops/{crop_id}/region/undo` | | the restored item | `404`, `409` nothing to undo |
| POST | `/crops/region/undo_batch` | `{crop_ids}` | `{items[], undone, nothing_to_undo, conflicts, not_found}` | `409` when no crop had anything to undo |
| POST | `/crops/{crop_id}/vlm_dismiss/undo` | | the restored item | `409` no dismissal to undo |

- A class snapshot holds `class_id`, `class_name`, `class_source`,
  `label_source`, `confidence`, `class_detector`, `class_detector_version`,
  `class_labeler`, `class_labeled_at`, `class_validated`, `cluster_id` and
  `cluster_subid`. Repeated undo steps back through successive human writes.
  A restored validated class sits in its class cluster
  (`cluster_id == class_id`); anything else returns to the cluster recorded
  before the write (`null` is the residual pool). Undo on an excluded item
  keeps it excluded.
- A region snapshot holds `region_boxes` with its summaries,
  `region_status`, `region_verified*`, `region_verifier*`,
  `region_validated`, `region_label_source`, `region_detected_at` and
  `region_rejection_reason`. Class fields are untouched.
- History entries (`GET /crops/{crop_id}/history`) add `writer` (for example
  `human:label_crop`, `human:discard_crop`, `class_merge`, `vlm_pipeline`) and
  `at`. Keys that were not recorded are `null`.
- Which route to call: label or confirm with `PUT /crops/{crop_id}/label`;
  hide or clear with `POST /crops/{crop_id}/discard`; revert with the undo
  routes.

## Regions: multi-box edits and queues

The region write routes, `GET /regions`, the region undo routes and the region
cluster card and refine routes need an active region profile. Without one they
answer `409` with the plain detail `no region profile is configured`.

### Per-box edits

These are the only human routes that write box geometry or per-box state.
Every one builds its update through the same code, snapshots the pre-write
region state for [region undo](#undo), and returns the post-write item so a
client adopts it instead of re-deriving it.

| Method | Path | Body | Response |
|---|---|---|---|
| PUT | `/crops/{crop_id}/regions` | `ItemRegionsRequest` | `{crop_id, item}` |
| PUT | `/crops/batch_regions` | `ItemBatchRegionsRequest` | `{updated, conflicts[], invalid[], items[]}` |
| PATCH | `/crops/{crop_id}/regions/{box_id}` | `BoxPatchRequest` | `{crop_id, box_id, item}` |
| POST | `/regions/batch_box_state` | `BatchBoxStateRequest` | `{updated, conflicts[], invalid[], items[]}` (rows: one per targeted box, with `region_box_id`) |
| PATCH | `/crops/{crop_id}/region_meta` | `ItemRegionMetaRequest` | `{crop_id, updated_fields[], item}` |
| POST | `/regions/batch_status` | `CropBatchStatusRequest` | `{updated, conflicts[], invalid[], items[]}` |

`PUT /crops/{crop_id}/regions` sets the full box list, in display order.
An element (`BoxWriteElement`: `box_id`, `bbox_norm`, `state`, `text`; extra
keys are `422`) is:

| Element | Effect |
|---|---|
| `{box_id}` alone | the stored box is untouched |
| `{box_id, bbox_norm}` | a moved box. Within `1e-4` per coordinate of the stored box it is a confirmation: the stored coordinates and provenance are kept. A different box is human geometry: `detector` is the human detector, `score` is 1.0, `source` is `human`, the verdict keys are cleared. State and text stay |
| `{box_id, state}` or `{box_id, bbox_norm, state}` | also changes the state |
| `{box_id: null, bbox_norm, state?}` | a new box. The server assigns the next `b<N>`. The default state is `accepted` |
| a stored box that is omitted | deleted |

Other fields: `frame` (`source` default; `parent` is the item crop's own
frame, projected through the item's stored `bbox_norm`, `422` if the item has
none), `region_status` (a whole-set status applied to the built list in the
same write), `region_label_source` (default `human`) and
`expected_region_revision`. An empty list is the human "no region visible".

`PUT /crops/batch_regions` replaces each crop's list with the same new boxes
(typically `[]`). Every element must have `box_id: null` (`422
box_id_in_batch`).

`PATCH /crops/{crop_id}/regions/{box_id}` takes `state`, `text`,
`region_label_source` and `expected_region_revision`. At least one of `state`
and `text` is required (`400`). Siblings are untouched. Reversing a rejection
of one box is `{"state": "accepted"}`, which keeps the box's provenance and
clears its rejection reason.

`POST /regions/batch_box_state` flips one state on named boxes across items
(`targets: [{crop_id, box_id}]`, `state`, `region_label_source`,
`expected_region_revisions` keyed by crop id). Only the named boxes change.

`PATCH /crops/{crop_id}/region_meta` takes `region_status`,
`region_rejection_reason` and `region_label_source`; only provided fields are
written. `POST /regions/batch_status` takes `crop_ids`, `region_status` and
`region_label_source` and flips every box of each item. The deprecated
`region_verified` key is accepted and ignored: the server derives it.

Errors for the box routes:

| Status | Cause |
|---|---|
| `400` | `PATCH` with neither `state` nor `text`; a `region_status` that is not human-writable on `region_meta` and `batch_status` |
| `404` | unknown crop |
| `409 region_conflict` | stale `expected_region_revision`; the body carries `current_region_revision`, `current_box_ids` and the current `item` |
| `409` | no region profile is active |
| `422 too_many_boxes` | more elements than `limits.max_boxes_per_write` (`OP_REGION_MAX_BOXES_PER_WRITE`, default 500; served on `GET /health` and `GET /regions/vocabulary`) |
| `422 region_text_disabled` | a `text` element on a profile whose `text_reader` is `none` |
| `422` | a `state` not allowed on that route, an out-of-range or degenerate box, a new box without `bbox_norm`, a duplicate or unknown `box_id`, `no_region_visible` together with boxes, `detected` with no accepted or confirmable box |

In the batch routes a per-crop refusal is reported in `invalid[]` (`{crop_id,
detail}`) and a stale revision in `conflicts[]`. The rest are written.

### Lifecycle rules

Every human region writer enforces, server-side:

- a status with `clears_box` (`no_region_visible`) empties `region_boxes`,
  whichever writer set it. An empty list written by `PUT` is that same
  status;
- `region_verified` is `true` exactly when `region_status` is `detected`. It
  is never taken from the request;
- `detected` on an item with no accepted or confirmable box is refused. A
  whole-set confirm settles the `proposed` boxes and never overrides a
  per-box decision. When nothing is `proposed` or `accepted` it reopens only
  the boxes the verifier rejected. A human's own reject and a sanity-gate
  reject are never reopened by it;
- human writes set `region_validated`. A box write that leaves a `proposed`
  box is a partial review and leaves it as stored;
- a box that becomes `false_positive` is parked in the permanent FP cluster
  (id `-100`); one that leaves it is released;
- a write that re-asserts the stored status re-derives nothing: cluster
  placement stays;
- a human reject stamps the box as human-owned; a human-typed text sets
  `text_source` to `human`.

Writable item statuses are the `human_writable` rows of
`GET /regions/statuses`: `detected`, `verify_rejected`, `no_region_visible`
and `false_positive`.

### Region browse, clusters and false positives

| Method | Path | Notes |
|---|---|---|
| GET | `/regions` | `RegionRowPage`. Filters: `page`, `page_size` (max 200), `class_id`, `cluster_id` (the item cluster), `region_cluster_id`, `region_cluster_subid`, `sort_by_subid`, `max_rank`, `min_score`, `max_score`, `verified`, `detector`, `text`, `box_state`, `status`, `include_test` |
| GET | `/regions/statuses` | the lifecycle vocabulary (see above) |
| GET | `/regions/vocabulary` | `RegionVocabularyResponse`: `detectors[]` (`id`, `label`, `role`, `filterable`), `region_sources[]`, `chain_actors[]`, `rejection_reasons[]`, `text_rules`, `text_choices`, `region_profile` summary. Built from the active profile, the ingest profiles and the VLM registry, never from a fixed model id |
| GET | `/regions/training_candidates` | `RegionRowPage`, `mode` required |
| GET | `/regions/suspected_false_positives` | `RegionRowPage`. `threshold` (0 to 2) is optional; omitted, the server applies `default_threshold` (0.35) and serves both |
| POST | `/regions/cluster` | job. Query `max_rank`, `auto_fp_threshold` (default 0.2), `rebuild_fp_centroids` (default true), `force_repartition` |
| GET | `/regions/cluster/status` | job state |
| GET | `/regions/clusters` | cluster cards (`max_clusters`, `per_cluster`, `max_rank`) |
| POST | `/regions/clusters/refine/{cluster_id}` | AHC refine of one region cluster |
| POST | `/regions/fp_centroids/build` | rebuild the false-positive sub-centroids from FP boxes |
| GET | `/regions/fp_centroids/status` | |
| GET | `/crops/{crop_id}/region_thumbnail` | `box_id` is required (`422 box_id_required`, `404 unknown_box_id`), `size` 32 to 512. Renders that box, also a rejected one |

Rows. `GET /regions`, `/regions/training_candidates`,
`/regions/suspected_false_positives`, the representatives of
`/regions/clusters` and the items of the batch box routes are rows: the full
wire item plus `region_box_id` (`null` for an item-level row). `total` counts
items (`hasMore = page * page_size < total`), `total_rows` counts rows, and a
page returns every row of its items. `rows_truncated` is `true` when an item
on the page matched more boxes than `index.max_inner_result_window` reports,
so some of its rows are missing.

`GET /regions` filters. The box filters (`detector`, `min_score`,
`max_score`, `text`, `region_cluster_id`, `region_cluster_subid`,
`box_state`) all apply to the same box, and each matching box is its own row.
With no box filter and no `status` the rows are the accepted and
`false_positive` boxes. The item filters (`status`, `class_id`, `cluster_id`,
`verified`, `max_rank`) select items. `status` is any value from
`GET /regions/statuses` (`400` otherwise) and lists every item in that status
as item rows. `box_state` is `proposed`, `accepted`, `rejected` or
`false_positive` (`400` otherwise).

`training_candidates` modes:

| Mode | Unit | Selects |
|---|---|---|
| `detector_blind_spots` | box | the detector missed and the segmenter found it (`<det>:miss` in the chain) |
| `low_conf_correct` | box | accepted boxes with a low score |
| `false_positives` | box | boxes a human marked false positive |
| `disagreement` | item | detector and segmenter both fired (`<det>:hit` and `<seg>:hit`) |
| `human_corrected` | item | a human moved or drew the box over a machine proposal |

An unknown mode is `400`. Each row's `selection_reason` states why it was
selected. `GET /training_cohorts` serves the same cohorts as links.

`GET /regions/suspected_false_positives` scores boxes: an accepted box with a
vector, of an item that is not test-holdout and not human-decided, and not
itself a human's or an import's box. Each row is one box with
`suspected_fp_distance` and `nearest_fp_subid`. It pages rows directly, so
`total == total_rows`. The matcher is double-layered: the build sub-types the
false-positive boxes into `k` sub-clusters and keeps one centroid per
sub-type. A candidate matches the nearest of them. Re-run the build after
marking a batch of false positives.

Region clustering is over boxes. A card's `size` is the number of items with
at least one box in the cluster and `box_count` is the number of boxes.
`representatives` are rows for the boxes nearest the centroid, next to
`representative_crop_ids`, `representative_box_ids` and
`representative_thumb_urls`. The permanent false-positive cluster (`-100`)
pins first. The count fields name their unit: the cluster job result has
`n_boxes`, `n_boxes_changed` and `n_items_written`; a region refine has
`n_boxes` and `n_boxes_updated`; an item refine has `n_items` and
`n_items_updated`; the centroid build has `n_boxes`; the automatic FP pull
has `n_boxes_scanned` and `n_boxes_moved`.

Automated region writers (worker, clustering, reprocess) never touch a locked
box or a human-final item (see [the lock rule](#the-lock-rule)).

### Cohorts

`GET /training_cohorts?class_id=` returns `{cohorts: [{id, label,
description, cutoffs, endpoint, params, row_kind}]}`. Fetch a cohort's rows
with `GET {endpoint}` and `params`. Core cohorts are `validated`,
`needs_labeling`, `low_confidence` (`cutoffs.classifier_conf_lt`, the
backend's review band) and `model_disagreements` (`row_kind: crop`). With a
region profile active, the `/regions/training_candidates` modes follow
(`row_kind: region`).

### Per-class thresholds

One definition (`src/services/curation/dataset_thresholds.py`) serves both
training preflight and every count display:

```json
"thresholds": {"block_below": 20, "warn_below": 500, "min_test_per_class": 5,
               "min_train_per_class": 1, "min_val_per_class": 1,
               "aug_target_min": 500, "aug_target_max": 3000}
```

`adequacy` is `block` below `block_below` validated items, `warn` below
`warn_below`, else `ok`. `aug_target` is the validated count clamped to
`[aug_target_min, aug_target_max]`; `aug_gap` is `aug_target` minus the
validated count. `trainable` is validated minus test-holdout minus excluded;
`trainable_gap` is `max(0, block_below - trainable)`. These appear on
`GET /classes` (`thresholds`, per-class `adequacy`, `trainable`,
`trainable_gap`), `GET /stats/classes` (also `aug_target`, `aug_gap`),
`POST /train/preflight` and `GET /test_holdout/stats`
(`min_test_per_class` and per-class `deficient`).

## Review tabs

`GET /review/tabs` serves the tab catalog:
`{tabs: [{id, label, description, filters, filter_defaults, filter_specs}], empty_state}`.
`filters` is the list of query parameters a tab honors (a parameter that is
not listed is accepted and ignored). `filter_defaults` is the value applied
when a parameter is omitted. `filter_specs` self-describes each enum filter
(`{param, kind: "enum", label, options: [{value, label}]}`) so a client
renders it generically. `empty_state` is `{has_probe_predictions,
has_item_scores, has_imported_labels}`. The `regions` tab is offered only
while a region profile is active.

| Method | Path | Notes |
|---|---|---|
| GET | `/review/tabs` | `ReviewTabsResponse` |
| GET | `/review/{tab}` | `items[]` (item plus `reason`), `total`, `page`, `page_size`, `sort_applied`, `sort_fallback_reason`, `empty_reason` |
| GET | `/review/{tab}/locate` | where one item sits in a tab. `crop_id` required. Same filters and `sort` as the queue plus `page_size` |
| GET | `/review/new_class_proposals/summary` | `size`, `samples`. Counts and terms behind the `new_class_proposals` tab |
| POST | `/review/new_class_proposals/resolve` | bulk-resolve every item proposing one name. `dry_run` query |
| GET | `/review/raw_label_clusters` | `size`, `samples_per_cluster`. Groups of unmatched raw VLM answers |
| GET | `/review/unmatched_terms` | `size`. Most common unmatched answers |

Tabs:

| Tab | Selects |
|---|---|
| `all` | the unified queue, most uncertain first. Includes `combine_conflict` items |
| `mismatches` | the VLM's reply did not match any registry class. `reason` says why: no match, a named registry class at low confidence that was not applied, or no class answer |
| `vlm_low_conf` | a VLM-sourced label whose `vlm_confidence` is `medium` or `low` |
| `outliers` | far from the cluster centroid |
| `uncertainty` | high probe entropy |
| `model_disagreements` | validated items where the probe disagrees with the human label |
| `regions` | items with an accepted but unvalidated box, plus verifier-rejected items (see `region_status`) |
| `primary_low_conf`, `classifier_blind_spots` | low-confidence and blind-spot items of the primary subject. `max_rank` defaults to 2 |
| `new_class_proposals` | items flagged `needs_new_class` by a human, or `class_source: vlm_new_class_pending`. Excludes items whose last VLM attempt had no answer and items that already have a class |
| `imported` | validated labels a dataset import wrote (`import_id`, `dataset_split`) |

Common filters: `include_test`, `max_rank`, `min_blur_ratio`,
`min_mistakenness`, `hide_near_duplicates`, `class_id`, `source`,
`conf_min`, `conf_max` (`400` if `conf_min` is above `conf_max`),
`combine_conflict`, `on_negative_frame`, `sort`, `page`, `page_size` (max
200). Tab-only filters: `text` and `region_status` on `regions`; `import_id`
and `dataset_split` on `imported`.

`region_status` on the `regions` tab:

| Value | Items served |
|---|---|
| `all` (default) | an accepted but unvalidated box, plus `verify_rejected` items with at least one rejected box |
| `detected` | an accepted box only |
| `verify_rejected` | items with only rejected boxes |
| `has_rejected_box` | any rejected box, whatever the item status (`region_rejected_count >= 1`) |

`false_positive` and `no_region_visible` items never appear in the default
mode. Any other value is `400`. A rejected candidate's `reason` names the
rejection (`needs human review: ...` for `needs_human` reasons, `rejected:
...` otherwise).

Sort. An explicit `sort` is honored even when its field has no coverage.
When omitted or `default`, the tab's own default applies (a deployment `sort`
default from `PUT /settings` applies only to a tab without one). If that
default orders by a field no item has, the queue falls back to the tab's next
covered sort, ending at `recent`. `sort_applied` names the sort that ran and
`sort_fallback_reason` says why the default was skipped. `PUT /settings`
refuses a `sort` default whose field has no coverage. A zero-result page
carries `empty_reason`, worded from live index state.

`GET /review/{tab}/locate` returns `{crop_id, in_queue, rank, page,
page_size, total, reason, sort_applied, sort_fallback_reason}`. `rank` is
0-based and `page` is 1-based. Outside the queue both are `null` and `reason`
is `not_found` or `filtered_out`. It counts items sorting before the crop,
so it works at any queue depth. Use it for deep links.

`GET /review/new_class_proposals/summary` returns `{total_pending,
without_term, top_terms, flagged_terms, term_rules}`. The summary, the queue
and the resolve route share one selection, so counts agree. Each term is
`{label, count, sample_crop_ids, flag, class_id}`. `flag` is:

| `flag` | Rule | Action |
|---|---|---|
| `null` | worth creating | create a class |
| `existing_class` | the normalized name is an active class (`class_id` set) | resolve with `class_id` |
| `generic_parent` | the whole name is in `OP_NEW_CLASS_GENERIC_TERMS`, or is a registry group name | assign a specific class |
| `non_object` | the name or one of its `_` tokens matches `OP_NEW_CLASS_NON_OBJECT_TERMS` (supports `prefix_*` and `*_suffix`) | discard or exclude |

Both term lists are empty by default. `term_rules` serves the active rules.

`POST /review/new_class_proposals/resolve` takes `{label, class_id | create:
{class_name, group, notes}, label_source}`. Exactly one of `class_id` and
`create` (`422` otherwise). An unknown `class_id` is `400`; a duplicate
`create.class_name` is `409` with no item writes. It matches every pending
item proposing `label` (never a validated, review-dismissed, excluded or
test-holdout item), writes each item independently and re-checks at write
time. Response: `{class_id, class_name, created, label, matched, matched_ids,
updated, updated_ids, conflicts[], skipped}`. `dry_run=true` reports the match
without writing or creating anything. Writes are undoable with
`POST /crops/label/undo_batch` on `updated_ids`. A resolve over more pending
items than its safety cap is `422`.

## Classes

A class has an integer id and a name inside one project. Names are unique
among active classes of a project. See [Class identity](#class-identity).

| Method | Path | Body | Response | Errors |
|---|---|---|---|---|
| GET | `/classes` | | `ClassListResponse`: `classes[]`, `thresholds`, `reserved_hotkeys` | `503` |
| POST | `/classes` | `ClassCreateRequest`: `name`, `group`, `notes`, `hotkey_letter` | `201` registry entry | `422` bad name, `409` name or hotkey taken, `400`/`422` hotkey rules |
| GET | `/classes/{class_id}` | | `ClassEntry` | `404` |
| PUT | `/classes/{class_id}` | `ClassUpdateRequest`: `name`, `group`, `hotkey_letter` | registry entry | `404`, `409`, `422` |
| POST | `/classes/merge` | `{source_id, target_id}`, query `dry_run` | merge report | `400`, `409` |
| POST | `/classes/{class_id}/deprecate` | | registry entry | `404`, `409 class_still_referenced` |
| POST | `/classes/{class_id}/restore` | | registry entry | `404`, `409`, `409 class_merged` |
| POST | `/classes/sync_to_opensearch` | | sync report | |
| GET | `/class_sources` | | `{class_sources[]}` | |

- `ClassEntry`: `class_id`, `class_name`, `group`, `sample_count`,
  `validated_count`, `cluster_size`, `deprecated`, `merged_into`,
  `hotkey_letter`, `adequacy`, `kind` (`item` or `region`), `trainable`,
  `trainable_gap`, `added_at`. A class named like the active profile's
  `region_class_name` has `kind: region`. The active region profile's class
  is seeded into the registry at project creation and API start, and it never
  labels a whole item.
- Names match `^[a-z0-9_]+$` on create and rename (`422`).
- Hotkeys, on create and update: one character (`400`), not reserved
  (`422 hotkey_reserved`), not bound to another active class (`409
  hotkey_taken`). `""` on update clears. `reserved_hotkeys` are the single
  keys bound in contexts where class hotkeys are live in the project's
  effective keymap. A create that fails a hotkey rule writes nothing.
- Merge relabels every non-holdout item of the source with `class_source:
  class_merge` and keeps `class_validated`. `dry_run=true` returns
  `{dry_run, source_id, target_id, would_relabel, validations_carried_over,
  holdout_blocking, blocked}` and writes nothing. `blocked` means the real
  merge would be `409` because of frozen test-holdout items. A self-merge or
  an unknown id is `400`. The merge is recorded in `class_id_history`
  (`writer: class_merge`). A merged class cannot be restored
  (`409 class_merged`); relabel manually.
- Deprecate retires a class that has no items and no confirmed labels. While
  items reference it, it is `409 class_still_referenced` with
  `item_count` and `confirmed_label_count`. It is idempotent and clears the
  hotkey.
- Restore is `409` if an active class already holds the name.

## Clusters

Cluster ids partition into ranges:

| `cluster_id` | `cluster_kind` | Meaning |
|---|---|---|
| `< 0` | `unassigned` | not clustered, noise, or excluded (`-2`) |
| `0` to the offset | `class` | equals the class id, for labeled items |
| `>= cluster_id_offset` | `candidate` | a residual cluster that needs a class |

| Method | Path | Notes |
|---|---|---|
| GET | `/clusters` | cluster cards. Query: `per_cluster`, `max_clusters`, `kind` (`class`, `candidate`, `all`), `class_id`, `cluster_id`, `max_rank`, `min_blur_ratio`, `class_source`, `offset`, `limit` |
| GET | `/clusters/representatives` | `per_cluster`, `max_clusters`, `class_id`, `offset` |
| POST | `/clusters/auto_promote` | `min_purity` (default 0.85), `min_members` (default 4), `dry_run`. Validates members of pure clusters |
| POST | `/clusters/refine/{cluster_id}` | AHC refine of one cluster. `distance_threshold`, `max_members`. Writes `cluster_subid` |
| POST | `/cluster/umap/rebuild` | refit the UMAP reducer on the residual pool and re-cluster it. Returns the clustering summary |

Cards. `labelled_count` counts members with any `class_name`. `dominant_count`
and `label_purity` describe the top class among them. For a candidate,
`dominant_class_name` is set only for a unique top class with at least 3
members and half of the labelled ones, else `null`, and `dominant_class_id`
is `null`.

`purity` is geometric and independent of labels: the share of members whose
nearest cluster centroid is their own cluster. `purity_n` is how many members
it covers and `purity_basis` is `nearest_centroid`; with `purity_n` of `0`
both `purity` and `purity_tier` are `null`. `purity_tier` is `pure`, `mixed`
or `noisy`. `promotable` is the auto-promote gate on labels (enough members,
enough of them labelled, `label_purity` at or above `pure_min`); never true
for a class cluster. The response serves `purity_thresholds` (`pure_min`
0.85, `mixed_min` 0.6, `promote_min_members` 4, `promote_min_labelled_share`
0.5) and `core_similarity_min` (0.75).

Representatives are paged. Every card is returned with `size`, `purity` and
the rest, but `representatives` is filled only for cards in the
`[offset, offset + limit)` window of the kind-filtered, member-count-descending
list (`limit` default 50, max 500). Other cards carry `representatives: []`.
`per_cluster=0` skips representatives. `GET /clusters/representatives`
pages the same way by `offset` and `max_clusters`.

Clustering runs inside the auto-label pipeline. Setting
`recluster_unvalidated=true` broadens the residual pool so candidate clusters
can fuse. The default pool is fresh and class-bucketed unvalidated items.

## VLM labeling and the auto-label pipeline

| Method | Path | Body or query | Response |
|---|---|---|---|
| POST | `/vlm/label_batch` | `{crop_ids}` (at most 64), `vlm`, `acknowledge_external` | label results |
| POST | `/vlm/verify_regions` | `{crop_ids}` (at most 64) | per-crop verification |
| POST | `/vlm/verify_region_batch` | `{items: [{crop_id, region_image_b64, candidate_text}]}` (at most 64) | `{results[]}`: `crop_id`, `is_region`, `confidence`, `reason`, `candidate_text` |
| POST | `/vlm/region_visible_batch` | `{items: [{crop_id, image_b64}]}` (at most 64) | `{visible: {crop_id: bool}}` |
| POST | `/vlm/label_cluster/{cluster_id}` | `prompt_pack`, `vlm`, `acknowledge_external` | the auto-label job state |
| POST | `/pipeline/auto_label` | query parameters | runs the pipeline in the request and returns each stage's counts. Idempotent |
| POST | `/pipeline/auto_label/start` | query parameters | starts a background job and returns at once; poll the status routes. One job at a time (`409` otherwise) |
| GET | `/pipeline/auto_label/status` | | the current job |
| GET | `/pipeline/auto_label/status/{job_id}` | | a job by id; `404` for an unknown id |
| POST | `/pipeline/auto_label/cancel` | | |
| GET | `/pipeline/events` | | SSE stream for a dashboard: `snapshot` on connect, then `state` on job changes and `stats` every 15 s |

- The VLM labeling routes honor the lock rule: an item whose class is locked
  is never overwritten. A project with no classes is `409 no_classes`. A call
  over 64 ids is `400`.
- A reply that resolves to a registry class also sets `cluster_id = class_id`
  and clears `cluster_subid`, unless the item is excluded. Undo restores the
  placement.
- A crop the VLM gave no usable verdict for is omitted from a batch
  response, or left untouched by a single-crop call. It is never written as a
  reject.
- `/pipeline/auto_label` and `/start` take: `train_clusters`,
  `promote_min_purity`, `promote_min_members`, `vlm_batch_size`,
  `vlm_concurrency`, `max_vlm_crops`, `classifier_confidence_skip_vlm`,
  `clustering_method`, `run_vlm`, `recluster_unvalidated`, `reassign_only`,
  `run_auto_promote`, `gate_max_rank`, `gate_min_blur_ratio`, `n_clusters`,
  `class_id`, `cluster_id`, `prompt_pack` (`<name>` or `<name>@<revision>`),
  `vlm` and `acknowledge_external`. A `detection_profile` parameter is `422`:
  no stage runs region detection. Region detection runs in the detection
  worker on the active profile.
- Job state: `job_id`, `status` (`queued`, `running`, `completed`, `failed`,
  `cancelled`, `interrupted`), `stage`, `processed`, `total`, `started_at`,
  `finished_at`, `error`, `result`, `args`, backend and VRAM telemetry,
  `stage_durations`, `eta_seconds`, `elapsed_seconds`. The newest 50 jobs are
  kept. A start while a job runs is `409`.
- `/vlm/label_cluster/{cluster_id}` runs only the VLM stage over every
  unvalidated, non-holdout, non-excluded member of one cluster. It does no
  re-clustering, no auto-promote and applies no cap. The index-wide stages are
  skipped even if requested, and writes touch only the selected members.
  `?cluster_id=` on `/pipeline/auto_label[/start]` gives the same scope.

## Ingest

| Method | Path | Body | Response |
|---|---|---|---|
| POST | `/ingest/image` | `IngestImageRequest`: `path`, `source` (`extra='forbid'`) | `IngestImageResponse` |
| POST | `/ingest/batch` | `IngestBatchRequest`: `items: [{path, source}]` (required, non-empty, `extra='forbid'`) | `BatchIngestResponse` |
| POST | `/ingest/upload` | multipart: `images` (files), `image_paths` (JSON list of identifiers, optional), `source`, `run_id` | `BatchIngestResponse` |
| POST | `/ingest/path_lookup` | `{image_paths}` (at most 10,000) | `{known_paths: {path: image_id}}` |
| GET | `/ingest/config` | | limits |
| GET | `/ingest/status` | query `run_id` | `{total, by_source[], by_day[]}` |
| GET | `/ingest/region_drain` | | drain state |

- `IngestImageResponse`: `status` (`success`, `duplicate`, `failed`),
  `image_id`, `image_path`, `source_identifier`, `imohash`, `n_crops`,
  `n_regions`, `error`, `error_kind`, `secondary_detector_error` (set when a
  configured secondary detector call failed; the image still ingests on the
  primary detector's output).
- `error_kind` is one of `empty`, `unservable_path`, `unsupported_type`,
  `decode_failed`, `detector_infer`, `bulk_index`. It is present with `error`
  when `status` is `failed`.
- `BatchIngestResponse`: `status` (`success`, `partial`, `error`),
  `summary` (`successful`, `duplicates`, `failed`, `crops_indexed`,
  `secondary_detector_failures`) and `results[]`.
- Duplicates are detected by content hash (`imohash`) against the index and
  within a batch. A byte-identical file later in the same request reports
  `duplicate` with the representative's `image_id`, and exactly one item set
  is created.
- Paths for `/ingest/image` and `/ingest/batch` must be servable: under a
  configured source root. An unservable path is `422` (single) or a failed
  row with `error_kind: unservable_path` (batch). An unreadable file is
  `404` (single).
- `/ingest/upload` stores the bytes content-addressed under the project's
  upload root: `<upload_root>/<imohash[:2]>/<imohash><ext>`, written
  atomically, stored once per content. `image_path` is the stored, servable
  path. The client's identifier is `source_identifier`. `path_lookup` matches
  either field and keys its result by the one that matched. `run_id` is
  recorded as `ingest_run_id` and scopes `GET /ingest/status?run_id=`.
- `GET /ingest/config`: `upload` (`enabled`, `max_images_per_request`,
  `max_bytes_per_request`, `accepted_extensions`, `persists_bytes`), `batch`
  (`enabled`, `max_items`, `source_roots`), `region_drain` (`poll_interval_s`,
  `stable_polls`). Limits: `OP_UPLOAD_MAX_IMAGES_PER_REQUEST`,
  `OP_UPLOAD_MAX_BYTES_PER_REQUEST`, `OP_UPLOAD_ACCEPTED_EXTENSIONS`,
  `OP_BATCH_MAX_ITEMS_PER_REQUEST`. Over a limit is `413`; an extension
  outside the accepted list fails that item with `error_kind:
  unsupported_type`.
- `GET /ingest/region_drain`: `pending_detection`, `pending_verification`,
  `total_unfinished`, `drained`, `stable_for_s`, `observed_at`,
  `region_dependencies[]` (`role`, `model`, `ready`, `unavailable_since`,
  `detail`) and `stall_reason`. `drained` is `true` only after
  `total_unfinished` has read `0` for `stable_polls`
  (`OP_REGION_DRAIN_STABLE_POLLS`, default 3) consecutive polls.
  `region_dependencies` is checked against Triton's repository index and is
  empty with no region profile. `stall_reason` is `null` when nothing is
  pending or every dependency is ready. Items stay `pending_detection` while a
  dependency is down; the worker never writes a terminal status on an
  infrastructure failure.
- `GET /ingest/region_drain` and `GET /ingest/status` answer `503` on a
  backend outage.
- Ingest answers `503` until the encoder has loaded.

## Dataset import

Import an already-labeled dataset (YOLO, COCO or an OpenProcessor export)
into a project. The importer is chunked, persisted, resumable and runs on the
shared jobs volume. Labels are written through the same single class writer
that human labels use, so the lock rule holds. A human edit made between
planning and writing wins.

| Method | Path | Body | Response | Errors |
|---|---|---|---|---|
| GET | `/datasets/formats` | | `DatasetFormatsResponse`: `formats[]`, `issues[]` (the issue catalog), `mapping_actions[]`, `match_kinds[]`, `parents_modes[]`, `processing_modes[]`, `trust_levels[]`, `status_labels`, `upload_limits` | |
| POST | `/datasets/uploads` | multipart `file` (zip or tar) | `201 {upload_id, dataset_path, bytes, files}` | `413 upload_too_large`, `422 archive_invalid` |
| POST | `/datasets/preview` | `DatasetPreviewRequest` | `DatasetPreview` | `422 dataset_path_not_allowed`, `422 format_undetected` |
| POST | `/datasets/imports` | `DatasetImportRequest` | `202 DatasetImportJob` (`200` with `reused: true` for a completed repeat) | `422 class_mapping_incomplete`, `422 class_mapping_invalid`, `422 import_blocked`, `409 import_busy`, `409 import_resumable`, `409 dataset_changed` |
| GET | `/datasets/imports` | `page`, `page_size`, `status` | `DatasetImportList` | |
| GET | `/datasets/imports/{import_id}` | | `DatasetImportJob` | `404 import_not_found` |
| GET | `/datasets/imports/{import_id}/issues` | `code`, `page`, `page_size` | `DatasetIssuePage` | |
| GET | `/datasets/imports/{import_id}/entries` | `split`, `label_state`, `status`, `page`, `page_size` | `DatasetImportEntryPage` | |
| POST | `/datasets/imports/{import_id}/cancel` | | `DatasetImportJob` | |
| POST | `/datasets/imports/{import_id}/resume` | | `202 DatasetImportJob` | `409 import_not_resumable` (only `interrupted`, `failed` or `cancelled`), `409 import_busy`, `409 dataset_changed` |
| POST | `/datasets/imports/{import_id}/undo` | `DatasetUndoRequest` | `200 DatasetUndoReportWire` for a dry run, else `202 DatasetImportJob` | `409 import_not_undoable`, `409 import_busy` |

`DatasetSource`: `path`, `format` (`auto`, `yolo`, `coco`,
`openprocessor_export`) and `coco_annotations[]` (`{path, images_dir,
split}`). Auto-detection tries an OpenProcessor export, then YOLO (a data
YAML), then COCO (`annotations/*.json` with no `labels/` directory, so
`images/` next to `annotations/` is COCO), then YOLO (an `images/` directory,
or a child with one). The path must lie under a source root, the project's
upload root or the project's export root, after symlinks are resolved
(`422 dataset_path_not_allowed`). References inside a dataset must stay
inside it. Archive uploads accept regular files and directories only and
cap member count, bytes written and compression ratio.

`DatasetImportOptions`:

| Option | Values | Effect |
|---|---|---|
| `processing` | `none` (default), `propose` | `propose` runs the detector on the imported images for proposals |
| `label_trust` | `validated` (default), `suggestion` | `validated` writes locked labels. `suggestion` writes unvalidated labels (and `proposed` boxes) that the machine pipeline may still replace |
| `parents` | `auto` (default), `labels`, `detect` | where the parent items of region labels come from |
| `freeze_test_split` | bool or unset | freeze the imported `test` split as the test holdout |
| `missing_label` | `unlabeled` (default), `negative` | how a frame with an empty label file is treated |
| `region_negatives` | bool, default true | |
| `region_containment` | 0.5 to 1.0, default 0.9 | how much of a region box must lie inside a parent |
| `name`, `source_tag` | strings | labels for the import and the `source` of created items |
| `force` | bool | bypass issues marked bypassable |

### Class mapping

Every dataset class that has boxes needs a decision. The mapping is a list of
`ClassMappingEntry`:

| Field | Meaning |
|---|---|
| `dataset_class` | the class name in the dataset |
| `action` | `map` to an existing class, `create` a class, `skip` the boxes, or `region` to treat the boxes as sub-regions of the active region profile |
| `class_id` | for `map` |
| `new_class_name`, `new_class_group` | for `create` (and `map` by name) |

The preview serves a `suggestion` per dataset class (`action`, `class_id`,
`class_name`, `match`: exact, case-insensitive or region). With
`accept_suggestions: true` the server takes exact, case-insensitive and
region matches. Matching is always by name, never by index. A class with boxes
and no decision is `422 class_mapping_incomplete` with `unmapped[]`. An
invalid row is `422 class_mapping_invalid`. A `region` action needs an active
region profile. Without one the preview reports the blocking issue
`region_profile_required` and a start is `422 import_blocked`.

### Preview

`DatasetPreview` writes nothing: `format`, `root`, `source_sha`,
`import_key`, `splits[]` (per split: `images`, `boxes`, `labeled`,
`unlabeled`, `negatives`), `classes[]` (`dataset_class`, `boxes`, `images`,
`suggestion`, `resolved`, `merged_from`), `totals`, `estimate`, `issues[]`,
`blocking`, `force_allowed`, `region` and `op_export`.
Issues carry `code`, `severity` (`error`, `warning`, `info`), `blocking`,
`bypassable`, `count` and `samples[]`. The catalog of codes is in
`DatasetImportJob`/`DatasetIssueWire.code` and `GET /datasets/formats`.

### Job

`DatasetImportJob`: `import_id`, `import_key`, `name`, `project`, `status`
(`queued`, `running`, `paused_backpressure`, `completed`,
`completed_with_errors`, `failed`, `cancelled`, `interrupted`, `undoing`,
`undone`), `progress` (`images_total`, `images_done`, `images_failed`,
`chunks_done`, `chunks_total`, `images_per_s`, `eta_s`), `report`, `issues_summary[]`,
`mapping[]` (resolved, with `created` for classes the import made),
`options`, `source`, `next_steps[]`, `waiting_for`, `poll_after_s`,
`reused`, timestamps and `error`.

`report` counts `images_created`, `images_reused`, `images_skipped`,
`images_failed`, `items_created`, `items_updated`, `items_noop`,
`items_reconciled_removed`, `labels_written`, `boxes_written`,
`label_conflicts_locked` (labels a human lock refused), `negatives`,
`unlabeled`, `parents_detected`, `standalone_regions`, `proposals_created`,
`proposals_merged`, `holdout_frozen` and `disagreements`.

- `import_key` hashes the project, the source content, the name-based
  mapping and the write-affecting options. Repeating a request is idempotent:
  a completed repeat answers `200` with `reused: true`. A prior interrupted
  import of the same dataset is `409 import_resumable`. `expected_import_key`
  (from the preview) guards against a changed dataset (`409 dataset_changed`).
- One import runs per project at a time (`409 import_busy`).
- When the region worker's backlog exceeds `OP_DATASET_IMPORT_MAX_PENDING`
  the job pauses (`paused_backpressure`).
- Importing a different version of a dataset over images an earlier import
  labeled removes items the new version no longer has, unless a human or a
  holdout freeze touched them, a box the dataset still has matches them, or
  the frame has no label file. The whole document is kept in the new
  import's ledger row, and undoing that import reinstates it.
- Undo (`dry_run` default `true`) restores each class snapshot or region edit
  history, deletes items the import created, optionally removes created
  images (`remove_images`) and deprecates created classes
  (`deprecate_created_classes`). Items a human edited or that another import
  shares are kept. A second undo reports zeros.
- Imported items carry `import_ids`, `dataset_split`, `imported_at`,
  `proposed_by_import`, `on_negative_frame` and `import_standalone_region`.
  Browse them with the `import_id`, `dataset_split`, `on_negative_frame` and
  `proposed_by_import` filters of `GET /crops` or the `imported` review tab.

## Reprocess

One route re-runs pipeline stages. It replaces every ad hoc requeue, clear
and retry path.

| Method | Path | Body | Response | Errors |
|---|---|---|---|---|
| POST | `/reprocess` | `ReprocessRequest` | `ReprocessWireResponse` | `422 reprocess_targets_invalid`, `409 reprocess_busy` |
| POST | `/images/{image_id}/reprocess` | `ReprocessOneRequest` | `ReprocessWireResponse` with the image's items | `404 image_not_found` |
| POST | `/crops/{crop_id}/reprocess` | `ReprocessOneRequest` | `ReprocessWireResponse` with the item | `404 not_found` |
| GET | `/reprocess/jobs/{job_id}` | | `ReprocessJobInfo` | `404 not_found` |
| POST | `/reprocess/jobs/{job_id}/cancel` | | `ReprocessJobInfo` | `404 not_found` |

- `ReprocessRequest`: `targets` (exactly one of `image_ids`, `crop_ids`,
  `filter`; at most 5000 ids), `scopes` (`detect`, `region`, `vlm`, `embed`;
  at least one), `region_mode` (`redetect` default, `reverify`) and
  `dry_run` (**default `true`**). The per-image and per-item routes drop
  `targets` (the path is the target) and apply by default (`dry_run`
  `false`).
- `filter` (`ReprocessFilter`): `region_status[]`, `detector[]`, `reason[]`,
  `profile_not`, `profile_revision_below`, `include_detected` (only valid with
  a profile selector), `missing_status`, `missing_provenance`, `import_id`,
  `source`, `class_id`, `dataset_split`. An empty filter is refused.
- Response: `dry_run`, `scopes[]` (`scope`, `selected`, `locked_skipped`,
  `queued`, `not_found`, `failed`, `breakdown[]`, `detail`), `items[]` (not on
  a dry run) and `job` for work that runs in the background.
- The lock rule applies: locked items and boxes are never written and are
  counted as `locked_skipped`.
- A `detect` or `embed` run over more than `OP_REPROCESS_SYNC_MAX` images
  (default 20) runs as a file-backed job. `job.status` is `queued`, `running`,
  `completed`, `completed_with_errors`, `cancelled` or `failed`. One job runs
  at a time (`409 reprocess_busy`). Cancel stops after the current chunk.
- `suggested_reprocess` in an activation impact is a ready `ReprocessRequest`.

## Export

| Method | Path | Body or query | Response |
|---|---|---|---|
| POST | `/export/yolo` | `ExportYoloRequest` | export summary |
| POST | `/export/single_class` | `ExportSingleClassRequest` | export summary |
| GET | `/export/status` | | `ExportStatusResponse` |
| GET | `/export/single_class/status` | `profile_name` (default `single_class`) | status of the last run for that profile |
| GET | `/export/datasets` | `kind` (`yolo`, `single_class`), `profile_name` | versions on disk |
| GET | `/export/registry/{artifact}` | | a file download |

`ExportYoloRequest`: `export_dir`, `version_tag`, `seed` (42),
`max_images`, `dedup_threshold`, `require_fully_labeled_images` (false),
`include_negative_frames` (true), `split_mode` (`keep_imported` default,
`recompute`).
`ExportSingleClassRequest`: `export_dir`, `version_tag`, `class_ids`,
`box_source` (`item` or `region`), `region_class_name`, `profile_name`,
`seed`, `skip_test_split`, `empty_bg_ratio`, `max_positive_images`,
`dedup_threshold`, `image_mode` (`whole_frame` or `item_crop`),
`img_max_side`, `copy_images`, `split_mode`.

Both exports refuse with `422 nothing to export: <reason>` when nothing is
exportable, write nothing and leave `current` on the previous export. Every
manifest records `items_index: {index, uuid, created_at}`.

### Multi-class layout

One image file and one label file per source image. Validated items are
grouped by `image_id`. Each image is written once as
`images/<split>/<image_id>.<ext>` next to `labels/<split>/<image_id>.txt`
with one `cls cx cy w h` line per validated, non-excluded, non-dismissed
object. `cls` is the dense `export_id`. The box comes from the item's
`bbox_norm`, normalized to the full source frame and clamped to `[0, 1]`.
Objects are ordered by item id so re-runs are byte-identical. An item with no
`image_id`, usable box or class is counted in `skipped_items`.

- Partial frames. An exported image can hold unlabeled objects (not
  validated, validated on a class with no dense id, or without a usable box).
  They stay in the pixels. By default the image is exported and the manifest
  records `unlabeled_items_on_exported_images` and
  `images_with_unlabeled_items`, and training preflight warns. With
  `require_fully_labeled_images: true` such images are left out
  (`images_dropped_not_fully_labeled`); if none remains the export is `422`.
- Negative frames. A frame an import marked a reviewed negative is written as
  an empty label file when `include_negative_frames` is true and the frame's
  `negative_for` covers every class that has objects in the export. The
  manifest records `negative_images` and `negative_frames_skipped_partial`.
- Order of operations: the partial-frame policy, then `dedup_threshold`
  (near-duplicate images collapse; a frozen-holdout image is preferred as
  survivor), then `max_images` (an even round-robin over each image's rarest
  class).
- Splits. The group is the source image (`group_key: image_id`). An image
  with a frozen `test_holdout` item goes to `test` with all its objects.
  With `split_mode: keep_imported`, the split a dataset import filed each
  frame under is kept. With `recompute`, it is ignored. Remaining groups count
  toward their most common class and are ordered within a class by
  `sha256(seed:class:group)`. A class with a frozen holdout item uses it as
  its test set and splits the rest 0.8 train to 0.1 val; a class without one
  splits 0.8, 0.1, 0.1. Every split with a positive ratio gets one group
  before any gets a second. The manifest records `split_mode` and how many
  groups were pinned or overridden.
- Manifest counts: `image_count`, `object_count`, `split_counts` (images per
  split), `split_object_counts`, `class_split_counts` (rows with `class_id`,
  `export_id`, `class_name`, `train`, `val`, `test`), `class_count` (the
  registry size written as `nc`), `classes_with_objects`, `skipped_items`,
  `dedup`, `dataset_sha`.
- `dataset_sha` hashes the on-disk content: for every `labels/<split>/*.txt`
  sorted by relative path, the path and the sha256 of the bytes, then the
  ordered `names:` list, so a class rename with no id change still changes
  it. The multi-class export records the full 64-hex digest and the
  single-class export the first 16.
- `POST /export/single_class` builds a narrowed dataset for one class or a
  subset. It adds `frozen_test_sha`, which hashes the identity of the test
  split (which frames, not their content). Each `profile_name` has its own
  output root and its own `current` link. `box_source: region` exports the
  region boxes of the active profile, with `image_mode: item_crop` cropping
  to the parent item. The region stratum key is the first accepted (positive)
  or `false_positive` (hard negative) box's cluster.
- `GET /export/status` serves the `current` manifest: `status` (`idle`,
  `unknown`, `success`), `path`, `export_dir`, `last_run`, `version_tag`,
  `dataset_sha`, `seed`, `group_key`, `image_count`, `object_count`,
  `class_count`, `classes_with_objects`, the split and class counts and the
  unlabeled-frame counters. A field the manifest does not record is `null`.
  `idle` nulls everything; `unknown` (manifest unreadable) sets only the
  path fields.
- `GET /export/datasets` rows: `kind`, `profile_name`, `export_dir`,
  `version_tag`, `image_count`, `object_count`, `split_counts`,
  `dataset_sha`, `exported_at`, `class_count`, `is_current`.
- `GET /export/registry/{artifact}` serves a frozen artifact of the current
  export as a file: `class_registry.json`, `data.yaml`, `manifest.json` or
  `label_stats.json`. Any other name is `404` before the filesystem is
  touched. It is scoped to the multi-class export.

Training preflight adds export checks (`export_readiness.py`):

| Check | `block` when |
|---|---|
| `export_not_empty` | the manifest has 0 images |
| `export_splits_nonempty` | train or val has 0 images |
| `export_class_split_coverage` | a trained class has fewer than `min_train_per_class` train or `min_val_per_class` val objects. Not applicable to a single-class export |
| `export_unlabeled_objects` | never; `warn` when unlabeled objects remain on exported images |
| `export_generation` | the items index was rebuilt since the export (its `items_index.uuid` differs) |

Each is `unknown` when the manifest lacks the data.

## Training, promote and models

Training runs in a separate trainer container. The API writes a job
(`job.json`), the trainer writes `status.json`, and the API serves both.
Training is project-scoped: the dataset is the project's export, the run is
tagged with the project, and a promoted model belongs to the project that
trained it.

| Method | Path | Body or query | Response |
|---|---|---|---|
| POST | `/train/preflight` | `TrainJobSpec` | `PreflightReport`: `blocked`, `checks[]` (`name`, `severity` of `ok`, `warn`, `block`, `unknown`, `message`, `detail`), `summary`, `thresholds` |
| POST | `/train/start` | `TrainJobSpec`, query `force` | `201 {job_id, preflight}` |
| POST | `/train/start_campaign` | `TrainCampaignSpec`, query `force` | `201 {campaign_id, job_ids[]}` |
| GET | `/train/status` | | the newest run's `TrainJobStatus`, or `null` |
| GET | `/train/status/{job_id}` | | `TrainJobStatus` |
| GET | `/train/runs` | `limit`, `offset` | `{items[], total}` |
| GET | `/train/log/tail/{job_id}` | `lines` (1 to 5000, default 200) | `{job_id, lines[]}` |
| POST | `/train/cancel/{job_id}` | | `{cancelled, job_id}` |
| POST | `/train/cancel_campaign/{campaign_id}` | | `{cancelled, campaign_id}` |
| GET | `/train/profiles` | | `{profiles[]}`: name, description, default hyperparameters |
| GET | `/train/presets` | | `{class_subset_presets[]}` |
| GET | `/train/gpus` | | `TrainGpuOptionsResponse`: `options[]` (`value`, `label`, `gpu_ids`, `advisory`, `stops_containers`, `default`), `allowed_ids`, `unrestricted` |
| GET | `/train/augmentation_presets` | | `{presets[], default}` |
| POST | `/train/promote/{job_id}` | `PromoteRequest` | `PromoteResponse` |
| POST | `/train/reload_promoted` | | `{status, reloaded[], failed[]}` |
| GET | `/train/manifest/{job_id}` | | the run's lineage envelope |
| GET | `/train/artifacts/{job_id}/{name}` | | a whitelisted artifact file |

### Start

- `TrainJobSpec` fields: `dataset_export_dir` (defaults to the current export;
  it must be under the project's export root, `422 export_outside_project`,
  not bypassable by `force`), `profile` (`probe`, `nano`, `small`, `medium`,
  `large`, `xlarge`, `custom`), `model_family` (`yolo26`), `model_size`,
  `hyperparameters`, `include_classes`, `single_cls`, `augmentation`,
  `stop_when`, `auto_promote_best`, `auto_quantize_bakeoff`,
  `cuda_visible_devices`, `mlflow_experiment`, `mlflow_run_name`,
  `submitted_by`. Provenance fields (`project`, `dataset_sha`,
  `frozen_test_sha`, `registry_snapshot_path`, image revision) are filled by
  the server.
- A blocked preflight is `422 {message: "preflight blocked", preflight}`
  unless `force=true` bypasses a bypassable check. An unknown augmentation
  preset is `422` with `field: augmentation.preset` and `valid_presets`,
  before any GPU claim or job write, even with `force`. A run that is already
  active is `409`. A GPU claim that cannot be satisfied is `409`.
- `GET /train/augmentation_presets` serves the preset catalog (`id`, `label`,
  `description`, `orientation_sensitive`) and the default
  (`balanced_default`). A client renders it and never hardcodes ids.
  `orientation_sensitive` means horizontal flip is off for the whole run.
- Preflight includes the export checks listed under
  [Export](#export), the `thresholds` and the class adequacy.

### Status

`TrainJobStatus`: `job_id`, `state` (`queued`, `starting`, `running`,
`exporting`, `finished`, `failed`, `cancelled`, `skipped`, `lost`),
`current_epoch`, `total_epochs`, `epoch_time_s`, `gpu[]`, `heartbeat_at`,
`started_at`, `finished_at`, `error`, `checkpoint_path`, `campaign_id`,
`last_epoch_metric`, `best_checkpoint_metric`, `eval`, `compare`,
`mlflow_run_id`, `mlflow_experiment_id`, `mlflow_run_url`.

- `last_epoch_metric` is `{epoch, map50, map50_95}` of the true last training
  epoch. `best_checkpoint_metric` is the same shape for the best checkpoint's
  own re-validation, as one coherent row. Both are `null` for a status
  written before they existed.
- `eval` states which split every number came from:

```json
{"map50": 0.62, "map50_95": 0.41, "precision": 0.71, "recall": 0.55, "split": "test",
 "val_last": {"map50": 0.9191, "map50_95": 0.742},
 "per_class": [{"class_id": 0, "name": "widget", "precision": 0.8, "recall": 0.7,
                "f1": 0.75, "ap50": 0.79, "support": 12}],
 "confusion_matrix_url": "/curation/projects/example/train/artifacts/<job_id>/confusion_matrix.png"}
```

  `split: "test"` means the frozen test split was scored. When the test pass
  fails or the export has no test split, the four overall keys carry the
  training-time validation numbers, `per_class` is absent and
  `split: "val"`. A consumer must check `split` before reading `map50` as
  held-out performance. `val_last` is always present when a result row
  exists. `head: "end2end"` records that the NMS-free one-to-one head was
  scored.
- `confusion_matrix_url` points at `/train/artifacts/{job_id}/{name}`.
  Whitelisted names: `confusion_matrix.png`, `confusion_matrix_normalized.png`,
  `results.png`, `results.csv`, `BoxP_curve.png`, `BoxR_curve.png`,
  `BoxF1_curve.png`, `BoxPR_curve.png`. The server path never reaches the
  wire.
- `mlflow_run_url` is rebuilt from `OP_MLFLOW_PUBLIC_URL`, the run id and the
  experiment id. It is `null` when the public base is not configured. The
  internal tracking host never reaches the wire.

### Promote

`POST /train/promote/{job_id}` copies the run's ONNX export into the Triton
model repository, writes the config and `labels.txt`, and asks Triton to load
it. Request: `triton_name`, `force`, `fp16`, `input_size`, `max_batch_size`,
`overwrite`. Response: `triton_name`, `onnx_path`, `config_path`,
`labels_path`, `triton_loaded`, `class_remap_source`, `force_used`,
`gate_report`, `lineage_stamped`, `cold_start_expected_on_first_inference`.

- The model is stored as `<slug>__<triton_name>` for every project except
  `default`. `triton_name` may not contain `__` (`422`). An existing name is
  `409` unless `overwrite`.
- Errors: `404` unknown job or no ONNX export; `422` the run is not
  `finished` or `exporting`, the promote gate failed, or class identity cannot
  be proven; `502` Triton refused the load.
- A `422` body is `{detail: {message, failures[], force_allowed, override,
  thresholds}}`, each failure `{code, message, class_name?}`. `force_allowed`
  says whether `force=true` can pass that failure. `force` bypasses only the
  score thresholds and a missing class remap; it cannot invent a finished
  run.
- The gate reads `eval`: a run whose test pass failed (`split: "val"`, no
  `per_class`) fails the per-class check outright.
- Class identity. A subset or single-class run's `class_remap.json` is carried
  from the trainer into the checkpoint's `weights/` directory.
  Promote resolves it (manifest `lineage.class_remap` first, then the weights
  directory file) before writing `labels.txt`. With no resolvable remap,
  a subset or single-class run is `422 class_remap_missing`, and a
  full-class run whose registry has a gap or deprecated class is
  `422 class_remap_missing_full_class`. See [Class identity](#class-identity).
  `force` bypasses both, logged distinctly.

### Models

| Method | Path | Body or query | Response | Errors |
|---|---|---|---|---|
| GET | `/models/status` | `include_other_projects` | `{models[]}`: Triton models and the segmenter and VLM services. Each Triton entry carries `project`, `shared`, `owned`, `sharing_revision`, `class_mapping`, `optional`. VLM rows are one per registered endpoint with `kind: vlm`, `active` and `active_in` (the bound project only) | |
| GET | `/models/{model_name}/class_mapping` | | `{model, model_project, project, entries[], unmapped[], not_covered[], labels}` | `404 model_not_found` |
| PUT | `/models/{model_name}/sharing` | `{shared, expected_revision}`, query `force` | `{name, project, shared, revision, used_by[]}` | `404 model_not_found`, `409 revision_conflict`, `409 in_use` (another project's active detection profile uses the model; typed `ModelSharingConflictResponse`: `detail.projects[]` slugs and `detail.used_by[]` rows `{project, profile}`; `force` bypasses), `503 config_store_unavailable` (typed `ModelSharingUnavailableResponse`; `force` bypasses) |
| DELETE | `/models/{model_name}` | query `force` | `{triton_name, triton_unloaded, directory_removed, forced, warning}` | `404`, `400`, `403`, `409` |

- A model's classes reach another project by name only. `class_mapping`
  matches them onto the bound project's registry; an entry has `match` of
  `exact`, `case_insensitive` or `none`.
- Only the owning project may share a model or delete it (`404` for anyone
  else). Sharing is opt-in.
- `status: not_installed` (instead of `not_ready`) marks an optional detector
  that is absent from Triton's repository index.
- `DELETE /models/{model_name}`: an external-service model is `400`. A
  region-detector or OCR model of the active profile is `403` always. Other
  core pipeline models need `force=true` (`409` without it).

## Bake-off

Model comparison on frozen test sets. Models are in
`src/routers/curation/_bakeoff_models.py`; every route has a response model.
Evaluator-written files are validated on read. A file that is not schema
version 2 is `409 bake-off result <file> has an unsupported schema`.

| Method | Path | Notes |
|---|---|---|
| GET | `/bakeoff/eval_datasets` | `source` (`export` or `external`) -> `{datasets[], count}`. A dataset has `id` (`export:<path>` or `external:<group>/<name>`), `dataset_kind` (`multi_class`, `single_class`, `external`), `nc`, `classes[]`, counts, `frozen_test_sha`, `test_label_sha`, `sha_source`, `dataset_sha`, `is_current` |
| GET | `/bakeoff/trained_models` | `dataset_id`, `limit` -> `{models[], count}`. A model has `run_id`, `display_name`, `model_family`, `model_size`, `imgsz`, `checkpoint_path`, `class_names`, `single_cls`, `trainer_map50`, `trainer_map50_split` and, with `dataset_id`, `for_dataset` (`same_export`, `same_frozen_test`, `n_classes_mapped`, `train_test_overlap`) |
| GET | `/bakeoff/profiles` | `{profiles[], count, default_profile, default_error}` |
| GET | `/bakeoff/baseline_models` | `profile` -> `{baselines[], count}` |
| POST | `/bakeoff/run` | `BakeoffRunRequest` -> `BakeoffRunAccepted` |
| GET | `/bakeoff/runs` | `{runs[]}` |
| GET | `/bakeoff/status/{job_id}` | `BakeoffStatus` (`state`: `queued`, `running`, `done`, `error`; `progress`, `completed[]`, `failed[]`) |
| GET | `/bakeoff/results/{job_id}` | `dataset_id` optional -> `BakeoffComparison` |
| GET | `/bakeoff/matrix/{job_id}` | `BakeoffMatrix` (`cells[model][dataset]`, `best[dataset][metric]` with every tied winner) |

`BakeoffRunRequest` (`extra='forbid'`): `job_id`, `profile`,
`datasets: [{id}]` (an `id` may be `run:<job_id>`), `models[]` and
`quantize`. A model is one of `{source: "run", run_id, backend, mode}`,
`{source: "baseline", name}` or `{source: "custom", name, backend, weights,
triton_model, imgsz, mode, class_map, backend_options}`. Errors: `400`
(no models, no datasets, unknown dataset, run or baseline id, duplicate model
keys, unknown profile), `422` (a single-class run over several classes on a
multi-class dataset, unknown fields), `409` (the job id exists, or
GPU-resident containers could not be stopped).

`BakeoffComparison`: `thresholds`, `dataset`, `eval_classes`,
`common_classes`, `rank_by`, `rank_scope` (`common` or `overall`), `models[]`
(`rank`, `overall`, `common`, `per_class[]`, `coverage`, `class_mapping`,
`train_test_overlap`, `latency_ms`, `fps`, `size_mb`, `per_stratum`),
`failed[]`, `warnings[]`, `n_models`. A model's classes map onto the eval
classes by name (`class_mapping.method`: `explicit`, `run_class_remap`,
`registry_ids`, `names`, `single_class_fallback`); unmapped and not-covered
classes are reported.

## Scores, selection, projection, probe, search, stats

| Method | Path | Notes |
|---|---|---|
| POST | `/scores/compute` | `{scorers?}`. Starts a scoring job. `400` when `OP_SCORES_ENABLED` is off or a scorer is unknown, `409` while one runs |
| GET | `/scores/status` | job state |
| POST | `/scores/cancel` | |
| GET | `/scores/coverage` | `{coverage}`: how many items carry each score field. Read-only routes work with the flag off |
| POST | `/select/diverse` | `{k, scope: {cluster_id, filters, review_tab}, seed_crop_id}`. Answers inline with `{crop_ids, method, version, n_pool}` (`200`) when the pool is small enough (`OP_SELECT_SYNC_MAX_OPS`), otherwise starts a background job (`202`). `400` when `OP_SELECT_DIVERSE_ENABLED` is off, `409` while a job runs |
| GET | `/select/status` | job state. `result` (`crop_ids`, `method`, `version`, `n_pool`) is set once `status` is `completed` |
| POST | `/select/cancel` | |
| GET | `/viz/projection` | `cluster_id`, `class_id`, `max_points` (default 50000). `400` when `OP_VIZ_PROJECTION_ENABLED` is off |
| POST | `/viz/projection/rebuild` | `scope` (`residual` default, or `cluster` with `cluster_id`). `202` |
| GET | `/viz/projection/status` | |
| POST | `/viz/projection/cancel` | |
| POST | `/probe/run` | `{job_id, architecture, gpu, resume}`. Runs the active-learning probe from a finished training job (`409` when that job is unknown, not `finished`, has no checkpoint, or the GPU cannot be claimed; one job at a time) |
| GET | `/probe/status` | `ProbeStatusResponse` (`status`, `job_id`, `train_job_id`, `model_path`, `gpu`, `updated_count`, `error`, `actionable_min_confidence`) |
| POST | `/probe/cancel` | |
| GET | `/search/text` | `q` (required), `page`, `page_size`, `class_id`, `cluster_id`, `tab`, `date_from`, `date_to`, `max_rank`, `min_blur_ratio`, `hide_near_duplicates`, `min_score`, `include_test`. Items plus `semantic_score`. `400` when `OP_SEMANTIC_SEARCH_ENABLED` is off, `503` while the text encoder is not ready |
| GET | `/stats` | project counts, indexes, disk, jobs |
| GET | `/stats/classes` | per-class counts with `thresholds`, `adequacy`, `trainable`, `trainable_gap`, `aug_target`, `aug_gap` |
| GET | `/stats/dataset` | the dashboard roll-up |
| POST | `/test_holdout/freeze` | `{percent}` (1 to 50, default 10), query `force` |
| GET | `/test_holdout/stats` | per-class holdout counts, `min_test_per_class`, `deficient` |
| GET | `/training_cohorts` | see [Cohorts](#cohorts) |

`POST /test_holdout/freeze` is deterministic: per class, the items with the
smallest `sha1(crop_id)`, `max(min_per_class, round(n * percent / 100))` of
them, capped at the class size. There is no seed (a `seed` key is `422`). A
second freeze without `force=true` is `409`. A selection that would pick zero
items is `422`. Response: `n_frozen`, `n_classes_covered`,
`test_holdout_sha`, `per_class_counts`, `selection` (`sha1_per_class`),
`percent`, `min_per_class` (5).

`GET /stats/dataset`:

- `labeled`: `by_human`, `by_vlm`, `by_classifier`, `by_import`, `other`.
  Built only from items that carry a `class_id`.
- `unlabeled`: `pending_detection`, `pending_verification`,
  `no_label_source`, `vlm_no_class` (a VLM answered or proposed but never
  landed a class) and `by_proposal` (the detector proposed it, nothing
  classified it). The last two are disjoint subsets of `no_label_source`.
- `regions`: `boxed`, `confirmed`, `total_detected`, `by_detector`,
  `by_segmenter`, `by_human`, `by_human_drew`, `by_import`,
  `verified_by_human`, `verified_by_vlm` (every non-human, non-import
  verifier), `verified_by_import`, `validated_by_human`,
  `validated_by_import`. The detector and segmenter buckets match the active
  profile's names.
- `validated_by_import`, `as_of`, `total_crops`, `validated`, `test_holdout`,
  `by_source`, `in_progress` (with `region_stall_reason`, the same text as
  `GET /ingest/region_drain`).
- `clusters`: `cluster_count` (distinct non-noise clusters now),
  `last_run_cluster_count`, `last_run_at`, `method`, `residual_count`,
  `noise_count`.

## Images

| Method | Path | Notes |
|---|---|---|
| GET | `/crops/{crop_id}/thumbnail` | JPEG, `size` 32 to 512 |
| GET | `/crops/{crop_id}/image` | the clean source render: EXIF-transposed, RGB, optional `max_dim` (128 to 8192). It draws no overlay, so the bytes do not depend on any box. Draw boxes from `GET /crops/{crop_id}/context` |
| GET | `/crops/{crop_id}/region_thumbnail` | see [Regions](#region-browse-clusters-and-false-positives) |
| GET | `/images/serve` | `path` (required). Streams a source image, choosing the configured root that matches the path |
| GET | `/images/root/{alias}` | `path` (required, relative to the alias root). `404` for an unknown alias |
| GET | `/images/cache/stats` | thumbnail cache counters |

The server serves clean images and database metadata. A client draws boxes,
labels and styling itself.

## Events

Server-sent events. A scoped stream carries only its own project's events. The
global stream carries the families `project.*`, `combine.*` and `vlm.*`.
Events are advisory: each subscriber has a bounded queue and the oldest drop
on overflow.

| Method | Path | Notes |
|---|---|---|
| GET | `/curation/events` | global stream; `topic` filter |
| GET | `/events` | the bound project's stream; `topic` and `class_id` filters |
| POST | `/events/publish` | used by the detection worker. `_PublishEvent`: `type`, `crop_id`, `class_id`, `class_name`, `class_source`, `region_status`, `region_count`, `image_path`, `topic`, `extra` (`extra='forbid'`). A global event type is `422`; `extra.project` must be the bound project |
| GET | `/events/stats` | `bus`, `log_path`, `subscribers`, `events_published`, `events_dropped` |
| GET | `/pipeline/events` | auto-label dashboard stream |

Event types: `project.created`, `project.updated`, `project.archived`,
`project.unarchived`, `project.deleted` (global, with `target`, `status` and
`revision`), `project.paused` and `project.resumed`, `combine.progress`
(`target`, `job_id`, `phase`, `done`, `total`, `status`), `vlm.changed`
(`axis`: `registry` or `local_vlm`), `config.changed` (project, with `axis`
of `prompt_pack`, `detection_profile`, `vlm`, `keymap`, ...),
`classes.changed`, `crop.created`, `crop.classified` and
`crop.region_verified`.

`crop.region_verified` data: `type`, `topic` (`region_status`), `crop_id`,
`region_status`, `region_count`, `ts`. The event hub is cross-process by
default (`OP_EVENT_BUS=file`): every API worker tails a shared bounded log, so
a client connected to any worker sees events from any other and from the
detection worker.

## The lock rule

A label or box that a human or a dataset import set is never touched by an
automated writer. The rule has one implementation (`src/clients/occ_locks.py`)
and is checked inside the write, not only before it, so an edit that lands
between planning and writing still wins.

| Level | Locked when |
|---|---|
| Class | a human set or confirmed it (`class_source` or `label_source` contains `human`); or it is a validated imported label (`class_source: external_label` and `class_validated`); or the item is in the frozen test holdout (`test_holdout`) |
| Box | a human created it, gave a verdict on it (a human reject, `rejection_reason` is the human reason) or typed its text (`text_source: human`); or it came from a dataset import (`source: import`) and is not a mere suggestion (`state` is not `proposed`) |
| Item | its class is locked, or any of its boxes is locked, or its box set is validated with `region_verifier` of `human` or `import` |

An import with `label_trust: suggestion` writes unvalidated labels and
`proposed` boxes, which the machine pipeline may still replace.

Automated writers that honor the rule: the detection worker (classification
gate, bulk writer and box merge), `POST /vlm/label_batch`, re-ingest, region
clustering and false-positive pulls, `POST /reprocess` (reported as
`locked_skipped`), the VLM stages of the auto-label pipeline, dataset import
and combine.

On the wire:

- `label_locked` on every item is the evaluation for the whole item. A client
  reads it instead of recomputing it.
- `locked` on every `region_boxes[]` element is the evaluation for that box.
- A dataset import reports labels it refused as `label_conflicts_locked`. A
  combine flags a conflicting box with `combine_conflict` and keeps the
  priority label.
- Human routes are not blocked by the rule. A human can always overwrite a
  human, imported or machine label, and can undo it.

## Class identity

A class is identified by its name inside one project. The integer id belongs
to a project's registry and never crosses a project boundary or a file format
unchanged.

| Hop | How the class is matched |
|---|---|
| Dataset import | the dataset's class names are mapped (`map`, `create`, `skip`, `region`) onto the project's registry by name, never by index. Two datasets with the same names in a different order land on the same classes |
| Combine | each source class is mapped by name onto a target class. The target owns its ids. Class ids, class-id history and cluster ids of a source do not cross |
| Export | the dataset uses a dense `export_id` (0 to n-1) assigned in registry order. `class_split_counts` and `class_registry.json` map `export_id` to `class_id` and `class_name`. `data.yaml` `names:` lists the names |
| Training | a subset or single-class run records `class_remap.json` (original id to trained id) |
| Promote | `labels.txt` is written from the remap, so line `i` is the name of trained class `i`. Without a resolvable remap, promote is refused (`class_remap_missing`) |
| Another project using a promoted model | the model's class names are matched by name onto that project's registry (`GET /models/{model_name}/class_mapping`); unmatched names are reported, never guessed |
| Bake-off | a model's classes map onto the eval dataset's classes by name or an explicit map; unmapped and uncovered classes are reported |

Rules that follow from this:

- Class names match `^[a-z0-9_]+$` and are unique among active classes.
- Renaming a class changes its name everywhere the next time the name is
  read. A merge records the relabel; a restore of a merged class is refused.
- `ClassEntry.kind` is `region` for the class named like the active region
  profile's `region_class_name`. That class labels a sub-box, not a whole
  item.
- `tests/integration/test_class_identity_e2e.py` and
  `tests/integration/test_class_identity_combine.py` assert the
  `(class_id, class_name)` pairing at every hop from import or combine through
  export, remap, promote and predict.

## Example: car to wheel

The public example detects wheels on cars. All data is public: COCO car
images. `examples/region_profiles/vehicle_wheel.json` is a text-free region
profile (no detector leg, a segmenter text prompt `wheel`, `parent_classes`
of `car`, `max_regions_per_item: 4`, `text_reader: none`).
`examples/prompt_packs/vehicle_wheel.json` is the matching prompt pack.
`examples/bakeoff/vehicle_wheel/profile.json` is the bake-off profile.
Fetch the images with `make sample-coco-cars`.

The walk, in order (it is what `scripts/examples/wheel_example_live.py`
runs against a live stack, and what
`tests/integration/test_wheel_example_e2e.py` runs offline with fakes at the
OpenSearch, Triton, segmenter and VLM boundaries):

| Step | Call |
|---|---|
| Create the project | `POST /curation/projects` with `{slug, display_name}` |
| Create the classes | `POST /classes` for `car` and `wheel` |
| Add the profile and pack | `POST /region_profiles` and `POST /prompt_packs` with the example bodies |
| Activate them | `POST /region_profiles/{name}/activate` and `POST /prompt_packs/{name}/activate` |
| Activate a VLM endpoint (optional) | `POST /vlm/endpoints/{name}/activate` |
| Ingest | `POST /ingest/batch` |
| Wait for the region worker | poll `GET /ingest/region_drain` until `drained` |
| Review | `GET /review/{tab}` with the `regions` tab, then `PATCH /crops/{crop_id}/regions/{box_id}` or `PUT /crops/{crop_id}/regions` |
| Export wheels cropped to their car | `POST /export/single_class` with `box_source: region`, `region_class_name: wheel`, `image_mode: item_crop` |

> Screenshot pending: Cropwright (the wheel review screen with several region boxes on one car)

## Breaking wire changes

v0.1.0 is a fresh wire. Deployments re-ingest; there is no data migration.
The table lists renamed or restructured wire surfaces.

| Surface | Now |
|---|---|
| Per-box item scalars (box geometry, score, detector, text, cluster, thumbnail, parent-frame box) | elements of `region_boxes[]`: `bbox_norm`, `score`, `detector`, `text*`, `cluster_*`, `thumbnail_url`, `bbox_in_parent`. The item carries only `region_*` set-level fields |
| Region attributes with a domain prefix | `region_<attr>`, fixed names independent of storage names |
| Vendor-named VLM and classifier fields, `class_source` and `label_source` values, review tab and history writer ids | `vlm_*` and `classifier_*`; ingest-profile-derived source ids (`{primary}_proposal`, `{secondary}_model`) |
| Domain-named statuses | `no_region_box`, `no_region_visible` |
| Single-box region writes | `PUT /crops/{crop_id}/regions`, `PUT /crops/batch_regions`, `PATCH /crops/{crop_id}/regions/{box_id}`, `POST /regions/batch_box_state` |
| Labeled-dataset import through ingest | `/datasets/*`. `IngestBatchItem` is `{path, source}` and `BatchIngestSummaryResponse` has no label-import counters |
| Requeue, clear-detection and retry paths | `POST /reprocess` |
| One configured VLM | the endpoint registry, a per-project `vlm` axis and per-run `?vlm=`. `GET /models/status` VLM rows have `kind: vlm` |
| `region_bbox_correct`, `region_confidence`, `region_text` as item keys | box keys `bbox_correct`, `confidence`, `text` (the VLM reply keys keep their protocol names) |
| Region clustering counts | `n_boxes`, `n_boxes_changed`, `n_items_written`, `n_boxes_updated`, `n_items`, `n_items_updated`, `n_boxes_scanned`, `n_boxes_moved`; cluster cards add `size` (items) and `box_count` |
| Ingest response region count | `n_regions` |
| `GET /stats/dataset` | `labeled.by_classifier`, `regions` block (was domain-named), `unlabeled.by_proposal`, `unlabeled.vlm_no_class`, `validated_by_import`, `labeled.by_import`, `regions.by_import` |
| Training status metric pair | `last_epoch_metric` and `best_checkpoint_metric` |
| Class merge dry run | `validations_carried_over` (a merge keeps `class_validated`) |
| `GET /crops/{crop_id}/image` | the clean render, no overlay |
| `GET /clusters` | `representatives` filled only for the `offset`/`limit` window |
| `GET /crops` | `classifier_conf_lt`, `limit`, `sort`, `conf_min`, `conf_max`, `k`, `item_text`, `import_id`, `dataset_split`, `on_negative_frame`, `proposed_by_import` |
| `GET /health` | `vlm` (the active endpoint), `region_profile`, `mlflow_public_url` |
| Environment variables | VLM and segmenter connection variables are `OP_VLM_*` and `OP_SEGMENTER_*`; retired names are listed in `src/config/retired_env.py` |
| Item key count | 106 keys; the list in `contracts/json/item_wire.json` is authoritative |

## Notes for consumers

- This document and the OpenAPI file are the shared contract. No wire change
  ships without a matching change here. `tests/curation/test_wire_contract.py`
  pins the item key set. `scripts/docs/check_docs_vs_code.py` checks that every
  route written in the docs exists.
- Gate optional UI on `GET /methods`, `GET /review/tabs` and
  `GET /regions/vocabulary`. They serve labels, options and defaults, so a
  client does not hardcode vocabulary.
- A backend outage is `503`. An empty list always means no matching data.
- Not on the wire: backend OpenSearch field names, the internal tracking
  host, API keys, and filesystem paths of training artifacts.
