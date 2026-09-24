# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

- A pre-push hook runs the full vitest suite (`vitest-pre-push` in
  `.pre-commit-config.yaml`), so a failing test can't be pushed. Install
  it with `pre-commit install --hook-type pre-push`.
- The `/train` GPU picker shows the backend's allowed GPUs
  (`GET /train/gpus`) with each option's advisory, and preselects the one
  the backend marks default. It shows a free-text field when the backend
  has no allowlist.
  - `trainGpuOptions.ts` and its hardcoded list are deleted. The list
    offered GPU 0, which is now reserved for another project.
  - The form no longer defaults to GPUs `0,2`, which the backend rejects.
    Until the options load, the request omits the claim and the backend
    picks from its allowlist.

### Added

- Tests now check the frontend against the backend's real API contract.
  - `contracts/openprocessor/` holds a copy of OpenProcessor's generated
    contract files (item and region fields, region statuses, class-source
    roles, the curation OpenAPI). `npm run contract:sync` refreshes it and
    `npm run contract:check` fails when it no longer matches the backend.
    The check runs in pre-commit and CI; CI skips it until the backend
    repo is available there.
  - Tests in `src/lib/contract/` read that copy instead of hand-copied
    lists. They check every field `RawCrop` reads, the slot profile's
    region fields and statuses, every class-source role, and every API
    call's path, method and query parameters. The call list is found by
    scanning the code, so a new call is checked without registering it.
  - `RAW_CROP_KEYS` in `api.ts` lists every `RawCrop` field, with a
    compile-time check that it stays complete.
- A shared test fixture, `src/lib/test/makeItem.ts`, sets every item field
  to a distinct value. `api.mapRawCrop.test.ts` uses it to check every
  mapped field. It catches the dropped `cluster_subid`, `test_holdout` and
  confidence mappings that no test caught before.
- Mutation testing with Stryker: `npm run test:mutation`, weekly in CI
  (`.github/workflows/mutation.yml`). It covers the API client, the undo
  store and eight helper modules. The build fails if the score drops
  below its current level.

### Changed

- Label writes, undo and discard now match OpenProcessor `main`
  (`d037be8`, see `docs/design/logic-moves-adoption-plan-2026-09-24.md`
  W1):
  - `bulkLabel`/`moveCropsToCluster` results carry `updated_ids`;
    `undoStore.recordWrites()` takes that served list directly instead
    of the request ids minus locally-tracked conflicts.
  - `/clusters/[id]`'s **D** (discard) now calls
    `POST {API_PREFIX}/crops/{id}/discard` (or `discard_batch` for a
    multi-select) instead of `DELETE {API_PREFIX}/crops/{id}/label`, and
    pushes an undo entry for the server-confirmed ids — Z restores a
    discarded crop.
  - Rejecting a VLM suggestion on `/clusters/[id]` calls
    `POST {API_PREFIX}/crops/{id}/vlm_dismiss` and renders the returned
    item, instead of only clearing the suggestion locally. A 409 (no
    suggestion) shows an info toast.
  - `/review` gets a minimal "Dismissed" panel
    (`GET {API_PREFIX}/crops?review_dismissed=true` +
    `POST {API_PREFIX}/crops/{id}/review_undismiss`) so a permanent
    dismiss (**D**) can be reversed.
- Region writes and statuses now match OpenProcessor `main` (`d037be8`,
  see `docs/design/logic-moves-adoption-plan-2026-09-24.md` W2):
  - `setSlotBox`/`patchSlotMeta`/`batchPlateStatus` render the item(s)
    the server actually wrote (`{..., item}` / `{..., items}`) instead
    of a hand-computed post-write state. `batchPlateStatus`'s `invalid[]`
    entries (e.g. `detected` with no box) show as a toast with the
    per-crop detail.
  - `setSlotBox` takes a `frame: 'source' | 'parent'` argument. Box
    edits (`/review`'s inline canvas, `SlotBboxEditor.svelte`) now send
    the box exactly as drawn with `frame: 'parent'` — no more
    client-side `projectFromParent` on the write path. Reads still
    project client-side when the server doesn't serve
    `region_bbox_in_parent`; `licensePlateSlot` now declares
    `bboxInParentField: 'region_bbox_in_parent'` and prefers it.
  - The ⚠ bbox-shape-plausibility warning is gone: `shapeGate.ts`,
    `PLATE_SHAPE_ENVELOPE`, `evaluateShapeGate`/
    `evaluateShapeEnvelopeOnFraction`/`describeEnvelope`, and the badge
    in `SlotCard.svelte`/`/review` are deleted — the backend never
    served this flag, and OpenProcessor's own geometry guard is a
    different check. `SubBoxCapability.envelope`/`ShapeEnvelope` and
    `SlotData.subBox.shapeWarning` are removed from the slot type model.
  - The region-status vocabulary is now served
    (`GET {API_PREFIX}/regions/statuses`, loaded once by
    `regionStatusesStore`) and drives `/review`'s status dropdown,
    clear/rejection-reason behavior and the confirm/reject/false-positive
    actions, via `slotPanel.ts`'s `humanWritableStates`/`statusClearsBox`/
    `statusWantsRejectionReason`. `licensePlateSlot`'s own
    `capabilities.lifecycle.states` is kept as the fallback for when the
    endpoint is unavailable.
- Item detail and OCR display now match OpenProcessor `main` (`d037be8`,
  see `docs/design/logic-moves-adoption-plan-2026-09-24.md` W7/W8):
  - `Crop.source` replaces the dead `hdd_source` field (`mapRawCrop`
    never populated it — the backend only ever emitted `source`).
    `/review`'s source-image panel reads `current.source`.
  - New `Crop` fields: `class_excluded`, `excluded_reason`,
    `excluded_at`, `item_text_lines`. New API calls: `getCropHistory`
    (`GET {API_PREFIX}/crops/{id}/history`) and `getCropImage`
    (`GET {API_PREFIX}/crops/{id}/image`).
  - `CropMetaPanel` (the `/clusters` detail modal, now also embedded
    behind a collapsed "Details" disclosure on `/review`) shows: an
    "Ignored" banner with `excluded_reason`/`excluded_at`; a lazy,
    on-open label-history list; the source image's metadata plus a
    sibling-crop thumbnail strip; `item_text_lines` with an optional
    box overlay (drawn directly from `box_norm`, already in the
    item-crop frame); and, for a slot with OCR candidates, the VLM/OCR
    readings plus a "readers disagree" flag and the text engine
    version.
  - `/clusters` gets an "Ignored" bucket toggle
    (`cluster_id=-2&include_excluded=true`) with a "Restore selected"
    action (`batch_unexclude`), and an item-text search box
    (`GET {API_PREFIX}/crops?item_text=`) — a 400 (no letter/digit in
    the query) renders as an inline hint, not a toast.
  - `TextCapability` gains `vlmValueField`/`ocrValueField`/
    `disagreementField`; `licensePlateSlot` wires them to
    `region_text_vlm`/`region_text_ocr`/`region_text_disagreement`.
    `SlotCard` and `/review`'s inline slot panel both show a "readers
    disagree" badge when set.
  - `ProvenanceChip`'s muted-tag pattern now includes
    `accepted_unverified`, so that chain step renders like a
    miss/reject rather than a confirmed one.

- The cluster-scoped VLM run now matches OpenProcessor `main` (`d037be8`,
  see `docs/design/logic-moves-adoption-plan-2026-09-24.md` W3):
  `runVlmOnCluster` (dashboard's "Run VLM on cluster" modal and
  `/clusters/[id]`'s "Run VLM" button) is a single
  `POST {API_PREFIX}/vlm/label_cluster/{id}[?prompt_pack=]` instead of a
  client-side fetch-200-crops-then-chunk-of-64 loop against
  `{API_PREFIX}/vlm/label_batch` — the server now selects and chunks the
  unvalidated members itself. Progress renders from a new
  `pollAutoLabelJob()` helper polling
  `GET {API_PREFIX}/pipeline/auto_label/status` (stage + processed/total)
  until the job leaves `running`; both pages show the live stage inline
  and the final toast reads `result.stages.vlm.predicted`/`.updated`.
- Cluster cards and the cluster-detail cut line now match OpenProcessor
  `main` (`d037be8`, see
  `docs/design/logic-moves-adoption-plan-2026-09-24.md` W6):
  - `Cluster` carries the served `purity_tier`/`promotable` and
    `core_similarity_min`; `/clusters`' `borderColor`/`purityBadge` branch
    on `purity_tier` and the card shows a `promotable` chip — the client
    0.8/0.6 purity thresholds are deleted.
  - `Crop.similarity_to_centroid` is mapped from the served
    `cluster_similarity` — the client `Math.max(0, 1 - cluster_distance)`
    estimate is deleted. A new `Crop.cluster_is_core` (served
    `cluster_is_core`) drives `/clusters/[id]`'s core/non-core cut line;
    the client `similarity_to_centroid <= 0.75` constant is deleted.
  - `/clusters/[id]`'s class-for-cluster lookup (`clsForCluster`) now
    reads the served `cluster.cluster_kind`/`cluster.dominant_class_id`
    instead of assuming `cluster_id === class_id`.
  - `/train`'s Training cohorts panel now sources its cohort definitions
    from `GET {API_PREFIX}/training_cohorts?class_id=` (already resolved
    for the class, loaded lazily per class group alongside the existing
    lazy counts) instead of the client-only `cohortsForClass()`
    (`CORE_COHORTS` + `license_plate`'s hardcoded 5 modes). A slot's own
    tier-2-declared `capabilities.trainingCohorts.cohorts` (see
    `parseSlotConfig.ts`) still fills in as a fallback for any cohort id
    the server didn't already send, so a deployment-registered slot the
    backend has no region profile for keeps working. `CORE_COHORTS`/
    `derivedCohorts`/`cohortsForClass` stay in `cohorts.ts` — the tier-2
    mechanism and its test coverage (`cohorts.test.ts`,
    `secondSlotIntegration.test.ts`, `exampleProfile.test.ts`) still need
    them.

### Removed

- `src/lib/annotations/regionWireContract.test.ts`, replaced by the
  contract tests.

### Fixed

- Dashboard/export stats resilience (frontend-coverage-audit-2026-09-24.md
  G1): `DatasetStats.svelte` no longer crashes when `GET /stats/dataset`
  (or its SSE `snapshot`/`stats` frames) returns an `{error}` envelope —
  it keeps the last-known-good stats on screen and shows the existing
  "Stats unavailable" banner instead of throwing `Cannot read properties
of undefined (reading 'sam_drain_total_unfinished')`. The guard is a
  new pure `resolveStatsUpdate()` (`src/lib/datasetStats.ts`).
  - `getStats()` now uses `Promise.allSettled` for `/stats/dataset` and
    `/stats/classes`, so a dataset-rollup failure no longer blanks
    `per_class` — `/export`'s class table renders again.
- "Validated" no longer conflates the class label with the region (G2):
  added `class_validated` to `Crop`/`RawCrop` and `mapRawCrop`.
  `CropCard`'s badge and VLM-accept chip, `CropMetaPanel`'s "validated"
  pill, and the accept-all-VLM / advance-to-next-unvalidated logic on
  `/clusters/[id]` now read `class_validated` instead of the OR-combined
  `label_validated`, which previously showed a crop as validated purely
  because its _region_ had been confirmed.
- `/review`'s Class, Source and Conf filter controls are re-enabled and
  sent to `GET {API_PREFIX}/review/{tab}` as `class_id`/`source`/
  `conf_min`/`conf_max` — the backend now honors them (verified live:
  `class_id` and `conf_min` both change the queue total). The old
  `REVIEW_SERVER_FILTERS_ENABLED` flag is deleted; the "HDD source"
  control is renamed to "Source" (`hddSource` → `sourceFilter`,
  reading/writing `Crop.source`, which replaces the dead `hdd_source`
  field). See `docs/design/logic-moves-adoption-plan-2026-09-24.md` W5.
- Scoped VLM-assist runs (`AssistScopeBar`) now actually run the VLM
  stage: `startAutoLabel()` sends `run_vlm: true` whenever a class or
  prompt pack is scoped, matching the toast copy that already claimed
  "VLM labeling limited to {class}" but never sent the flag (G5). Added
  an explicit "Run VLM labeling stage" checkbox to `AutoLabelPanel`, off
  by default to match the backend's own default.

### Added

- `/review`'s item panel shows a "Model predicts" row
  (`probe_pred_class` + an entropy `ScoreChip`) when the backend serves
  a probe prediction — previously computed but never rendered (G4). An
  "Accept model's class" button now assigns `probe_pred_class_id`
  directly (closes G4 — the backend ships the id, so no
  name-to-id lookup is needed).
- `/review` W5 (`docs/design/logic-moves-adoption-plan-2026-09-24.md`):
  - `/review?crop_id=` deep links now call
    `GET {API_PREFIX}/review/{tab}/locate` and jump straight to the
    crop's served `page`/`rank`, instead of paging forward up to 300
    items hoping to find it. A crop the backend reports as not in the
    queue shows its `reason` (e.g. "filtered_out") in the toast.
  - The client-side `proposed_class_id`/`proposed_class_name` fill-ins
    in the diverse-selection hydration, semantic-search results and
    undo-restore are deleted — `Crop`/`mapRawCrop` now carry the
    server's own `proposed_class_id`/`_name` (served on every
    crop-shaped item, not just review rows), so every one of those
    paths already has the real value.
  - The StrategyBar summary chip shows the server's `sort_applied`
    (e.g. "sort: Default order → atypicality") next to whatever the
    operator picked, so a tab-default or a sort fallback is visible,
    not just implied.
  - A new `new_class_proposals` review tab (not a preset — a distinct
    triage workflow) surfaces crops the VLM flagged as needing a class
    the registry doesn't have yet; flagged items show
    `needs_new_class_note` inline.
  - `/classes` gets a "Proposals" section
    (`GET {API_PREFIX}/review/new_class_proposals/summary`) — the top
    VLM-proposed-but-unmatched terms with sample thumbnails, each with
    "Create class & assign" (`POST {API_PREFIX}/classes` then
    `PUT {API_PREFIX}/crops/batch_label` on the served
    `sample_crop_ids`) and "Map to existing" actions. Degrades to an
    inline error banner (verified live against a real opensearch
    aggregation 503) rather than breaking the page.

### Changed

- Undo (Z) on `/review`, `/clusters` and `/clusters/[id]` now calls the
  backend's `POST /crops/{id}/label/undo` and renders the item it
  returns.
  - The frontend no longer chooses between re-applying an old label and
    deleting the current one. That choice restored the wrong class after
    a relabel of an unconfirmed crop.
  - A `409` shows "Nothing left to undo".
- Undo entries are recorded only after the server confirms a label
  write, and never for conflicted crops. Before, a failed or conflicted
  drop-to-label left entries behind, and Z on them would undo an older,
  unrelated write.
- Discard (`D`) on `/clusters/[id]` is no longer undoable. The backend's
  `DELETE /crops/{id}/label` is itself an undo of the last human write,
  so a Z entry for it would step back a second write.

### Changed

- Removed the remaining `legacy`/`op` names from code, scripts and docs.
  - `legacyDetectors.ts` is now `builtinDetectors.ts`
    (`builtinDetectorRegistry`). Its entries for `legacy_vehicle_v6_trt`
    and `ingest_v6` are gone, since OpenProcessor no longer emits those
    detector ids.
  - The nginx upstream variable `$kbapi` is now `$api_upstream`.
    `.env.example` documents the `/curation` default.
  - Comments and page copy name OpenProcessor's current files and its
    `curation-trainer`/`curation-evaluator` containers.
  - README, CONTRIBUTING, SECURITY, RUBRIC and CLAUDE.md describe the
    OpenProcessor deployment. README no longer covers the retired compose
    overlay or the GPU-sharing make targets.

### Fixed

- The stubbed Playwright scripts intercepted `/curation/**`. The app calls
  `/curation/**`, so their stubs never matched. They now stub `/curation`.
  `playwright_curation_settings.py` also expects the server's `settable`
  flag and the current `/settings` copy, and it passes against the built
  container. `playwright_smoke.py` finds a cluster through
  `/curation/clusters`; the `/clusters/stats/op_vehicles` route it used
  was removed.

### Changed

- The frontend now uses the backend's `class_source` catalog
  (`GET /class_sources`, loaded once in the layout) instead of its own
  copies of backend rules.
  - The `/clusters/[id]` source filter is a dropdown of the deployment's
    catalog, including the config-derived ingest sources, and is absent
    without one.
  - The crop card's badge takes its text from the catalog label and its
    color from the catalog role. An unknown id renders verbatim and
    neutral, never inferred from its name.
  - `Enter`/Confirm uses the backend's `proposed_class_id` as-is. The
    client-side fallback to `class_id` was a second copy of review.py's
    rule.
  - The card shows a VLM new-class proposal
    (`vlm_proposed_class_name` with no id) read-only.

### Changed

- Removed the `op` (legacy) naming from the frontend.
  - The `Kb`-prefixed types lose the prefix, or get a descriptive name
    where a bare one would clash or read vaguely: `Crop`, `RegistryClass`
    (+ `Create`/`Update`/`Merge`), `Cluster`, `MethodsResponse`,
    `ApiHealth`, `StatsSummary`, `ModelInfo`, `ExportDataset`,
    `CurationEvent*`, `subscribeCurationEvents`, and so on.
  - Comments now reference `{API_PREFIX}/…` paths, main's module names and
    `OP_*` env vars.
  - The drag-and-drop item type is now `crop-card`.
  - The saved `/clusters` filter key is now `clusters_filter_v1`, so a
    previously saved filter resets once.
  - The prefix scan tests' fixtures use a real `/curation/…` literal again.
    The rename had briefly turned them into `{API_PREFIX}` strings that
    could never match, which made three of them vacuous.

### Fixed

- `/review`'s `Enter` no longer confirms an unrelated class on a
  `vlm_new_class_pending` item. `resolveConfirmClassId` used to fall back
  to the crop's current class when there was no proposal, and for these
  items the backend now reports `proposed_class_id: null` precisely
  because that class is unrelated. `Enter` opens the class picker there
  instead.

### Fixed

- `/review` now honors `?tab=` and `?crop_id=` deep links. It opens that
  tab and jumps to that crop, paging forward up to 300 items and saying so
  when the crop isn't in the queue. Switching tabs keeps `?tab=` in the
  address bar. Before this, bookmarks such as `?tab=plates` and `/train`'s
  cohort-preview links always landed on the first item of All:
  `tabFromUrlId` existed but nothing called it.
- `/review` fetched the first queue page twice on every load. The
  debounced filter effect treated its initial state as a change 250 ms
  after the immediate load, and the second fetch reset the cursor.
- The review panel's proposer label reads "Proposal hint" instead of
  "COCO hint".

### Changed

- **BREAKING (B3, ships with OpenProcessor's `cutover/wire-contract`):**
  the frontend now speaks OpenProcessor's generic wire vocabulary, with
  no fallback to the old names (deployments re-ingest fresh). Plan:
  `docs/design/b3-wire-rename-frontend-plan-2026-09-23.md`.
  - Every region attribute is read and written as `region_<attr>` (was
    `plate_<attr>`), including `region_thumbnail_url` and the
    `/regions` `region_cluster_id`/`region_cluster_subid` filters.
    Region box and batch-status writes use the slot's own wire fields
    (new optional `lifecycle.labelSourceField`), so writes and reads can
    never use different names.
  - `classifier_raw_confidence`, `proposal_name`, `classifier_conf_lt`.
    The auto-label params and stage move to `vlm_*`, the review preset to
    `vlm_low_conf`, and the stats rollups to
    `labeled.by_classifier`/`by_proposal` plus a `regions` block
    (`by_detector`/`by_segmenter`/`verified_by_vlm`) with generic row
    labels. `OpHealth` models the real `{status, triton, opensearch, vlm,
registry}` payload.
  - The `/clusters/[id]` source filter offers B3's fixed writer values
    (`human`, `vlm`, `vlm_unmatched`, `vlm_new_class_pending`,
    `classifier_vlm_agreement`). `LabelSource` follows the same
    vocabulary, and the crop badge shows `vlm?`/`vlm`.
  - UI copy says "VLM" instead of "Gemma" wherever it describes the
    pluggable VLM role.

### Fixed

- The accept-VLM-suggestion flow on `/clusters/[id]` (card chip, `G`,
  `Shift+Enter`, meta-panel row) could never fire, because no field ever
  populated the suggestion. `mapRawCrop` now maps `vlm_proposed_class_id`
  / `vlm_proposed_class_name` and the categorical `vlm_confidence`. The
  meta panel's VLM-confidence row, which read an unmapped field through a
  cast, now renders too.

### Changed

- `/settings` decides which axes get a control from the server's
  per-entry `settable` flag on `/methods`, not from a hardcoded
  `SETTINGS_AXES.kind`. The table now carries only labels, buckets and
  blurbs, so it can't drift from what the backend honors. The store's
  client-side "advisory axis" guard is gone too: OpenProcessor rejects a
  non-settable axis with a 422, and that detail is surfaced.

### Fixed

- `/settings`'s prompt-pack blurb understated its effect. The shared
  default also drives the always-on background VLM labeler
  (`/vlm/label_batch`), not only auto-label runs.

### Fixed

- `/bakeoff` preselects the deployment's default profile (`default` /
  `default_profile` on `/bakeoff/profiles`) and warns when the configured
  default is invalid (`default_error`). Profiles now load before
  baselines, so an unscoped baseline lookup can no longer land after the
  scoped one and show the wrong baselines.
- `/bakeoff` shows why a run failed. It shows the job-level `error` when
  `state` is `error`, and every failed stage or dataset × model cell from
  the status `failed` list. Before, a failed cell silently vanished from
  the matrix.
- Dropped the progress line's "auto-stops SAM3/Gemma" claim, which
  described one deployment's GPU handling rather than main's.

### Removed

- The dashboard's per-run detection-profile picker, along with
  `detection_profile` on `startAutoLabel` and the
  `isDetectionProfileAvailable` gate. OpenProcessor confirmed that no
  auto-label stage runs region detection: it's the detection worker's
  startup config, so the picker was a silent no-op, and main now rejects
  the param with a 422. The scope bar is gated on the `prompt_pack` axis
  alone and shows a chosen pack in its collapsed summary.

### Changed

- Adopted OpenProcessor's per-run auto-label contract (`profile-arbiter`):
  - An unknown `prompt_pack`/`detection_profile` now produces a readable
    "unknown prompt pack "x" — valid: …" toast from the 422's
    `{axis, requested, valid_ids}` instead of a bare "API 422".
    `ApiError` also reads the text of any structured `{detail: {error}}`
    body.
  - The class scope's copy now says what it really does: it limits the
    VLM labeling sweep to that class, while clustering still covers the
    whole pool.
  - `/settings` makes the VLM prompt pack a real, settable deployment
    default. It's honored by every auto-label run that doesn't pick its
    own pack. The detection profile stays display-only, because nothing
    that runs reads it.

### Fixed

- The embedding plot's projection rebuild is no longer fire-and-forget. It
  polls `GET /viz/projection/status`, shows progress, offers Cancel
  (`POST /viz/projection/cancel`) and reloads the plot when the job
  completes, or shows an error if it fails. It also picks up a rebuild
  already running when the page opens. Once a projection exists there's
  now a Rebuild button; before this, a projection could only be built
  once and never refreshed as new crops arrived.

### Changed

- The single-class plate dataset export now runs on OpenProcessor's generic
  narrowed export (`POST /export/single_class`,
  `GET /export/single_class/status?profile_name=`) instead of the
  never-ported `/export/lpr`, which had kept the `/train` panel hidden on
  `main`.
  - A slot's `extras.datasetExport` now declares its wire identity:
    `profileName`, `boxSource` (`item`|`region`), optional
    `regionClassName`, and `classIds`. `datasetExportForSlot` validates
    them, including the backend's rule that `item` needs `classIds`.
  - `exportSingleClass`/`exportSingleClassStatus` post to the spec's own
    paths, so the `api.ts`/profile drift-ratchet test is gone. There's
    only one copy of each path now.
  - The `vehicle_crop` image mode is now `item_crop`.
  - `/train`'s dataset picker filters `/export/datasets` rows on `kind`
    (`yolo` | `single_class`) plus `profile_name`, per the contract agreed
    with the backend owner. The multi-class toggle is no longer labeled
    "vehicles".

### Removed

- The dashboard's "Snapshot op\_\* indexes" button. It was a placeholder that
  only showed a "coming in v1.1" toast, and no backend route exists for
  it. The API-health banner and tooltip no longer name `openprocessor`.

### Added

- `/bakeoff` profile picker, backed by OpenProcessor's B1 `BakeoffProfile`
  (`GET /bakeoff/profiles`). The chosen profile scopes the baseline model
  list (`/bakeoff/baseline_models?profile=`) and is sent on
  `POST /bakeoff/run`. "Deployment default" omits it. `BakeoffModelSpec`
  now matches main's model spec: per-model `profile`,
  `primary_*`/`secondary_*` coarse-stage fields replacing
  `vehicle_weights`, and the `onnxruntime`/`coreml` backends. Trained
  contenders are labeled "trained here" instead of a deployment-specific
  corpus string.

### Changed

- Retargeted at OpenProcessor `main` as the only backend (E2E contract
  audit, `docs/design/e2e-contract-audit-2026-09-23.md`):
  - `API_PREFIX` now defaults to `/curation` (T-E2), in `api.ts` and
    `docker-entrypoint.sh`.
  - nginx's API upstream is configurable at container start via
    `API_UPSTREAM` (default `http://op-api:8000`). It was hardcoded to
    `op-api:8000`.
  - Added a standalone `docker-compose.yml` that joins the API's docker
    network (`OP_DOCKER_NETWORK`), plus a `.dockerignore` so
    `node_modules`, `.git` and `.env` files stay out of the build
    context.
  - Adopted main's renamed response keys:
    - dataset stats `labeled.by_vlm` (was `by_gemma`, which made the
      labeled total NaN)
    - model `is_region_protected` (was `is_lpr`, which exposed Unload on
      protected models)
    - class sync `upserted`
    - region clustering `n_regions`
    - SSE `crop.region_verified`, replacing the retired
      `crop.plate_verified` literal
  - `getPlates` takes the slot's declared `queue.browsePath` instead of a
    hidden route constant.

### Fixed

- `/clusters/[id]` no longer opens an SSE subscription on candidate
  clusters. They have no class, and the backend filters events on exact
  `class_id`, so that stream could never deliver anything.
- Corrected the `startAutoLabel` doc comment claiming
  `detection_profile`/`prompt_pack` were confirmed live. `main` silently
  dropped them; per-run support is landing backend-side.

### Added

- Annotation slots are now wired all the way through the crop pipeline
  instead of stopping at the adapter layer (Wave 0 + Wave 1 of
  `docs/design/slot-generic-crop-mapping-plan-2026-09-21.md`, C1-C11).
  `OpCrop` gains a `slots?: Record<SlotKey, SlotData>` map, populated by
  `mapRawCrop`/`getPlates` via the existing `readSlot` adapter
  (`src/lib/annotations/cropSlots.ts`'s `mapCropSlots`/`slotOf`).
  `SlotState` gains an optional `aliases?: string[]` (the read-tolerance
  half of a future wire-vocabulary rename), and `readSlot.ts` exports
  `projectFromParent`, the inverse of its private forward projection.
  Every consumer that used to read a hardcoded `crop.plate_*` field or
  import `licensePlateSlot` directly — `CropMetaPanel`, `CropCard`,
  `BboxCanvas`, `/review`'s entire inline slot panel (Finding D),
  `sse.ts`'s verify-event dispatch, `SlotBboxEditor`, `SlotCard`, and
  `DatasetStats`' detection panel — now reads through the active slot's
  own capabilities, so a second registered slot renders correctly with
  zero further code change (proved by
  `src/lib/annotations/secondSlotIntegration.test.ts`).

### Changed

- **BREAKING (backend-contract):** adopted OpenProcessor's (openprocessor)
  region wire-vocabulary rename, merged to its `main` at `b3f928d`
  (Wave 2, C12-C14 of
  `docs/design/slot-generic-crop-mapping-plan-2026-09-21.md`). This app
  will 404 against any backend older than `b3f928d`.
  - Routes: `/plates` and its 9 sub-routes (`cluster`,
    `cluster/status`, `clusters`, `clusters/refine/{id}`,
    `fp_centroids/build`, `fp_centroids/status`,
    `suspected_false_positives`, `training_candidates`,
    `batch_status`) all move to `/regions`. `/crops/{id}/plate` →
    `/crops/{id}/region`, `/crops/{id}/plate_meta` →
    `/crops/{id}/region_meta`.
  - The `/review` slot-tab endpoint id `plates` → `regions` (the
    bookmark `?tab=plates` URL itself is unaffected — that's a separate,
    permanently-frozen contract, see `reviewTabs.ts`).
  - Status values `no_plate_box`/`no_plate_visible` →
    `no_region_box`/`no_region_visible`. The old values are still
    accepted on read forever via `SlotState.aliases` (added in Wave 0,
    C1) — no OpenSearch reindex required.
  - Training-cohort `?mode=` values `lpr_blind_spots`/
    `lpr_low_conf_correct` → `detector_blind_spots`/`low_conf_correct`.
  - Every `plate_*` OpenSearch **document field name**
    (`plate_bbox_norm`, `plate_status`, `plate_detector`, …), the
    `plates` key in `GET /stats/dataset`'s response, and
    `plate_thumbnail_url`'s JSON key are explicitly UNCHANGED — jointly
    agreed WONTFIX with the backend team (a 347k-document reindex was
    not worth it).
  - `runCohortQuery` (`/train`) now dispatches on a compiled cohort
    query's final path segment (`cohortEndpointKind()`,
    `src/lib/annotations/cohorts.ts`) instead of a
    `path === '/plates/training_candidates'` string-equality check,
    which would have silently returned an empty training-cohort
    preview against the renamed backend.
  - Deleted `src/lib/plateStatus.ts` (an auto-generated, zero-importer
    file superseded by `licensePlateSlot.capabilities.lifecycle.states`)
    rather than renaming its now-doubly-dead members.

### Fixed

- `sse.ts`'s `subscribeKbEvents` used to hardcode a fixed list of known
  SSE event types; any type outside that list was silently never
  dispatched. A second queue-capable slot's own verify event would have
  refreshed nothing on `/review`, with no error anywhere. Event types
  are now derived from `slotRegistry.queues` at call time.
- `plateGalleryController.svelte.ts`'s `savePlateBbox` (the plate
  gallery's bbox-editor save handler) was re-sending the box to the
  backend a second time on every save, even though `SlotBboxEditor` had
  already performed the write — a redundant PUT on every plate-gallery
  bbox save. It is now a pure local-state patch.

- `/bakeoff` is now gated on backend availability instead of assuming
  `/curation/bakeoff/*` is always mounted. A new one-shot probe
  (`src/lib/bakeoffAvailability.svelte.ts`) calls the idempotent
  `GET {API_PREFIX}/bakeoff/runs`: a 404/501 hides the nav link and
  swaps the page body for a calm "not available" note; any other
  failure (network error, 5xx, abort) leaves the route visible, since a
  transient outage must not look like an absent capability. This was the
  last backend-optional surface in the app with no availability gate at
  all — the nav link rendered unconditionally and the page fired four
  GETs on every mount regardless of whether the router existed. The
  module is explicitly provisional and documents its own replacement:
  once the backend ships an `evaluation` axis on `GET {API_PREFIX}/methods`
  (mirroring the existing `export` axis), this probe is deleted in favor
  of the same capability-discovery pattern every other gate in the app
  already uses. See `docs/design/bakeoff-train-genericization-plan-2026-09-21.md`.

### Changed

- `/train`'s single-class dataset-export panel now renders its
  remaining copy (heading, dataset-kind toggle label, the `current`
  symlink name, the description blurb, the "no export yet" message, the
  build button, and the success/failure toasts) from the active slot's
  already-resolved `datasetExportSpec` instead of nine hand-written
  LPR-specific strings. `spec.blurb` — declared on the profile and
  validated by `datasetExportForSlot`, but rendered nowhere until now —
  is the one genuinely dead config value this fixes. `exportLpr`/
  `exportLprStatus`, the `/export/lpr` wire path, `spec.options[]`, and
  `OpLprExportResponse` are all unchanged; this is a copy-only pass, not
  a new capability.
- The four production comments asserting `cluster_id == class_id` as a
  "legacy ensemble" or "legacy convention" (`src/routes/+layout.svelte`,
  `src/routes/clusters/[id]/+page.svelte`, `src/routes/clusters/+page.svelte`)
  now name the invariant's real, backend-contractual scope
  (`cluster_kind === 'class'`, per `ClusterKind` in `src/lib/types.ts`)
  and cite in-repo sources instead of a private backend filename
  (`legacy_ingest.py`) a public reader can't open. No logic change —
  the invariant was already correct, just mis-attributed and
  under-qualified. One unrelated stray reference to "the legacy_sorter
  UX" is reworded to "the sorter-app UX" in the same pass.

### Fixed

- `src/routes/train/datasetExportGate.test.ts`'s highest-value
  assertion referenced the dead identifier `refreshLprStatus` (renamed to
  `refreshSingleClassExportStatus` in an earlier pass), making the test
  unfailable — it could never have caught the "never 404 unconditionally
  on mount" regression it exists to guard. Renamed to the live
  identifier; verified it fails when the regression is reintroduced.

- A new `/settings` page gives deployment operators one place to set the
  shared, backend-side defaults for two curation strategies — the
  clustering method used by every auto-label run this app starts, and the
  review-queue sort applied to every `/review` tab that does not request
  its own — persisted via `GET,PUT {API_PREFIX}/settings`. This is stored
  backend-side rather than per-browser because the product has no user
  accounts: one operator's pick is every operator's pick, on every
  session, until changed again. The page also lists two more axes the
  backend advertises but does not yet act on, `detection_profile` and
  `prompt_pack`, as a read-only "Advertised but not yet wired" section —
  no dropdown, no Save button — because no backend request path reads a
  shared default for either one yet, and offering a control that silently
  does nothing would be worse than not offering one. Against a backend
  that predates this endpoint, the page degrades to an explicit "not
  supported" note with no controls rendered, rather than erroring or
  guessing.
- The `/settings` page now has a **Clear** control next to **Save** for
  every settable axis, sending `PUT {defaults: {[axis]: null}}` to
  remove that axis's pinned override entirely — every caller falls back
  to its own built-in default afterward. This closes the backend gap
  (H-1) noted when the page first shipped: once a shared review-sort
  default was pinned, there was no way to un-pin it through this API at
  all, since a `PUT` had to name a currently-advertised id and "each tab
  uses its own default" wasn't one. The backend added a `null`-clears
  contract for exactly this; every save and every clear still goes
  through the same explicit confirm dialog, since both remain
  deployment-wide, no-undo-visible writes. Live-verified against a real
  backend: pinned a sort default out of band, cleared it through the UI,
  confirmed `GET /settings` reflects the clear.
- Deployment operators can now register their own annotation slot —
  without forking the repo or touching a single line of application
  code — by dropping an `annotation-profiles.json` file next to the
  built app: in `static/` before a build, or bind-mounted over
  `/usr/share/nginx/html/annotation-profiles.json` in a running
  container. The file is fetched once in the root layout's `load()` and
  validated by a new hardened parser
  (`src/lib/annotations/config/parseSlotConfig.ts`) against the schema
  already published in `docs/annotation-slots-contract-draft.md`,
  treating the file as untrusted operator input rather than reviewed
  source: template paths are prefix-relative only and their placeholders
  are drawn from a closed allow-list, regex patterns reject the `g`/`y`
  flags and a conservative ReDoS shape, Tailwind ring classes must come
  from a pre-declared preset or allow-list (a runtime-mounted class
  string is invisible to Tailwind's JIT scan regardless), keymaps cannot
  claim a hotkey the review page already owns, and any object carrying a
  `__proto__`/`constructor`/`prototype` key is rejected outright. A
  missing or malformed file degrades silently to this deployment's
  built-in `license_plate` slot — a console warning plus one toast
  surface a broken config, but the app never crashes and the currently
  running legacy deployment's behavior is completely unchanged, since
  it ships no live `annotation-profiles.json`. A worked example
  (`static/annotation-profiles.example.json`, a pallet-shipping-label
  slot) is shipped and covered by an integration test proving it renders
  a real review tab with zero further code changes. Tier 3
  (server-declared slots) remains deliberately unbuilt — this closes
  steps 1 and 2 of the sequencing `docs/annotation-slots-contract-draft.md`
  §9 already committed to.
- The dashboard's auto-label run can be scoped to a single class — "just
  help me with pallets right now" — instead of always sweeping the whole
  pool. A collapsed-by-default `<AssistScopeBar>` on `/dashboard` picks
  one class (fuzzy-searched through the same `searchClasses` ranking
  `/review`'s class picker uses) and contributes `class_id` to
  `POST {API_PREFIX}/pipeline/auto_label/start`; when the backend
  advertises them, it also offers a detection-profile and a prompt-pack
  selector, from two new `/methods` axes (`detection_profile`,
  `prompt_pack`) parsed into `detection_profiles`/`prompt_packs` and
  gated by `isDetectionProfileAvailable`/`isPromptPackAvailable` with the
  same stable/experimental-only bar as every other axis. Leaving the bar
  alone is byte-identical to the previous one-click run: `toStartParams()`
  returns `{}` and the composed URL is unchanged. The whole bar is absent
  — not disabled — unless `/methods` advertises at least one assist axis,
  because `class_id` has no capability signal of its own and an unknown
  query param is silently dropped server-side, which on an hours-long run
  would mean an unscoped sweep while the UI claimed otherwise. Built
  against a contract agreed with the backend session but not yet landed
  there; verified statically, against both `/methods` fixtures, and in a
  route-stubbed browser (`scripts/playwright_assist_scope.py`). A live
  integration pass is still owed once the backend ships its half.
- `docs/annotation-slots-contract-draft.md` — the P4.2 wire contract draft
  for server-declared annotation slots (H3 opening offer; proposal only,
  nothing implemented on either side).
- `docs/FEATURES.md` — a full visual feature tour (screenshot + explanation
  for every route), and a `docs/screenshots/demo.gif` slideshow now leading
  the README instead of a static image grid.
- `docs/README.md` — an index distinguishing current/maintained docs from
  historical/reference ones (the 2026-09-11 audit, the curation-strategy
  design doc, the two market-research docs).
- `src/lib/annotations/` — additive foundation for genericizing the
  `license_plate` vertical into a reusable "annotation slot" mechanism
  (`docs/genericization-plan-2026-09-13.md`, Phase 1): the `SlotSpec`
  capability model (`subBox`/`text`/`provenance`/`lifecycle`/`queue`),
  a field-mapping adapter (`readSlot`) that reads whichever wire field
  names a slot declares, a merge-by-replace `resolveSlotRegistry`, the
  legacy `license_plate` profile decomposing today's ~30 `plate_*`
  fields, and a config-driven detector label/palette registry proven
  equivalent to `DetectorChip.svelte`'s hand-written switch/if-chain via
  a 23-case snapshot test. Not yet wired into any route or component —
  this is the additive Phase 1 slice; `mapRawCrop`/`DetectorChip`/
  `PlateCard`/`/review`/`/clusters` migrations (Phase 2) are follow-up
  work.
- Two example slot profiles (`aircraft_tail_number`, `defect_code`,
  under `src/lib/annotations/profiles/`) plus a falsification test
  proving the capability model above handles a disjoint capability
  subset (no sub-bbox at all, for `defect_code`), a different stored
  bbox frame and shape envelope (`aircraft_tail_number`), and a
  closed-vocabulary text field — with zero changes to `types.ts`,
  `registry.ts`, or `readSlot.ts` beyond the model, and zero
  special-casing of either example outside `profiles/`. Neither example
  is bound to a real route or class; they are proof-of-concept configs
  only.
- `src/routes/review/plateReviewCharacterization.test.ts` — Phase 0
  characterization tests (`docs/genericization-plan-2026-09-13.md`
  §5.1) pinning today's Plates-tab behavior in `review/+page.svelte`
  before any Phase 2 refactor touches it: the scan-mode keymap, the
  class-drop tab guard invariant behind Finding C.2, the per-crop
  (not per-cursor) save-abort map, the `$state.raw` undo-stack identity
  semantics, the frozen-viewport `untrack()` seed read, and today's
  closed `REVIEW_TABS`/`license_plate`-literal baseline in
  `clusters/+page.svelte`. Source-scan style (no `@testing-library/svelte`
  harness exists in this repo) rather than the plan's preferred
  extract-then-test approach — see the file's doc comment for why.
- `src/lib/review/slotQueueOps.ts` and `src/lib/review/abortRegistry.ts`
  — the real P0.1/P0.3 extraction of the Plates-tab's undo-stack and
  per-crop abort-map logic out of `review/+page.svelte`, with executable
  unit tests. `review/+page.svelte` now delegates to both; behavior is
  unchanged.

### Changed

- `/train/+page.svelte`'s own internal state for the single-class dataset
  export panel is renamed off vehicle/lpr-specific naming: `datasetKind`
  is now typed `string` instead of the hardcoded `'vehicles' | 'lpr'`
  union (removing a `datasetExportSpec.datasetKind as 'lpr'` cast that
  forced a generically-typed spec value back into a hardcoded literal),
  and the LPR-export local state/handlers (`lprExportDir`, `lprExporting`,
  `lprMessage`, `lprImageMode`, `lprImgSize`, `lprDedup`,
  `lprMaxPositives`, `refreshLprStatus`, `runLprExport`) are renamed to
  `singleClassExportDir`/`singleClassExporting`/`singleClassExportMessage`/
  `singleClassImageMode`/`singleClassImgSize`/`singleClassDedup`/
  `singleClassMaxPositives`/`refreshSingleClassExportStatus`/
  `runSingleClassExport`. Pure identifier/type rename with the
  capability-gating mechanism (`extras.datasetExport`, the `/methods`
  `export` axis check) completely unchanged; `TrainForm.svelte` already
  used the generic `singleClassExport` prop name from T-C3 and needed no
  further change. The `exportLpr`/`exportLprStatus` API functions and the
  `/export/lpr` wire path are untouched — they name the one dataset
  export this deployment's backend actually implements today, not a
  hardcoded assumption on Cropwright's side.
- `/train`'s LPR export panel is now gated on server capability instead
  of being hardcoded (P2.15). `GET {API_PREFIX}/methods` grew an `export`
  axis listing the dataset-export kinds a deployment can actually produce;
  `strategies.ts` parses it into a new `dataset_exports` bucket and
  `isDatasetExportAvailable()` gates on it with the same
  stable/experimental-only bar as the `diverse`/`viz_projection`/
  `semantic_search` overlays. When the kind is absent — which is the
  case on OpenProcessor, where the proprietary LPR exporter was never
  ported — the panel and its dataset-kind toggle are _absent_, not
  disabled, and `GET {API_PREFIX}/export/lpr/status` is never requested.
  Capability is never probed by calling the export endpoint and reading
  the 404: that endpoint is a write that kicks off a real dataset build.
  The panel now reads its kind/label/paths from the `license_plate`
  profile's `extras.datasetExport` (finally consuming what P2.13 declared)
  and `TrainForm`'s `lpr` prop is renamed `singleClassExport`. On a
  `/methods` failure the fallback advertises no export kinds at all, so
  the optional panel stays hidden rather than rendering a button that
  404s. Rendering `extras.datasetExport.options[]` as a generic form
  (rather than the four typed bound controls `/train` still uses) remains
  follow-up work.
- Every backend URL is now composed from a single exported `API_PREFIX`
  (`src/lib/api.ts`) instead of 92 hardcoded `/curation/…` literals across
  `api.ts`, `sse.ts` and `/export`. Default is transitionally `/curation`, so
  traffic is unchanged; the prefix is now settable at build time via
  `PUBLIC_API_PREFIX` and at container start via the `__API_PREFIX__`
  entrypoint substitution (bundle **and** `nginx.conf`'s proxy `location`).
- **The `genericization-wip` line of work landed on `master`** (2026-09-20,
  merge of 26 commits implementing Phases 0-4 of
  `docs/genericization-plan-2026-09-13.md`; see
  `docs/design/genericization-wip-merge-plan-2026-09-20.md` for the merge
  record). Everything under this heading and the Added/Fixed entries below
  that reference `src/lib/annotations/` arrived together in that merge. No
  backend change was required for any of it — the whole mechanism reads
  through the frontend-side field-mapping adapter (`readSlot`), so
  openprocessor's wire format is untouched.
- Rebranded the app from "legacy Labeler" to **Cropwright** (Track N1 of
  `docs/genericization-plan-2026-09-13.md`, cosmetic-only — no behavior
  change): `package.json` name → `cropwright`, browser tab title, top-bar
  wordmark/badge (now `Cropwright` / `CW`, and env-configurable via
  `PUBLIC_APP_NAME` / `PUBLIC_APP_BADGE`, same convention as
  `PUBLIC_TRITON_API_URL`), README framing, and `CLAUDE.md`'s project
  description. Also corrected `CLAUDE.md`'s component references, which
  still named the pre-genericization `DetectorChip.svelte`/`PlateCard.svelte`
  — the actual current files are `ProvenanceChip.svelte`/`SlotCard.svelte`.
- `PlateBboxCanvas.svelte` renamed to `BboxCanvas.svelte` and
  `plate_geometry.ts` renamed to `bboxFrames.ts` (P2.1,
  `docs/genericization-plan-2026-09-13.md` §3.1) — both were already
  fully generic (no plate-specific logic), so this is a pure rename:
  `vehicleBbox` params renamed to `parentBbox`, and `BboxCanvas` gains a
  `ringColor` prop (default unchanged, sky-blue matching the
  server-rendered plate overlay) so a future non-plate slot can use its
  own ring color. No behavior change; verified live (edit-bbox mode on
  `/review`'s Plates tab renders and drags correctly).
- `DetectorChip.svelte` renamed to `ProvenanceChip.svelte` (P2.2) —
  its hand-written 20-arm label `switch` and 10-branch palette
  if-chain are deleted in favor of the config-driven
  `legacyDetectorRegistry` (`src/lib/annotations/`) built in Phase 1,
  proven equivalent by a 23-case snapshot test before the old functions
  were removed. All 4 consumer files updated. No behavior change;
  verified live against the real backend — LPR/Gemma/⚠-shape chips
  render with identical colors to before the migration.
- `PlateEditor.svelte` renamed to `SlotBboxEditor.svelte` and
  `PlateCard.svelte` renamed to `SlotCard.svelte` (P2.3/P2.4). Both are
  renames only — `SlotBboxEditor` keeps its direct `setCropPlate` call
  rather than the plan's ideal injected-`onsave`-performs-the-write
  contract (that changes both call sites' behavior and was judged out
  of scope for this pass), and `SlotCard` keeps reading
  `PlateBrowseItem`'s hardcoded `plate_*` fields rather than a generic
  `slots[key]` lookup (that needs `getPlates`/`PlateBrowseItem` routed
  through the slot adapter, not done yet). Both deviations are
  documented in the files' doc comments as follow-up work.
  `plateThumbUrlScan.test.ts`'s hardcoded `PlateCard.svelte` path
  (flagged by the plan as a rename hazard) updated in the same commit.
  No behavior change; verified live — the `/clusters?class=license_plate`
  SlotCard gallery and the SlotBboxEditor edit-bbox modal both render
  and interact identically to before.
- `src/lib/review/plateKeymap.ts`, `slotTabGuard.ts`, and `viewBox.ts` —
  the real Phase 0 seams for the pieces P2.8 (`reviewTabs.ts`
  data-driving) most directly needs, extracted from
  `review/+page.svelte`'s inline keymap `$effect`, the class-drop tab
  guard, and the frozen-viewport zoom math, each with its own unit-test
  suite. `review/+page.svelte` now delegates to all three; no behavior
  change. Verified live: Skip (`n`), entering edit mode (`e`), and
  canceling edit (`Escape`) all still work through the extracted
  keymap table, and the frozen zoom still renders correctly in edit
  mode.
- `src/lib/components/slots/SlotGallery.svelte` +
  `src/routes/clusters/plateGalleryController.svelte.ts` (P2.6) — the
  ~660-line plate-gallery view (plate pager, multi-select, the AHC
  secondary-clustering sub-system: bucket grid, sub-cluster refine, FP
  centroid build, suspected-FP triage, and the bbox-editor modal)
  extracted verbatim out of `clusters/+page.svelte`'s
  `{:else if isLicensePlateFilter}` branch. The ~30-item state/function
  surface now lives in a controller factory (following this codebase's
  existing `createPager`/`createSelection` convention rather than
  prop-drilling 30 individual bindables), and the route passes one
  `gallery` object to the new component. Verbatim move only — no
  parameterization by slot yet (P2.7). No behavior change; verified
  live against the real backend: bucket grid → sub-cluster drill-down →
  suspected-FP view all render correctly (real, non-mutating reads),
  and the three mutating actions (Cluster plates, Build FP centroids,
  plate edit + Save) were confirmed to dispatch the correct
  request/method/body to the correct endpoint via intercepted routes,
  without starting a real multi-minute background job or writing to
  the live 347k-crop index.
- `SlotCard.svelte` parameterized by slot (P2.7) — now calls `readSlot()`
  on the raw crop and renders through the resulting `SlotData`
  (`text`/`subBox`/`provenance`/`lifecycle`) instead of `PlateBrowseItem`'s
  hardcoded `plate_*` fields, with a `slot: SlotSpec` prop (defaults to
  the legacy `license_plate` profile). `PlateBrowseItem`'s flat
  `plate_*` properties already match the profile's wire-field names, so
  this needed no change to `getPlates`/`api.ts`. No behavior change;
  verified live — the `/clusters?class=license_plate` gallery renders
  pixel-identically (detector chips, scores, plate text, shape warnings,
  false-positive badges) through the new adapter-driven read path.
- `src/lib/reviewTabs.ts` data-driving (P2.8) — `REVIEW_TABS` is now
  `CORE_REVIEW_TABS` (the 4 core cohorts) plus `buildReviewTabs(slots)`,
  which derives a tab from each queue-capable slot's `QueueCapability`
  instead of a hand-maintained `{ id: 'plates', label: 'Plates' }`
  literal — a new deployment configuring a second queue-capable slot
  gets a real `/review` tab with zero `reviewTabs.ts` edits. Adds
  `isSlotTab`/`endpointForTab` per the plan's §3.3. Scoped down from the
  plan's full design: the internal tab id stays `'plates'` (not
  `slot:license_plate`) since widening it would require touching 11+
  `tab === 'plates'` call sites plus `getReviewQueue`'s and
  `selectDiverse`'s tab-forwarding params, for zero behavior gain while
  only one queue-capable slot exists — tracked as real, separate
  follow-up work rather than bundled in under time pressure.
  `review/+page.svelte`'s `PLATE_STATUS_OPTIONS` (Finding C.4's second
  hand-copy of the human-writable status whitelist) is now derived from
  `licensePlateSlot.capabilities.lifecycle.states` instead of a
  hand-copied array — same 4 values, minor reordering (now matches the
  profile's state order rather than the old array's hand-picked order).
  No other behavior change; verified live — the Plates tab still
  activates correctly via nav click, and the status dropdown shows the
  correct 4 derived options.
- P2.10: `src/lib/annotations/registeredSlots.ts` — the one deployment-
  config file listing which slots are actually live in this app (today
  just `[licensePlateSlot]`). Every remaining hardcoded
  `class === 'license_plate'` / `'license_plate'` string comparison
  outside `profiles/` now goes through `slotForClassName()`/
  `registeredSlots` instead: `clusters/+page.svelte`'s
  `isLicensePlateFilter`, `loadLicensePlateCard`, the synthetic pinned
  card's `dominant_class_name` (now `lp.name`, not a literal), and its
  click-through routing; `+layout.svelte`'s sidebar-click routing;
  `reviewTabs.ts`'s `REVIEW_TABS` (built from `buildReviewTabs(registeredSlots)`
  instead of a hardcoded `[licensePlateSlot]` array). This closes the
  plan's Phase 3 ship-gate to its honest form: registering a new live
  slot (a real domain becoming pluggable via config, not route-code
  edits) is exactly "add one entry to `registeredSlots.ts`" — verified
  by temporarily adding `aircraft_tail_number` and `defect_code` to that
  array, confirming `REVIEW_TABS` and `slotForClassName` picked them up
  with zero other file changes (check/build green), then reverting the
  registry back to just `licensePlateSlot` (those two profiles stay as
  proof-of-genericity in `profiles/` + `profiles.falsification.test.ts`,
  not enabled in the live app — enabling either for real would also
  need backend wire-field/endpoint support this deployment doesn't
  have). `registeredSlots.ts` living outside `profiles/` is expected and
  correct — it's the one-line-per-slot registration point the plan's
  "zero diffs outside profiles/" bar was always going to need; the
  honest gate is "zero diffs outside `profiles/` + `registeredSlots.ts`".
  `plateReviewCharacterization.test.ts`'s pinned pre-migration
  characterization test (asserting `clusters/+page.svelte` still
  hardcoded >= 4 `'license_plate'` literals) was updated in place to its
  post-migration form (asserting 0 such literals and a
  `slotForClassName(` call site), per that test's own documented intent
  to flip red — not be deleted — the moment this migration landed.
  Verified live: `/clusters` sidebar click on `license_plate` still
  routes to `/clusters?class={id}` and renders the plate gallery; the
  `/review` Plates tab still renders identically; `npm run
check`/`test`/`lint`/`build` all green.

- `scripts/playwright_backend_integration.py` — the manual write-path
  integration runbook (12 steps from the cross-repo plan §5.2:
  navigation, thumbnail render, single + batch label, cluster refine,
  review dismiss, region write/restore, batch region status, VLM label
  batch, both SSE channels, export gating, train read path),
  parameterized on frontend origin / API origin / `API_PREFIX` with a
  `--dry-run` mode that proves URL composition at both prefixes without
  a backend. **Not run end to end** — no OpenProcessor instance is
  available yet; D3-RUN remains open.
- `src/lib/apiPrefixScan.test.ts` — CI ratchet: no `.ts`/`.svelte` file
  under `src/` may compose a backend URL from a bare `/curation` or
  `/curation` literal (comment-stripped scan, one pinned exception:
  `normalizeApiPrefix`'s own fallback), and every `apiFetch` path plus
  every `${apiBase}` template builder must start with `${API_PREFIX}`
  — the second guard being prefix-name-agnostic, so it survives the
  `/curation` flip unchanged.

### Fixed

- `/review`'s keyboard-shortcut hint strip hardcoded the reject/"no
  {label} visible" glyph to the literal `"D"` for every slot, instead of
  reading the active slot's own `capabilities.queue.keymap.reject` —
  the same lookup `buildSlotKeymap()` already used correctly for real
  key dispatch, so pressing the actual bound key always worked even
  though the on-screen hint could lie. New `rejectKeyGlyph()`
  (`src/lib/review/slotKeymap.ts`) resolves the displayed glyph from
  that one keymap declaration, so a slot bound to something other than
  `d` shows its real key. Behavior-neutral for `licensePlateSlot`
  (still shows "D").
- `scripts/playwright_round_trip.py` — composes its four backend URLs
  through a configurable `--api-prefix` instead of a hardcoded `/curation`;
  picks its target cluster via `GET {prefix}/clusters` rather than the
  legacy `GET /clusters/stats/op_vehicles`, which rejects that index
  name and is no longer proxied by the labeler's nginx; and drops an
  absolute repo path plus a `DISPLAY=:11 source …` invocation that was
  never valid shell.
- Region thumbnails no longer 404. `getPlateThumbUrl` is renamed
  `getRegionThumbUrl` and builds `{API_PREFIX}/crops/{id}/region_thumbnail`
  — the only region-thumbnail route OpenProcessor registers. The old
  `plate_thumbnail` segment had no route on either side and no alias will
  ever be added (`cropwright_backend_integration_plan.md` §3.2), so every
  client-built region thumbnail was dead: the `SlotCard` fallback, the
  `/clusters` synthetic plate card's four tiles, and — most visibly — the
  cache-busted URL the plate gallery substitutes after every bbox save.
  The backend's matching fix (T-A1) makes the _server-supplied_
  `plate_thumbnail_url` value correct; this is the client-built half.
  The JSON key `plate_thumbnail_url` is frozen wire contract and is
  deliberately unchanged — only the path inside its value is generic.
  `licensePlate.ts`'s parallel (as-yet-unconsumed) slot-profile path
  template is fixed in the same pass, and `plateThumbUrlScan.test.ts`'s
  guard now matches both segments so a revert to the dead one is caught
  anywhere in `src/`.
- Cluster VLM labeling called `/gemma/label_batch`, a route the backend
  does not register; it is `{prefix}/vlm/label_batch`. The `gemma_*`
  payload fields and the `gemma_low_conf` review-tab id are unchanged —
  only the URL path segment moved.
- `CropCard`'s plate ring color rendered green ("human confirmed") for
  every machine-detected plate, not just human-verified ones. The
  predicate checked `plate_status === 'human_confirmed' ||
plate_status === 'detected'` — `'human_confirmed'` is not a value the
  backend can ever produce (openprocessor's `PlateStatus` has 8 members,
  none of them that), so that branch was dead, and `'detected'` (written
  for every machine detection, verified or not) matched the second
  branch. The correct predicate is the boolean `plate_verified` field,
  which the component now reads. **Visible behavior change:**
  machine-detected-but-unverified plates now render a yellow ring
  instead of green. `types.ts`'s `plate_status` docstring, which invented
  `human_confirmed`/`pending_verify` and omitted three real pipeline
  states, is corrected to match openprocessor's `PlateStatus`.
- Plate-bbox shape-envelope check (the ⚠ "implausible shape" warning) had
  two independent implementations that disagreed on corrupt/non-finite
  input: `api.ts`'s `_platesShapeWarning` guarded with `Number.isFinite`
  and warned; `PlateCard.svelte`'s inline `shapeWarning()` had no such
  guard, so a `NaN` box compared `false` against every bound and silently
  reported no warning. A corrupt row could show ⚠ on `/review` but not on
  `/clusters`. Both call sites now share one `evaluateShapeGate()`
  (`src/lib/shapeGate.ts`), so a corrupt row warns consistently everywhere.
  The `/review` tooltip's hardcoded envelope sentence is also now generated
  from the same numbers instead of being a third hand-copy.
- `GET /curation/clusters` and `/curation/clusters/representatives` (openprocessor) were
  hardcoded to the pre-cutover `op_vehicle_crops` index, which was left
  empty after the 2026-09-12 kNN reindex — this silently broke the
  `/clusters` and `/clusters/[id]` pages entirely (always zero clusters)
  despite `/review` and other pages working fine against the same data.
  Fixed on the openprocessor side to use `OP_VEHICLE_CROPS_INDEX` like every
  other endpoint; verified live (589 clusters return correctly).

### Removed

- The dead `location ~ ^/clusters/(train|assign|stats)/` nginx proxy
  block. No frontend code has called those legacy FAISS endpoints
  through the labeler's nginx.
- `docs/audit-2026-09-11/` (old audit screenshot set, superseded by
  `docs/screenshots/` + `docs/FEATURES.md`) and two untracked stray
  directories (`frontend/`, empty; `diagnostics/plate_bbox_audit/`, old
  ad-hoc scratch output) — none were referenced from any doc.

- Curation-strategy selector bar (`StrategyBar.svelte`) across `/clusters`,
  `/clusters/[id]`, and `/review` — cluster-method picker, review-sort
  dropdown, score chips, diverse overlay, and an embedding-plot toggle.
- Embedding-plot lasso-select-and-act tool on `/clusters`: 2-d UMAP
  visualization (never feeds clustering), lasso-select crops, bulk
  assign/move directly from the plot. Selected-crop preview thumbnails
  render side-by-side with the plot, each clickable to a full-size view.
- Fuzzy-search class picker (`/` key) on `/review`, reaching every
  non-deprecated class instead of only the top-10 quick-assign row.
- Single-GPU-2 picker on `/train`'s GPU selector, and an unload action for
  promoted Triton models on `/models`.
- `/export` can now download the frozen export's `class_registry.json`,
  `data.yaml`, and `manifest.json` directly.
- Semantic text search over vehicle crops (P2-14), backed by the PE-Core
  text/image encoder and a real OpenSearch kNN index: a search box on
  `/clusters/[id]` (scoped to that cluster), `/review` (scoped to the
  active tab), and `/clusters` (fully global, unscoped across the whole
  dataset) — each result carries a similarity-score badge, and global
  results additionally show a cluster-origin badge (`#id · dominant
class`) so an operator can see where a crop lives before relabeling
  it. Results feed through the existing `CropCard` grid, so labeling/
  drag-drop/bulk-select all work unchanged on search results.
- Diverse-selection overlay (k-center-greedy core-set) exposed on
  `/review`, reusing the pattern already shipped on `/clusters/[id]`
  (pool-scale job with 200/202 polling for large cohorts).
- Back-to-all-clusters link on `/clusters/[id]`'s header.

### Changed

- A shared three-tier control-sizing scale (`.btn`/`.select`/`.input`
  at 32px, `.btn-sm`/`.select-sm`/`.input-sm` at 26px, `.chip` at
  20px), driven by CSS custom properties, replacing ad-hoc per-page
  Tailwind sizing strings that had drifted into 7 different control
  heights app-wide with no shared scale.
- Dropdown-caret glyphs (the Ignore-reason toggle, StrategyBar's
  collapsed-chip indicator) are now real SVG icons instead of a plain
  unicode "▾", which rendered inconsistently thin/small across
  browsers.
- Semantic search result page size raised to the API's max (200,
  up from 30-60) across all three integrations — the search box
  fetches a single page with no load-more, so page size was the
  effective result cap.

- `/review` consolidated from 9 top-level tabs to 5 (All, Uncertainty,
  Model Disagreements, COCO Blind Spots, Plates), with the three
  removed tabs (Mismatches, Gemma Low-Conf, Primary · Low-Conf) moved to
  quick-filter chips on the All tab. Outliers retired in favor of the
  `atypicality` sort, which covers the same need correctly.
- Rank-scope (Largest/+2nd) and clarity-slider controls on `/review` are
  now available on every tab and preset, not just two of them.

### Fixed

- Drag-and-drop on `/clusters/[id]` no longer resurrects a crop that was
  just successfully moved — the grid is now derived live from the crop
  list instead of a manually-rebuilt snapshot that could go stale.
  Relatedly, a shared pager race could resurrect a stale page after
  Refine (AHC); now guarded with a monotonic epoch counter.
- Toast notifications (and therefore several optimistic-UI actions) no
  longer fail when accessed over plain HTTP via a LAN IP, where
  `crypto.randomUUID` is unavailable (secure-context-only browser API) —
  this previously caused a successful backend action to look like it
  failed and get reverted client-side.
- Crop thumbnails are no longer natively draggable in Safari, which could
  hijack a drag gesture before the app's own pointer-based drag library
  saw it.
- Plate thumbnail URLs now resolve against the configured API base
  instead of a bare relative path — fixes 404s whenever the labeler
  points at a openprocessor instance on a different host.
- Removed a stale `/export` panel referencing a shell script that no
  longer exists; fixed the `data.yaml` download requesting the wrong
  filename.
- Deprecated classes (and, briefly, `license_plate`) excluded from the
  sidebar/hotkey dispatch — the `license_plate` exclusion was reverted
  the same day it shipped, since it should remain a normal, reachable
  class.
- Semantic search's similarity-score badge was rendering 0% on every
  page it shipped on: the frontend read `similarity_score`/`score` but
  the live backend sends `semantic_score`. Fixed the field precedence.
- The confirm-class `<select>` on `/clusters/[id]`'s bulk-action row,
  and every other ad-hoc toolbar `<select>`/button in the app, had
  drifted into 7 different heights with no shared sizing convention —
  root-caused across four passes to (a) a shared control not matching
  its own box model, (b) an inner flex div silently letting an
  11-control row overflow off-screen at realistic viewport widths with
  no scrollbar, and (c) `.btn`/`.select` being defined unlayered in
  `app.css`, which silently defeated every per-instance Tailwind
  override (e.g. the Ignore-dropdown caret's `rounded-l-none`/`px-1.5`
  did nothing — fixed by wrapping the shared classes in `@layer
components` and introducing the three-tier scale above).
- The cluster-detail page (`/clusters/[id]`) showed no human-readable
  class name in its header for any "candidate" (unlabeled) cluster —
  the lookup used `class_id` as a stand-in for the cluster's own
  identity, which only coincidentally works for "class" clusters
  (`cluster_id == class_id` by design); every candidate cluster has no
  matching class at all, so the lookup silently returned nothing. Now
  uses a real `cluster_id` filter (new backend query param) that works
  regardless of cluster kind.

### Changed

- Class adequacy, thresholds and hotkeys now come entirely from the
  backend (`docs/design/logic-moves-adoption-plan-2026-09-24.md` W4,
  OpenProcessor `main` @ `d037be8`):
  - `adequacy.ts` renders whichever tier (`block`/`warn`/`ok`) the
    server puts on each class (`GET /classes`, `GET /stats/classes`),
    instead of recomputing it from `validated_count` against hardcoded
    500/100 thresholds. The old `ADEQUACY_OK`/`ADEQUACY_LOW` constants
    are gone.
  - `/export`'s dataset table reads the server's `aug_target`/`aug_gap`
    per class and the server's per-class `deficient` flag on
    `GET /test_holdout/stats` (falling back to comparing against the
    served `min_test_per_class` only when a bucket omits the flag) —
    the client-side `clamp(validated, 500, 3000)` augmentation target
    and the hardcoded "< 5 test crops" minimum are gone. The row-building
    logic moved to a pure `src/lib/export/exportDatasetRows.ts` module.
  - `classHotkey.ts`'s `reservedHotkeyLetters()` now unions
    `classesStore.reservedHotkeys` (the server's own `reserved_hotkeys`
    field) with any registered slot's own keymap letters, instead of a
    hardcoded `RESERVED_HOTKEY_LETTERS` constant unioned with the same
    slot letters. Verified live: the server's set (`/abdefgmnuxz`)
    already matches what the old constant + union produced.
    `setClassHotkey` now shows the server's 400/409/422 detail text
    verbatim on a rejected bind.
  - `/classes` and `AddClassModal` no longer validate the class-name
    slug pattern client-side — `POST`/`PUT /classes` 422s with the
    `^[a-z0-9_]+$` pattern in its detail, which now renders inline.
  - `errorDetail()` (`api.ts`) now also extracts the `msg` field from a
    FastAPI/Pydantic validation-error array (`{detail: [{msg, ...}]}`),
    not just a plain string `detail` — this is what makes the class-name
    422 above legible instead of falling through to "API 422 ...".
  - `/classes` shows a warning banner listing any class whose bound
    hotkey has since become reserved. Live on this deployment: `bmw` is
    bound to `b`, which the backend now reserves for the `license_plate`
    slot's keymap — the binding is kept (matches the backend), not
    auto-cleared.
  - The merge dialog on `/classes` calls
    `POST /classes/merge?dry_run=true` (new `previewClassMerge()`)
    whenever the source/target selection changes, and shows
    `would_relabel`/`would_unvalidate`/`holdout_blocking` before the
    real merge. Confirm is disabled when the dry run reports `blocked`.
  - `GET /classes` now returns `{classes, thresholds, reserved_hotkeys}`;
    `getClasses()`'s return type changed from `RegistryClass[]` to
    `ClassesResponse`, and `classesStore` gained `thresholds` and
    `reservedHotkeys` state alongside `classes`. The only caller
    (`classesStore.refresh()`) was updated; no other frontend code
    called `getClasses()` directly.
  - `RegistryClass.added_at` and the `/classes` "Added" column were
    already wired end to end — verified live, no change needed.
