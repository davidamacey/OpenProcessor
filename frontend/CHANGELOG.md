# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Adopted OpenProcessor `main` 840beb8 (rejection-reason vocabulary +
  review `filter_specs`, contracts synced from 1bea18b):
  - **Labeled rejection reasons.** `GET {API_PREFIX}/regions/vocabulary`
    gained `rejection_reasons` (`{id, label, kind, match, label_template}`)
    — `regionVocabularyStore.rejectionReasonLabel` now resolves an exact
    match first, then a longest-prefix match (`label_template`'s
    `{detail}` filled from the rest of the stored value, e.g.
    `sanity_reject:degenerate_zero_size` → "Box failed the geometry check
    (degenerate_zero_size)"), and falls back to the raw stored value
    verbatim (never titlecased) when nothing matches. The old
    client-side "Sanity check failed:" prefix and titlecase placeholder
    for rejection reasons are gone. New `rejectionReasonKind(id)` drives
    styling everywhere a rejected candidate renders (`/review`'s inline
    slot panel, `SlotCard`, `CropMetaPanel`): `model_verdict` reads red
    ("the model rejected it"), `automatic` reads amber (a geometry
    gate), and `needs_human` reads neutral zinc and is never worded as a
    rejection ("candidate · needs review" / "Needs review", not
    "rejected"/"Rejection").
  - **`region_bbox_correct`.** New `LifecycleCapability.boxCorrectField`
    (declared on `licensePlateSlot`, read by `readSlot` into
    `SlotData.lifecycle.boxCorrect`) — the verifier's own box-correctness
    verdict, `false` being the actual "model said wrong box" signal.
    Folded into the existing Validation/status row as a "model: box
    wrong" chip in `/review`'s inline panel and `CropMetaPanel`, not a
    new row.
  - **Generic served-enum review filter bar.** `GET {API_PREFIX}/review/tabs`
    entries gained `filter_specs` (`{param, kind: 'enum', label, options}`)
    — `/review` renders one `<select>` per entry generically (no
    tab/param-specific markup), sends the picked value as its own query
    param on both `GET {API_PREFIX}/review/{tab}` and its `/locate` route,
    resets on tab change, and persists in the URL like `?preset=` does.
    The Plates tab's `region_status` (all / detected only /
    verifier-rejected candidates only) is the first live spec.
    `reviewTabsVocabularyStore` gained `filterSpecsFor`.
  - **Per-item reason wording fix.** A slot-tab item's own
    `region_rejection_reason`, when present, now wins over the generic
    per-item `reason` string in `/review`'s Reason row — the backend's
    `reason` always reads "verifier rejected this candidate (…) — needs
    human review" even for a `needs_human` item with no actual verdict,
    which is wrong wording for that case. The generic `reason` still
    renders as before on core tabs (e.g. the Mismatches preset) that
    don't carry a `region_rejection_reason` at all.
- Adopted OpenProcessor `main` f7171cc (cutover/dq-region — region verdict
  integrity: null≠reject, rejected candidates kept + reviewable,
  auto-confirm split from human validation, region-text rules):
  - **Rejected candidate boxes.** `verify_rejected` items no longer carry
    `region_bbox_norm` at all — the box lives in
    `region_candidate_bbox_norm`/`_score`/`_detector`/`_detector_version`/
    `_source` (+ `_bbox_in_parent`) until a human accepts it. `SlotSpec`'s
    `subBox` capability gained `candidateBboxField`/
    `candidateBboxInParentField`/`candidateScoreField`/
    `candidateDetectorField`/`candidateDetectorVersionField`/
    `candidateSourceField`; `readSlot`/`SlotData.subBox.candidate` carry
    it generically. `/review`'s slot panel seeds `editedSlotBox` from the
    candidate when the main box is absent, so an unchanged Confirm still
    promotes it server-side (the frontend never computes the promotion
    itself); the candidate box renders dashed (`BboxCanvas`'s new
    `dashed` prop) with a "rejected candidate · confirm to accept" hint
    (the server's own rejection reason in its tooltip, never worded as a
    model verdict). `SlotCard` gets a matching amber "candidate" badge.
  - **Human-validated vs auto-confirmed.** `region_validated` is now
    human-only; `region_auto_confirmed` is the new "machine accepted,
    unreviewed" signal — `region_verified` keeps its prior, distinct
    meaning ("a verification pass ran"). `LifecycleCapability` gained
    `validatedField`/`autoConfirmedField`; `/review`, `SlotCard` and
    `CropMetaPanel` all render a validated/auto-confirmed badge instead
    of overloading `region_verified`.
  - **Text choice / VLM-invalid reason.** `region_text_choice`
    (`readers_agree`/`vlm_preferred`/`vlm_only`/`ocr_only`/`ocr_mode`/
    `vlm_invalid`/`no_valid_reading`/`human`) and
    `region_text_vlm_invalid` (`placeholder`/`no_reading`/`sequence`/
    `charset`/`too_short`/`too_long`/`format`) render next to the plate
    text on `/review` and `CropMetaPanel`, labeled through
    `regionVocabularyStore`'s new `text_choices`/`text_rules` (from `GET
{API_PREFIX}/regions/vocabulary`) — titlecased id placeholder until the
    backend serves real labels.
  - **Region-status filter.** `GET {API_PREFIX}/regions` gains a `status=`
    filter (400 on an unknown value); the `/clusters` plate gallery's
    `SlotGallery` renders it as a `<select>` sourced from
    `regionStatusesStore.list`, matching the existing Detector filter's
    served-vocabulary pattern.
  - `regionVocabularyStore` gained `rejectionReasonLabel` — a single
    lookup point for `region_rejection_reason` ids, currently a
    titlecase-id placeholder (no served label yet; `openprocessor` fix #29
    will add one) that deliberately never infers "wrong box" from
    `verify_rejected` alone (`verifier_no_verdict` means "needs human",
    not a model rejection — `region_bbox_correct === false` is the
    actual "model said wrong box" signal).
- Adopted OpenProcessor `main` 7254ec4 (cutover/dq-queues):
  - **Item confidence fields.** `class_confidence`/`class_confidence_source`
    (VLM high/medium/low mapped to 0.92/0.70/0.40 server-side, or the
    classifier's own score; null for a human label), `vlm_raw_class`,
    `vlm_class_attempted_at`/`vlm_class_empty_reason`, and
    `cluster_nearest_id`/`cluster_distance` now map onto `Crop`
    (`mapRawCrop`) and render on `CropCard`, the `/review` item panel and
    `CropMetaPanel` — a "Label confidence" row separate from the existing
    detector-score `confidence` row, "VLM said: …", and the empty reason
    when set.
  - **Per-tab review filters.** `GET {API_PREFIX}/review/tabs` now serves
    each tab's `filters`/`filter_defaults`. `reviewTabsVocabularyStore`
    gained `filtersFor`/`filterSupported`/`filterDefault`; `/review`'s
    filter bar renders only the controls the active tab's served list
    names, and the subject/max_rank toggle's "unset" label reads the
    served default (`Top ${n}` / "All ranks") instead of a hardcoded
    "Top 2". Unknown/absent (older backend) still shows every control.
  - **New-class proposals (DQ-M11).** `GET {API_PREFIX}/review/
new_class_proposals/summary` now serves `without_term`, `term_rules`,
    and splits terms into `top_terms` (actionable) and `flagged_terms`
    (`existing_class`/`generic_parent`/`non_object`). `/classes` renders
    flagged terms in a collapsed section with the served reason and no
    create action — `existing_class` gets a one-click map-to-class_id
    instead.
  - **Cluster purity (DQ-M2).** `purity` is now nearest-centroid geometry
    purity (`purity_basis: 'nearest_centroid'`, `purity_n` measured),
    independent of the new `label_purity`/`labelled_share`. `/clusters`
    cards show "purity NN% · n=NNN" with the rest in the chip tooltip;
    `/clusters/[id]`'s header gains a purity line it previously lacked
    entirely. `purity_tier` (the pure/mixed/noisy badge) is unchanged.
  - **Cluster moves on accept.** `acceptVlmForCrop`/`acceptAllVlmOnPage`
    (`/clusters/[id]`) now drop a crop from the grid — via the same
    `exclusionGuard` optimistic-removal pattern as a manual label move —
    when the accepted class differs from the current cluster's own,
    since labeling a crop now moves it into its class cluster
    server-side; a suggestion matching the cluster's own class still
    applies in place.
  - Verified already-correct with no source change: `/review/mismatches`'
    per-item `reason`, `/export` and `/train`'s generic 422-detail and
    preflight-check rendering (new `export_not_empty`/`export_generation`
    checks render with zero component changes).
- Adopted OpenProcessor `main` 1327181's naming-sweep waves W0/W1 (finding
  m9) and the small F4-F11 renames that shipped alongside it:
  - **W0 — `GET {API_PREFIX}/regions/vocabulary`**: a new `regionVocabularyStore`
    (`$stores/regionVocabulary.svelte`), loaded once from the root
    layout, serves this deployment's detector/segmenter/verifier
    vocabulary (`{detectors, region_sources, chain_actors}`, each
    `{id, label, role, filterable?}`). `ProvenanceChip.svelte` now reads
    a chain entry's label from the store and derives chip color from its
    served `role` via a new `paletteForRole` (`detectorRegistry.ts`) —
    an id the vocabulary doesn't know about renders verbatim with the
    neutral chip, same as before. `SlotGallery.svelte`'s plate-gallery
    Detector `<select>` now renders the vocabulary's `filterableDetectors`
    instead of a hardcoded `lpr_nanov11_640`/`sam3` option list.
  - **W0 — `GET {API_PREFIX}/review/tabs`**: a new `reviewTabsVocabularyStore`
    (`$stores/reviewTabsVocabulary.svelte`), also loaded once from the
    root layout, serves each review tab/preset's `{id, label,
description}`. `/review`'s tab bar and preset chips render the
    served label (falling back to the existing static label) and the
    served description as a `title` tooltip; the tab id `coco_blind_spots`
    is unchanged for now.
  - New e2e stub defaults for both endpoints in `e2e/conftest.py`
    (shapes lifted from the vendored OpenAPI description), plus a new
    e2e test (`test_plate_gallery_detector_filter.py`) proving the
    real browser-rendered plate-gallery detector filter reflects a
    served vocabulary rather than a hardcoded list.
- Adopted the OpenProcessor `main` findings pass (07cc061 — B2/M6/M8/V1
  region+VLM write integrity and undo), per
  docs/design/interactive-pass-2026-09-24.md §6:
  - **D-4** (`docs/design/curation_query_performance_audit.md`):
    `getClusters` forwards `representatives_offset`/`representatives_limit`
    as `offset`/`limit` to `GET {API_PREFIX}/clusters`. `/clusters`' card
    grid now requests representatives only for the currently-visible
    window (`clusterQuery`'s first `pageSize` cards), backfilling later
    windows via a new `loadMoreRepresentatives()` as the operator scrolls,
    merged into the already-loaded cards in place rather than a second
    full re-fetch.
  - **M7**: `pollAutoLabelJob`'s `expectedJobId` path now polls the new
    `GET {API_PREFIX}/pipeline/auto_label/status/{job_id}`
    (`getAutoLabelJobStatus`) directly instead of the old client-side
    job-id-match workaround over the single "current job"
    `.../status` slot. `/dashboard`'s `runVlm` now also passes its
    started job's id (previously only `/clusters/[id]` did).
  - **M11**: `StrategyBar`'s summary chip gained a `fallbackReason` prop
    rendering `sort_fallback_reason` next to `sort_applied` (both
    collapsed and expanded views); `/review` wires it from both
    `GET {API_PREFIX}/review/{tab}` and `.../locate` (the latter's result
    type gained `sort_fallback_reason`), replacing the old page-level
    banner. Verified `/settings`' existing generic `ApiError` 422 handling
    already surfaces a 0-coverage-sort rejection with no code change
    needed.
  - **M6**: region-write undo — `POST {API_PREFIX}/crops/{id}/region/undo`,
    `.../crops/region/undo_batch`, and VLM-dismiss undo
    (`.../crops/{id}/vlm_dismiss/undo`, closing V1) via new
    `undoCropRegion`/`undoCropRegionBatch`/`undoVlmDismiss` in `api.ts`.
    `UndoEntry` gained a `kind: 'label' | 'region' | 'vlm_dismiss'` tag
    (default `'label'`) so the existing `undoStore` ring buffer — kept as
    ONE stack, not a separate one per kind, so Z reverses whatever
    actually happened last chronologically across kinds — routes `Z` to
    the matching backend undo route. Wired into `/review`'s
    confirm/reject/false-positive/box-edit slot actions (Z now works on
    the Plates/slot tabs, previously a no-op there), `/clusters/[id]`'s
    Reject-VLM (`rejectVlmForCrop`), and the plate gallery's bulk status
    change / bbox edit (new `Z` binding + `undoLastPlateAction()` in
    `plateGalleryController.svelte.ts`). Renamed the Plates tab's
    misleading "← back"/"← to go back" copy (implied Back itself undid
    the write) to "step back" throughout — Back only re-queues the crop
    locally; Z undoes the server write.
  - **m21**: verified live that `cluster_distance` is mid-rollout on this
    deployment's index — `GET /clusters`' `representatives` now carry it
    for some cards (background residual runs since 07cc061 landed) but
    `GET /crops` still returns `null` for other clusters' members not yet
    re-run. `/clusters/[id]`'s cut-line (`cutLineIndex`) already degrades
    correctly either way (stops at index 0 when `cluster_is_core` is
    `null`), so no frontend change was needed — it renders correctly for
    a cluster as soon as that cluster's own residual run backfills the
    field.
  - Contracts re-synced to OpenProcessor main 07cc061 (from f934f32).
- Component-mount vitest support: `vite.config.ts` sets
  `resolve.conditions: ['browser']` under `VITEST` so Svelte 5's
  `mount`/`unmount`/`flushSync` work under jsdom (no new dependency).
  Real DOM-rendering tests replace source-text scans for `CropCard`,
  `DatasetStats`, `SlotCard`, `TrainForm`'s GPU picker and
  `StrategyBar`'s applied-sort summary — each verified to fail against a
  mutated copy of the component it covers before being counted as
  passing (docs/design/test-audit-2026-09-24.md recommendation 7).
- `src/lib/review/reviewController.svelte.ts`: assign/discard/skip/undo
  extracted out of `/review`'s `+page.svelte` into a testable controller
  (`plateGalleryController.svelte.ts`'s existing factory-function
  pattern). Closes three mutations the audit found surviving under the
  old source-scan-only suite — a failed assign/discard no longer
  restoring the item, and skip no longer advancing the cursor — with a
  98.63% Stryker mutation score on the new controller (the one
  remaining survivor is a documented equivalent mutant).
- `src/lib/clusters/clusterController.svelte.ts`: the same extraction for
  `/clusters/[id]`, the other labeling hot path — assignClassToSelected,
  the drop-on-class handler, accept/reject-VLM, accept-all-VLM-on-page,
  discard (D), undoLast (Z), ignore/undoIgnore (X/U), and moveCropIds,
  plus the `excludedCropIds` stale-fetch-race guard (now `ExclusionGuard`,
  via `createExclusionGuard()`) all move out of `+page.svelte`, which is
  now wiring and markup only. Closes the surviving `if (false)` mutant in
  the old undo path (docs/design/test-audit-2026-09-24.md §2.2, old
  `+page.svelte:634`) — both the in-place-replace and prepend branches of
  `undoLast`'s `if (cropPager.items.some(...))` are now exercised
  directly — with a 98.70% Stryker mutation score on the new controller
  (three documented equivalent-mutant survivors, same convention as
  `reviewController.svelte.ts`'s).
- `src/lib/testing/sourceScan.ts`: whitespace-insensitive,
  brace-balanced helpers (`normalize`, `extractFunction`,
  `extractBalanced`) for the remaining source-scan tests that guard
  real wiring a mount/controller test can't reach (recommendation 8).
  `clusterMoveRace.test.ts`'s wiring scans now use these instead of
  `\n {2}\}`-anchored regexes that broke on a harmless prettier re-wrap.
- A stubbed-backend end-to-end suite, `e2e/` (pytest + Playwright, its own
  gitignored `e2e/.venv/`), gated in CI by a new `e2e-stubbed` job and
  runnable locally via `npm run test:e2e`
  (docs/design/test-audit-2026-09-24.md recommendations 5 and 8).
  - `e2e/conftest.py`'s `Stub` fixture routes every `{API_PREFIX}` request
    and fails **closed**: an unstubbed path gets `501` and is recorded, a
    request to the retired `/curation/` prefix is intercepted and flagged, and
    every test asserts at teardown that nothing went unhandled and that
    at least one stub actually fired — the old
    `scripts/playwright_*.py` stubs' catch-all `return ok({})` (and their
    silent fall-through to the real network for anything outside
    `**/curation/**`) is what let them drift to `/curation/**` for weeks
    without anyone noticing.
  - `e2e/fixtures/wire.py`'s `make_item()` builds a full item-wire payload
    from `contracts/openprocessor/json/item_wire.json` instead of a
    hand-copied shape, and fails at import time if the vendored contract
    grows a key it doesn't cover.
  - Ported and rewrote `scripts/playwright_labeling_flow.py`,
    `playwright_assist_scope.py`, `playwright_curation_settings.py` and
    `playwright_tier2_profile.py` as real pytest tests under
    `e2e/stubbed/` (expectations updated against the current UI; the
    scripts are deleted).
  - New coverage that had none before: `/review` Enter-assign then
    Z-undo (`test_review_actions.py`), `/clusters/[id]` D-discard
    (`test_cluster_discard.py`), the `/train` GPU picker rendering
    `GET /train/gpus` (`test_train_gpus.py`), and `/dashboard` rendering
    "Stats unavailable" on a stats-fetch error without a page error
    (`test_dashboard_stats.py`).
- A `py_compile` gate for `scripts/*.py` and `e2e/**/*.py`, in both
  pre-commit (`py-compile` local hook) and CI (`check` job) — nothing
  previously compiled or linted the runbook scripts, so a syntax error
  shipped silently.
- `/classes`'s Proposals section now bulk-resolves a VLM new-class term
  against OpenProcessor's `POST {API_PREFIX}/review/new_class_proposals/
resolve` (backend `main` `af3a580`): "Create class & assign" / "Map to
  existing" resolve **every** pending item proposing the term, not just
  the summary's capped `sample_crop_ids` sample. Each action dry-runs
  first (`?dry_run=true`) and shows the real served `matched` count in a
  confirm dialog before writing; a 400 (unknown class), 409 (duplicate
  class name) or 422 (missing/conflicting `class_id`/`create`, or over
  the backend's per-label match cap) shows the server's detail text in
  the failure toast. `resolveNewClassProposal`/`undoLabelBatch`
  (`api.ts`) are typed from the vendored OpenAPI contract.
  - Replaces the old sample-only flow (`addClass` + `bulkLabel` on
    `sample_crop_ids`), deleted from this page — `bulkLabel` itself
    stays (still used by `/clusters` drag-and-drop assignment).
  - A successful resolve records its `updated_ids` via
    `undoStore.recordWrites()`, the same ring-buffer `Z`-undo
    `bulkLabel`/`moveCropsToCluster` already use elsewhere, rather than
    a new bulk-undo affordance — `POST {API_PREFIX}/crops/label/
undo_batch` restores each crop to its prior `vlm_new_class_pending`
    proposal state.

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

- `/bakeoff` profile picker, backed by OpenProcessor's B1 `BakeoffProfile`
  (`GET /bakeoff/profiles`). The chosen profile scopes the baseline model
  list (`/bakeoff/baseline_models?profile=`) and is sent on
  `POST /bakeoff/run`. "Deployment default" omits it. `BakeoffModelSpec`
  now matches main's model spec: per-model `profile`,
  `primary_*`/`secondary_*` coarse-stage fields replacing
  `vehicle_weights`, and the `onnxruntime`/`coreml` backends. Trained
  contenders are labeled "trained here" instead of a deployment-specific
  corpus string.

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

- OpenProcessor 1327181 naming sweep, F6/F9: `DatasetStats.in_progress`'s
  `sam_drain_total_unfinished` field is renamed
  `region_drain_total_unfinished` (`DatasetStats.svelte`'s in-flight-
  pipeline panel, ETA/drain-rate math, and copy text updated); `GET
{API_PREFIX}/crops`'s `?hdd_source=` param was removed outright — every
  frontend site that sent it (`CropFilter.source`, `/review`'s source
  filter and its diverse-selection scope's term filters) now sends
  `?source=` instead.
- The pre-push hook also runs the stubbed e2e suite (`e2e-pre-push`), so a
  failing e2e test can't be pushed. It adds about a minute per push.
- The crop detail panel reads the source image's metadata and sibling
  crops from `GET /crops/{id}/context`, which the backend added to end the
  route collision on `/crops/{id}/image`. `getCropImage` is now
  `getCropContext`. `/crops/{id}/image` serves the image itself again.

- `clusterMoveRace.test.ts`'s `describe('wiring: ... actually uses the
exclusion set', ...)` block (6 source-scan tests regexing
  `excludedCropIds.add`/`.delete` call sites across
  `+page.svelte`) is deleted — that logic moved into
  `clusterController.svelte.ts`'s `ExclusionGuard`/`handleClassDrop`/
  `moveCropIds`/`ignoreSelected`/`undoIgnore`/`discardSelected`/
  `undoLast`, now covered behaviorally by `clusterController.test.ts`
  (see that file's `describe` blocks for the 1:1 replacement mapping,
  documented inline in `clusterMoveRace.test.ts`). `logicMovesW1.test.ts`
  (3 more source-scan tests, over the same discard/reject-VLM/
  bulk-label-undo logic that also moved into the controller) is deleted
  outright for the same reason. `clusterMoveRace.test.ts`'s other block
  (`gridGroups`/`onGroupFinalize` derivation, unrelated to this
  extraction) is untouched.
- Trimmed/deleted several source-scan tests now superseded by real
  mount or controller tests, or that never failed for an actual
  regression (test-audit-2026-09-24.md T2/P2-2): the
  `regionRouteScan.test.ts` `REGION_BASE === '/regions'`
  constant-reassertion, `plateGalleryController.statusConstants
.test.ts`'s "today's values match the pre-C4b literals" block, and
  `TrainForm.test.ts`'s render-shape checks now covered by
  `TrainForm.gpuPicker.test.ts`'s mount test. `stripComments` moved
  from `apiPrefixScan.test.ts` to `$lib/testing/sourceScan.ts` as the
  one shared implementation.
- `stryker.config.json` gets an `ignorePatterns` excluding `e2e/` —
  its gitignored Python venv's `lib64` symlink was crashing Stryker's
  sandbox copy with `EISDIR`, making `npm run test:mutation` unrunnable.
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

- **One Z now undoes one action, however many crops it touched**, not
  one crop at a time. `UndoStore`'s `UndoEntry` is now `{ crop_ids:
string[]; at: number }`; `recordWrites(updatedIds)` pushes ONE entry
  per confirmed write (skipped when the write touched no crop), so a
  bulk label, a move, or a new-class-proposal resolve over N crops is
  one undo, not N. `undoLast()` routes a single-crop entry through
  `POST {API_PREFIX}/crops/{id}/label/undo` (`undoCropLabel`) as before
  and a multi-crop entry through the new
  `POST {API_PREFIX}/crops/label/undo_batch` (`undoLabelBatch` in
  `api.ts`), toasting "Reverted N." plus nothing-to-undo/conflict counts
  when non-zero; a 409 on either route means nothing left to undo
  (no re-push), a transport/5xx failure re-pushes the whole entry so Z
  stays retryable. `clusters/+page.svelte` (search mode),
  `clusters/[id]/+page.svelte` (grid replace/re-insert, releasing
  `excludedCropIds` per restored crop), and `review/+page.svelte`
  (re-inserting every restored item at the cursor) all now render the
  array `undoLast()` returns instead of a single `Crop | null`.
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

- `/settings` decides which axes get a control from the server's
  per-entry `settable` flag on `/methods`, not from a hardcoded
  `SETTINGS_AXES.kind`. The table now carries only labels, buckets and
  blurbs, so it can't drift from what the backend honors. The store's
  client-side "advisory axis" guard is gone too: OpenProcessor rejects a
  non-settable axis with a 422, and that detail is surfaced.

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

### Removed

- W0 finding m9: `builtinDetectors.ts`'s hardcoded per-model-id
  `labels`/`palettes`/`prefixes` tables and `detectorRegistry.ts`'s
  `labelForDetector`/`paletteForDetector` — both a private-model-id →
  label/color naming table the backend now serves via `GET
{API_PREFIX}/regions/vocabulary`. `builtinDetectors.ts` keeps only the
  `mutedTagPattern` (outcome/tag business logic, not a naming table);
  `detectorRegistry.ts` keeps only the display-only role→palette map
  (`paletteForRole`) and `isMutedTag`.
- `src/lib/annotations/regionWireContract.test.ts`, replaced by the
  contract tests.

- The dashboard's per-run detection-profile picker, along with
  `detection_profile` on `startAutoLabel` and the
  `isDetectionProfileAvailable` gate. OpenProcessor confirmed that no
  auto-label stage runs region detection: it's the detection worker's
  startup config, so the picker was a silent no-op, and main now rejects
  the param with a 422. The scope bar is gated on the `prompt_pack` axis
  alone and shows a chosen pack in its collapsed summary.

- The dashboard's "Snapshot op\_\* indexes" button. It was a placeholder that
  only showed a "coming in v1.1" toast, and no backend route exists for
  it. The API-health banner and tooltip no longer name `openprocessor`.

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

### Fixed

- FRONTEND fixes from the 2026-09-24 interactive-pass follow-up
  (`docs/design/interactive-pass-2026-09-24.md` §6):
  - **M1** — the `/review` Dismissed panel sent `sort=recent`, which
    400s (`GET {API_PREFIX}/crops`'s `sort` is a closed field list; the
    default is `updated_at:desc`). Fixed to request `updated_at:desc`;
    a failed load now shows an inline error state instead of falling
    through to "No dismissed crops."
  - **M2, M12** — opening `/review`'s Details disclosure used to
    collapse the crop/plate image to 0px (a `flex-1` image container
    shrinking to make room for the growing Details content). The image
    now sits in a `shrink-0` panel with a 300px floor height; everything
    below it (metadata, slot fields, Details) scrolls in its own
    `overflow-y-auto` region instead.
  - **B2 (frontend half)** — confirming a plate/region box that the
    operator didn't touch used to always `PUT region` with the same
    box, which the backend treats as a human-drawn box and uses to
    overwrite `region_detector`/`region_score`, destroying detector
    provenance. `confirmSlot()` now compares the edited box against the
    seeded (served) box and, when unchanged and the deployment serves a
    `confirm_status` (`GET {API_PREFIX}/regions/statuses`), sends a
    status-only `PATCH region_meta` instead. A box that actually
    changed still goes through `PUT region` (`frame: 'parent'`).
  - **M13** — the dashboard's "Export Dataset (YOLO)" quick action fired
    `POST {API_PREFIX}/export/yolo` with no confirmation and toasted
    "Export job started", but that endpoint is synchronous (verified
    against openprocessor's `export_yolo` handler — it awaits the full
    export pipeline before returning; there is no `job_id`). It now
    opens a confirm dialog and renders the real returned
    `export_dir`/`dataset_sha`/`split_counts`/`finished_at` fields (or
    the error) once the request resolves. `ExportResult` (`types.ts`)
    now matches the served shape.
  - **m6** — the dashboard's class-balance legend hardcoded "green ≥500
    · orange 100–499 · red <100", contradicting the served
    `block_below`/`warn_below` thresholds the bar colors are actually
    computed from. Now rendered from `legacyStats.thresholds`.
  - **M14** — `/train`'s cohort preview used to render once, after
    every class group (~4,700px below a chip clicked near the top),
    which looked like the click did nothing. It now renders inline
    under the class group whose chip was clicked.
  - **m11** — the global `` ` ``/`~` shortcut-overlay toggle only
    worked on pages that had registered at least one shortcut of their
    own (`/review`, `/clusters/[id]`); `keyboardStore`'s window listener
    installed lazily on the first `register()` call, so `/clusters`,
    `/classes` and `/dashboard` never got one. It now installs at
    construction. `Shift+`` `` (`~`) also normalizes to `"shift+~"`,
which the old check (`combo === 'shift+\``) never matched — fixed
    to match on the physical key too.
  - **m12** — added a shared `trapFocus` action
    (`src/lib/actions/trapFocus.ts`) that Tab-wraps focus inside a
    dialog and restores it to the trigger on close, wired into every
    modal named in the pass doc: the `/review` class picker, the
    `/classes` merge dialog, the `/settings` shared-default confirm,
    the dashboard Run-VLM/export confirms and the `AutoLabelPanel`
    recluster confirm (new, see p2 below), `/export`'s progress and
    freeze-holdout dialogs, `/clusters/[id]`'s move picker, the Add
    Class modal and the crop detail modal. Previously Tab could walk
    out of any of these into the page behind it.
  - **p8** — the shortcut overlay printed the raw combo string
    (`arrowright`) instead of a glyph. Added
    `src/lib/keyboardDisplay.ts`'s `formatShortcutKey()` (→ ← ↑ ↓,
    title-cased modifiers) and wired it into `ShortcutOverlay.svelte`.
  - **p6** — the crop detail modal's "Labeled at" ISO timestamp had no
    whitespace to wrap on and got visually clipped inside the
    fixed-width metadata panel. Added `break-all` to that value.
  - **p2** — `AutoLabelPanel`'s "Recluster now" fired with no
    confirmation (Run VLM on `/clusters/[id]` already gained one in the
    earlier merge pass fix). It now opens a confirm dialog first.
  - **m16** — `/export`'s served per-class `adequacy` tier (`block`/
    `warn`/`ok`) wasn't rendered anywhere; added an Adequacy column.
    The `by_source` HDD distribution fetch also failed silently on a
    503/non-JSON response — it now shows an inline "Dataset totals
    unavailable" notice instead of just rendering nothing.
  - **m15** — the `class_registry.json`/`data.yaml`/`manifest.json`
    download buttons gated on the shared export-job-status slot
    (`exportState?.status === 'success'`), which stayed "success" with
    no _current_ frozen multi-class (`yolo`) export on disk and 404ed.
    Added `hasCurrentMulticlassExport()` (`exportDatasetRows.ts`),
    which reads the served `GET {API_PREFIX}/export/datasets` list, and
    gated the three buttons on that instead.
  - **m10** — `/settings`' sort-axis blurb claimed the pinned default
    "REPLACES each tab's own tuned default", which is backwards — a
    tab with its own tuned default (Uncertainty, Model Disagreements,
    COCO Blind Spots, Plates) keeps it regardless of this setting; only
    All and New Class Proposals fall back to the pinned sort. Fixed the
    copy, and every settable axis's `<select>` now marks a
    zero-`field_coverage` option "· no coverage yet" (shared
    `hasFieldCoverage()`, same predicate `StrategyBar` uses), plus an
    inline warning when the currently-selected/effective option has no
    coverage.
  - **m7** — `/clusters`' purity-legend chips hardcoded "≥80% / ≥60% /
    <60%", contradicting the served `purity_thresholds`
    (`pure_min: 0.85`, `mixed_min: 0.6`). `GET {API_PREFIX}/clusters`'
    `purity_thresholds` is now surfaced on `PaginatedResponse` and the
    legend renders the real percentages.
  - **m9 (frontend half)** — the plate gallery's bulk confirm/reject/
    mark-false-positive buttons read `PLATE_CONFIRM_STATE`/etc. from
    the static `licensePlateSlot` profile only, unlike the review tab's
    same buttons, which already prefer the served
    `GET {API_PREFIX}/regions/statuses`. `plateGalleryController.svelte.ts`'s
    exports are now functions that check `regionStatusesStore` first
    and fall back to the profile literal. The plate detector filter
    list and `FP_PLATE_CLUSTER_ID` stay hardcoded — no backend endpoint
    serves either today (needs backend data).
  - **m20** — the embedding plot's point-count caption always read
    "{n} points", even when `{API_PREFIX}/methods`' `viz_projection`
    overlay reports a larger `field_coverage_total` (e.g. 116 of a
    422-crop pool). Added `embeddingCoverageSuffix()`
    (`$lib/embeddingPlot.ts`) and a `coveragePoolTotal` prop so it now
    reads "116 of 422 projected — Rebuild to include the rest".
  - **p7** — the MATCH/cluster-origin chips on `/clusters` search
    results sat directly on the crop image with no backdrop, unreadable
    against busy content on a tall/portrait crop. Added a top gradient
    scrim behind the chip row in `CropResultGrid.svelte`.
  - **m2** — after Z (undo), the `/review` Details "Reason" row read the
    frontend-invented string "restored by undo" — the backend never
    serves that. The queue controller now caches each item's real
    served `reason` at removal time and restores it on undo, falling
    back to `null` (rendered "—") when no cache entry exists (e.g. the
    undo didn't originate from this controller's own remove).
  - **m1** — a name-only proposal (`proposed_class_name` set,
    `proposed_class_id: null` — e.g. a non-registry term like
    "motorcycle") rendered in the same yellow confirmable style as a
    real proposal, though Enter opens the class picker instead of
    confirming it. It now renders dimmed with a "(hint only)" label.
  - **m5** — Reject (`D` on a slot tab) cleared the box immediately with
    no chance to record why, even when the served reject status has
    `wants_reason: true`. It now prompts for a reason first (checked
    against the served `regionStatusesStore` vocabulary) and persists
    it via `patchSlotMeta` alongside the box clear.
  - **m31** — the active quick-filter preset chip on `/review`'s All tab
    wasn't reflected in the URL, so it couldn't be bookmarked or
    shared, and reloading always silently dropped back to plain All.
    `reviewDeepLink()` now also parses a validated `?preset=`, and
    `togglePreset` keeps it in sync. Relabeled the "Quick filter" row
    "Queue:" — Primary · low-conf returns more rows (374) than plain
    All (114), so calling it a narrowing filter was backwards.
  - **p9** — at 1920×1080 the `/review` crop thumbnail rendered at its
    tiny natural size (~134px) inside a ~900px-tall panel, because
    `max-h-full max-w-full` only ever shrinks an image, never upscales
    one smaller than its container. Switched to `h-full w-full` (still
    `object-contain`), which fills the panel at any viewport.
  - **m14** — `/classes`' "Added" column parsed the date-only `added_at`
    ("2026-04-29") through `new Date(...).toLocaleDateString()`, which
    reads it as UTC midnight and rolls it back a day in any timezone
    behind UTC (4/28/2026). Added `formatDateOnly()` (`$lib/formatDate.ts`),
    which reads the Y-M-D components directly with no `Date`/timezone
    conversion for a bare date string.
  - **m23** — `/train`'s `<ClassSubsetPicker>` ("Classes to train: All
    84 classes · N validated crops") rendered unconditionally even for a
    single-class export, whose actual submission already forces
    `include_classes: null, single_cls: true` regardless of any
    selection there — so the summary named a registry-wide count with
    no bearing on what would train. Replaced with a plain note when
    `singleClassExport` is true.
  - **m27** — the dashboard recluster panel's blurb said the residual
    stage uses "AHC" (the live method is IVF, with AHC only as a refine
    step) and that it "runs the VLM over remaining unvalidated crops",
    though `run_vlm` defaults off. Fixed both claims.
  - **m29** — the dashboard's stats badge stayed green "LIVE" even
    while the panel below showed "Stats unavailable" (it only tracked
    SSE transport connectivity, not payload validity) — now reads
    "degraded" (amber) whenever the last frame was an error envelope.
    The banner also dumped the raw served exception in full; added
    `summarizeStatsError()` (`$lib/datasetStats.ts`) to cap it to a
    headline, with the full string still available via `title`.
  - **p4** — the VLM source badge suffixed an unconfirmed label with a
    bare "?" straight into the badge text ("Labeled by the VLM?"),
    which read as a literal question. `sourceBadge()` now returns an
    `unvalidated` flag instead of baking "?" into the text; the crop
    card puts the real explanation ("— not yet validated") in the
    title tooltip. (Long source-label truncation, the other half of
    this finding, needs a served `short_label` — not fixed here.)
  - **p5** — `batchPlateStatus()` always sent `region_verified: null`
    when the caller didn't pass `plateVerified` (true of every bulk
    reject/false-positive call from the plate gallery) — a field the
    backend ignores. The key is now omitted entirely unless the caller
    actually sets it.
  - **m26** — every page load fired `/health` and `/classes` 3x each
    (2 and 1 respectively aborted), verified live via
    `healthStore.acquire()`/`classesStore.acquire()` tracing: the root
    layout mounts/releases/remounts during app boot (an SPA-boot
    quirk), and `#release()` tore down polling (aborting the in-flight
    request) synchronously on every cycle, even when immediately
    followed by a re-acquire. Both stores now defer the actual teardown
    by one macrotask, cancelled by a re-acquire before it fires — a
    burst of release-then-reacquire now coalesces into a no-op.
  - **p1** — a 422 validation-error toast/modal message showed the raw
    pydantic sentence verbatim ("String should match pattern
    '^[a-z0-9_]+$'"), no field context. `errorDetail()`'s array branch
    now runs each `{loc, msg}` entry through `formatValidationEntry()`,
    which prefixes the real field name from `loc` ("name: must match
    pattern ^[a-z0-9_]+$") and reworks that one common phrasing — still
    entirely the server's own field/regex, nothing invented.
  - **p10** — CLAUDE.md's routes table still described `/` as a
    reachable "legacy" stats/recent-crops/Gemma-run page; it's a bare
    307 redirect to `/dashboard` today (`src/routes/+page.ts`, merged
    2026-09), and the top-bar logo/"Dashboard" link both point at
    `/dashboard` directly, not `/`. Removed the stale table row and
    fixed the routes-section intro sentence.
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

- The stubbed Playwright scripts intercepted `/curation/**`. The app calls
  `/curation/**`, so their stubs never matched. They now stub `/curation`.
  `playwright_curation_settings.py` also expects the server's `settable`
  flag and the current `/settings` copy, and it passes against the built
  container. `playwright_smoke.py` finds a cluster through
  `/curation/clusters`; the `/clusters/stats/op_vehicles` route it used
  was removed.

- `/review`'s `Enter` no longer confirms an unrelated class on a
  `vlm_new_class_pending` item. `resolveConfirmClassId` used to fall back
  to the crop's current class when there was no proposal, and for these
  items the backend now reports `proposed_class_id: null` precisely
  because that class is unrelated. `Enter` opens the class picker there
  instead.

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

- The accept-VLM-suggestion flow on `/clusters/[id]` (card chip, `G`,
  `Shift+Enter`, meta-panel row) could never fire, because no field ever
  populated the suggestion. `mapRawCrop` now maps `vlm_proposed_class_id`
  / `vlm_proposed_class_name` and the categorical `vlm_confidence`. The
  meta panel's VLM-confidence row, which read an unmapped field through a
  cast, now renders too.

- `/settings`'s prompt-pack blurb understated its effect. The shared
  default also drives the always-on background VLM labeler
  (`/vlm/label_batch`), not only auto-label runs.

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

- The embedding plot's projection rebuild is no longer fire-and-forget. It
  polls `GET /viz/projection/status`, shows progress, offers Cancel
  (`POST /viz/projection/cancel`) and reloads the plot when the job
  completes, or shows an error if it fails. It also picks up a rebuild
  already running when the page opens. Once a projection exists there's
  now a Rebuild button; before this, a projection could only be built
  once and never refreshed as new crops arrived.

- `/clusters/[id]` no longer opens an SSE subscription on candidate
  clusters. They have no class, and the backend filters events on exact
  `class_id`, so that stream could never deliver anything.
- Corrected the `startAutoLabel` doc comment claiming
  `detection_profile`/`prompt_pack` were confirmed live. `main` silently
  dropped them; per-run support is landing backend-side.

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
- `/clusters` and `/clusters/[id]` frontend findings from
  `docs/design/interactive-pass-2026-09-24.md` §6 FRONTEND:
  - **M5**: Z did not undo a Move (M) — `moveCropIds`
    (`clusterController.svelte.ts`) now records the server's own
    `updated_ids` as one undo entry, same as `bulkLabel`/discard, so Z
    restores a moved crop through the existing label undo/undo_batch
    route.
  - **M3**: the license_plate plate gallery only ever rendered
    region-cluster cards — when the only bucket was the permanent
    false-positive one, every other plate was unreachable (233 of 234,
    live), with no "unclustered" entry and no flat-grid fallback. Added
    a "Browse all plates" entry point (`viewingAllPlates` on
    `plateGalleryController.svelte.ts`, `openAllPlates()`) that opens
    the flat gallery with no `region_cluster_id` filter. Also deleted
    the stray literal `gallery.platePager.items` template text leaking
    into the counter strip (`SlotGallery.svelte`).
  - **M4**: the synthetic license_plate inventory card on `/clusters`
    only rendered in about 1 of 8 loads (a load-order race against
    `classesStore`'s own fetch), showed an invented "noisy 0%" purity
    badge and "· 0%" dominant-pct, and hid the real class-kind cluster
    sharing its id. `loadLicensePlateCard` now builds a fully-typed
    `Cluster` (no `as Cluster` cast masking missing fields), re-runs
    whenever `classesStore.classes` changes rather than once after the
    first load, and the card is marked `isSlotCard` so the grid renders
    no purity badge and no dominant-pct for it, keys it independently
    of a same-id real cluster (`slot-${id}` vs `id`), and no longer
    filters that real cluster out of the grid.
  - **M7 (frontend half)**: Run VLM on a cluster could read the
    _previous_ auto-label job's already-terminal status in the same
    tick it started a new one (the status endpoint has one slot for
    "the current job", not one per job), toasting a stale "0 crops (0
    updated)". `pollAutoLabelJob` (`api.ts`) now takes an optional
    `expectedJobId` and skips any status whose `job_id` doesn't match,
    bounded by a 5-minute timeout; `runVlm` passes the id the `POST
/vlm/label_cluster/{id}` response just returned. Added a
    `window.confirm` before starting the run — it's a no-undo
    background sweep over the whole cluster, unlike the one-keystroke
    label/move/discard actions Z already reverses instantly.
  - Invalid move target: entering a negative cluster id in the move
    picker used to be rejected by a client-side rule before the
    server's own 400 could ever show. Deleted that rule (only a
    genuinely non-numeric entry is still caught client-side); the
    server's 400 detail now surfaces via the existing
    `Move failed: …` toast.
  - m22: `/clusters?class=license_plate` (a name-form deep link) was
    silently ignored — `classFilter` only ever parsed a numeric id. It
    now falls back to a case-insensitive name lookup against the loaded
    class registry.
  - m18: the item-text search's empty state ("No crops matched that
    text.") rendered at the same time as the server's 400 detail,
    reading as two contradictory messages. The empty-state copy is now
    gated on there being no error.
