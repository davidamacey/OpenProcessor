# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

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
