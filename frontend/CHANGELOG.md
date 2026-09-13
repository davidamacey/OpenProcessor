# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

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

### Fixed

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
